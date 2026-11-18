"""Operational active-time limits, independent of the training algorithm."""

import math
import warnings
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic, perf_counter, sleep
from typing import Protocol

from civ5rl.engine.runtime import is_int_at_least
from civ5rl.provenance.common import read_json, write_json


class ExecutionBudget(Protocol):
    def expired(self) -> bool: ...

    def remaining_seconds(self) -> float: ...


class DeadlineBudget:
    """An absolute perf_counter deadline, optionally refreshed by its owner."""

    def __init__(self, deadline: float | Callable[[], float]):
        self.deadline = deadline

    def remaining_seconds(self) -> float:
        return max(0.0, resolve_deadline(self.deadline) - perf_counter())

    def expired(self) -> bool:
        return deadline_reached(self.deadline)


def budget_base(record):
    kind, config = record["kind"], record["config"]
    if kind == "rl01.ppo_after_target_bc":
        return float(config["experiment"]["active_seconds"])
    if kind in {
        "rl01.cp_imitation_target_scale",
        "rl01.cp_imitation_target_more_data",
        "rl01.cp_imitation_target_c2",
    }:
        return float(config["training_seconds"])
    raise ValueError(f"Run kind {kind!r} does not support operational budget extension")


def read_extension(directory, record):
    path = Path(directory) / "active-budget.json"
    value = read_json(path) if path.exists() else {}
    if value and (
        value.get("schema") != 1
        or value.get("run_id") != record["run_id"]
        or value.get("base_active_seconds") != budget_base(record)
        or not is_int_at_least(value.get("additional_active_seconds"), 0)
    ):
        raise ValueError("Invalid recorded active-budget extension")
    return value


class ActiveBudget:
    def __init__(self, directory, state=None, *, record=None, elapsed=None):
        self.directory = Path(directory)
        self.record = record or read_json(self.directory / "run.json")
        self.base = budget_base(self.record)
        self.state = state
        started = perf_counter()
        self.elapsed = (
            elapsed if elapsed is not None else lambda: perf_counter() - started
        )
        self.revision = -1
        self.effective = self.base
        self.next_check = 0.0

    def limit(self):
        now = monotonic()
        if now >= self.next_check:
            extension = read_extension(self.directory, self.record)
            revision = len(extension.get("requests", []))
            effective = self.base + extension.get("additional_active_seconds", 0)
            if revision < self.revision or effective < self.effective:
                raise ValueError("Operational budget must not decrease")
            if revision != self.revision:
                self.revision, self.effective = revision, effective
                write_json(
                    self.directory / "budget-applied.json",
                    {
                        "schema": 1,
                        "run_id": self.record["run_id"],
                        "revision": revision,
                        "effective_limit_seconds": effective,
                        "applied_at": datetime.now(UTC).isoformat(),
                    },
                )
            self.next_check = now + 1.0
        if self.state is not None:
            self.state.update(
                active_seconds_limit=self.effective,
                budget_revision_applied=self.revision,
            )
        return self.effective

    def remaining_seconds(self) -> float:
        return max(0.0, self.limit() - self.elapsed())

    def expired(self) -> bool:
        return self.elapsed() >= self.limit()


def resolve_deadline(deadline: ExecutionBudget | float | Callable[[], float] | None):
    """Resolve an absolute perf_counter deadline for bounded waits."""
    if deadline is None:
        return math.inf
    if callable(deadline):
        return deadline()
    if isinstance(deadline, DeadlineBudget):
        return resolve_deadline(deadline.deadline)
    if isinstance(deadline, (float, int)):
        return deadline
    return perf_counter() + deadline.remaining_seconds()


def deadline_reached(deadline):
    if (
        deadline is not None
        and not callable(deadline)
        and not isinstance(deadline, (float, int))
    ):
        return deadline.expired()
    return perf_counter() >= resolve_deadline(deadline)


class PhaseBudget:
    """Persistent phase time excludes downtime; a heartbeat bounds accounting loss."""

    def __init__(self, out, phase, limit, *, initial_seconds=0.0):
        self.path = Path(out) / "phase-budgets" / f"{phase}.json"
        self.base = (
            read_json(self.path)["seconds"] if self.path.exists() else initial_seconds
        )
        self.started = perf_counter()
        self.limit = limit
        self.lock = Lock()
        self.stop = Event()
        self.thread = None

    def elapsed(self):
        return self.base + perf_counter() - self.started

    def expired(self):
        return self.elapsed() >= self.limit

    def remaining_seconds(self) -> float:
        return max(0.0, self.limit - self.elapsed())

    def publish(self):
        """Retry short Windows reader/rename conflicts without resetting clocks."""
        with self.lock:
            for attempt in range(6):
                try:
                    write_json(
                        self.path,
                        dict(
                            seconds=self.elapsed(),
                            limit=self.limit,
                            crash_accounting_resolution_seconds=1,
                        ),
                    )
                    return
                except PermissionError:
                    if attempt == 5:
                        raise
                    sleep(0.01 * 2**attempt)

    def __enter__(self):
        self.publish()

        def record():
            while not self.stop.wait(1):
                try:
                    self.publish()
                except PermissionError:
                    warnings.warn(
                        "Budget heartbeat publication denied; retrying on the next tick",
                        RuntimeWarning,
                    )

        self.thread = Thread(target=record, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.publish()
