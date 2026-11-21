"""Shared experiment failure recording and active-time accounting."""

from __future__ import annotations

import traceback
from collections.abc import Callable, Iterator, Mapping, MutableMapping
from contextlib import AbstractContextManager, contextmanager, nullcontext
from pathlib import Path
from time import perf_counter
from typing import Any

from civ5rl.engine.runtime.budget import ActiveBudget
from civ5rl.engine.runtime.control import RunLock
from civ5rl.provenance.common import write_json


@contextmanager
def record_experiment_errors(
    out: Path,
    *,
    phase: str | None,
    on_error_report: Callable[[], None] | None = None,
    passthrough: tuple[type[BaseException], ...] = (),
    error_message_only: bool = False,
) -> Iterator[None]:
    """Persist a technical failure inside the runner's acquired lock."""
    try:
        yield
    except passthrough:
        raise
    except BaseException as error:
        if not error_message_only:
            (out / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        status = {"status": "technical_error"}
        if phase is not None:
            status["phase"] = phase
        if error_message_only:
            status["error"] = f"{type(error).__name__}: {error}"
        write_json(out / "status.json", status)
        if on_error_report is not None:
            on_error_report()
        raise


@contextmanager
def experiment_run_guard(
    out: Path,
    *,
    phase: str | None,
    on_error_report: Callable[[], None] | None = None,
    preflight: Callable[[], None] | None = None,
    session: Callable[[], AbstractContextManager[object]] | None = None,
    passthrough: tuple[type[BaseException], ...] = (),
) -> Iterator[None]:
    """Run a protected phase after acquiring its lock and checking admission.

    Lock/session entry and preflight errors retain the existing status. A runner
    with independently handled error types can re-raise them through passthrough.
    """
    with RunLock(out), nullcontext() if session is None else session():
        if preflight is not None:
            preflight()
        with record_experiment_errors(
            out,
            phase=phase,
            on_error_report=on_error_report,
            passthrough=passthrough,
        ):
            yield


class ActiveSessionClock:
    """Training clock with a restored offset and separately timed exclusions."""

    def __init__(self, offset: float = 0.0, *, timer: Callable[[], float] = perf_counter) -> None:
        self._timer = timer
        self.started = timer()
        self.excluded = 0.0
        self.offset = offset

    def seconds(self) -> float:
        return self.offset + self._timer() - self.started - self.excluded

    @contextmanager
    def pause(self) -> Iterator[None]:
        start = self._timer()
        try:
            yield
        finally:
            self.excluded += self._timer() - start


class PersistedActiveSessionClock:
    """Accumulate active time in the run state and honor budget revisions."""

    def __init__(
        self,
        state: MutableMapping[str, Any],
        out: Path,
        *,
        timer: Callable[[], float] = perf_counter,
    ) -> None:
        self.state = state
        self.out = out
        self._timer = timer
        self.started: float | None = timer()
        self.budget = ActiveBudget(out, state)

    def sync(self) -> None:
        if self.started is not None:
            now = self._timer()
            self.state["active_seconds"] += now - self.started
            self.started = now

    def progress(self) -> None:
        self.sync()
        write_json(self.out / "state.json", self.state)

    def pause(self) -> None:
        self.sync()
        self.started = None

    def resume(self) -> None:
        self.started = self._timer()

    def limit(self, config: Mapping[str, object] | None = None) -> float:
        # The historical argument remains accepted; operational limits come from the run record.
        limit: float = self.budget.limit()
        return limit

    def deadline(self, config: Mapping[str, object] | None = None) -> float:
        self.sync()
        return self._timer() + max(0, self.limit(config) - self.state["active_seconds"])
