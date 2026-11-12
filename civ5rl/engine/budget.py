"""Operational active-time limits, independent of the training algorithm."""

import math
from pathlib import Path
from time import monotonic, perf_counter
from datetime import UTC, datetime

from civ5rl.provenance.common import read_json, write_json


def budget_base(record):
    kind, config = record["kind"], record["config"]
    if kind == "rl01.ppo_after_target_bc":
        return float(config["experiment"]["active_seconds"])
    if kind in {"rl01.cp_imitation_target_scale", "rl01.cp_imitation_target_more_data", "rl01.cp_imitation_target_c2"}:
        return float(config["training_seconds"])
    raise ValueError(f"Run kind {kind!r} does not support operational budget extension")


def read_extension(directory, record):
    path = Path(directory) / "active-budget.json"
    value = read_json(path) if path.exists() else {}
    if value and (value.get("schema") != 1 or value.get("run_id") != record["run_id"]
                  or value.get("base_active_seconds") != budget_base(record)
                  or type(value.get("additional_active_seconds")) is not int
                  or value["additional_active_seconds"] < 0):
        raise ValueError("Invalid recorded active-budget extension")
    return value


class ActiveBudget:
    def __init__(self, directory, state=None, *, record=None):
        self.directory = Path(directory)
        self.record = record or read_json(self.directory / "run.json")
        self.base = budget_base(self.record)
        self.state = state
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
                write_json(self.directory / "budget-applied.json", {
                    "schema": 1, "run_id": self.record["run_id"], "revision": revision,
                    "effective_limit_seconds": effective, "applied_at": datetime.now(UTC).isoformat(),
                })
            self.next_check = now + 1.0
        if self.state is not None:
            self.state.update(active_seconds_limit=self.effective, budget_revision_applied=self.revision)
        return self.effective


def resolve_deadline(deadline):
    """Resolve an absolute perf_counter deadline, optionally supplied by callback."""
    return deadline() if callable(deadline) else math.inf if deadline is None else deadline


def deadline_reached(deadline):
    return perf_counter() >= resolve_deadline(deadline)
