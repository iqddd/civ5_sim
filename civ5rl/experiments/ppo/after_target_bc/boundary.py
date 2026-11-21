"""Persist and recover committed rollout phase boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from time import perf_counter
from typing import Any

from civ5rl.engine.ppo.adam import (
    expected_counters,
    uses_grouped_adam,
)
from civ5rl.engine.runtime.control import TrainingCounters
from civ5rl.experiments._shared.lifecycle import PersistedActiveSessionClock
from civ5rl.protocols.ppo_after_target_bc import KIND
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import write_json
from civ5rl.provenance.jsonl import append_jsonl, read_jsonl


class Phase(str, Enum):
    COLLECT = "collect"
    CRITIC = "critic"
    PPO = "ppo"


def heal_interrupted_update(out: Path, state: dict) -> bool:
    """If pending_update is present and not flushed to updates.jsonl, flush it."""
    out = Path(out)
    pending = state.get("pending_update")
    if not pending:
        return False
    ledger = out / "updates.jsonl"
    committed = read_jsonl(ledger) if ledger.is_file() else []
    healed = False
    if not committed or committed[-1].get("rollout") != pending.get("rollout"):
        append_jsonl(ledger, pending)
        healed = True
    state.pop("pending_update", None)
    write_json(out / "state.json", state)
    return healed


@dataclass
class RolloutBoundary:
    checkpoint: str
    phase: Phase
    rollout_number: int
    pending_rollout: str | None = None
    decisions: int = 0
    optimizer_steps: int = 0

    def __post_init__(self):
        if isinstance(self.phase, str) and not isinstance(self.phase, Phase):
            self.phase = Phase(self.phase)

    @classmethod
    def from_state(cls, state: dict) -> RolloutBoundary:
        raw_phase = state.get("phase", Phase.COLLECT)
        phase = raw_phase if isinstance(raw_phase, Phase) else Phase(raw_phase)
        counters = TrainingCounters.from_dict(state, protocol=KIND)
        return cls(
            checkpoint=state.get("checkpoint", state.get("latest", "")),
            phase=phase,
            rollout_number=state.get("rollout_number", counters.rollouts_collected),
            pending_rollout=state.get("pending_rollout"),
            decisions=state.get("decisions", 0),
            optimizer_steps=state.get("optimizer_steps", counters.adam_steps_accepted),
        )

    def apply_to_state(self, state: dict) -> dict:
        state["phase"] = self.phase.value if isinstance(self.phase, Phase) else str(self.phase)
        state["latest"] = self.checkpoint
        state["rollouts"] = self.rollout_number
        state["rollouts_collected"] = self.rollout_number
        state["pending_rollout"] = self.pending_rollout
        state["decisions"] = self.decisions
        state["optimizer_steps"] = self.optimizer_steps
        state["adam_steps_accepted"] = self.optimizer_steps
        if "adam_steps_attempted" not in state or state["adam_steps_attempted"] < self.optimizer_steps:
            state["adam_steps_attempted"] = self.optimizer_steps
            state["attempted_steps"] = self.optimizer_steps
        if "rollouts_with_accepted_steps" not in state:
            state["rollouts_with_accepted_steps"] = state.get(
                "accepted_updates",
                0 if self.optimizer_steps == 0 else state.get("updates", 0),
            )
        if "accepted_updates" not in state:
            state["accepted_updates"] = state["rollouts_with_accepted_steps"]
        return state

    def to_state(self) -> dict:
        return self.apply_to_state({})

    def validate_sufficiency(self, out: Path) -> bool:
        """Validate that necessary files for the current phase exist on disk before proceeding."""
        out = Path(out)
        ckpt_path = out / self.checkpoint
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file does not exist: {ckpt_path}")
        if self.phase in (Phase.CRITIC, Phase.PPO):
            if not self.pending_rollout:
                raise ValueError(f"Pending rollout path is required for phase '{self.phase.value}'")
            rollout_path = out / self.pending_rollout
            if not rollout_path.exists():
                raise FileNotFoundError(f"Pending rollout artifact does not exist: {rollout_path}")
        if self.phase == Phase.COLLECT and self.rollout_number > 0:
            ledger = out / "updates.jsonl"
            if not ledger.is_file():
                raise FileNotFoundError(f"Update ledger does not exist: {ledger}")
            rows = read_jsonl(ledger)
            if not rows:
                raise ValueError(f"Update ledger is empty: {ledger}")
            last = rows[-1]
            if last.get("rollout") != self.rollout_number:
                raise ValueError(
                    f"Update ledger rollout ({last.get('rollout')}) disagrees with boundary ({self.rollout_number})"
                )
            if "total_decisions" in last and last["total_decisions"] != self.decisions:
                raise ValueError(
                    f"Update ledger decisions ({last['total_decisions']}) disagrees with boundary ({self.decisions})"
                )
        return True

    @classmethod
    def heal_interrupted_update(cls, out: Path, state: dict) -> bool:
        return heal_interrupted_update(out, state)


class ActiveClock(PersistedActiveSessionClock):
    """One elapsed-time counter; evaluations explicitly pause it."""

    def __init__(self, state: dict[str, Any], out: Path) -> None:
        super().__init__(state, out, timer=lambda: perf_counter())


def save_boundary(out, policy, optimizer, state, shuffle, config, identity, clock=None):
    if clock:
        clock.sync()
    if isinstance(state, RolloutBoundary):
        state_dict = state.to_state()
    else:
        state_dict = state
    counters = TrainingCounters.from_dict(state_dict, protocol=KIND)
    state_dict.update(counters.to_state_dict(protocol=KIND))
    state_dict.setdefault("updates", 0)
    if uses_grouped_adam(config):
        state_dict.pop("Adam_counter_total", None)
        state_dict["Adam_counters"] = expected_counters(state_dict)
    else:
        state_dict["Adam_counter_total"] = (
            int(state_dict.get("origin_Adam_counter", 0)) + counters.adam_steps_accepted
        )
    if isinstance(state_dict.get("phase"), Phase):
        state_dict["phase"] = state_dict["phase"].value
    checkpoint.save(
        out / state_dict["latest"],
        policy,
        optimizer,
        state_dict,
        shuffle,
        config,
        identity,
    )
    if clock:
        clock.sync()
    write_json(out / "state.json", state_dict)
