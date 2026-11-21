"""Shared episode and inference state for collection schedulers."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from civ5rl.core.contracts import Episode, ScenarioSpec, WorkerState
from civ5rl.protocols.diagnostic_metrics import decision_metrics

if TYPE_CHECKING:
    from torch import Tensor

    from civ5rl.core.model import Policy
    from civ5rl.core.observation import ObservationEncoder


WORKER_TIMEOUT_SECONDS = 300


class EpisodeBuilder:
    """Accumulate one episode while preserving the learner's dictionary format."""

    def __init__(
        self,
        spec: ScenarioSpec,
        variant,
        reuse_encoded,
        hidden_size,
        temperature=1.0,
        *,
        record_hidden=True,
    ):
        self.reuse_encoded = reuse_encoded
        self.hidden_size = hidden_size
        self.episode: Episode = {
            "policy_temperature": temperature,
            "spec": spec,
            "observations": [],
            "actions": [],
            "chosen": [],
            "old_logp": [],
            "old_log_probs": [],
            "old_value": [],
            "decision_metrics": [],
        }
        if record_hidden:
            self.episode["old_hidden"] = []
        if reuse_encoded:
            self.episode.update(encoded=[], encoded_variant=variant)

    def record_step(self, state: WorkerState, encoded, chosen, logp, probabilities, value, old_hidden):
        ep = self.episode
        action = state["actions"][chosen]
        ep["observations"].append(state["observation"])
        if self.reuse_encoded:
            ep["encoded"].append(encoded)
        ep["actions"].append(state["actions"])
        ep["chosen"].append(chosen)
        ep["old_logp"].append(float(logp[chosen]))
        ep["old_log_probs"].append(logp.copy())
        ep["decision_metrics"].append(decision_metrics(state["actions"], chosen, probabilities))
        ep["old_value"].append(float(value))
        if "old_hidden" in ep:
            ep["old_hidden"].append(
                None
                if old_hidden is None or len(ep["chosen"]) == 1
                else old_hidden.reshape(1, 1, self.hidden_size).copy()
            )
        return action

    def finalize(self, state: WorkerState):
        self.episode["result"] = state["result"]
        self.episode["joint_commands"] = state["joint_commands"]
        return self.episode


@dataclass(kw_only=True)
class CollectionSlot:
    """State specific to one live environment and its two policies."""

    state: WorkerState | None = None
    builder: EpisodeBuilder | None = None
    encoder: ObservationEncoder | None = None
    opp_encoder: ObservationEncoder | None = None
    rng: random.Random | None = None
    opp_rng: random.Random | None = None
    opponent_id: str | None = None
    generation: int | None = None
    completed_at: float | None = None


@dataclass
class CollectionWave:
    """Live slots, recurrent state, and pending synchronous workers."""

    wave_id: int
    slots: Sequence[CollectionSlot]
    hidden: Tensor | None
    pending: set[int] = field(default_factory=set)
    decisions: int = 0


class CollectionScheduler(Protocol):
    supports_diagnostics: bool

    def worker_setup(self, reply_size: int): ...

    def collect(
        self,
        collector,
        policy: Policy,
        stage,
        next_episode: int,
        monitor,
        fixed_specs,
        deadline,
        max_decisions: int | None,
        episode_length_hint: float | None,
    ) -> tuple[list[Episode], int, dict]: ...
