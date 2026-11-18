"""Shared names and result types for replay diagnostics exposed by PPO."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

REPLAY_DIAGNOSTIC_KEYS = (
    "sampled_kl",
    "sampled_kl_free",
    "full_kl",
    "full_kl_free",
    "free_decisions",
    "all_decisions",
    "critic_mean",
    "critic_mse",
    "critic_explained_variance",
)
KL_DIAGNOSTIC_KEYS = ("full_kl", "full_kl_free", "sampled_kl", "sampled_kl_free")
STEP_REPLAY_DIAGNOSTIC_KEYS = REPLAY_DIAGNOSTIC_KEYS[:6]


@dataclass(frozen=True)
class StepReplay:
    """Whole-rollout replay measurement after an optimizer step or rollback."""

    acceptance_kl: float | None
    diagnostics: dict[str, Any] = field(default_factory=dict)
