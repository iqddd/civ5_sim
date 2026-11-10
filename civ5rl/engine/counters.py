"""Canonical training counter definitions and per-protocol legacy JSON translation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any

_TARGET_BC_PROTOCOLS = frozenset({
    "rl01.ppo_after_target_bc",
    "ppo_after_target_bc",
})

_CONTROLLED_PPO_PROTOCOLS = frozenset({
    "rl01.ppo_controlled_updates",
    "ppo_controlled_updates",
    "rl01.ppo_scale_up",
    "ppo_scale_up",
    "rl01.ppo_target_scale",
    "ppo_target_scale",
})

_ORDINARY_PPO_PROTOCOLS = frozenset({
    "rl01.ppo_after_imitation",
    "ppo_after_imitation",
})

_BC_PROTOCOLS = frozenset({
    "bc",
    "imitation_base",
    "rl01.cp_imitation",
    "cp_imitation",
    "rl01.cp_imitation_own_states",
    "cp_imitation_own_states",
    "imitation_own_states",
    "rl01.cp_imitation_target_scale",
    "cp_imitation_target_scale",
    "imitation_target_scale",
    "rl01.cp_imitation_target_more_data",
    "cp_imitation_target_more_data",
    "imitation_target_more_data",
    "rl01.cp_imitation_target_c2",
    "cp_imitation_target_c2",
    "imitation_target_c2",
})


def _normalize_protocol(
    protocol: str | None, data: Mapping[str, Any] | None = None
) -> str | None:
    raw = protocol
    if raw is None and data is not None:
        kind = data.get("kind")
        if isinstance(kind, str):
            raw = kind
    if raw is None:
        return None
    if raw in _TARGET_BC_PROTOCOLS:
        return "ppo_after_target_bc"
    if raw in _CONTROLLED_PPO_PROTOCOLS:
        return "ppo_controlled"
    if raw in _ORDINARY_PPO_PROTOCOLS:
        return "ppo_after_imitation"
    if raw in _BC_PROTOCOLS or raw.startswith(("rl01.cp_imitation", "cp_imitation", "imitation")):
        return "bc"
    raise ValueError(f"Unsupported counter protocol: {protocol!r}")


def _first_present(
    data: Mapping[str, Any], keys: tuple[str, ...], default: Any = None
) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


@dataclass(frozen=True)
class TrainingCounters:
    rollouts_collected: int = 0
    rollouts_with_accepted_steps: int = 0
    adam_steps_attempted: int = 0
    adam_steps_accepted: int = 0

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None,
        *,
        protocol: str | None = None,
    ) -> TrainingCounters:
        """Translate canonical target names or protocol-specific legacy JSON keys into TrainingCounters."""
        if not data:
            return cls()

        mode = _normalize_protocol(protocol, data)

        if mode == "bc":
            asa = _first_present(
                data,
                (
                    "adam_steps_accepted",
                    "Adam_steps",
                    "steps_performed",
                    "optimizer_steps",
                    "total_Adam_steps",
                    "Adam_steps_total",
                    "accepted_optimizer_steps",
                    "optimizer_steps_total",
                    "adam_steps",
                ),
                default=0,
            )
            att = _first_present(
                data,
                (
                    "adam_steps_attempted",
                    "attempted_steps",
                    "total_attempted_steps",
                    "attempted_optimizer_steps",
                ),
                default=asa,
            )
            return cls(
                rollouts_collected=0,
                rollouts_with_accepted_steps=0,
                adam_steps_attempted=int(att) if att is not None else 0,
                adam_steps_accepted=int(asa) if asa is not None else 0,
            )

        if mode == "ppo_after_target_bc":
            rc = _first_present(
                data,
                (
                    "rollouts_collected",
                    "rollouts",
                    "rollout_number",
                    "rollout",
                    "updates",
                    "update",
                ),
                default=0,
            )
            asa = _first_present(
                data,
                (
                    "adam_steps_accepted",
                    "optimizer_steps",
                    "accepted_optimizer_steps",
                    "total_Adam_steps",
                    "Adam_steps",
                    "steps_performed",
                    "Adam_steps_total",
                    "optimizer_steps_total",
                ),
                default=0,
            )
            att = _first_present(
                data,
                (
                    "adam_steps_attempted",
                    "attempted_steps",
                    "total_attempted_steps",
                    "attempted_optimizer_steps",
                ),
                default=None,
            )
            if att is None:
                rejected = (
                    int(data.get("rejected_steps", 0) or 0)
                    if "rejected_steps" in data
                    else 0
                )
                att = int(asa or 0) + rejected
            rwa = _first_present(
                data,
                ("rollouts_with_accepted_steps", "accepted_updates"),
                default=None,
            )
            if rwa is None:
                if (
                    "optimizer_steps" in data
                    and data["optimizer_steps"] is not None
                    and int(data["optimizer_steps"]) == 0
                ):
                    rwa = 0
                else:
                    rwa = _first_present(data, ("updates", "update"), default=0)
            return cls(
                rollouts_collected=int(rc) if rc is not None else 0,
                rollouts_with_accepted_steps=int(rwa) if rwa is not None else 0,
                adam_steps_attempted=int(att) if att is not None else 0,
                adam_steps_accepted=int(asa) if asa is not None else 0,
            )

        if mode == "ppo_controlled":
            rc = _first_present(
                data,
                (
                    "rollouts_collected",
                    "rollouts",
                    "updates",
                    "rollout_index",
                    "rollout_number",
                    "rollout",
                    "update",
                ),
                default=0,
            )
            asa = _first_present(
                data,
                (
                    "adam_steps_accepted",
                    "optimizer_steps",
                    "accepted_optimizer_steps",
                    "total_Adam_steps",
                    "Adam_steps",
                    "steps_performed",
                    "Adam_steps_total",
                    "optimizer_steps_total",
                ),
                default=0,
            )
            att = _first_present(
                data,
                (
                    "adam_steps_attempted",
                    "attempted_steps",
                    "total_attempted_steps",
                    "attempted_optimizer_steps",
                ),
                default=None,
            )
            if att is None:
                rejected = (
                    int(data.get("rejected_steps", 0) or 0)
                    if "rejected_steps" in data
                    else 0
                )
                att = int(asa or 0) + rejected
            rwa = _first_present(
                data,
                ("rollouts_with_accepted_steps", "accepted_updates"),
                default=None,
            )
            if rwa is None:
                if (
                    "optimizer_steps" in data
                    and data["optimizer_steps"] is not None
                    and int(data["optimizer_steps"]) == 0
                ):
                    rwa = 0
                else:
                    rwa = _first_present(data, ("updates", "update"), default=0)
            return cls(
                rollouts_collected=int(rc) if rc is not None else 0,
                rollouts_with_accepted_steps=int(rwa) if rwa is not None else 0,
                adam_steps_attempted=int(att) if att is not None else 0,
                adam_steps_accepted=int(asa) if asa is not None else 0,
            )

        if mode == "ppo_after_imitation":
            rc = _first_present(
                data,
                (
                    "rollouts_collected",
                    "rollouts",
                    "rollout_number",
                    "rollout",
                    "updates",
                    "update",
                ),
                default=0,
            )
            rwa = _first_present(
                data,
                (
                    "rollouts_with_accepted_steps",
                    "accepted_updates",
                    "updates",
                    "update",
                ),
                default=0,
            )
            asa = _first_present(
                data,
                (
                    "adam_steps_accepted",
                    "optimizer_steps",
                    "accepted_optimizer_steps",
                    "total_Adam_steps",
                    "Adam_steps",
                    "steps_performed",
                    "Adam_steps_total",
                    "optimizer_steps_total",
                ),
                default=0,
            )
            att = _first_present(
                data,
                (
                    "adam_steps_attempted",
                    "attempted_steps",
                    "total_attempted_steps",
                    "attempted_optimizer_steps",
                ),
                default=asa,
            )
            return cls(
                rollouts_collected=int(rc) if rc is not None else 0,
                rollouts_with_accepted_steps=int(rwa) if rwa is not None else 0,
                adam_steps_attempted=int(att) if att is not None else 0,
                adam_steps_accepted=int(asa) if asa is not None else 0,
            )

        # Fallback when protocol is None and not specified in data["kind"]
        rc = _first_present(
            data,
            (
                "rollouts_collected",
                "rollouts",
                "rollout_index",
                "rollout_number",
                "rollout",
                "updates",
                "update",
            ),
            default=0,
        )
        rwa = _first_present(
            data,
            ("rollouts_with_accepted_steps", "accepted_updates"),
            default=None,
        )
        if rwa is None:
            if (
                "optimizer_steps" in data
                and data["optimizer_steps"] is not None
                and int(data["optimizer_steps"]) == 0
            ):
                rwa = 0
            else:
                rwa = _first_present(data, ("updates", "update"), default=0)

        asa = _first_present(
            data,
            (
                "adam_steps_accepted",
                "optimizer_steps",
                "Adam_steps",
                "total_Adam_steps",
                "steps_performed",
                "accepted_optimizer_steps",
                "Adam_steps_total",
                "optimizer_steps_total",
                "adam_steps",
            ),
            default=0,
        )
        att = _first_present(
            data,
            (
                "adam_steps_attempted",
                "attempted_steps",
                "total_attempted_steps",
                "attempted_optimizer_steps",
            ),
            default=None,
        )
        if att is None:
            if "rejected_steps" in data and data["rejected_steps"] is not None:
                att = int(asa or 0) + int(data["rejected_steps"])
            else:
                att = asa

        return cls(
            rollouts_collected=int(rc) if rc is not None else 0,
            rollouts_with_accepted_steps=int(rwa) if rwa is not None else 0,
            adam_steps_attempted=int(att) if att is not None else 0,
            adam_steps_accepted=int(asa) if asa is not None else 0,
        )

    def to_state_dict(
        self,
        *,
        protocol: str | None = None,
        include_legacy_bc: bool = False,
    ) -> dict[str, int]:
        """Dual-write canonical counter fields and protocol-appropriate legacy keys."""
        mode = _normalize_protocol(protocol)
        base: dict[str, int] = {
            "rollouts_collected": self.rollouts_collected,
            "rollouts_with_accepted_steps": self.rollouts_with_accepted_steps,
            "adam_steps_attempted": self.adam_steps_attempted,
            "adam_steps_accepted": self.adam_steps_accepted,
        }
        rejected = max(0, self.adam_steps_attempted - self.adam_steps_accepted)

        if mode == "bc" or (mode is None and include_legacy_bc):
            return {
                **base,
                "Adam_steps": self.adam_steps_accepted,
                "steps_performed": self.adam_steps_accepted,
                "optimizer_steps": self.adam_steps_accepted,
            }

        if mode == "ppo_after_target_bc":
            result = {
                **base,
                "rollouts": self.rollouts_collected,
                "accepted_updates": self.rollouts_with_accepted_steps,
                "attempted_steps": self.adam_steps_attempted,
                "optimizer_steps": self.adam_steps_accepted,
                "rejected_steps": rejected,
            }
            if include_legacy_bc:
                result["Adam_steps"] = self.adam_steps_accepted
                result["steps_performed"] = self.adam_steps_accepted
            return result

        if mode == "ppo_controlled":
            result = {
                **base,
                "rollouts": self.rollouts_collected,
                "updates": self.rollouts_collected,
                "accepted_updates": self.rollouts_with_accepted_steps,
                "attempted_steps": self.adam_steps_attempted,
                "optimizer_steps": self.adam_steps_accepted,
                "rejected_steps": rejected,
            }
            if include_legacy_bc:
                result["Adam_steps"] = self.adam_steps_accepted
                result["steps_performed"] = self.adam_steps_accepted
            return result

        if mode == "ppo_after_imitation":
            result = {
                **base,
                "updates": self.rollouts_with_accepted_steps,
                "optimizer_steps": self.adam_steps_accepted,
            }
            if include_legacy_bc:
                result["Adam_steps"] = self.adam_steps_accepted
                result["steps_performed"] = self.adam_steps_accepted
            return result

        return {
            **base,
            "rollouts": self.rollouts_collected,
            "updates": self.rollouts_with_accepted_steps,
            "accepted_updates": self.rollouts_with_accepted_steps,
            "attempted_steps": self.adam_steps_attempted,
            "optimizer_steps": self.adam_steps_accepted,
            "Adam_steps": self.adam_steps_accepted,
            "rejected_steps": rejected,
            "steps_performed": self.adam_steps_accepted,
        }

    def replace(self, **kwargs: Any) -> TrainingCounters:
        """Return a copy with specified fields updated."""
        return replace(self, **kwargs)
