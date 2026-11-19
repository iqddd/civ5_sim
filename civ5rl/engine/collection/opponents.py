"""Deterministic mixed CP + frozen self-play opponent pool for PPO collection."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from civ5rl.core.model import Policy
from civ5rl.provenance.common import seed_for


def self_play_config(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return validated self_play settings, or None when disabled."""
    settings = config.get("self_play")
    if not settings or not settings.get("enabled", True):
        return None
    cp_share = float(settings.get("cp_share", 0.5))
    current_share = float(settings.get("current_share", 0.3))
    parent_share = float(settings.get("parent_share", 0.1))
    archive_share = float(settings.get("archive_share", 0.1))
    total = cp_share + current_share + parent_share + archive_share
    if any(x < 0 for x in (cp_share, current_share, parent_share, archive_share)) or abs(total - 1.0) > 1e-9:
        raise ValueError(f"Self-play shares must be nonnegative and sum to 1.0 (got {total})")
    interval = int(settings.get("archive_interval_updates", 10))
    if interval < 1:
        raise ValueError("archive_interval_updates must be >= 1")
    temperature = float(settings.get("opponent_temperature", config.get("policy_temperature", 0.5)))
    if temperature <= 0:
        raise ValueError("opponent_temperature must be positive")
    return {
        "enabled": True,
        "cp_share": cp_share,
        "current_share": current_share,
        "parent_share": parent_share,
        "archive_share": archive_share,
        "archive_interval_updates": interval,
        "opponent_temperature": temperature,
        "namespace": str(
            settings.get(
                "namespace",
                config.get("experiment", {}).get("opponent_namespace")
                or f"{config.get('experiment', {}).get('training_namespace', 'selfplay')}_opponent",
            )
        ),
    }


def select_opponent(
    episode_id: int,
    config: dict[str, Any],
    archive: list[dict[str, Any]],
) -> tuple[str, str | None]:
    """Deterministically choose ('cp', None) or ('policy', opponent_id) for one episode.

    When the archive is empty (updates 1..archive_interval), archive_share is
    added to parent_share (u0 = u171), preserving the 50% CP / 50% neural split.
    """
    settings = self_play_config(config)
    if settings is None:
        return "cp", None
    seed = int(config.get("seed", 17))
    rng = random.Random(
        seed_for(seed, int(episode_id), "opponent_pool", namespace=settings["namespace"])
    )
    u = rng.random()
    if u < settings["cp_share"]:
        return "cp", None
    u -= settings["cp_share"]
    if u < settings["current_share"]:
        return "policy", "current"
    u -= settings["current_share"]
    if not archive or u < settings["parent_share"]:
        return "policy", "u0"
    entry = archive[rng.randrange(len(archive))]
    return "policy", str(entry["id"])


def maybe_archive_snapshot(
    state: dict[str, Any],
    config: dict[str, Any],
    *,
    checkpoint_relpath: str,
    policy_hash_value: str,
) -> dict[str, Any] | None:
    """Append a pool snapshot every archive_interval updates when >=1 Adam step occurred."""
    settings = self_play_config(config)
    if settings is None:
        return None
    updates = int(state.get("updates", 0))
    interval = settings["archive_interval_updates"]
    if updates <= 0 or updates % interval != 0:
        return None
    accepted = int(state.get("optimizer_steps", 0))
    last_archived_steps = int(state.get("self_play_last_archived_optimizer_steps", 0))
    if accepted <= last_archived_steps:
        return None
    archive = state.setdefault("self_play_archive", [])
    snapshot_id = f"archive:{updates}"
    if any(
        item["id"] == snapshot_id or item.get("policy_hash") == policy_hash_value
        for item in archive
    ):
        return None
    parent_hash = config.get("experiment", {}).get("parent_policy_hash")
    if parent_hash and policy_hash_value == parent_hash:
        return None
    entry = {
        "id": snapshot_id,
        "update": updates,
        "decisions": int(state.get("decisions", 0)),
        "path": checkpoint_relpath,
        "policy_hash": policy_hash_value,
        "optimizer_steps": accepted,
    }
    archive.append(entry)
    state["self_play_last_archived_optimizer_steps"] = accepted
    return entry


def _build_collection_policy(
    checkpoint_path: Path,
    config: dict[str, Any],
    device: str | torch.device,
) -> Policy:
    """Load a frozen opponent policy matching the collector's exact precision/compile mode."""
    from civ5rl.experiments.imitation_helpers import load_policy

    policy = load_policy(config, checkpoint_path, str(device))
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    return policy


class OpponentPool:
    """GPU-resident frozen opponent pool with zero recompilation across archive snapshots.

    - 'current' aliases the frozen learner policy (batched directly alongside learner turns).
    - 'u0' stays permanently resident in its own compiled Policy instance.
    - 'archive:*' snapshots cache their FP32 parameter tuples on device (~8.7 MiB each)
      and reuse one compiled archive runner via fused _foreach_copy_ on snapshot switch.
    """

    def __init__(
        self,
        config: dict[str, Any],
        device: str | torch.device,
        run_dir: Path | str,
        *,
        parent_checkpoint: Path | str | None = None,
    ):
        self.config = config
        self.device = torch.device(device)
        self.run_dir = Path(run_dir)
        settings = self_play_config(config)
        self.temperature = (
            settings["opponent_temperature"]
            if settings is not None
            else float(config.get("policy_temperature", 0.5))
        )
        u0_path = (
            Path(parent_checkpoint)
            if parent_checkpoint is not None
            else self.run_dir / "checkpoints/initial.pt"
        )
        self.u0_policy = _build_collection_policy(u0_path, config, self.device)
        self._archive_runner: Policy | None = None
        self._archive_runner_params: list[torch.Tensor] = []
        self._archive_weights: dict[str, list[torch.Tensor]] = {}
        self._active_archive_id: str | None = None

    def sync_archive(self, archive: list[dict[str, Any]]) -> None:
        """Load any newly archived snapshot weights onto device."""
        for entry in archive:
            snap_id = str(entry["id"])
            if snap_id in self._archive_weights:
                continue
            path = Path(entry["path"])
            if not path.is_absolute():
                path = self.run_dir / path
            if self._archive_runner is None:
                self._archive_runner = _build_collection_policy(path, self.config, self.device)
                self._archive_runner_params = list(self._archive_runner.parameters())
                self._archive_weights[snap_id] = [
                    p.detach().clone() for p in self._archive_runner_params
                ]
                self._active_archive_id = snap_id
            else:
                saved = torch.load(path, map_location="cpu", weights_only=False)
                state_dict = saved["model"]
                weights = [
                    state_dict[name].to(device=self.device, dtype=param.dtype)
                    for name, param in self._archive_runner.named_parameters()
                ]
                self._archive_weights[snap_id] = weights

    def model_for(self, opponent_id: str, current_policy: Policy) -> Policy:
        if opponent_id == "current":
            return current_policy
        if opponent_id == "u0":
            return self.u0_policy
        weights = self._archive_weights.get(opponent_id)
        if weights is None or self._archive_runner is None:
            raise KeyError(f"Unknown self-play opponent snapshot: {opponent_id!r}")
        if self._active_archive_id != opponent_id:
            with torch.no_grad():
                torch._foreach_copy_(self._archive_runner_params, weights)
            self._active_archive_id = opponent_id
        return self._archive_runner
