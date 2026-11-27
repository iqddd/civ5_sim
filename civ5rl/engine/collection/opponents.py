"""Deterministic mixed CP + frozen self-play opponent pool for PPO collection."""

from __future__ import annotations

import random
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from civ5rl.core.model import Policy
from civ5rl.core.preview import load_parent_model, model_extension
from civ5rl.core.spatial import StaticSpatialGraph
from civ5rl.provenance.common import file_sha256, seed_for
from civ5rl.provenance.identity import policy_hash


U0_CHECKPOINT = "checkpoints/reference-u0.pt"


def u0_checkpoint(config):
    """C2PA uses its native C2P parent as the frozen match reference."""
    extension = model_extension(config)
    return U0_CHECKPOINT if extension and extension.adds_action_context else "checkpoints/initial.pt"


def self_play_config(config: dict[str, Any]) -> dict[str, Any] | None:
    """Return validated self_play settings, or None when disabled."""
    settings = config.get("self_play")
    if not settings or not settings.get("enabled", True):
        return None
    cp_share = float(settings.get("cp_share", 0.5))
    cpv2_share = float(settings.get("cpv2_share", 0.0))
    current_share = float(settings.get("current_share", 0.3))
    parent_share = float(settings.get("parent_share", 0.1))
    archive_share = float(settings.get("archive_share", 0.1))
    shares = (cp_share, cpv2_share, current_share, parent_share, archive_share)
    total = sum(shares)
    if not all(0 <= x <= 1 for x in shares) or abs(total - 1.0) > 1e-9:
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
        "cpv2_share": cpv2_share,
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
    rng = random.Random(seed_for(seed, int(episode_id), "opponent_pool", namespace=settings["namespace"]))
    u = rng.random()
    if u < settings["cp_share"]:
        return "cp", None
    u -= settings["cp_share"]
    if u < settings["cpv2_share"]:
        return "cpv2", None
    u -= settings["cpv2_share"]
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
    policy_hash_value: str | Callable[[], str],
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
    if any(item["id"] == snapshot_id for item in archive):
        return None
    if callable(policy_hash_value):
        policy_hash_value = policy_hash_value()
    if any(item.get("policy_hash") == policy_hash_value for item in archive):
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
    checkpoint: Path | str | dict[str, Any] | None,
    config: dict[str, Any],
    device: str | torch.device,
    *,
    live_policy: Policy | None = None,
    native_parent: bool = False,
) -> Policy:
    """Load a frozen inference-only opponent policy with one bounded 4-row spatial graph."""
    device_str = str(device)
    if live_policy is None:
        saved = checkpoint if isinstance(checkpoint, dict) else torch.load(checkpoint, weights_only=False, map_location="cpu")
        extension = model_extension(config)
        variant = saved.get("variant") or config.get("model", {}).get("variant", "C")
        if extension:
            if native_parent and variant != extension.parent:
                raise ValueError("Frozen u0 must use the native parent architecture")
            if not native_parent:
                variant = extension.child
        with torch.random.fork_rng(devices=[]):
            policy = Policy(variant)
        load_parent_model(policy, saved, config)
        policy = policy.to(device_str)
    else:
        with torch.random.fork_rng(devices=[]):
            policy = Policy(live_policy.variant).to(device_str)
        policy.load_state_dict(live_policy.state_dict())
    perf = config.get("performance", {})
    policy.spatial_cell_padding = perf.get("spatial_cell_padding", 0)
    amp = device_str.startswith("cuda") and config.get("numerics", {}).get("amp", False)
    # Opponent moves never enter behavior_replay/precheck: block size 4 eliminates
    # 32-row padding waste while giving StaticSpatialGraph one fixed padded shape.
    use_static_graph = amp and perf.get("compile", False) and policy.spatial_cell_padding > 0
    policy.enable_amp(
        amp,
        spatial_batch_size=4 if use_static_graph else 0,
    )
    if perf.get("inverse_routing", False):
        policy.enable_inverse_routing()
    if perf.get("optimized_routing", False):
        policy.enable_inverse_routing()
        policy.enable_action_routing()
        policy.enable_maximum_routing()
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    if device_str.startswith("cuda") and perf.get("compile", False):
        policy.enable_compile(
            dynamic=perf.get("compile_dynamic", True),
            recompile_limit=perf.get("compile_recompile_limit", 32),
        )
        if use_static_graph:
            policy._spatial_runner = StaticSpatialGraph(
                policy.spatial_runner, policy.spatial_batch_size, policy.spatial_cell_padding
            )
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
        reference_path = u0_checkpoint(config)
        u0_path = (
            Path(parent_checkpoint)
            if parent_checkpoint is not None
            else self.run_dir / reference_path
        )
        self.native_parent = reference_path == U0_CHECKPOINT
        if self.native_parent and file_sha256(u0_path) != config["experiment"]["parent_sha256"]:
            raise ValueError("Frozen u0 hash differs from the native parent checkpoint")
        self.u0_policy = _build_collection_policy(
            u0_path, config, self.device, native_parent=self.native_parent,
        )
        self._archive_runner: Policy | None = None
        self._archive_runner_params: list[torch.Tensor] = []
        self._archive_weights: dict[str, list[torch.Tensor]] = {}
        self._active_archive_id: str | None = None

    def add_archive_snapshot(self, entry: dict[str, Any], policy: Policy) -> None:
        """Own a detached copy of the learner just written at the checkpoint boundary."""
        snap_id = str(entry["id"])
        if snap_id in self._archive_weights:
            return
        if self._archive_runner is None:
            runner = _build_collection_policy(None, self.config, self.device, live_policy=policy)
            self._set_archive_runner(snap_id, runner)
            return
        weights = dict(policy.named_parameters())
        self._archive_weights[snap_id] = [
            weights[name].detach().to(device=self.device, dtype=param.dtype, copy=True)
            for name, param in self._archive_runner.named_parameters()
        ]

    def _set_archive_runner(self, snap_id, runner):
        self._archive_runner = runner
        self._archive_runner_params = list(runner.parameters())
        self._archive_weights[snap_id] = [p.detach().clone() for p in self._archive_runner_params]
        self._active_archive_id = snap_id

    def sync_archive(self, archive: list[dict[str, Any]]) -> None:
        """Load any newly archived snapshot weights onto device."""
        for entry in archive:
            snap_id = str(entry["id"])
            if snap_id in self._archive_weights:
                continue
            path = Path(entry["path"])
            if not path.is_absolute():
                path = self.run_dir / path
            saved = torch.load(path, map_location="cpu", weights_only=False)
            state_dict = saved["model"]
            if self.native_parent and (
                saved["variant"] != self.config["model"]["variant"]
                or policy_hash(state_dict) != entry.get("policy_hash")
            ):
                raise ValueError(f"Archive {snap_id} differs from the recorded C2PA snapshot")
            if self._archive_runner is None:
                self._set_archive_runner(snap_id, _build_collection_policy(saved, self.config, self.device))
            else:
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
