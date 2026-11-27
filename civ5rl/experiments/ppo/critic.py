"""Value-only fitting primitives and behavior invariants."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, overload

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch.nn import functional as F

from civ5rl.core.model import batch_inputs
from civ5rl.engine.ppo.adam import expected_counters
from civ5rl.engine.ppo.core import encode_episode, episode_returns, raw_gae_advantages, targets
from civ5rl.engine.runtime.budget import DeadlineBudget, deadline_reached, resolve_deadline
from civ5rl.engine.runtime.sequence import DEFAULT_PREFIX_BLOCK_SIZE
from civ5rl.engine.runtime.transactions import atomic_legacy_torch_save, cpu_snapshot
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import seed_for
from civ5rl.provenance.identity import policy_hash as actor_hash

if TYPE_CHECKING:
    from civ5rl.core.contracts import Episode

Device = str | torch.device
FeatureBlock = tuple[int, int, torch.Tensor]
PolicyHash = Callable[[Mapping[str, torch.Tensor]], str]
BehaviorHash = Callable[[Sequence["Episode"]], str]


class ValuePolicy(Protocol):
    @property
    def value(self) -> torch.nn.Module: ...

    def parameters(self) -> Iterator[torch.nn.Parameter]: ...


class CriticStatistics(TypedDict):
    prediction_mean: float
    return_mean: float
    mse: float
    constant_return_mean_mse: float
    explained_variance: float | None


def critic_statistics(predictions: ArrayLike, returns: ArrayLike) -> CriticStatistics:
    predictions = np.asarray(predictions, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    residual = returns - predictions
    variance = returns.var()
    return {
        "prediction_mean": float(predictions.mean()),
        "return_mean": float(returns.mean()),
        "mse": float(np.mean(residual**2)),
        "constant_return_mean_mse": float(np.mean((returns - returns.mean()) ** 2)),
        "explained_variance": float(1 - residual.var() / variance) if variance > 0 else None,
    }


def cpv2_contact_mask(episodes):
    return np.repeat([ep["spec"]["opponent"] == "cpv2" and ep["spec"].get("placement") == "contact"
                      for ep in episodes], [len(ep["chosen"]) for ep in episodes])


def behavior_digest(episodes: Sequence[Episode]) -> str:
    """Hash sampled choices and their probabilities, independently of critic values."""
    result = hashlib.sha256()
    for episode in episodes:
        result.update(np.asarray(episode["chosen"], dtype=np.int64).tobytes())
        result.update(np.asarray(episode["old_logp"], dtype=np.float64).tobytes())
        for row in episode["old_log_probs"]:
            result.update(row.tobytes())
    return result.hexdigest()


@contextmanager
def value_only_grad_mode(policy: ValuePolicy, parameters=None) -> Iterator[None]:
    """Temporarily enable gradients on the value head and restore all flags."""
    saved = [(parameter, parameter.requires_grad) for parameter in policy.parameters()]
    try:
        for parameter, _ in saved:
            parameter.requires_grad_(False)
        for parameter in policy.value.parameters() if parameters is None else parameters:
            parameter.requires_grad_(True)
        yield
    finally:
        for parameter, requires_grad in saved:
            parameter.requires_grad_(requires_grad)


def _cached_blocks(
    episodes: Sequence[Episode], cache: torch.Tensor, batch_size: int
) -> Iterator[FeatureBlock]:
    offset = 0
    for episode_index, episode in enumerate(episodes):
        length = len(episode["old_value"])
        for start in range(0, length, batch_size):
            size = min(batch_size, length - start)
            yield episode_index, start, cache[offset : offset + size]
            offset += size


@overload
def refresh_episode_old_values(
    policy: ValuePolicy,
    episodes: Sequence[Episode],
    features_or_stream: torch.Tensor,
    device: Device,
    *,
    batch_size: int = 128,
    bulk_transfer: Literal[True],
    progress: Callable[[], None] | None = None,
    expected_positions: int | None = None,
) -> NDArray[np.floating]: ...


@overload
def refresh_episode_old_values(
    policy: ValuePolicy,
    episodes: Sequence[Episode],
    features_or_stream: torch.Tensor | Iterable[FeatureBlock],
    device: Device,
    *,
    batch_size: int = 128,
    bulk_transfer: Literal[False] = False,
    progress: Callable[[], None] | None = None,
    expected_positions: int | None = None,
) -> list[float]: ...


def refresh_episode_old_values(
    policy: ValuePolicy,
    episodes: Sequence[Episode],
    features_or_stream: torch.Tensor | Iterable[FeatureBlock],
    device: Device,
    *,
    batch_size: int = 128,
    bulk_transfer: bool = False,
    progress: Callable[[], None] | None = None,
    expected_positions: int | None = None,
) -> list[float] | NDArray[np.floating]:
    """Refresh stored values using the fit's original arithmetic and transfer order.

    Preparation uses episode-local blocks and returns values in traversal order.
    Adaptation passes its flat, device-resident cache, disables autocast, and
    performs one host transfer of the concatenated predictions.
    """
    if bulk_transfer:
        if not isinstance(features_or_stream, torch.Tensor):
            raise TypeError("Bulk value refresh requires a device-resident tensor")
        with torch.no_grad(), torch.autocast(torch.device(device).type, enabled=False):
            refreshed: NDArray[np.floating] = (
                torch.cat([_value_predictions(policy, part) for part in features_or_stream.split(batch_size)])
                .cpu()
                .numpy()
            )
        offset = 0
        for episode in episodes:
            size = len(episode["old_value"])
            episode["old_value"][:] = refreshed[offset : offset + size].tolist()
            offset += size
        if progress is not None:
            progress()
        return refreshed

    blocks = (
        _cached_blocks(episodes, features_or_stream, batch_size)
        if isinstance(features_or_stream, torch.Tensor)
        else features_or_stream
    )
    after_values: list[float] = []
    with torch.no_grad():
        for episode_index, start, features in blocks:
            values: list[float] = policy.value(features.to(device)).squeeze(-1).cpu().tolist()
            episodes[episode_index]["old_value"][start : start + len(values)] = values
            after_values.extend(values)
            if progress is not None:
                progress()
    if expected_positions is not None and len(after_values) != expected_positions:
        raise ValueError("Incomplete first-rollout old_value refresh")
    return after_values


def verify_critic_invariance(
    policy: torch.nn.Module,
    episodes: Sequence[Episode],
    orig_policy_hash: str,
    orig_behavior_hash: str,
    *,
    mode: Literal["preparation", "adaptation"] = "preparation",
    policy_hash_fn: PolicyHash | None = None,
    behavior_hash_fn: BehaviorHash = behavior_digest,
) -> None:
    if policy_hash_fn is None:
        policy_hash_fn = actor_hash
    if policy_hash_fn(policy.state_dict()) != orig_policy_hash:
        if mode == "adaptation":
            raise RuntimeError("Critic adaptation changed policy or recorded behavior")
        raise RuntimeError("Critic preparation changed encoder/GRU/policy parameters")
    if behavior_hash_fn(episodes) != orig_behavior_hash:
        if mode == "adaptation":
            raise RuntimeError("Critic adaptation changed policy or recorded behavior")
        raise RuntimeError("Critic preparation changed recorded behavior")


@torch.no_grad()
def recurrent_features(policy, episodes, device):
    """Yield bounded FP32 GRU features, preserving complete episode prefixes."""
    for episode_index, episode in enumerate(episodes):
        encoded = encode_episode(episode, policy.variant)
        hidden = None
        for start in range(0, len(encoded), 128):
            if policy.spatial_cuda_graphs:
                torch.compiler.cudagraph_mark_step_begin()
            part = encoded[start : start + 128]
            batch = batch_inputs(part, device, non_blocking=True, spatial_only=True)
            _, context = policy.encode_spatial(batch, return_cells=False)
            features, hidden = policy.sequence_memory(context, [len(part)], hidden)
            # Own the host data: compiled/CUDA runners may reuse their buffers.
            yield episode_index, start, features.detach().cpu().clone()


@torch.no_grad()
def batched_recurrent_features(policy, episodes, device):
    """Stream bounded, episode-indexed FP32 feature blocks on the target device."""
    action_context = policy.variant_spec.action_context
    block = DEFAULT_PREFIX_BLOCK_SIZE
    group_size = 8
    for first in range(0, len(episodes), group_size):
        encoded = [encode_episode(ep, policy.variant) for ep in episodes[first : first + group_size]]
        hidden = (
            torch.zeros((1, len(encoded), policy.hidden_size), device=device) if policy.is_recurrent else None
        )
        for start in range(0, max(map(len, encoded)), block):
            parts = [(i, rows[start : start + block]) for i, rows in enumerate(encoded) if start < len(rows)]
            lengths = [len(rows) for _, rows in parts]
            slots = None
            if hidden is not None and len(parts) != len(encoded):
                slots = torch.tensor([i for i, _ in parts], device=device)
            if policy.spatial_cuda_graphs:
                torch.compiler.cudagraph_mark_step_begin()
            batch = batch_inputs(
                [row for _, rows in parts for row in rows],
                device,
                non_blocking=True,
                spatial_only=not action_context,
            )
            cells, context = policy.encode_spatial(batch, return_cells=action_context)
            features, last = policy.sequence_memory(
                context,
                lengths,
                hidden if slots is None else hidden.index_select(1, slots),
            )
            if slots is None:
                hidden = last
            else:
                hidden.index_copy_(1, slots, last)
            if action_context:
                _, features = policy.heads(batch, cells, features, return_value_features=True)
            for (i, _), values in zip(parts, features.split(lengths)):
                yield first + i, start, values


def _value_predictions(policy, features):
    if features.shape[-1] != policy.hidden_size:
        return policy.value_from_features(features)
    return policy.value(features).squeeze(-1)


@dataclass(frozen=True)
class _CriticFit:
    cache: torch.Tensor
    cached: int
    processed: int
    steps: int
    completed_epochs: int
    fit_seconds: float


def _fit_cached_critic_epochs(fit_batch, cached, batch_size, epochs, fit_deadline):
    completed_epochs = 1
    for _ in range(1, epochs):
        for first in range(0, cached, batch_size):
            if deadline_reached(fit_deadline):
                return completed_epochs
            fit_batch(first, min(first + batch_size, cached))
        completed_epochs += 1
    return completed_epochs


def _fit_initial_critic(
    policy, episodes, settings, device, returns, fit_deadline, started, progress
) -> _CriticFit:
    with value_only_grad_mode(policy):
        temporary = torch.optim.Adam(
            policy.value.parameters(),
            lr=settings["lr"],
            betas=tuple(settings["betas"]),
            eps=settings["eps"],
            weight_decay=settings["weight_decay"],
            fused=str(device).startswith("cuda"),
        )
        cache = torch.empty((len(returns), policy.hidden_size), dtype=torch.float32)
        cached = processed = steps = completed_epochs = 0
        batch_size = settings["batch_size"]

        def fit_batch(first, stop):
            nonlocal processed, steps
            temporary.zero_grad(set_to_none=True)
            features = cache[first:stop].to(device)
            labels = torch.from_numpy(returns[first:stop]).to(device)
            with torch.autocast(torch.device(device).type, enabled=False):
                prediction = policy.value(features).squeeze(-1)
                loss = F.mse_loss(prediction, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite critic preparation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.value.parameters(),
                settings["max_gradient_norm"],
                error_if_nonfinite=True,
            )
            temporary.step()
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            processed += stop - first
            steps += 1
            if progress:
                progress()

        portions = iter(recurrent_features(policy, episodes, device))
        trained = 0
        while cached < len(returns) and not deadline_reached(fit_deadline):
            _, _, features = next(portions)
            cache[cached : cached + len(features)] = features
            cached += len(features)
            if progress:
                progress()
            while cached - trained >= batch_size and not deadline_reached(fit_deadline):
                fit_batch(trained, trained + batch_size)
                trained += batch_size
        if cached == len(returns):
            if trained < cached and not deadline_reached(fit_deadline):
                fit_batch(trained, cached)
                trained = cached
            if trained == cached:
                completed_epochs = _fit_cached_critic_epochs(
                    fit_batch, cached, batch_size, settings["epochs"], fit_deadline
                )
        fit_seconds = perf_counter() - started
    return _CriticFit(cache, cached, processed, steps, completed_epochs, fit_seconds)


def prepare_critic(
    policy,
    episodes,
    settings,
    device,
    *,
    gamma=1.0,
    gae_lambda=1.0,
    deadline=None,
    progress=None,
    preparation_started=None,
):
    """Fit only value parameters, then refresh old_value without rewriting behavior."""
    started = preparation_started if preparation_started is not None else perf_counter()
    fit_deadline = DeadlineBudget(
        lambda: min(started + settings["active_seconds"], resolve_deadline(deadline))
    )
    original_policy_hash = actor_hash(policy.state_dict())
    original_behavior = behavior_digest(episodes)
    returns = np.concatenate(episode_returns(episodes, gamma=gamma))
    before = critic_statistics(np.concatenate([ep["old_value"] for ep in episodes]), returns)
    fit = _fit_initial_critic(policy, episodes, settings, device, returns, fit_deadline, started, progress)
    # The full refresh follows the fitting time limit.
    refresh_started = perf_counter()
    features = fit.cache if fit.cached == len(returns) else recurrent_features(policy, episodes, device)
    after_values = refresh_episode_old_values(
        policy,
        episodes,
        features,
        device,
        progress=progress if progress else None,
        expected_positions=len(returns),
    )
    verify_critic_invariance(
        policy,
        episodes,
        original_policy_hash,
        original_behavior,
        policy_hash_fn=actor_hash,
        behavior_hash_fn=behavior_digest,
    )
    raw_advantages = np.concatenate(raw_gae_advantages(episodes, gamma=gamma, gae_lambda=gae_lambda))
    advantages = targets(episodes, gamma=gamma, gae_lambda=gae_lambda)
    return {
        "steps": fit.steps,
        "positions_processed": fit.processed,
        "positions_cached": fit.cached,
        "epochs_completed": fit.completed_epochs,
        "fit_seconds": fit.fit_seconds,
        "fit_time_overshoot_seconds": max(0, fit.fit_seconds - settings["active_seconds"]),
        "refresh_seconds": perf_counter() - refresh_started,
        "before": before,
        "after": critic_statistics(after_values, returns),
        "raw_advantage_std": float(np.std(raw_advantages)),
        "normalized_advantage_std": float(np.concatenate(advantages).std()),
        "behavior_parameters_unchanged": True,
        "recorded_behavior_unchanged": True,
    }


atomic_save = atomic_legacy_torch_save


def _restore_critic_progress(
    capsule_path,
    policy,
    optimizer,
    value_groups,
    identity,
    behavior,
    device,
    state,
    shuffle,
):
    progress = {"steps": 0, "cursor": 0, "fit_seconds": 0.0, "positions_processed": 0}
    if capsule_path.exists():
        capsule = torch.load(capsule_path, map_location="cpu", weights_only=False)
        if capsule["identity"] != identity or capsule["behavior"] != behavior:
            raise ValueError("Critic adaptation capsule belongs to another rollout/run")
        policy.value.load_state_dict(capsule["value"])
        if len(value_groups) == 2:
            policy.value_actions.load_state_dict(capsule["value_actions"])
        members = [(group, p) for group in value_groups for p in group["params"]]
        for (value_group, parameter), item in zip(members, capsule["adam"], strict=True):
            optimizer.state.pop(parameter, None)
            if item:
                optimizer.state[parameter] = {
                    name: value.to(
                        "cpu"
                        if name == "step" and not (value_group.get("fused") or value_group.get("capturable"))
                        else device
                    )
                    if isinstance(value, torch.Tensor)
                    else value
                    for name, value in item.items()
                }
        progress = capsule["progress"]
        checkpoint.restore_rng(capsule["rng"], shuffle)
        state["active_seconds"] = max(state["active_seconds"], capsule["active_seconds"])
    # Heartbeats may include interrupted extraction/compilation since the last batch.
    progress["fit_seconds"] = max(progress["fit_seconds"], state.get("critic_fit_seconds", 0.0))
    return progress


def _load_or_extract_critic_features(
    out,
    policy,
    episodes,
    device,
    returns,
    behavior,
    original_policy,
    features_for,
    shuffle,
    tick,
):
    cache_path = out / "critic-features.pt"
    width = policy.hidden_size + (2 * policy.hex_dim if getattr(policy, "value_actions", None) is not None else 0)
    if cache_path.exists():
        saved = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            saved["behavior"] != behavior
            or saved["policy_hash"] != original_policy
            or saved["features"].shape != (len(returns), width)
        ):
            raise ValueError("Critic feature cache differs from frozen rollout/encoder")
        features = saved["features"]
    else:
        features = torch.empty((len(returns), width), dtype=torch.float32, device=device)
        offsets = np.cumsum([0, *[len(ep["chosen"]) for ep in episodes]])
        cached = 0
        last_progress = perf_counter()
        extraction_rng = checkpoint.capture_rng(shuffle)
        try:
            for episode_index, start, part in features_for(policy, episodes, device):
                offset = int(offsets[episode_index]) + start
                features[offset : offset + len(part)].copy_(part, non_blocking=True)
                cached += len(part)
                # Keep budget recovery current without writing JSON for every block.
                now = perf_counter()
                if now - last_progress >= 1.0:
                    tick()
                    last_progress = now
        finally:
            checkpoint.restore_rng(extraction_rng, shuffle)
        if cached != len(returns):
            raise ValueError("Incomplete critic feature cache")
        atomic_save(
            {
                "behavior": behavior,
                "policy_hash": original_policy,
                "features": features.cpu(),
            },
            cache_path,
        )
    return features


def _fit_critic_batches(
    policy,
    optimizer,
    value_groups,
    features,
    labels,
    config,
    settings,
    device,
    progress,
    capsule_path,
    deadline,
    tick,
    commit,
):
    position_count = len(labels)
    optimizer.zero_grad(set_to_none=True)
    old_lrs = [group["lr"] for group in value_groups]
    for group in value_groups:
        group["lr"] = settings["lr"]
    parameters = [p for group in value_groups for p in group["params"]]
    batches = (position_count + settings["batch_size"] - 1) // settings["batch_size"]
    shuffle_epochs = bool(settings.get("shuffle", bool(config.get("self_play", {}).get("enabled"))))
    critic_namespace = config.get("experiment", {}).get("critic_namespace", "critic")
    base_seed = int(config.get("seed", 17))
    cached_epoch = None
    epoch_order = None
    try:
        while progress["cursor"] < batches * settings["epochs"]:
            tick(persist=False)
            if progress["fit_seconds"] >= settings["active_seconds"] or deadline_reached(deadline):
                break
            epoch_idx, batch_idx = divmod(progress["cursor"], batches)
            first = batch_idx * settings["batch_size"]
            stop = min(first + settings["batch_size"], position_count)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(torch.device(device).type, enabled=False):
                if shuffle_epochs:
                    if cached_epoch != epoch_idx:
                        permutation = list(range(position_count))
                        random.Random(
                            seed_for(
                                base_seed,
                                epoch_idx,
                                "critic_shuffle",
                                namespace=critic_namespace,
                            )
                        ).shuffle(permutation)
                        epoch_order = torch.tensor(permutation, device=device, dtype=torch.long)
                        cached_epoch = epoch_idx
                    indices = epoch_order[first:stop]
                    loss = F.mse_loss(_value_predictions(policy, features[indices]), labels[indices])
                else:
                    loss = F.mse_loss(
                        _value_predictions(policy, features[first:stop]),
                        labels[first:stop],
                    )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, settings["max_gradient_norm"])
            finite_loss, finite_norm = torch.isfinite(torch.stack((loss.detach(), norm))).tolist()
            if not finite_loss:
                raise FloatingPointError("Non-finite critic adaptation loss")
            if not finite_norm:
                raise FloatingPointError("Non-finite critic adaptation gradient norm")
            optimizer.step()
            progress["steps"] += 1
            progress["cursor"] += 1
            progress["positions_processed"] += stop - first
            commit()
        if not capsule_path.exists():
            commit()
    finally:
        for group, lr in zip(value_groups, old_lrs, strict=True):
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
    return batches


def _critic_horizon_statistics(episodes, refreshed, returns, advantages, statistics):
    remaining = np.concatenate([np.arange(len(ep["chosen"]) - 1, -1, -1) for ep in episodes])
    horizon_statistics = {}
    for label, mask in (
        ("0_99", remaining < 100),
        ("100_299", (remaining >= 100) & (remaining < 300)),
        ("300_plus", remaining >= 300),
    ):
        if mask.any():
            horizon_statistics[label] = {
                "positions": int(mask.sum()),
                **statistics(refreshed[mask], returns[mask]),
                "raw_advantage_mean": float(advantages[mask].mean()),
            }
    return horizon_statistics


def _critic_adaptation_result(
    config,
    episodes,
    progress,
    batches,
    refresh_started,
    before,
    refreshed,
    returns,
    gamma,
    statistics,
):
    advantages = np.concatenate(
        raw_gae_advantages(episodes, gamma=gamma, gae_lambda=config["ppo"]["gae_lambda"])
    )
    horizon_statistics = _critic_horizon_statistics(episodes, refreshed, returns, advantages, statistics)
    contact_stats = {}
    if config.get("experiment", {}).get("diagnostic_training_only"):
        contact = cpv2_contact_mask(episodes)
        contact_stats["cpv2_contact_after"] = statistics(refreshed[contact], returns[contact]) if contact.any() else None
    return {
        "steps": progress["steps"],
        "positions_processed": progress["positions_processed"],
        "positions_cached": len(returns),
        "epochs_completed": progress["cursor"] // batches,
        "fit_seconds": progress["fit_seconds"],
        "refresh_seconds": perf_counter() - refresh_started,
        "before": before,
        "after": statistics(refreshed, returns),
        **contact_stats,
        "raw_advantage_std": float(advantages.std()),
        "return_min": float(returns.min()),
        "return_max": float(returns.max()),
        "remaining_decisions": horizon_statistics,
        "behavior_parameters_unchanged": True,
        "recorded_behavior_unchanged": True,
    }


def adapt_critic(
    policy,
    optimizer,
    episodes,
    config,
    device,
    out,
    state,
    shuffle,
    identity,
    clock,
    deadline,
    *,
    features_for=batched_recurrent_features,
    behavior_hash=behavior_digest,
    statistics=critic_statistics,
    policy_hash=actor_hash,
):
    """Each completed batch commits only the value head, its Adam and RNG."""
    started = perf_counter()
    out = Path(out)
    settings = config["critic"]
    behavior = behavior_hash(episodes)
    original_policy = policy_hash(policy.state_dict())
    value_groups = [group for group in optimizer.param_groups if group["name"] in ("value", "value_actions")]
    parameters = [p for group in value_groups for p in group["params"]]
    capsule_path = out / "critic-adaptation.pt"
    progress = _restore_critic_progress(
        capsule_path,
        policy,
        optimizer,
        value_groups,
        identity,
        behavior,
        device,
        state,
        shuffle,
    )
    state["critic_progress"] = progress
    elapsed_before = progress["fit_seconds"]

    def tick(*, persist=True):
        progress["fit_seconds"] = elapsed_before + perf_counter() - started
        state["critic_fit_seconds"] = progress["fit_seconds"]
        state["Adam_counters"] = expected_counters(state)
        if persist:
            clock.progress()
        else:
            clock.sync()

    def commit():
        tick(persist=False)
        model_adam = cpu_snapshot(
            {
                "value": policy.value.state_dict(),
                **({"value_actions": policy.value_actions.state_dict()} if len(value_groups) == 2 else {}),
                "adam": [optimizer.state.get(parameter, {}) for parameter in parameters],
            }
        )
        atomic_save(
            {
                "identity": identity,
                "behavior": behavior,
                **model_adam,
                "progress": dict(progress),
                "rng": checkpoint.capture_rng(shuffle),
                "active_seconds": state["active_seconds"],
            },
            capsule_path,
        )
        clock.progress()

    fit_finished = False
    try:
        gamma = float(config.get("ppo", {}).get("gamma", 1.0))
        returns = np.concatenate(episode_returns(episodes, gamma=gamma))
        before = statistics(np.concatenate([ep["old_value"] for ep in episodes]), returns)
        features = _load_or_extract_critic_features(
            out,
            policy,
            episodes,
            device,
            returns,
            behavior,
            original_policy,
            features_for,
            shuffle,
            tick,
        )
        # Move the cache and labels once for the entire fit.
        features = features.to(device)
        labels = torch.from_numpy(returns).to(device)
        with value_only_grad_mode(policy, parameters):
            batches = _fit_critic_batches(
                policy,
                optimizer,
                value_groups,
                features,
                labels,
                config,
                settings,
                device,
                progress,
                capsule_path,
                deadline,
                tick,
                commit,
            )
        tick()
        fit_finished = True
        refresh_started = perf_counter()
        refreshed = refresh_episode_old_values(
            policy,
            episodes,
            features,
            device,
            batch_size=settings["batch_size"],
            bulk_transfer=True,
            progress=clock.progress,
        )
        verify_critic_invariance(
            policy,
            episodes,
            original_policy,
            behavior,
            mode="adaptation",
            policy_hash_fn=policy_hash,
            behavior_hash_fn=behavior_hash,
        )
        return _critic_adaptation_result(
            config,
            episodes,
            progress,
            batches,
            refresh_started,
            before,
            refreshed,
            returns,
            gamma,
            statistics,
        )
    finally:
        if not fit_finished:
            tick()
