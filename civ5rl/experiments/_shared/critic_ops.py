"""Value-only fitting primitives and behavior invariants."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, overload

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray

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
def value_only_grad_mode(policy: ValuePolicy) -> Iterator[None]:
    """Temporarily enable gradients on the value head and restore all flags."""
    saved = [(parameter, parameter.requires_grad) for parameter in policy.parameters()]
    try:
        for parameter, _ in saved:
            parameter.requires_grad_(False)
        for parameter in policy.value.parameters():
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
                torch.cat([policy.value(part).squeeze(-1) for part in features_or_stream.split(batch_size)])
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
        from civ5rl.experiments.ppo_after_imitation import policy_hash

        policy_hash_fn = policy_hash
    if policy_hash_fn(policy.state_dict()) != orig_policy_hash:
        if mode == "adaptation":
            raise RuntimeError("Critic adaptation changed policy or recorded behavior")
        raise RuntimeError("Critic preparation changed encoder/GRU/policy parameters")
    if behavior_hash_fn(episodes) != orig_behavior_hash:
        if mode == "adaptation":
            raise RuntimeError("Critic adaptation changed policy or recorded behavior")
        raise RuntimeError("Critic preparation changed recorded behavior")
