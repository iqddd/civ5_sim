"""Execute exact current-weight rollout replay and collect KL diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, partial
from typing import Literal, NamedTuple, Protocol

import numpy as np
import torch

from civ5rl.core.distributions import masked_kl_rows, policy_logits, require_rollout_temperature
from civ5rl.core.schema import MIN_FREE_ACTIONS
from civ5rl.engine.runtime.sequence import DEFAULT_PREFIX_BLOCK_SIZE, Span


class ReplayCache(Protocol):
    """Input operations needed by on-demand replay."""

    episodes: list
    offsets: np.ndarray
    action_counts: np.ndarray
    device: str | torch.device
    has_behavior_distribution: bool

    def batch(self, spans): ...


class _ReplayPlanCache(ReplayCache, Protocol):
    """Transfer state for plans that can stream pinned host tensors."""

    cpu_labels: bool
    pinned_replay_hits: int
    _replay_h2d_stream: torch.cuda.Stream | None


ReplayMode = Literal["on_demand", "resident_indices", "host", "device"]


class ReplayPart(NamedTuple):
    data: dict[str, torch.Tensor]
    labels: dict[str, torch.Tensor]
    lengths: list[int]
    slots: torch.Tensor | None
    free_count: int


class HostReplayPart(NamedTuple):
    data: dict[str, torch.Tensor]
    labels: dict[str, torch.Tensor]
    lengths: list[int]
    slots: torch.Tensor | None
    free_count: int
    is_pinned: bool


class ResidentReplaySpec(NamedTuple):
    spans: list[Span]
    indices: torch.Tensor
    width: int
    slots: torch.Tensor | None
    free_count: int


@dataclass
class ReplayPlan:
    """Prepared rollout replay plan with an explicit placement mode."""

    mode: ReplayMode
    sequences_per_batch: int
    groups: list[tuple[int, list]]
    cached_bytes: int = 0
    _count_host_hits: bool = False

    def iter_groups(self, cache: _ReplayPlanCache):
        if self.mode == "device":
            yield from self.groups
            return
        if self.mode == "host":
            count_hits = self._count_host_hits
            self._count_host_hits = True
            yield from _stream_pinned_groups(cache, self.groups, count_hits=count_hits)
            return
        if self.mode == "resident_indices":
            for group_size, specs in self.groups:
                yield group_size, [
                    ReplayPart(
                        *cache.batch(spec.spans, indices=spec.indices, width=spec.width,
                                     min_cells=cache.spatial_cell_padding),
                        spec.slots, spec.free_count,
                    )
                    for spec in specs
                ]
            return
        for group_size, specs in self.groups:
            parts = replay_parts(cache, specs)
            if not getattr(cache, "cpu_labels", False):
                parts = list(parts)
            yield group_size, parts


def categorical_distance(behavior_log_probs, logits, mask, behavior_log_normalizer=None):
    """Exact categorical KL and total variation; double avoids KL cancellation."""
    behavior = behavior_log_probs.double()
    logp = (
        behavior.log_softmax(-1)
        if behavior_log_normalizer is None
        else behavior - behavior_log_normalizer[:, None]
    )
    logq = logits.double().log_softmax(-1)
    p, q = logp.exp(), logq.exp()
    # Equivalent distributions can differ by FP64 roundoff after normalization.
    kl = masked_kl_rows(logp, logq, mask, probabilities=p).clamp_min(0)
    return kl, (p - q).abs().sum(-1) * 0.5


@lru_cache(maxsize=1)
def compiled_categorical_distance():
    return torch.compile(categorical_distance, dynamic=True, fullgraph=True)


def validate_behavior_replay(
    pre_kl,
    max_logp_error,
    *,
    amp_enabled,
    numerics,
    full_kl=None,
    max_total_variation=None,
):
    # old_logp always remains the probability used to select the real command.
    # With complete behavior distributions, BF16 is checked by probability mass,
    # rather than the relative error of a potentially very rare command.
    tolerance = (
        numerics.get("bf16_replay_max_logp_error", 0.005)
        if amp_enabled
        else numerics.get("fp32_replay_max_logp_error", 0.001)
    )
    kl_tolerance = numerics.get("replay_max_kl", 1e-6)
    has_distribution = full_kl is not None and max_total_variation is not None
    use_distribution = amp_enabled and has_distribution
    if amp_enabled and numerics.get("require_full_behavior_distribution", False) and not use_distribution:
        raise RuntimeError("BF16 replay verification requires complete behavior distributions")
    distribution_bad = has_distribution and (
        not np.isfinite([full_kl, max_total_variation]).all()
        or full_kl > kl_tolerance
        or max_total_variation > numerics.get("replay_max_total_variation", 0.005)
    )
    if (
        not np.isfinite([pre_kl, max_logp_error]).all()
        or (not use_distribution and max_logp_error > tolerance)
        or pre_kl > kl_tolerance
        or distribution_bad
    ):
        mode_name = "full_distribution" if use_distribution else "sampled_logp"
        raise RuntimeError(
            f"Behavior/replay mismatch: max_logp_error={max_logp_error}, "
            f"sampled_KL={pre_kl}, full_KL={full_kl}, "
            f"max_TV={max_total_variation}; mode={mode_name}"
        )
    return None if use_distribution else tolerance


def ppo_replay_forward(policy, *, spatial_graph=True):
    """Select the no-grad forward for PPO, independent of actor training."""
    replay_runner = getattr(policy, "spatial_replay_runner", None)
    if spatial_graph and replay_runner is not None:
        if (runner := replay_runner()) is not None:
            return partial(policy.forward_sequences, spatial_runner=runner)
    return policy.forward_sequences


class _ReplaySums(NamedTuple):
    sampled_kl: torch.Tensor
    sampled_kl_free: torch.Tensor
    critic_sum: torch.Tensor
    residual_square_sum: torch.Tensor
    residual_sum: torch.Tensor
    return_sum: torch.Tensor
    return_square_sum: torch.Tensor


class _ReplayTotals(NamedTuple):
    sampled_kl: float
    sampled_kl_free: float
    critic_sum: float
    residual_square_sum: float
    residual_sum: float
    return_sum: float
    return_square_sum: float
    max_logp_error: float
    full_kl: float
    max_total_variation: float
    full_kl_free: float


def _replay_sums(logits, value, indices, old_logp, returns, action_mask):
    logp = logits.log_softmax(-1).gather(1, indices[:, None])[:, 0]
    delta = logp - old_logp
    sampled = delta.exp() - 1 - delta
    free = action_mask.sum(-1) >= MIN_FREE_ACTIONS
    residual = returns - value
    sums = _ReplaySums(
        sampled.sum(),
        sampled.masked_fill(~free, 0).sum(),
        value.sum(),
        residual.square().sum(),
        residual.sum(),
        returns.sum(),
        returns.square().sum(),
    )
    return torch.stack(sums), delta.abs().max()


@lru_cache(maxsize=1)
def compiled_replay_sums():
    return torch.compile(_replay_sums, dynamic=True, fullgraph=True)


def _part_free_count(cache: ReplayCache, spans):
    return sum(
        int(
            (
                cache.action_counts[
                    int(cache.offsets[span.episode]) + span.start : int(cache.offsets[span.episode])
                    + span.stop
                ]
                >= MIN_FREE_ACTIONS
            ).sum()
        )
        for span in spans
    )


def replay_group_specs(cache: ReplayCache, sequences_per_batch):
    lengths = np.diff(cache.offsets)
    for first in range(0, len(cache.episodes), sequences_per_batch):
        episodes = list(range(first, min(first + sequences_per_batch, len(cache.episodes))))
        group_size = len(episodes)
        max_length = max(lengths[episode] for episode in episodes)
        specs = []
        for offset in range(0, max_length, DEFAULT_PREFIX_BLOCK_SIZE):
            active = [(slot, episode) for slot, episode in enumerate(episodes) if lengths[episode] > offset]
            spans = [
                Span(
                    episode,
                    offset,
                    min(
                        offset + DEFAULT_PREFIX_BLOCK_SIZE,
                        int(lengths[episode]),
                    ),
                )
                for _, episode in active
            ]
            slots_list = None if len(active) == group_size else [slot for slot, _ in active]
            specs.append((spans, slots_list, _part_free_count(cache, spans)))
        yield group_size, specs


def _stream_pinned_groups(cache: _ReplayPlanCache, pinned_groups, *, count_hits):
    if cache._replay_h2d_stream is None:
        cache._replay_h2d_stream = torch.cuda.Stream(device=cache.device)
    h2d_stream = cache._replay_h2d_stream
    default_stream = torch.cuda.current_stream(cache.device)
    flat = [part for _, g_parts in pinned_groups for part in g_parts]

    def _enqueue(part):
        cpu_data, cpu_labels, lengths, cpu_slots, free_count, is_pinned = part
        h2d_stream.wait_stream(default_stream)
        with torch.cuda.stream(h2d_stream):
            gpu_data = {k: v.to(cache.device, non_blocking=is_pinned) for k, v in cpu_data.items()}
            gpu_labels = {k: v.to(cache.device, non_blocking=is_pinned) for k, v in cpu_labels.items()}
            gpu_slots = None if cpu_slots is None else cpu_slots.to(cache.device, non_blocking=is_pinned)
            done = torch.cuda.Event()
            done.record(h2d_stream)
        return gpu_data, gpu_labels, lengths, gpu_slots, free_count, is_pinned, done

    state = {"idx": 0, "inflight": _enqueue(flat[0]) if flat else None}
    for group_size, g_parts in pinned_groups:

        def _iter_group(num_parts=len(g_parts)):
            for _ in range(num_parts):
                cur = state["inflight"]
                nxt_idx = state["idx"] + 1
                state["idx"] = nxt_idx
                state["inflight"] = _enqueue(flat[nxt_idx]) if nxt_idx < len(flat) else None
                (
                    gpu_data,
                    gpu_labels,
                    lengths,
                    gpu_slots,
                    free_count,
                    is_pinned,
                    done,
                ) = cur
                default_stream.wait_event(done)
                for t in (*gpu_data.values(), *gpu_labels.values()):
                    t.record_stream(default_stream)
                if gpu_slots is not None:
                    gpu_slots.record_stream(default_stream)
                if count_hits and is_pinned:
                    cache.pinned_replay_hits += 1
                yield ReplayPart(gpu_data, gpu_labels, lengths, gpu_slots, free_count)

        yield group_size, _iter_group()


def replay_parts(cache: ReplayCache, specs):
    min_cells = getattr(cache, "spatial_cell_padding", 0)
    has_replay_batch = hasattr(cache, "replay_batch")
    for spans, slots_list, free_count in specs:
        if has_replay_batch:
            batch, labels, lengths = cache.replay_batch(spans)
        elif min_cells:
            batch, labels, lengths = cache.batch(spans, min_cells=min_cells)
        else:
            batch, labels, lengths = cache.batch(spans)
        slots = (
            None if slots_list is None else torch.tensor(slots_list, dtype=torch.long, device=cache.device)
        )
        yield ReplayPart(batch, labels, lengths, slots, free_count)


def _replay_groups(cache: ReplayCache, sequences_per_batch):
    # Immutable inputs only: no prefix states or outputs from previous weights.
    if hasattr(cache, "prepare_replay_plan"):
        plan = cache.prepare_replay_plan(sequences_per_batch)
        yield from plan.iter_groups(cache)
        return
    for group_size, specs in replay_group_specs(cache, sequences_per_batch):
        parts = replay_parts(cache, specs)
        if not getattr(cache, "cpu_labels", False):
            parts = list(parts)
        yield group_size, parts


@torch.no_grad()
def measure_replay(
    policy,
    cache: ReplayCache,
    sequences_per_batch=8,
    *,
    full_distribution=False,
    temperature=1.0,
    forward=None,
):
    require_rollout_temperature(cache.episodes, temperature)
    forward = forward or ppo_replay_forward(policy)
    device = cache.device
    totals = torch.zeros(len(_ReplaySums._fields), device=device)
    maximum = torch.zeros((), device=device)
    free_count = count = 0
    full_total = torch.zeros((), device=device, dtype=torch.float64)
    full_free_total = torch.zeros_like(full_total)
    max_tv = torch.zeros_like(full_total)
    runner = getattr(cache, "_replay_sums_runner", None) or _replay_sums
    distance = getattr(cache, "_categorical_distance_runner", None) or categorical_distance
    for group_size, parts in _replay_groups(cache, sequences_per_batch):
        hidden = (
            torch.zeros((1, group_size, policy.hidden_size), device=device) if policy.is_recurrent else None
        )
        for batch, labels, lengths, slots, part_free in parts:
            if policy.spatial_cuda_graphs:
                torch.compiler.cudagraph_mark_step_begin()
            part_hidden = (
                None if hidden is None else (hidden if slots is None else hidden.index_select(1, slots))
            )
            logits, value, last = forward(batch, lengths, part_hidden)
            logits = policy_logits(logits, temperature)
            if hidden is not None:
                if slots is None:
                    hidden.copy_(last)
                else:
                    hidden.index_copy_(1, slots, last)
            sums, max_error = runner(
                logits,
                value,
                labels["indices"],
                labels["old_logp"],
                labels["returns"],
                batch["action_mask"],
            )
            totals += sums
            maximum = torch.maximum(maximum, max_error)
            free_count += part_free
            count += sum(lengths)
            if full_distribution and cache.has_behavior_distribution:
                kl, tv = distance(
                    labels["old_log_probs"][:, : logits.shape[1]],
                    logits,
                    batch["action_mask"],
                    labels.get("behavior_log_normalizer"),
                )
                full_total += kl.sum()
                full_free_total += kl.masked_fill(batch["action_mask"].sum(-1) < MIN_FREE_ACTIONS, 0).sum()
                max_tv = torch.maximum(max_tv, tv.max())
    # One host transfer for all diagnostics; full-distribution sums stay FP64.
    host_metrics = (
        torch.cat(
            (
                totals.double(),
                maximum[None].double(),
                full_total[None],
                max_tv[None],
                full_free_total[None],
            )
        )
        .cpu()
        .tolist()
    )
    replay = _ReplayTotals(*host_metrics)
    variance = replay.return_square_sum / count - (replay.return_sum / count) ** 2
    return {
        "sampled_kl": replay.sampled_kl / count,
        "sampled_kl_free": replay.sampled_kl_free / free_count if free_count else None,
        "free_decisions": free_count,
        "all_decisions": count,
        "critic_mean": replay.critic_sum / count,
        "critic_mse": replay.residual_square_sum / count,
        "critic_explained_variance": (
            1 - (replay.residual_square_sum / count - (replay.residual_sum / count) ** 2) / variance
            if variance > 1e-12
            else None
        ),
        "max_logp_error": replay.max_logp_error,
        "full_kl": replay.full_kl / count if full_distribution and cache.has_behavior_distribution else None,
        "full_kl_free": replay.full_kl_free / free_count
        if full_distribution and cache.has_behavior_distribution and free_count
        else None,
        "max_total_variation": replay.max_total_variation
        if full_distribution and cache.has_behavior_distribution
        else None,
    }
