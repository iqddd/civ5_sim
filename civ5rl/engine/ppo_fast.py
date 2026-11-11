"""GPU-resident rollout and batched recurrent chunks; same PPO minibatch/targets."""

from dataclasses import dataclass
from time import perf_counter
from typing import Literal, NamedTuple

import numpy as np
import torch

from civ5rl.core.distributions import (
    policy_logits,
    require_rollout_temperature,
    validate_temperature,
)
from civ5rl.core.schema import MIN_FREE_ACTIONS
from civ5rl.engine.constants import (
    DEFAULT_GPU_RESERVE_BYTES,
    DEFAULT_PINNED_REPLAY_BYTES,
    DEFAULT_PREFIX_BLOCK_SIZE,
)
from civ5rl.engine.ppo import encode_episode, episode_returns, objective, targets
from civ5rl.engine.replay_metrics import REPLAY_DIAGNOSTIC_KEYS, StepReplay
from civ5rl.engine.sequence import SequenceBuffer, allocation_with_reserve, cuda_memory_snapshot
from civ5rl.engine.spans import Span
from civ5rl.engine.training import GradientAccumulator


ReplayMode = Literal["on_demand", "host", "device"]


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


def _part_tensor_bytes(data, labels, slots) -> int:
    return sum(
        t.numel() * t.element_size()
        for t in (
            *data.values(),
            *labels.values(),
            *((slots,) if slots is not None else ()),
        )
    )


@dataclass
class ReplayPlan:
    """Prepared rollout replay plan with an explicit placement mode."""

    mode: ReplayMode
    sequences_per_batch: int
    groups: list[tuple[int, list]]
    cached_bytes: int = 0
    _count_host_hits: bool = False

    def iter_groups(self, cache):
        if self.mode == "device":
            yield from self.groups
            return
        if self.mode == "host":
            count_hits = self._count_host_hits
            self._count_host_hits = True
            yield from _stream_pinned_groups(cache, self.groups, count_hits=count_hits)
            return
        for group_size, specs in self.groups:
            parts = _replay_parts(cache, specs)
            if not getattr(cache, "cpu_labels", False):
                parts = list(parts)
            yield group_size, parts


def categorical_distance(behavior_log_probs, logits, mask):
    """Exact categorical KL and total variation; double avoids KL cancellation."""
    logp = behavior_log_probs.double().log_softmax(-1)
    logq = logits.double().log_softmax(-1)
    p, q = logp.exp(), logq.exp()
    delta = (logp - logq).masked_fill(~mask, 0)
    return (p * delta).sum(-1), (p - q).abs().sum(-1) * 0.5


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


class RolloutCache(SequenceBuffer):
    def __init__(
        self,
        episodes,
        variant,
        device,
        *,
        resident=True,
        cpu_labels=False,
        gamma=1.0,
        gae_lambda=1.0,
    ):
        self.episodes, self.device = episodes, device
        super().__init__([encode_episode(e, variant) for e in episodes], device, resident=resident)
        flat = [x for episode in self.encoded for x in episode]
        self.resident = resident
        self.cpu_labels = cpu_labels
        label_device = "cpu" if cpu_labels else device
        returns = episode_returns(episodes, gamma=gamma)
        advantages = targets(episodes, gamma=gamma, gae_lambda=gae_lambda)
        self.labels = {
            "indices": torch.tensor([v for ep in episodes for v in ep["chosen"]], device=label_device),
            "old_logp": torch.tensor([v for ep in episodes for v in ep["old_logp"]], device=label_device),
            "returns": torch.tensor(
                np.concatenate(returns),
                dtype=torch.float32,
                device=label_device,
            ),
            "advantages": torch.tensor(np.concatenate(advantages), device=label_device),
        }
        has_full = ["old_log_probs" in ep for ep in episodes]
        if any(has_full) and not all(has_full):
            raise ValueError("Mixed complete/incomplete behavior distributions")
        self.has_behavior_distribution = all(has_full)
        self.ragged_behavior = None
        if self.has_behavior_distribution and cpu_labels:
            self.ragged_behavior = [values for ep in episodes for values in ep["old_log_probs"]]
            if len(self.ragged_behavior) != len(flat):
                raise ValueError("Behavior distribution count differs from decisions")
            for i, (values, count) in enumerate(zip(self.ragged_behavior, self.action_counts)):
                if (
                    len(values) != count
                    or not np.isfinite(values).all()
                    or float(values[int(self.labels["indices"][i])]) != float(self.labels["old_logp"][i])
                    or not np.isclose(np.exp(values).sum(), 1.0, atol=1e-5, rtol=0)
                ):
                    raise ValueError("Invalid ragged behavior distribution")
        elif self.has_behavior_distribution:
            width = int(self.action_counts.max())
            behavior = np.full((len(flat), width), -np.inf, np.float32)
            full = [values for ep in episodes for values in ep["old_log_probs"]]
            if len(full) != len(flat):
                raise ValueError("Behavior distribution count differs from decisions")
            for index, (values, count) in enumerate(zip(full, self.action_counts)):
                if len(values) != count or not np.isfinite(values).all():
                    raise ValueError("Invalid behavior action probabilities")
                behavior[index, :count] = values
            behavior = torch.tensor(behavior, device=device)
            selected = behavior.gather(1, self.labels["indices"][:, None])[:, 0]
            if not torch.equal(selected, self.labels["old_logp"]):
                raise ValueError("Recorded chosen logp differs from behavior distribution")
            if not torch.allclose(
                behavior.exp().sum(-1),
                torch.ones(len(flat), device=device),
                atol=1e-5,
                rtol=0,
            ):
                raise ValueError("Behavior probabilities do not sum to one")
            self.labels["old_log_probs"] = behavior
        self.resident_bytes = sum(v.numel() * v.element_size() for v in (self.data or {}).values()) + sum(
            v.numel() * v.element_size() for v in self.labels.values()
        )
        self.spatial_cell_padding = 0
        self.replay_inputs = False
        self._replay_plan: ReplayPlan | None = None
        self._replay_h2d_stream = None
        self._replay_sums_runner = None
        self._pinned_replay = None
        self._pinned_replay_limit = 0
        self.pinned_replay_bytes = 0
        self.pinned_replay_hits = 0
        self.replay_memory_admission = None

    @property
    def replay_mode(self) -> ReplayMode:
        return self._replay_plan.mode if self._replay_plan is not None else "on_demand"

    @property
    def cached_replay_bytes(self) -> int:
        if self._replay_plan is not None and self._replay_plan.mode == "device":
            return self._replay_plan.cached_bytes
        return 0

    def enable_cached_replay(self):
        """Cache inputs only; GRU state and logits always use current weights."""
        if self.data is None:
            return
        if str(self.device).startswith("cuda"):
            self._replay_sums_runner = torch.compile(_replay_sums, dynamic=True, fullgraph=True)
        self.replay_inputs = True
        self._replay_plan = None

    def enable_pinned_replay(self, limit_bytes=DEFAULT_PINNED_REPLAY_BYTES):
        """Keep immutable replay inputs on CPU; stream each part to the GPU."""
        if self.data is None and self.cpu_labels and str(self.device).startswith("cuda"):
            self._pinned_replay = {}
            self._pinned_replay_limit = limit_bytes
            self._replay_plan = None

    def _build_host_groups(self, sequences_per_batch, *, cpu_groups=None):
        if cpu_groups is None:
            cpu_groups, _ = self._collect_cpu_replay_groups(sequences_per_batch)
        host_groups = []
        for group_size, parts in cpu_groups:
            host_parts = []
            for key, data, labels, lengths, slots, free_count, part_bytes in parts:
                is_pinned = False
                cached = self._pinned_replay.get(key) if self._pinned_replay is not None else None
                if cached is not None:
                    data, labels, lengths = cached
                    is_pinned = True
                elif (
                    self._pinned_replay is not None
                    and self.pinned_replay_bytes + part_bytes <= self._pinned_replay_limit
                ):
                    data = {k: v.pin_memory() for k, v in data.items()}
                    labels = {k: v.pin_memory() for k, v in labels.items()}
                    if slots is not None:
                        slots = slots.pin_memory()
                    self._pinned_replay[key] = data, labels, lengths
                    self.pinned_replay_bytes += part_bytes
                    is_pinned = True
                host_parts.append(HostReplayPart(data, labels, lengths, slots, free_count, is_pinned))
            host_groups.append((group_size, host_parts))
        return host_groups

    def _collect_cpu_replay_groups(self, sequences_per_batch):
        cpu_groups = []
        total_bytes = 0
        for group_size, specs in _replay_group_specs(self, sequences_per_batch):
            parts = []
            for spans, slots_list, free_count in specs:
                data, labels, lengths = self.batch(spans, device="cpu", min_cells=self.spatial_cell_padding)
                slots = None if slots_list is None else torch.tensor(slots_list, dtype=torch.long)
                part_bytes = _part_tensor_bytes(data, labels, slots)
                total_bytes += part_bytes
                parts.append((tuple(spans), data, labels, lengths, slots, free_count, part_bytes))
            cpu_groups.append((group_size, parts))
        return cpu_groups, total_bytes

    def prepare_replay_plan(
        self,
        sequences_per_batch=8,
        *,
        reserve_bytes=2 * DEFAULT_GPU_RESERVE_BYTES,
        prefer_device=False,
    ) -> ReplayPlan:
        """Try real cached replay allocations; fall back only on CUDA OOM."""
        if (
            self._replay_plan is not None
            and self._replay_plan.sequences_per_batch == sequences_per_batch
            and (not prefer_device or self._replay_plan.mode == "device")
        ):
            return self._replay_plan

        specs_by_group = list(_replay_group_specs(self, sequences_per_batch))
        if self.replay_inputs or (prefer_device and str(self.device).startswith("cuda")):
            device_groups, parts, part = [], None, None
            total_bytes = 0
            try:
                with allocation_with_reserve(self.device, reserve_bytes) as admission:
                    self.replay_memory_admission = admission
                    for group_size, specs in specs_by_group:
                        parts = []
                        device_groups.append((group_size, parts))
                        for part in _replay_parts(self, specs):
                            total_bytes += _part_tensor_bytes(part.data, part.labels, part.slots)
                            parts.append(part)
            except torch.cuda.OutOfMemoryError:
                # No exception/traceback is retained in the fallback plan.
                # Drop both completed groups and the most recent local aliases.
                device_groups.clear()
                if parts is not None:
                    parts.clear()
                part = None
                self.replay_inputs = False
            else:
                self.replay_inputs = True
                self._replay_plan = ReplayPlan(
                    mode="device",
                    sequences_per_batch=sequences_per_batch,
                    groups=device_groups,
                    cached_bytes=total_bytes,
                )
                return self._replay_plan

        if self._pinned_replay is not None:
            if (
                self._replay_plan is not None
                and self._replay_plan.sequences_per_batch == sequences_per_batch
                and self._replay_plan.mode == "host"
            ):
                self._replay_plan._count_host_hits = True
                return self._replay_plan
            host_groups = self._build_host_groups(sequences_per_batch)
            self._replay_plan = ReplayPlan(
                mode="host",
                sequences_per_batch=sequences_per_batch,
                groups=host_groups,
                _count_host_hits=prefer_device,
            )
            return self._replay_plan

        self._replay_plan = ReplayPlan(
            mode="on_demand",
            sequences_per_batch=sequences_per_batch,
            groups=specs_by_group,
        )
        return self._replay_plan

    def stage_gpu_replay(self, sequences_per_batch=8, *, reserve_bytes=2 * DEFAULT_GPU_RESERVE_BYTES):
        """Stage pre-packed immutable replay parts in VRAM when reserve permits."""
        if not str(self.device).startswith("cuda"):
            return None
        return self.prepare_replay_plan(
            sequences_per_batch,
            reserve_bytes=reserve_bytes,
            prefer_device=True,
        )

    def replay_batch(self, spans):
        if self._pinned_replay is None:
            return self.batch(spans, min_cells=self.spatial_cell_padding)
        key = tuple(spans)
        part = self._pinned_replay.get(key)
        if part is None:
            data, labels, lengths = self.batch(spans, device="cpu", min_cells=self.spatial_cell_padding)
            size = sum(v.numel() * v.element_size() for v in (*data.values(), *labels.values()))
            if self.pinned_replay_bytes + size <= self._pinned_replay_limit:
                data = {k: v.pin_memory() for k, v in data.items()}
                labels = {k: v.pin_memory() for k, v in labels.items()}
                self._pinned_replay[key] = data, labels, lengths
                self.pinned_replay_bytes += size
        else:
            data, labels, lengths = part
            self.pinned_replay_hits += 1
        return (
            {k: v.to(self.device, non_blocking=True) for k, v in data.items()},
            {k: v.to(self.device, non_blocking=True) for k, v in labels.items()},
            lengths,
        )

    def batch(self, spans, *, device=None, include_behavior=True, min_cells=0):
        device = self.device if device is None else device
        indices = (
            torch.tensor(
                [
                    i
                    for episode, start, stop in spans
                    for i in range(
                        int(self.offsets[episode]) + start,
                        int(self.offsets[episode]) + stop,
                    )
                ]
            )
            if self.cpu_labels
            else self.indices(spans)
        )
        data = self.gather_inputs(spans, indices, device=device, min_cells=min_cells)
        if min_cells and "cells" in data and data["cells"].shape[1] < min_cells:
            extra = min_cells - data["cells"].shape[1]
            data = {
                **data,
                "cells": torch.nn.functional.pad(data["cells"], (0, 0, 0, extra)),
                "neighbors": torch.nn.functional.pad(data["neighbors"], (0, 0, 0, extra), value=-1),
                "cell_mask": torch.nn.functional.pad(data["cell_mask"], (0, extra), value=False),
            }
        selected = indices.cpu() if self.cpu_labels else indices
        labels = {
            k: v.index_select(0, selected).to(device)
            for k, v in self.labels.items()
            if include_behavior or k != "old_log_probs"
        }
        if include_behavior and self.ragged_behavior is not None:
            rows = [self.ragged_behavior[i] for i in selected.tolist()]
            behavior = np.full((len(rows), data["action_mask"].shape[1]), -np.inf, np.float32)
            for i, values in enumerate(rows):
                behavior[i, : len(values)] = values
            labels["old_log_probs"] = torch.from_numpy(behavior).to(device)
        return data, labels, [stop - start for _, start, stop in spans]


@torch.no_grad()
def prefixes(policy, cache, spans, block_size=DEFAULT_PREFIX_BLOCK_SIZE):
    if not policy.is_recurrent:
        return None
    hidden = torch.zeros((1, len(spans), policy.hidden_size), device=cache.device)
    is_cuda = str(cache.device).startswith("cuda")
    for offset in range(0, max((start for _, start, _ in spans), default=0), block_size):
        active = [(slot, episode, start) for slot, (episode, start, _) in enumerate(spans) if start > offset]
        subspans = [Span(episode, offset, min(start, offset + block_size)) for _, episode, start in active]
        if hasattr(cache, "gather_spatial_inputs"):
            batch = cache.gather_spatial_inputs(subspans)
            lengths = [span.length for span in subspans]
        else:
            batch, _, lengths = cache.batch(subspans)
        if policy.spatial_cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        if len(active) == len(spans):
            hidden = policy.forward_sequences(batch, lengths, hidden, memory_only=True)
        else:
            slots = torch.tensor(
                [slot for slot, _, _ in active],
                dtype=torch.long,
                pin_memory=is_cuda,
            ).to(cache.device, non_blocking=is_cuda)
            last = policy.forward_sequences(batch, lengths, hidden.index_select(1, slots), memory_only=True)
            hidden.index_copy_(1, slots, last)
    return hidden


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


def _part_free_count(cache, spans):
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


def _replay_group_specs(cache, sequences_per_batch):
    for first in range(0, len(cache.episodes), sequences_per_batch):
        episodes = list(range(first, min(first + sequences_per_batch, len(cache.episodes))))
        group_size = len(episodes)
        max_length = max(len(cache.encoded[episode]) for episode in episodes)
        specs = []
        for offset in range(0, max_length, DEFAULT_PREFIX_BLOCK_SIZE):
            active = [
                (slot, episode)
                for slot, episode in enumerate(episodes)
                if len(cache.encoded[episode]) > offset
            ]
            spans = [
                Span(
                    episode,
                    offset,
                    min(
                        offset + DEFAULT_PREFIX_BLOCK_SIZE,
                        len(cache.encoded[episode]),
                    ),
                )
                for _, episode in active
            ]
            slots_list = None if len(active) == group_size else [slot for slot, _ in active]
            specs.append((spans, slots_list, _part_free_count(cache, spans)))
        yield group_size, specs


def _stream_pinned_groups(cache, pinned_groups, *, count_hits):
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


def _replay_parts(cache, specs):
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


def _replay_groups(cache, sequences_per_batch):
    # Immutable inputs only: no prefix states or outputs from previous weights.
    if hasattr(cache, "prepare_replay_plan"):
        plan = cache.prepare_replay_plan(sequences_per_batch)
        yield from plan.iter_groups(cache)
        return
    for group_size, specs in _replay_group_specs(cache, sequences_per_batch):
        parts = _replay_parts(cache, specs)
        if not getattr(cache, "cpu_labels", False):
            parts = list(parts)
        yield group_size, parts


@torch.no_grad()
def measure_replay(policy, cache, sequences_per_batch=8, *, full_distribution=False, temperature=1.0):
    require_rollout_temperature(cache.episodes, temperature)
    device = cache.device
    totals = torch.zeros(len(_ReplaySums._fields), device=device)
    maximum = torch.zeros((), device=device)
    free_count = count = 0
    full_total = torch.zeros((), device=device, dtype=torch.float64)
    full_free_total = torch.zeros_like(full_total)
    max_tv = torch.zeros_like(full_total)
    runner = getattr(cache, "_replay_sums_runner", None) or _replay_sums
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
            logits, value, last = policy.forward_sequences(
                batch,
                lengths,
                part_hidden,
            )
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
                kl, tv = categorical_distance(
                    labels["old_log_probs"][:, : logits.shape[1]],
                    logits,
                    batch["action_mask"],
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


def measure_kl(policy, cache, sequences_per_batch=8):
    result = measure_replay(policy, cache, sequences_per_batch)
    return result["sampled_kl"], result["max_logp_error"]


class BatchedBackend:
    def __init__(
        self,
        policy,
        episodes,
        config,
        device,
        monitor=None,
        *,
        force_batched=False,
        pinned_replay=False,
    ):
        self.policy, self.config, self.device, self.monitor = (
            policy,
            config,
            device,
            monitor,
        )
        options = config.get("performance", {})
        self.temperature = validate_temperature(config.get("policy_temperature", 1.0))
        require_rollout_temperature(episodes, self.temperature)
        tick = perf_counter()
        batched = force_batched or options.get("batched_ppo", False)
        ppo_cfg = config.get("ppo", {})
        self.cache = RolloutCache(
            episodes,
            policy.variant,
            device,
            resident=options.get("resident_rollout", True) if batched else False,
            cpu_labels=options.get("cpu_rollout_labels", False),
            gamma=float(ppo_cfg.get("gamma", 1.0)),
            gae_lambda=float(ppo_cfg.get("gae_lambda", 1.0)),
        )
        self.cache.spatial_cell_padding = int(getattr(policy, "spatial_cell_padding", 0) or 0)
        self.kl_batch = options.get("kl_sequences_per_batch", 8)
        if str(device).startswith("cuda") and options.get("ppo_optimized_inputs", False):
            self.cache.enable_compiled_gather()
            self.cache.enable_async_transfers()
        if options.get("ppo_cached_replay", False):
            self.cache.enable_cached_replay()
        if pinned_replay or options.get("ppo_pinned_replay", False):
            self.cache.enable_pinned_replay()
            if str(device).startswith("cuda"):
                self.cache.stage_gpu_replay(self.kl_batch)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        self.cache_seconds = perf_counter() - tick
        self.names = (
            "policy_loss",
            "value_loss",
            "entropy",
            "normalized_entropy",
            "clip_fraction",
        )
        self.totals = torch.zeros(len(self.names), device=device)
        self.gradients = []
        self.prefix_seconds = self.backward_seconds = 0.0
        self.last_minibatch_approx_kl = 0.0
        self.track_minibatch_kl = (
            ppo_cfg.get("kl_measurement")
            == "whole_rollout_after_epoch_or_minibatch_spike"
        )
        self.micro_chunks = config["ppo"].get("sequences_per_microbatch", 1) if batched else 1

    def precheck(self):
        if self.monitor:
            self.monitor.mark("behavior_replay")
        self.replay = measure_replay(
            self.policy,
            self.cache,
            self.kl_batch,
            full_distribution=True,
            temperature=self.temperature,
        )
        pre_kl, pre_error = self.replay["sampled_kl"], self.replay["max_logp_error"]
        self.replay_tolerance = validate_behavior_replay(
            pre_kl,
            pre_error,
            amp_enabled=self.policy.amp_enabled,
            numerics=self.config.get("numerics", {}),
            full_kl=self.replay["full_kl"],
            max_total_variation=self.replay["max_total_variation"],
        )
        if self.monitor:
            self.monitor.mark("ppo", ppo_epoch=0)
        if not 1 <= self.micro_chunks <= self.config["ppo"]["sequences_per_minibatch"]:
            raise ValueError("Microbatch must contain 1..minibatch chunks")
        return dict(self.replay)

    def backward(self, minibatch, count, entropy_coefficient):
        if self.policy.spatial_cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        tick = perf_counter()
        hidden = prefixes(self.policy, self.cache, minibatch)
        self.prefix_seconds += perf_counter() - tick
        finite = torch.ones((), dtype=torch.bool, device=self.device)
        k3_free_sum = (
            torch.zeros((), dtype=torch.float64, device=self.device)
            if self.track_minibatch_kl
            else None
        )
        free_tokens_total = (
            torch.zeros((), dtype=torch.long, device=self.device)
            if self.track_minibatch_kl
            else None
        )
        tokens = 0
        accumulator = GradientAccumulator(
            self.policy.parameters(), coalesced=self.policy.coalesced_accumulation
        )
        for micro_start in range(0, len(minibatch), self.micro_chunks):
            spans = minibatch[micro_start : micro_start + self.micro_chunks]
            tick = perf_counter()
            batch, labels, lengths = self.cache.batch(spans, include_behavior=False)
            logits, value, _ = self.policy.forward_sequences(
                batch,
                lengths,
                None if hidden is None else hidden[:, micro_start : micro_start + len(spans)].contiguous(),
            )
            loss, stats = objective(
                logits,
                value,
                labels["indices"],
                labels["old_logp"],
                labels["returns"],
                labels["advantages"],
                entropy_coefficient,
                clip_ratio=self.config["ppo"]["clip_ratio"],
                value_coefficient=self.config["ppo"]["value_coefficient"],
                temperature=self.temperature,
            )
            if self.track_minibatch_kl:
                log_ratio = (stats["logp"] - labels["old_logp"]).double()
                k3 = (log_ratio.exp() - 1.0) - log_ratio
                free_mask = batch["action_mask"].sum(-1) >= MIN_FREE_ACTIONS
                k3_free_sum += k3.masked_fill(~free_mask, 0.0).sum()
                free_tokens_total += free_mask.sum()
            finite &= torch.isfinite(loss.detach())
            valid_tokens = sum(lengths)
            weighted_loss = loss * (valid_tokens / count)
            accumulator.backward(weighted_loss)
            self.backward_seconds += perf_counter() - tick
            self.totals += torch.stack([stats[key] for key in self.names]) * valid_tokens
            tokens += valid_tokens
        if not finite.item():
            raise FloatingPointError("Non-finite batched PPO loss")
        if self.track_minibatch_kl:
            free_count = int(free_tokens_total.item())
            self.last_minibatch_approx_kl = (
                float(k3_free_sum.item()) / free_count if free_count > 0 else 0.0
            )
        else:
            self.last_minibatch_approx_kl = 0.0
        return tokens

    def measure_step_replay(self) -> StepReplay:
        exact = self.config["ppo"].get("step_acceptance") == "full_categorical_kl_rollback"
        if exact and not self.cache.has_behavior_distribution:
            raise ValueError("Controlled PPO requires behavior distributions")
        diagnostics = measure_replay(
            self.policy,
            self.cache,
            self.kl_batch,
            full_distribution=exact,
            temperature=self.temperature,
        )
        self.last_replay = diagnostics
        if exact:
            acceptance_kl = diagnostics["full_kl_free"]
        elif self.config["ppo"].get("kl_population") == "free":
            acceptance_kl = diagnostics["sampled_kl_free"]
        else:
            acceptance_kl = diagnostics["sampled_kl"]
        return StepReplay(acceptance_kl=acceptance_kl, diagnostics=diagnostics)

    def kl(self):
        return self.measure_step_replay().acceptance_kl

    def record_gradient(self, norm):
        self.gradients.append(norm.detach())

    def statistics(self, tokens):
        if not tokens:
            return dict.fromkeys(self.names, None), None, None
        stats = (self.totals / tokens).cpu().tolist()
        gradient_values = torch.stack(self.gradients).cpu().numpy()
        return (
            dict(zip(self.names, stats)),
            float(gradient_values.mean()),
            float(gradient_values.max()),
        )

    def extras(self, optimizer_seconds):
        return {
            **{
                k: v
                for k, v in getattr(self, "last_replay", self.replay).items()
                if k in REPLAY_DIAGNOSTIC_KEYS
            },
            "pre_update_full_kl": self.replay["full_kl"],
            "pre_update_max_total_variation": self.replay["max_total_variation"],
            "behavior_check": "full_distribution" if self.replay_tolerance is None else "sampled_logp",
            "behavior_logp_tolerance": self.replay_tolerance,
            "optimizer_seconds": optimizer_seconds,
            "cache_seconds": self.cache_seconds,
            "rollout_device_bytes": self.cache.resident_bytes,
            "resident_memory_admission": self.cache.memory_admission,
            "replay_memory_admission": self.cache.replay_memory_admission,
            "cuda_memory_after_update": (
                cuda_memory_snapshot(self.device)
                if str(self.device).startswith("cuda") else None
            ),
            "replay_mode": self.cache.replay_mode,
            "cached_replay_inputs": self.cache.replay_inputs,
            "pinned_replay_input_bytes": self.cache.pinned_replay_bytes,
            "pinned_replay_hits": self.cache.pinned_replay_hits,
            "cached_replay_input_bytes": self.cache.cached_replay_bytes,
            "microbatch_sequences": self.micro_chunks,
        }


def update_fast(policy, optimizer, episodes, config, decisions, device, shuffle_rng, monitor=None):
    """Compatibility entry point; use the batched backend regardless of config."""
    from functools import partial

    from civ5rl.engine.ppo import update

    return update(
        policy,
        optimizer,
        episodes,
        config,
        decisions,
        device,
        shuffle_rng,
        monitor,
        _backend=partial(BatchedBackend, force_batched=True),
    )
