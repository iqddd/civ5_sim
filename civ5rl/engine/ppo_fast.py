"""GPU-resident rollout and batched recurrent chunks; same PPO minibatch/targets."""

from time import perf_counter
from typing import NamedTuple

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
from civ5rl.engine.ppo import encode_episode, objective, targets
from civ5rl.engine.replay_metrics import REPLAY_DIAGNOSTIC_KEYS
from civ5rl.engine.sequence import SequenceBuffer
from civ5rl.engine.spans import Span
from civ5rl.engine.training import GradientAccumulator


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
    if (
        amp_enabled
        and numerics.get("require_full_behavior_distribution", False)
        and not use_distribution
    ):
        raise RuntimeError(
            "BF16 replay verification requires complete behavior distributions"
        )
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
        raise RuntimeError(
            f"Behavior/replay mismatch: max_logp_error={max_logp_error}, sampled_KL={pre_kl}, full_KL={full_kl}, max_TV={max_total_variation}; mode={'full_distribution' if use_distribution else 'sampled_logp'}"
        )
    return None if use_distribution else tolerance


class RolloutCache(SequenceBuffer):
    def __init__(self, episodes, variant, device, *, resident=True, cpu_labels=False):
        self.episodes, self.device = episodes, device
        super().__init__(
            [encode_episode(e, variant) for e in episodes], device, resident=resident
        )
        flat = [x for episode in self.encoded for x in episode]
        self.resident = resident
        self.cpu_labels = cpu_labels
        label_device = "cpu" if cpu_labels else device
        advantages = targets(episodes)
        self.labels = {
            "indices": torch.tensor(
                [v for ep in episodes for v in ep["chosen"]], device=label_device
            ),
            "old_logp": torch.tensor(
                [v for ep in episodes for v in ep["old_logp"]], device=label_device
            ),
            "returns": torch.tensor(
                [
                    float(ep["result"]["reward"])
                    for ep in episodes
                    for _ in ep["chosen"]
                ],
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
            self.ragged_behavior = [
                values for ep in episodes for values in ep["old_log_probs"]
            ]
            if len(self.ragged_behavior) != len(flat):
                raise ValueError("Behavior distribution count differs from decisions")
            for i, (values, count) in enumerate(
                zip(self.ragged_behavior, self.action_counts)
            ):
                if (
                    len(values) != count
                    or not np.isfinite(values).all()
                    or float(values[int(self.labels["indices"][i])])
                    != float(self.labels["old_logp"][i])
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
                raise ValueError(
                    "Recorded chosen logp differs from behavior distribution"
                )
            if not torch.allclose(
                behavior.exp().sum(-1),
                torch.ones(len(flat), device=device),
                atol=1e-5,
                rtol=0,
            ):
                raise ValueError("Behavior probabilities do not sum to one")
            self.labels["old_log_probs"] = behavior
        self.resident_bytes = sum(
            v.numel() * v.element_size() for v in (self.data or {}).values()
        ) + sum(v.numel() * v.element_size() for v in self.labels.values())
        self.replay_inputs = False
        self._replay_plan = None
        self._replay_sums_runner = None
        self._pinned_replay = None
        self.pinned_replay_bytes = 0
        self.pinned_replay_hits = 0

    def enable_cached_replay(self):
        """Cache inputs only; GRU state and logits always use current weights."""
        if self.data is None:
            return
        if str(self.device).startswith("cuda"):
            free, _ = torch.cuda.mem_get_info(self.device)
            if free < self.resident_bytes + DEFAULT_GPU_RESERVE_BYTES:
                return
            self._replay_sums_runner = torch.compile(
                _replay_sums, dynamic=True, fullgraph=True
            )
        self.replay_inputs = True

    def enable_pinned_replay(self, limit_bytes=DEFAULT_PINNED_REPLAY_BYTES):
        """Keep immutable replay inputs on CPU; stream each part to the GPU."""
        if (
            self.data is None
            and self.cpu_labels
            and str(self.device).startswith("cuda")
        ):
            self._pinned_replay = {}
            self._pinned_replay_limit = limit_bytes

    def replay_batch(self, spans):
        if self._pinned_replay is None:
            return self.batch(spans)
        key = tuple(spans)
        part = self._pinned_replay.get(key)
        if part is None:
            data, labels, lengths = self.batch(spans, device="cpu")
            size = sum(
                v.numel() * v.element_size() for v in (*data.values(), *labels.values())
            )
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

    def batch(self, spans, *, device=None, include_behavior=True):
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
        data = self.gather_inputs(spans, indices, device=device)
        selected = indices.cpu() if self.cpu_labels else indices
        labels = {
            k: v.index_select(0, selected).to(device)
            for k, v in self.labels.items()
            if include_behavior or k != "old_log_probs"
        }
        if include_behavior and self.ragged_behavior is not None:
            rows = [self.ragged_behavior[i] for i in selected.tolist()]
            behavior = np.full(
                (len(rows), data["action_mask"].shape[1]), -np.inf, np.float32
            )
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
        active = [
            (slot, episode, start)
            for slot, (episode, start, _) in enumerate(spans)
            if start > offset
        ]
        subspans = [
            Span(episode, offset, min(start, offset + block_size))
            for _, episode, start in active
        ]
        if hasattr(cache, "gather_spatial_inputs"):
            batch = cache.gather_spatial_inputs(subspans)
            lengths = [span.length for span in subspans]
        else:
            batch, _, lengths = cache.batch(subspans)
        if policy.spatial_cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        if len(active) == len(spans):
            hidden = policy.forward_sequences(
                batch, lengths, hidden, memory_only=True
            )
        else:
            slots = torch.tensor(
                [slot for slot, _, _ in active],
                dtype=torch.long,
                pin_memory=is_cuda,
            ).to(cache.device, non_blocking=is_cuda)
            last = policy.forward_sequences(
                batch, lengths, hidden.index_select(1, slots), memory_only=True
            )
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


def _replay_parts(cache, episodes):
    max_length = max(len(cache.encoded[episode]) for episode in episodes)
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
                min(offset + DEFAULT_PREFIX_BLOCK_SIZE, len(cache.encoded[episode])),
            )
            for _, episode in active
        ]
        batch, labels, lengths = cache.replay_batch(spans)
        slots = torch.tensor([slot for slot, _ in active], device=cache.device)
        free_count = sum(
            int(
                (
                    cache.action_counts[
                        int(cache.offsets[span.episode]) + span.start : int(
                            cache.offsets[span.episode]
                        )
                        + span.stop
                    ]
                    >= MIN_FREE_ACTIONS
                ).sum()
            )
            for span in spans
        )
        yield batch, labels, lengths, slots, free_count


def _replay_groups(cache, sequences_per_batch):
    # Immutable inputs only: no prefix states or outputs from previous weights.
    previous = getattr(cache, "_replay_plan", None)
    if previous is not None and previous[0] == sequences_per_batch:
        yield from previous[1]
        return
    plan = []
    for first in range(0, len(cache.episodes), sequences_per_batch):
        episodes = list(
            range(first, min(first + sequences_per_batch, len(cache.episodes)))
        )
        parts = _replay_parts(cache, episodes)
        if not getattr(cache, "cpu_labels", False):
            parts = list(parts)
        group = len(episodes), parts
        if getattr(cache, "replay_inputs", False) and not getattr(
            cache, "cpu_labels", False
        ):
            plan.append(group)
        yield group
    if getattr(cache, "replay_inputs", False) and not getattr(
        cache, "cpu_labels", False
    ):
        cache._replay_plan = sequences_per_batch, plan


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
            torch.zeros((1, group_size, policy.hidden_size), device=device)
            if policy.is_recurrent
            else None
        )
        for batch, labels, lengths, slots, part_free in parts:
            if policy.spatial_cuda_graphs:
                torch.compiler.cudagraph_mark_step_begin()
            logits, value, last = policy.forward_sequences(
                batch,
                lengths,
                None if hidden is None else hidden.index_select(1, slots),
            )
            logits = policy_logits(logits, temperature)
            if hidden is not None:
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
                full_free_total += kl.masked_fill(
                    batch["action_mask"].sum(-1) < MIN_FREE_ACTIONS, 0
                ).sum()
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
            1
            - (replay.residual_square_sum / count - (replay.residual_sum / count) ** 2)
            / variance
            if variance > 1e-12
            else None
        ),
        "max_logp_error": replay.max_logp_error,
        "full_kl": replay.full_kl / count
        if full_distribution and cache.has_behavior_distribution
        else None,
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
        self.cache = RolloutCache(
            episodes,
            policy.variant,
            device,
            resident=options.get("resident_rollout", True) if batched else False,
            cpu_labels=options.get("cpu_rollout_labels", False),
        )
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        self.cache_seconds = perf_counter() - tick
        if str(device).startswith("cuda") and options.get(
            "ppo_optimized_inputs", False
        ):
            self.cache.enable_compiled_gather()
            self.cache.enable_async_transfers()
        if options.get("ppo_cached_replay", False):
            self.cache.enable_cached_replay()
        if pinned_replay or options.get("ppo_pinned_replay", False):
            self.cache.enable_pinned_replay()
        self.kl_batch = options.get("kl_sequences_per_batch", 8)
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
        self.micro_chunks = (
            config["ppo"].get("sequences_per_microbatch", 1) if batched else 1
        )

    def precheck(self):
        if self.monitor:
            self.monitor.mark("behavior_replay")
        self.replay = measure_replay(
            self.policy, self.cache, self.kl_batch, full_distribution=True, temperature=self.temperature
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
        return pre_kl, pre_error

    def backward(self, minibatch, count, entropy_coefficient):
        if self.policy.spatial_cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        tick = perf_counter()
        hidden = prefixes(self.policy, self.cache, minibatch)
        self.prefix_seconds += perf_counter() - tick
        finite = torch.ones((), dtype=torch.bool, device=self.device)
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
                None
                if hidden is None
                else hidden[:, micro_start : micro_start + len(spans)].contiguous(),
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
            finite &= torch.isfinite(loss.detach())
            valid_tokens = sum(lengths)
            weighted_loss = loss * (valid_tokens / count)
            accumulator.backward(weighted_loss)
            self.backward_seconds += perf_counter() - tick
            self.totals += (
                torch.stack([stats[key] for key in self.names]) * valid_tokens
            )
            tokens += valid_tokens
        if not finite.item():
            raise FloatingPointError("Non-finite batched PPO loss")
        return tokens

    def kl(self):
        exact = (
            self.config["ppo"].get("step_acceptance") == "full_categorical_kl_rollback"
        )
        if exact and not self.cache.has_behavior_distribution:
            raise ValueError("Controlled PPO requires behavior distributions")
        self.last_replay = measure_replay(
            self.policy, self.cache, self.kl_batch, full_distribution=exact, temperature=self.temperature
        )
        if exact:
            return self.last_replay["full_kl_free"]
        return (
            self.last_replay["sampled_kl_free"]
            if self.config["ppo"].get("kl_population") == "free"
            else self.last_replay["sampled_kl"]
        )

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
            "behavior_check": "full_distribution"
            if self.replay_tolerance is None
            else "sampled_logp",
            "behavior_logp_tolerance": self.replay_tolerance,
            "optimizer_seconds": optimizer_seconds,
            "cache_seconds": self.cache_seconds,
            "rollout_device_bytes": self.cache.resident_bytes,
            "cached_replay_inputs": self.cache.replay_inputs,
            "pinned_replay_input_bytes": self.cache.pinned_replay_bytes,
            "pinned_replay_hits": self.cache.pinned_replay_hits,
            "cached_replay_input_bytes": sum(
                tensor.numel() * tensor.element_size()
                for _, parts in (
                    self.cache._replay_plan[1]
                    if self.cache._replay_plan is not None
                    else []
                )
                for data, labels, _, slots, _ in parts
                for tensor in (*data.values(), *labels.values(), slots)
            ),
            "microbatch_sequences": self.micro_chunks,
        }


def update_fast(
    policy, optimizer, episodes, config, decisions, device, shuffle_rng, monitor=None
):
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
