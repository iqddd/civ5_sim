"""Batched recurrent PPO training with current-weight prefix reconstruction."""

from __future__ import annotations

from functools import lru_cache
from time import perf_counter

import numpy as np
import torch

from civ5rl.core.distributions import require_rollout_temperature, validate_temperature
from civ5rl.core.schema import MIN_FREE_ACTIONS
from civ5rl.engine.ppo.cache import RolloutCache
from civ5rl.engine.ppo.metrics import REPLAY_DIAGNOSTIC_KEYS, StepReplay
from civ5rl.engine.ppo.objective import objective
from civ5rl.engine.ppo.replay import (
    ReplayPart,
    categorical_distance,
    measure_replay,
    ppo_replay_forward,
    validate_behavior_replay,
)
from civ5rl.engine.runtime.sequence import cuda_memory_snapshot, prefixes
from civ5rl.engine.runtime.transactions import GradientAccumulator

__all__ = (
    "BatchedBackend",
    "ReplayPart",
    "RolloutCache",
    "categorical_distance",
    "measure_replay",
    "ppo_replay_forward",
    "prefixes",
    "validate_behavior_replay",
)


_LOSS_METRICS = ("policy_loss", "value_loss", "entropy", "normalized_entropy", "clip_fraction")


def _minibatch_diagnostics(loss, stats, old_logp, action_mask, track_kl, tokens):
    k3_sum = free_tokens = None
    if track_kl:
        log_ratio = (stats["logp"] - old_logp).double()
        k3 = (log_ratio.exp() - 1.0) - log_ratio
        free_mask = action_mask.sum(-1) >= MIN_FREE_ACTIONS
        k3_sum = k3.masked_fill(~free_mask, 0.0).sum()
        free_tokens = free_mask.sum()
    totals = torch.stack([stats[key] for key in _LOSS_METRICS]) * tokens
    return torch.isfinite(loss), totals, k3_sum, free_tokens


@lru_cache(maxsize=1)
def _compiled_minibatch_diagnostics():
    return torch.compile(_minibatch_diagnostics, dynamic=True, fullgraph=True)


class BatchedBackend:
    def __init__(
        self,
        policy,
        episodes,
        config,
        device,
        monitor=None,
        *,
        pinned_replay=False,
        consume_encoded=False,
        sequence_observer=None,
    ):
        self.policy, self.config, self.device, self.monitor = (
            policy,
            config,
            device,
            monitor,
        )
        options = config.get("performance", {})
        self.replay_forward = ppo_replay_forward(
            policy, spatial_graph=options.get("ppo_spatial_cuda_graph", True)
        )
        self.training_forward = policy.forward_sequences
        if sequence_observer is not None:
            self.replay_forward = self._observe_sequences(self.replay_forward, sequence_observer)
            self.training_forward = self._observe_sequences(self.training_forward, sequence_observer)
        self.temperature = validate_temperature(config.get("policy_temperature", 1.0))
        require_rollout_temperature(episodes, self.temperature)
        tick = perf_counter()
        batched = options.get("batched_ppo", False)
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
        if consume_encoded and self.cache.resident:
            for episode in episodes:
                episode.pop("encoded", None)
                episode.pop("encoded_variant", None)
        self.cache.spatial_cell_padding = int(getattr(policy, "spatial_cell_padding", 0) or 0)
        self.kl_batch = options.get("kl_sequences_per_batch", 8)
        self.diagnostics_runner = _minibatch_diagnostics
        if str(device).startswith("cuda") and options.get("ppo_optimized_inputs", False):
            self.cache.enable_compiled_gather()
            self.cache.enable_async_transfers()
            self.cache.enable_compiled_replay_sums()
            self.diagnostics_runner = _compiled_minibatch_diagnostics()
        if options.get("ppo_cached_replay", False):
            self.cache.enable_cached_replay()
        if pinned_replay or options.get("ppo_pinned_replay", False):
            self.cache.enable_pinned_replay()
            if str(device).startswith("cuda") and (
                self.cache.data is None or options.get("ppo_cached_replay", False)
            ):
                self.cache.stage_gpu_replay(self.kl_batch)
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        self.cache_seconds = perf_counter() - tick
        self.names = _LOSS_METRICS
        self.totals = torch.zeros(len(self.names), device=device)
        self.gradients = []
        self.prefix_seconds = self.backward_seconds = 0.0
        self.last_minibatch_approx_kl = 0.0
        self.track_minibatch_kl = (
            ppo_cfg.get("kl_measurement") == "whole_rollout_after_epoch_or_minibatch_spike"
        )
        self.micro_chunks = config["ppo"].get("sequences_per_microbatch", 1) if batched else 1

    @staticmethod
    def _observe_sequences(forward, observer):
        """Trace scored logits without observing recurrent prefix-only forwards."""

        def observed(*args, **kwargs):
            output = forward(*args, **kwargs)
            if not kwargs.get("memory_only"):
                observer(output[0], training=torch.is_grad_enabled())
            return output

        return observed

    def precheck(self):
        if self.monitor:
            self.monitor.mark("behavior_replay")
        self.replay = measure_replay(
            self.policy,
            self.cache,
            self.kl_batch,
            full_distribution=True,
            temperature=self.temperature,
            forward=self.replay_forward,
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
        tokens, diagnostics = self.backward_minibatch(minibatch, count, entropy_coefficient)
        self.read_backward(diagnostics)
        return tokens

    def backward_minibatch(self, minibatch, count, entropy_coefficient):
        if self.policy.spatial_cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        tick = perf_counter()
        hidden = prefixes(self.policy, self.cache, minibatch, forward=self.replay_forward)
        self.prefix_seconds += perf_counter() - tick
        finite = torch.ones((), dtype=torch.bool, device=self.device)
        k3_free_sum = (
            torch.zeros((), dtype=torch.float64, device=self.device) if self.track_minibatch_kl else None
        )
        free_tokens_total = (
            torch.zeros((), dtype=torch.long, device=self.device) if self.track_minibatch_kl else None
        )
        tokens = 0
        accumulator = GradientAccumulator(
            self.policy.parameters(), coalesced=self.policy.coalesced_accumulation
        )
        for micro_start in range(0, len(minibatch), self.micro_chunks):
            spans = minibatch[micro_start : micro_start + self.micro_chunks]
            tick = perf_counter()
            batch, labels, lengths = self.cache.batch(spans, include_behavior=False)
            logits, value, _ = self.training_forward(
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
            valid_tokens = sum(lengths)
            valid, totals, k3_sum, free_tokens = self.diagnostics_runner(
                loss.detach(),
                stats,
                labels["old_logp"],
                batch["action_mask"],
                self.track_minibatch_kl,
                valid_tokens,
            )
            if self.track_minibatch_kl:
                k3_free_sum += k3_sum
                free_tokens_total += free_tokens
            finite &= valid
            accumulator.backward(loss * (valid_tokens / count))
            self.backward_seconds += perf_counter() - tick
            self.totals += totals
            tokens += valid_tokens
        return tokens, (finite, k3_free_sum, free_tokens_total)

    def read_backward(self, diagnostics, norm=None):
        """Read the loss guard, pre-clip norm and optional KL proxy together."""
        finite, k3_sum, free_tokens = diagnostics
        fields = [finite]
        if norm is not None:
            fields.append(norm.detach())
        if k3_sum is not None:
            fields.extend((k3_sum, free_tokens))
        valid, *metrics = torch.stack([v.double() for v in fields]).tolist()
        if not valid:
            raise FloatingPointError("Non-finite batched PPO loss")
        norm_value = metrics.pop(0) if norm is not None else None
        if norm_value is not None and not np.isfinite(norm_value):
            raise FloatingPointError("Non-finite PPO gradient norm")
        self.last_minibatch_approx_kl = metrics[0] / metrics[1] if metrics and metrics[1] else 0.0
        return norm_value

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
            forward=self.replay_forward,
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
                cuda_memory_snapshot(self.device) if str(self.device).startswith("cuda") else None
            ),
            "replay_mode": self.cache.replay_mode,
            "cached_replay_inputs": self.cache.replay_inputs,
            "pinned_replay_input_bytes": self.cache.pinned_replay_bytes,
            "pinned_replay_hits": self.cache.pinned_replay_hits,
            "cached_replay_input_bytes": self.cache.cached_replay_bytes,
            "microbatch_sequences": self.micro_chunks,
        }
