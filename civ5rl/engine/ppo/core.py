"""Episodic recurrent PPO with exact current-weight prefix reconstruction."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from functools import partial
from time import perf_counter
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import torch

from civ5rl.core.model import batch_inputs
from civ5rl.engine.ppo.backend import BatchedBackend
from civ5rl.engine.ppo.metrics import (
    KL_DIAGNOSTIC_KEYS,
    STEP_REPLAY_DIAGNOSTIC_KEYS,
    StepReplay,
)
from civ5rl.engine.ppo.objective import (
    encode_episode as encode_episode,
)
from civ5rl.engine.ppo.objective import (
    episode_returns as episode_returns,
)
from civ5rl.engine.ppo.objective import (
    objective as objective,
)
from civ5rl.engine.ppo.objective import (
    raw_gae_advantages as raw_gae_advantages,
)
from civ5rl.engine.ppo.objective import (
    targets as targets,
)
from civ5rl.engine.runtime.budget import deadline_reached
from civ5rl.engine.runtime.sequence import (
    DEFAULT_PREFIX_BLOCK_SIZE,
    Span,
    pack_minibatches,
)
from civ5rl.engine.runtime.transactions import AdamSnapshot, require_finite_state

if TYPE_CHECKING:
    from civ5rl.core.contracts import Episode


def schedule(spec, decisions):
    fraction = min(decisions / spec["linear_decay_decisions"], 1)
    return spec["initial"] + fraction * (spec["final"] - spec["initial"])


@torch.no_grad()
def prefix_hidden(policy, encoded, end, device, block_size=DEFAULT_PREFIX_BLOCK_SIZE):
    hidden = None
    if policy.is_recurrent:
        for start in range(0, end, block_size):
            _, _, hidden = policy(
                batch_inputs(encoded[start : min(start + block_size, end)], device),
                hidden,
                sequence=True,
            )
    return hidden


class RolloutBackend(Protocol):
    """Backend-specific replay and minibatch operations for a PPO update."""

    prefix_seconds: float
    backward_seconds: float

    def precheck(self) -> Mapping[str, Any]: ...
    def backward(self, minibatch, count, entropy_coefficient) -> int: ...
    def measure_step_replay(self) -> StepReplay: ...
    def record_gradient(self, norm) -> None: ...
    def statistics(self, tokens) -> tuple[dict, float, float]: ...
    def extras(self, optimizer_seconds) -> dict: ...


def token_minibatches(order, valid_tokens):
    """Partition shuffled recurrent spans without losing or joining decisions."""
    for batch, _ in pack_minibatches(order, None, valid_tokens):
        yield batch


@dataclass
class _UpdateMetrics:
    tokens: int = 0
    steps: int = 0
    attempts: int = 0
    rejected: int = 0
    transaction_seconds: float = 0.0
    kl_seconds: float = 0.0
    optimizer_seconds: float = 0.0
    kl: float | None = None
    pre_kl: float | None = None
    pre_error: float = 0.0
    lr: float = 0.0
    entropy_coefficient: float = 0.0
    start_update: float = 0.0
    started_epochs: int = 0
    completed_epochs: int = 0
    step_records: list[dict] = field(default_factory=list)


@dataclass
class _MinibatchStep:
    epoch: int
    tokens: int
    norm: float
    started: float

    def record(self, attempt):
        return {
            "step": attempt,
            "epoch": self.epoch + 1,
            "valid_tokens": self.tokens,
            "gradient_norm_pre_clip": self.norm,
            "seconds": perf_counter() - self.started,
        }


class _KLGate:
    """Owns acceptance, stop decisions and committed-step notifications."""

    transactional = False

    def __init__(self, policy, optimizer, backend, options, metrics, on_step):
        self.policy = policy
        self.optimizer = optimizer
        self.backend = backend
        self.options = options
        self.metrics = metrics
        self.on_step = on_step

    def snapshot(self):
        started = perf_counter()
        snapshot = AdamSnapshot(self.policy, self.optimizer)
        self.metrics.transaction_seconds += perf_counter() - started
        return snapshot

    def measure(self):
        replay = self.backend.measure_step_replay()
        kl = replay.acceptance_kl
        if kl is not None and not np.isfinite(kl):
            raise FloatingPointError("Non-finite post-step KL")
        if self.transactional:
            diagnostics = [
                value
                for key, value in replay.diagnostics.items()
                if key in KL_DIAGNOSTIC_KEYS and value is not None
            ]
            if not np.isfinite(diagnostics).all():
                raise FloatingPointError("Non-finite post-step distribution diagnostics")
        return replay

    def rollback(self, snapshot):
        tick = perf_counter()
        snapshot.restore()
        self.metrics.transaction_seconds += perf_counter() - tick
        restored_kl = self.backend.measure_step_replay().acceptance_kl
        accepted_kl = self.metrics.kl
        if accepted_kl is None:
            if restored_kl is not None:
                raise RuntimeError("Rollback changed KL population")
        elif restored_kl is None or not np.isclose(restored_kl, accepted_kl, rtol=1e-5, atol=1e-7):
            raise RuntimeError(f"Rollback replay mismatch: {restored_kl} != {accepted_kl}")
        self.optimizer.zero_grad(set_to_none=True)

    def publish(self, pending, replay, accepted, *, final_kl):
        attempted_kl = replay.acceptance_kl
        committed_steps = self.metrics.steps + (len(pending) if accepted else 0)
        for index, step in enumerate(pending):
            if accepted:
                self.metrics.steps += 1
            else:
                self.metrics.rejected += 1
            record = {
                **step,
                "accepted": accepted,
                "accepted_step": self.metrics.steps,
                "committed_accepted_steps": committed_steps,
                "commit_batch_steps": len(pending),
                "full_kl_checked": index == len(pending) - 1,
                "attempted_kl": attempted_kl,
                "kl": attempted_kl,
                "final_accepted_kl": final_kl,
                **{
                    key: value
                    for key, value in replay.diagnostics.items()
                    if key in STEP_REPLAY_DIAGNOSTIC_KEYS
                },
            }
            self.metrics.step_records.append(record)
            if self.on_step is not None:
                self.on_step(record)

    def before_step(self):
        pass

    def finish_epoch(self, stop_reason):
        return stop_reason


class _EpochEndKLGate(_KLGate):
    def after_step(self, step, last, deadline):
        self.metrics.steps += 1
        return None, False

    def finish_epoch(self, stop_reason):
        tick = perf_counter()
        self.metrics.kl = self.backend.measure_step_replay().acceptance_kl
        self.metrics.kl_seconds += perf_counter() - tick
        if self.metrics.kl is not None and self.metrics.kl > self.options["stop_kl"]:
            return "stop_kl"
        return stop_reason


class _PerStepKLGate(_KLGate):
    def __init__(self, *args, controlled):
        super().__init__(*args)
        self.transactional = controlled
        self.pre_step = None

    def before_step(self):
        if self.transactional:
            self.pre_step = self.snapshot()

    def after_step(self, step, last, deadline):
        tick = perf_counter()
        replay = self.measure()
        kl = replay.acceptance_kl
        above_limit = kl is not None and kl > self.options["stop_kl"]
        accepted = not self.transactional or not above_limit
        if accepted:
            self.metrics.kl = kl
        else:
            self.rollback(self.pre_step)
        self.pre_step = None
        self.metrics.kl_seconds += perf_counter() - tick
        self.publish(
            [step.record(self.metrics.attempts)],
            replay,
            accepted,
            final_kl=self.metrics.kl if self.transactional else None,
        )
        if deadline_reached(deadline):
            return "active_time_limit", last and accepted
        if not accepted:
            return "rejected_full_kl", False
        if above_limit:
            return "stop_kl", last
        return None, False


class _SpikeOrEpochKLGate(_KLGate):
    transactional = True

    def __init__(self, *args):
        super().__init__(*args)
        self.threshold = float(
            self.options.get("minibatch_kl_spike_threshold", self.options.get("target_kl", 0.01))
        )
        self.verified = self.snapshot()
        self.pending = []
        self.yellow_zone = False

    def verify(self):
        tick = perf_counter()
        replay = self.measure()
        kl = replay.acceptance_kl
        accepted = kl is None or kl <= self.options["stop_kl"]
        if accepted:
            self.verified = self.snapshot()
            self.metrics.kl = kl
        else:
            self.rollback(self.verified)
        self.publish(self.pending, replay, accepted, final_kl=self.metrics.kl)
        self.pending.clear()
        self.metrics.kl_seconds += perf_counter() - tick
        return accepted

    def after_step(self, step, last, deadline):
        record = step.record(self.metrics.attempts)
        approximate_kl = float(getattr(self.backend, "last_minibatch_approx_kl", 0.0))
        record["minibatch_approx_kl"] = approximate_kl
        self.pending.append(record)
        self.yellow_zone |= approximate_kl > self.threshold
        timed_out = deadline_reached(deadline)
        if self.yellow_zone or last or timed_out:
            if not self.verify():
                return "rejected_full_kl", False
            self.yellow_zone |= self.metrics.kl is not None and self.metrics.kl > self.threshold
            if timed_out:
                return "active_time_limit", last
        return None, False

    def finish_epoch(self, stop_reason):
        if self.pending and not self.verify():
            return "rejected_full_kl"
        return stop_reason


def _select_kl_gate(options) -> tuple[Callable[..., _KLGate], bool]:
    controlled = options.get("step_acceptance") == "full_categorical_kl_rollback"
    mode = options.get("kl_measurement", "whole_rollout_after_epoch")
    if mode == "whole_rollout_after_epoch_or_minibatch_spike":
        return _SpikeOrEpochKLGate, controlled
    if mode == "whole_rollout_after_optimizer_step":
        return partial(_PerStepKLGate, controlled=controlled), controlled
    if controlled:
        raise ValueError(
            "full_categorical_kl_rollback requires whole_rollout_after_optimizer_step "
            "or whole_rollout_after_epoch_or_minibatch_spike"
        )
    return _EpochEndKLGate, controlled


def _scheduled_hyperparameters(config, decisions):
    optimizer_options, ppo_options = config["optimizer"], config["ppo"]
    lr = optimizer_options.get("lr")
    if lr is None:
        lr = schedule(optimizer_options["lr_schedule"], decisions)
    entropy_coefficient = ppo_options.get("entropy_coefficient")
    if entropy_coefficient is None:
        entropy_coefficient = schedule(ppo_options["entropy_schedule"], decisions)
    return lr, entropy_coefficient


def _optimizer_step(policy, optimizer, backend, minibatch, options, metrics, gate, epoch):
    started = perf_counter()
    count = sum(stop - start for _, start, stop in minibatch)
    optimizer.zero_grad(set_to_none=True)
    backward_minibatch = getattr(backend, "backward_minibatch", None)
    if backward_minibatch is not None:
        processed, diagnostics = backward_minibatch(minibatch, count, metrics.entropy_coefficient)
    else:
        processed = backend.backward(minibatch, count, metrics.entropy_coefficient)
    if processed != count:
        raise ValueError("PPO backend valid-token count mismatch")
    metrics.tokens += processed
    tick = perf_counter()
    norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        options["max_gradient_norm"],
        error_if_nonfinite=backward_minibatch is None,
    )
    norm_value = backend.read_backward(diagnostics, norm) if backward_minibatch is not None else float(norm)
    backend.record_gradient(norm)
    gate.before_step()
    optimizer.step()
    metrics.attempts += 1
    if gate.transactional:
        require_finite_state(policy, optimizer)
    metrics.optimizer_seconds += perf_counter() - tick
    return _MinibatchStep(epoch, count, norm_value, started)


def update(
    policy,
    optimizer,
    episodes: list[Episode],
    config,
    decisions,
    device,
    shuffle_rng,
    monitor=None,
    *,
    _backend=None,
    deadline=None,
    on_step=None,
):
    options = config["ppo"]
    gate_factory, controlled = _select_kl_gate(options)
    policy.train()
    start_update = perf_counter()
    if monitor:
        monitor.mark("rollout_preparation")
    if _backend is None:
        _backend = BatchedBackend
    backend: RolloutBackend = _backend(policy, episodes, config, device, monitor)
    length = options["sequence_length"]
    chunks = [
        Span(i, start, min(start + length, len(e["chosen"])))
        for i, e in enumerate(episodes)
        for start in range(0, len(e["chosen"]), length)
    ]
    lr, entropy_coefficient = _scheduled_hyperparameters(config, decisions)
    for group in optimizer.param_groups:
        group["lr"] = lr
    initial_replay = backend.precheck()
    pre_kl = initial_replay["sampled_kl"]
    pre_error = initial_replay["max_logp_error"]
    metrics = _UpdateMetrics(
        lr=lr,
        entropy_coefficient=entropy_coefficient,
        pre_kl=pre_kl,
        pre_error=pre_error,
        start_update=start_update,
        kl=initial_replay.get("full_kl_free") if controlled else pre_kl,
    )
    batch_chunks = options["sequences_per_minibatch"]
    token_count = options.get("valid_tokens_per_minibatch")
    stop_reason = "epochs_completed"
    gate = gate_factory(policy, optimizer, backend, options, metrics, on_step)

    for epoch in range(options["epochs"]):
        order = list(chunks)
        shuffle_rng.shuffle(order)
        batches = (
            list(token_minibatches(order, token_count))
            if token_count
            else [order[first : first + batch_chunks] for first in range(0, len(order), batch_chunks)]
        )
        for index, minibatch in enumerate(batches):
            if deadline_reached(deadline):
                stop_reason = "active_time_limit"
                break
            metrics.started_epochs = epoch + 1
            step = _optimizer_step(policy, optimizer, backend, minibatch, options, metrics, gate, epoch)
            reason, epoch_completed = gate.after_step(step, index == len(batches) - 1, deadline)
            if reason is not None:
                stop_reason = reason
                metrics.completed_epochs += int(epoch_completed)
                break
        else:
            metrics.completed_epochs += 1
        stop_reason = gate.finish_epoch(stop_reason)
        if monitor:
            monitor.mark("ppo", ppo_epoch=epoch + 1, optimizer_steps=metrics.steps)
        if stop_reason != "epochs_completed":
            break
    del gate
    gamma = float(config.get("ppo", {}).get("gamma", 1.0))
    gae_lambda = float(config.get("ppo", {}).get("gae_lambda", 1.0))
    summary = _summarize_ppo_update(backend, episodes, metrics, gamma=gamma, gae_lambda=gae_lambda)
    summary.update(
        attempted_optimizer_steps=metrics.attempts,
        accepted_optimizer_steps=metrics.steps,
        rejected_optimizer_steps=metrics.rejected,
        final_accepted_kl=metrics.kl if controlled else None,
        max_attempted_kl=max(
            (r["attempted_kl"] for r in metrics.step_records if r["attempted_kl"] is not None),
            default=None,
        ),
        transaction_seconds=metrics.transaction_seconds,
        epochs_started=metrics.started_epochs,
        epochs_completed=metrics.completed_epochs,
        optimizer_step_records=metrics.step_records,
        stop_reason=stop_reason,
        max_kl=max(
            v
            for v in [
                metrics.pre_kl,
                metrics.kl,
                *[r["kl"] for r in metrics.step_records],
            ]
            if v is not None
        ),
        valid_tokens_per_minibatch=token_count,
        update_complete=stop_reason != "active_time_limit",
    )
    return summary


def _summarize_ppo_update(backend, episodes, metrics: _UpdateMetrics, *, gamma=1.0, gae_lambda=1.0):
    stats, gradient_mean, gradient_max = backend.statistics(metrics.tokens)
    critic_predictions = np.concatenate([e["old_value"] for e in episodes])
    returns = np.concatenate(episode_returns(episodes, gamma=gamma))
    raw_advantages = np.concatenate(raw_gae_advantages(episodes, gamma=gamma, gae_lambda=gae_lambda))
    variance = np.var(returns)
    extras = backend.extras(metrics.optimizer_seconds)
    return {
        **stats,
        "lr": metrics.lr,
        "entropy_coefficient": metrics.entropy_coefficient,
        "epochs": metrics.started_epochs,
        "optimizer_steps": metrics.steps,
        "objective_valid_tokens": metrics.tokens,
        "kl": metrics.kl,
        "pre_update_kl": metrics.pre_kl,
        "pre_update_max_logp_error": metrics.pre_error,
        **extras,
        "gradient_norm_mean": gradient_mean,
        "gradient_norm_max": gradient_max,
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "value_mean": float(critic_predictions.mean()),
        "value_std": float(critic_predictions.std()),
        "raw_advantage_mean": float(raw_advantages.mean()),
        "raw_advantage_std": float(raw_advantages.std()),
        "explained_variance": None
        if variance == 0
        else float(1 - np.var(returns - critic_predictions) / variance),
        "prefix_seconds": backend.prefix_seconds,
        "backward_seconds": backend.backward_seconds,
        "kl_seconds": metrics.kl_seconds,
        "ppo_seconds": perf_counter() - metrics.start_update,
    }
