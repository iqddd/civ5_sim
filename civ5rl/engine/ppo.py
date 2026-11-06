"""Episodic recurrent PPO with exact current-weight prefix reconstruction."""

from dataclasses import dataclass, field
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from civ5rl.core.contracts import Episode

import numpy as np
import torch

from civ5rl.core.context import context_from_spec
from civ5rl.core.model import batch_inputs, distribution_stats
from civ5rl.core.observation import ObservationEncoder
from civ5rl.engine.constants import DEFAULT_PREFIX_BLOCK_SIZE
from civ5rl.engine.replay_metrics import KL_DIAGNOSTIC_KEYS, STEP_REPLAY_DIAGNOSTIC_KEYS
from civ5rl.engine.sequence import pack_minibatches
from civ5rl.engine.spans import Span


def encode_episode(episode: "Episode", variant, *, reuse_cached=True):
    if reuse_cached and "encoded" in episode:
        if episode.get("encoded_variant") != variant:
            raise ValueError("Encoded rollout variant differs from requested policy")
        if len(episode["encoded"]) != len(episode["chosen"]):
            raise ValueError("Encoded rollout count differs from decisions")
        return episode["encoded"]
    s = episode["spec"]
    encoder = ObservationEncoder(
        context_from_spec(s),
        variant,
    )
    result = []
    for obs, actions, chosen in zip(
        episode["observations"], episode["actions"], episode["chosen"]
    ):
        result.append(encoder.encode(obs, actions))
        encoder.remember_action(actions[chosen], obs)
    return result


def targets(episodes):
    advantages = [
        np.asarray([e["result"]["reward"]] * len(e["chosen"]), np.float32)
        - np.asarray(e["old_value"], np.float32)
        for e in episodes
    ]
    all_advantages = np.concatenate(advantages)
    if not len(all_advantages):
        raise ValueError("Empty PPO rollout")
    mean, std = all_advantages.mean(), all_advantages.std()
    return [(a - mean) / (std + 1e-8) for a in advantages]


def objective(
    logits,
    values,
    indices,
    old_logp,
    returns,
    advantages,
    entropy_coefficient,
    *,
    clip_ratio=0.2,
    value_coefficient=0.5,
):
    logp, entropy, normalized_entropy = distribution_stats(logits, indices)
    ratio = (logp - old_logp).exp()
    policy = -torch.minimum(
        ratio * advantages, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages
    ).mean()
    value = (values - returns).square().mean()
    loss = (
        policy
        + value_coefficient * value
        - entropy_coefficient * normalized_entropy.mean()
    )
    return loss, {
        "policy_loss": policy.detach(),
        "value_loss": value.detach(),
        "entropy": entropy.mean().detach(),
        "normalized_entropy": normalized_entropy.mean().detach(),
        "clip_fraction": ((ratio - 1).abs() > clip_ratio).float().mean().detach(),
    }


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

    def precheck(self) -> tuple[float, float]: ...
    def backward(self, minibatch, count, entropy_coefficient) -> int: ...
    def kl(self) -> float | None: ...
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


def _optimizer_attempt(
    policy,
    optimizer,
    backend,
    minibatch,
    config,
    metrics,
    *,
    controlled,
    per_step_kl,
    epoch,
    on_step,
):
    """Run one complete optimizer transaction, including any KL rollback."""
    step_start = perf_counter()
    count = sum(stop - start for _, start, stop in minibatch)
    optimizer.zero_grad(set_to_none=True)
    processed = backend.backward(minibatch, count, metrics.entropy_coefficient)
    if processed != count:
        raise ValueError("PPO backend valid-token count mismatch")
    metrics.tokens += processed
    tick = perf_counter()
    norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), config["ppo"]["max_gradient_norm"], error_if_nonfinite=True
    )
    backend.record_gradient(norm)
    snapshot = None
    if controlled:
        from civ5rl.engine.optimizer_transaction import (
            AdamSnapshot,
            require_finite_state,
        )

        transaction_start = perf_counter()
        snapshot = AdamSnapshot(policy, optimizer)
        metrics.transaction_seconds += perf_counter() - transaction_start
    optimizer.step()
    metrics.attempts += 1
    if controlled:
        require_finite_state(policy, optimizer)
    metrics.optimizer_seconds += perf_counter() - tick
    accepted = True
    if not controlled:
        metrics.steps += 1
    if per_step_kl:
        tick = perf_counter()
        attempted_kl = backend.kl()
        attempted_replay = dict(getattr(backend, "last_replay", {}))
        if attempted_kl is not None and not np.isfinite(attempted_kl):
            raise FloatingPointError("Non-finite post-step KL")
        if controlled:
            kl_diagnostics = [
                v
                for k, v in attempted_replay.items()
                if k in KL_DIAGNOSTIC_KEYS and v is not None
            ]
            if not np.isfinite(kl_diagnostics).all():
                raise FloatingPointError(
                    "Non-finite post-step distribution diagnostics"
                )
            accepted = attempted_kl is None or attempted_kl <= config["ppo"]["stop_kl"]
            if accepted:
                metrics.steps += 1
                metrics.kl = attempted_kl
            else:
                transaction_start = perf_counter()
                snapshot.restore()
                metrics.transaction_seconds += perf_counter() - transaction_start
                metrics.rejected += 1
                restored_kl = backend.kl()
                if metrics.kl is None:
                    if restored_kl is not None:
                        raise RuntimeError("Rollback changed KL population")
                elif restored_kl is None or not np.isclose(
                    restored_kl, metrics.kl, rtol=1e-5, atol=1e-7
                ):
                    raise RuntimeError(
                        f"Rollback replay mismatch: {restored_kl} != {metrics.kl}"
                    )
                optimizer.zero_grad(set_to_none=True)
            del snapshot
        else:
            metrics.kl = attempted_kl
        metrics.kl_seconds += perf_counter() - tick
        if metrics.kl is not None and not np.isfinite(metrics.kl):
            raise FloatingPointError("Non-finite post-step KL")
        record = {
            "step": metrics.attempts,
            "accepted": accepted,
            "accepted_step": metrics.steps,
            "attempted_kl": attempted_kl,
            "epoch": epoch + 1,
            "valid_tokens": count,
            "kl": attempted_kl,
            "final_accepted_kl": metrics.kl if controlled else None,
            "gradient_norm_pre_clip": float(norm),
            "seconds": perf_counter() - step_start,
            **{
                k: v
                for k, v in attempted_replay.items()
                if k in STEP_REPLAY_DIAGNOSTIC_KEYS
            },
        }
        metrics.step_records.append(record)
        if on_step is not None:
            on_step(record)
    return accepted


def update(
    policy,
    optimizer,
    episodes: list["Episode"],
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
    policy.train()
    start_update = perf_counter()
    if _backend is None:
        from civ5rl.engine.ppo_fast import BatchedBackend

        _backend = BatchedBackend
    backend: RolloutBackend = _backend(policy, episodes, config, device, monitor)
    length = config["ppo"]["sequence_length"]
    chunks = [
        Span(i, start, min(start + length, len(e["chosen"])))
        for i, e in enumerate(episodes)
        for start in range(0, len(e["chosen"]), length)
    ]
    lr = config["optimizer"].get("lr")
    if lr is None:
        lr = schedule(config["optimizer"]["lr_schedule"], decisions)
    entropy_coefficient = config["ppo"].get("entropy_coefficient")
    if entropy_coefficient is None:
        entropy_coefficient = schedule(config["ppo"]["entropy_schedule"], decisions)
    for group in optimizer.param_groups:
        group["lr"] = lr
    pre_kl, pre_error = backend.precheck()
    controlled = config["ppo"].get("step_acceptance") == "full_categorical_kl_rollback"
    metrics = _UpdateMetrics(
        lr=lr,
        entropy_coefficient=entropy_coefficient,
        pre_kl=pre_kl,
        pre_error=pre_error,
        start_update=start_update,
        kl=backend.replay["full_kl_free"] if controlled else pre_kl,
    )
    batch_chunks = config["ppo"]["sequences_per_minibatch"]
    token_count = config["ppo"].get("valid_tokens_per_minibatch")
    per_step_kl = (
        config["ppo"].get("kl_measurement", "whole_rollout_after_epoch")
        == "whole_rollout_after_optimizer_step"
    )
    stop_reason = "epochs_completed"
    for epoch in range(config["ppo"]["epochs"]):
        order = list(chunks)
        shuffle_rng.shuffle(order)
        batches = (
            list(token_minibatches(order, token_count))
            if token_count
            else [
                order[first : first + batch_chunks]
                for first in range(0, len(order), batch_chunks)
            ]
        )
        for batch_index, minibatch in enumerate(batches):
            if deadline is not None and perf_counter() >= deadline:
                stop_reason = "active_time_limit"
                break
            metrics.started_epochs = epoch + 1
            accepted = _optimizer_attempt(
                policy,
                optimizer,
                backend,
                minibatch,
                config,
                metrics,
                controlled=controlled,
                per_step_kl=per_step_kl,
                epoch=epoch,
                on_step=on_step,
            )
            if per_step_kl:
                timed_out = deadline is not None and perf_counter() >= deadline
                if (
                    timed_out
                    or not accepted
                    or (
                        not controlled
                        and metrics.kl is not None
                        and metrics.kl > config["ppo"]["stop_kl"]
                    )
                ):
                    stop_reason = (
                        "active_time_limit"
                        if timed_out
                        else ("rejected_full_kl" if controlled else "stop_kl")
                    )
                    if batch_index == len(batches) - 1 and (not controlled or accepted):
                        metrics.completed_epochs += 1
                    break
        else:
            metrics.completed_epochs += 1
        if not per_step_kl:
            tick = perf_counter()
            metrics.kl = backend.kl()
            metrics.kl_seconds += perf_counter() - tick
            if metrics.kl is not None and metrics.kl > config["ppo"]["stop_kl"]:
                stop_reason = "stop_kl"
        if monitor:
            monitor.mark("ppo", ppo_epoch=epoch + 1, optimizer_steps=metrics.steps)
        if stop_reason in ("stop_kl", "rejected_full_kl", "active_time_limit"):
            break
    summary = _summarize_ppo_update(backend, episodes, metrics)
    summary.update(
        attempted_optimizer_steps=metrics.attempts,
        accepted_optimizer_steps=metrics.steps,
        rejected_optimizer_steps=metrics.rejected,
        final_accepted_kl=metrics.kl if controlled else None,
        max_attempted_kl=max(
            (
                r["attempted_kl"]
                for r in metrics.step_records
                if r["attempted_kl"] is not None
            ),
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


def _summarize_ppo_update(backend, episodes, metrics: _UpdateMetrics):
    stats, gradient_mean, gradient_max = backend.statistics(metrics.tokens)
    critic_predictions = np.concatenate([e["old_value"] for e in episodes])
    returns = np.concatenate(
        [[e["result"]["reward"]] * len(e["chosen"]) for e in episodes]
    )
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
        "raw_advantage_mean": float((returns - critic_predictions).mean()),
        "raw_advantage_std": float((returns - critic_predictions).std()),
        "explained_variance": None
        if variance == 0
        else float(1 - np.var(returns - critic_predictions) / variance),
        "prefix_seconds": backend.prefix_seconds,
        "backward_seconds": backend.backward_seconds,
        "kl_seconds": metrics.kl_seconds,
        "ppo_seconds": perf_counter() - metrics.start_update,
    }
