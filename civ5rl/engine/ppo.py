"""Episodic recurrent PPO with exact current-weight prefix reconstruction."""

from time import perf_counter
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from civ5rl.core.contracts import Episode

import numpy as np
import torch

from civ5rl.core.context import context_from_spec
from civ5rl.core.model import batch_inputs, distribution_stats
from civ5rl.core.observation import ObservationEncoder
from civ5rl.engine.sequence import pack_minibatches


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
def prefix_hidden(policy, encoded, end, device, block_size=128):
    hidden = None
    if policy.variant == "C":
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
):
    policy.train()
    start_update = perf_counter()
    if _backend is None:
        from civ5rl.engine.ppo_fast import BatchedBackend

        _backend = BatchedBackend
    backend: RolloutBackend = _backend(policy, episodes, config, device, monitor)
    length = config["ppo"]["sequence_length"]
    chunks = [
        (i, start, min(start + length, len(e["chosen"])))
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
    if controlled:
        from civ5rl.engine.optimizer_transaction import AdamSnapshot, require_finite_state
    tokens = steps = attempts = rejected = 0
    transaction_seconds = 0.0
    kl_seconds = optimizer_seconds = 0.0
    kl = backend.replay["full_kl_free"] if controlled else pre_kl
    batch_chunks = config["ppo"]["sequences_per_minibatch"]
    token_count = config["ppo"].get("valid_tokens_per_minibatch")
    per_step_kl = (
        config["ppo"].get("kl_measurement", "whole_rollout_after_epoch")
        == "whole_rollout_after_optimizer_step"
    )
    step_records = []
    completed_epochs = 0
    started_epochs = 0
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
            started_epochs = epoch + 1
            step_start = perf_counter()
            count = sum(stop - start for _, start, stop in minibatch)
            optimizer.zero_grad(set_to_none=True)
            processed = backend.backward(minibatch, count, entropy_coefficient)
            if processed != count:
                raise ValueError("PPO backend valid-token count mismatch")
            tokens += processed
            tick = perf_counter()
            norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                config["ppo"]["max_gradient_norm"],
                error_if_nonfinite=True,
            )
            backend.record_gradient(norm)
            snapshot = None
            if controlled:
                transaction_start = perf_counter()
                snapshot = AdamSnapshot(policy, optimizer)
                transaction_seconds += perf_counter() - transaction_start
            optimizer.step()
            attempts += 1
            if controlled:
                require_finite_state(policy, optimizer)
            optimizer_seconds += perf_counter() - tick
            accepted = True
            attempted_kl = None
            if not controlled:
                steps += 1
            if per_step_kl:
                tick = perf_counter()
                attempted_kl = backend.kl()
                attempted_replay = dict(getattr(backend, "last_replay", {}))
                if attempted_kl is not None and not np.isfinite(attempted_kl):
                    raise FloatingPointError("Non-finite post-step KL")
                if controlled:
                    values = [v for k, v in attempted_replay.items()
                              if k in ("full_kl", "full_kl_free", "sampled_kl", "sampled_kl_free")
                              and v is not None]
                    if not np.isfinite(values).all():
                        raise FloatingPointError("Non-finite post-step distribution diagnostics")
                    accepted = attempted_kl is None or attempted_kl <= config["ppo"]["stop_kl"]
                    if accepted:
                        steps += 1
                        kl = attempted_kl
                    else:
                        transaction_start = perf_counter()
                        snapshot.restore()
                        transaction_seconds += perf_counter() - transaction_start
                        rejected += 1
                        restored_kl = backend.kl()
                        if kl is None:
                            if restored_kl is not None:
                                raise RuntimeError("Rollback changed KL population")
                        elif restored_kl is None or not np.isclose(restored_kl, kl, rtol=1e-5, atol=1e-7):
                            raise RuntimeError(f"Rollback replay mismatch: {restored_kl} != {kl}")
                        optimizer.zero_grad(set_to_none=True)
                    del snapshot
                else:
                    kl = attempted_kl
                kl_seconds += perf_counter() - tick
                if kl is not None and not np.isfinite(kl):
                    raise FloatingPointError("Non-finite post-step KL")
                step_records.append(
                    {
                        "step": attempts,
                        "accepted": accepted,
                        "accepted_step": steps,
                        "attempted_kl": attempted_kl,
                        "epoch": epoch + 1,
                        "valid_tokens": count,
                        "kl": attempted_kl,
                        "final_accepted_kl": kl if controlled else None,
                        "gradient_norm_pre_clip": float(norm),
                        "seconds": perf_counter() - step_start,
                        **{
                            k: v
                            for k, v in attempted_replay.items()
                            if k
                            in (
                                "sampled_kl",
                                "sampled_kl_free",
                                "full_kl",
                                "full_kl_free",
                                "free_decisions",
                                "all_decisions",
                            )
                        },
                    }
                )
                timed_out = deadline is not None and perf_counter() >= deadline
                if timed_out or not accepted or (not controlled and kl is not None and kl > config["ppo"]["stop_kl"]):
                    stop_reason = "active_time_limit" if timed_out else ("rejected_full_kl" if controlled else "stop_kl")
                    if batch_index == len(batches) - 1 and (not controlled or accepted):
                        completed_epochs += 1
                    break
        else:
            completed_epochs += 1
        if not per_step_kl:
            tick = perf_counter()
            kl = backend.kl()
            kl_seconds += perf_counter() - tick
            if kl is not None and kl > config["ppo"]["stop_kl"]:
                stop_reason = "stop_kl"
        if monitor:
            monitor.mark("ppo", ppo_epoch=epoch + 1, optimizer_steps=steps)
        if stop_reason in ("stop_kl", "rejected_full_kl", "active_time_limit"):
            break
    summary = _summarize_ppo_update(
        backend,
        episodes,
        tokens,
        steps,
        started_epochs,
        lr,
        entropy_coefficient,
        kl,
        pre_kl,
        pre_error,
        kl_seconds,
        optimizer_seconds,
        start_update,
    )
    summary.update(
        attempted_optimizer_steps=attempts,
        accepted_optimizer_steps=steps,
        rejected_optimizer_steps=rejected,
        final_accepted_kl=kl if controlled else None,
        max_attempted_kl=max((r["attempted_kl"] for r in step_records if r["attempted_kl"] is not None), default=None),
        transaction_seconds=transaction_seconds,
        epochs_started=started_epochs,
        epochs_completed=completed_epochs,
        optimizer_step_records=step_records,
        stop_reason=stop_reason,
        max_kl=max(
            v for v in [pre_kl, kl, *[r["kl"] for r in step_records]] if v is not None
        ),
        valid_tokens_per_minibatch=token_count,
        update_complete=stop_reason != "active_time_limit",
    )
    return summary


def _summarize_ppo_update(
    backend,
    episodes,
    tokens,
    steps,
    epochs,
    lr,
    entropy_coefficient,
    kl,
    pre_kl,
    pre_error,
    kl_seconds,
    optimizer_seconds,
    start_update,
):
    stats, gradient_mean, gradient_max = backend.statistics(tokens)
    values = np.concatenate([e["old_value"] for e in episodes])
    returns = np.concatenate(
        [[e["result"]["reward"]] * len(e["chosen"]) for e in episodes]
    )
    variance = np.var(returns)
    extras = backend.extras(optimizer_seconds)
    return {
        **stats,
        "lr": lr,
        "entropy_coefficient": entropy_coefficient,
        "epochs": epochs,
        "optimizer_steps": steps,
        "objective_valid_tokens": tokens,
        "kl": kl,
        "pre_update_kl": pre_kl,
        "pre_update_max_logp_error": pre_error,
        **extras,
        "gradient_norm_mean": gradient_mean,
        "gradient_norm_max": gradient_max,
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "value_mean": float(values.mean()),
        "value_std": float(values.std()),
        "raw_advantage_mean": float((returns - values).mean()),
        "raw_advantage_std": float((returns - values).std()),
        "explained_variance": None
        if variance == 0
        else float(1 - np.var(returns - values) / variance),
        "prefix_seconds": backend.prefix_seconds,
        "backward_seconds": backend.backward_seconds,
        "kl_seconds": kl_seconds,
        "ppo_seconds": perf_counter() - start_update,
    }
