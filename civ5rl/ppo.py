"""Episodic recurrent PPO with exact current-weight prefix reconstruction."""

from time import perf_counter

import numpy as np
import torch

from .corpus import context_for
from .model import batch_inputs, distribution_stats
from .observation import ObservationEncoder


def encode_episode(episode, variant, *, reuse_cached=True):
    if reuse_cached and "encoded" in episode:
        if episode.get("encoded_variant") != variant:
            raise ValueError("Encoded rollout variant differs from requested policy")
        if len(episode["encoded"]) != len(episode["chosen"]):
            raise ValueError("Encoded rollout count differs from decisions")
        return episode["encoded"]
    s = episode["spec"]
    encoder = ObservationEncoder(
        context_for(s["scenario"], s["armies"], s["learner_side"], s["max_rounds"], 0),
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
    logits, values, indices, old_logp, returns, advantages, entropy_coefficient
):
    logp, entropy, normalized_entropy = distribution_stats(logits, indices)
    ratio = (logp - old_logp).exp()
    policy = -torch.minimum(
        ratio * advantages, ratio.clamp(0.8, 1.2) * advantages
    ).mean()
    value = (values - returns).square().mean()
    loss = policy + 0.5 * value - entropy_coefficient * normalized_entropy.mean()
    return loss, {
        "policy_loss": policy.detach(),
        "value_loss": value.detach(),
        "entropy": entropy.mean().detach(),
        "normalized_entropy": normalized_entropy.mean().detach(),
        "clip_fraction": ((ratio - 1).abs() > 0.2).float().mean().detach(),
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


@torch.no_grad()
def rollout_kl(policy, episodes, encoded, device):
    total, count, max_logp_error = 0.0, 0, 0.0
    for ep, inputs in zip(episodes, encoded):
        hidden = None
        for start in range(0, len(inputs), 128):
            stop = min(start + 128, len(inputs))
            logits, _, hidden = policy(
                batch_inputs(inputs[start:stop], device), hidden, sequence=True
            )
            indices = torch.tensor(ep["chosen"][start:stop], device=device)
            logp, _, _ = distribution_stats(logits, indices)
            delta = logp - torch.tensor(ep["old_logp"][start:stop], device=device)
            total += (delta.exp() - 1 - delta).sum().item()
            max_logp_error = max(max_logp_error, delta.abs().max().item())
            count += stop - start
    return total / count, max_logp_error


def update(
    policy, optimizer, episodes, config, decisions, device, shuffle_rng, monitor=None
):
    if config.get("performance", {}).get("batched_ppo", False):
        from .ppo_fast import update_fast

        return update_fast(
            policy, optimizer, episodes, config, decisions, device, shuffle_rng, monitor
        )
    policy.train()
    start_update = perf_counter()
    encoded = [encode_episode(e, policy.variant) for e in episodes]
    advantages = targets(episodes)
    length = config["ppo"]["sequence_length"]
    chunks = [
        (i, start, min(start + length, len(e["chosen"])))
        for i, e in enumerate(episodes)
        for start in range(0, len(e["chosen"]), length)
    ]
    lr = schedule(config["optimizer"]["lr_schedule"], decisions)
    entropy_coefficient = schedule(config["ppo"]["entropy_schedule"], decisions)
    for group in optimizer.param_groups:
        group["lr"] = lr
    pre_kl, pre_error = rollout_kl(policy, episodes, encoded, device)
    if pre_error > 0.001:
        raise RuntimeError(f"Behavior/replay mismatch before PPO: {pre_error}")
    totals, tokens, gradients = {}, 0, []
    prefix_seconds = backward_seconds = kl_seconds = 0.0
    kl = pre_kl
    batch_chunks = config["ppo"]["sequences_per_minibatch"]
    for epoch in range(config["ppo"]["epochs"]):
        order = list(chunks)
        shuffle_rng.shuffle(order)
        for batch_start in range(0, len(order), batch_chunks):
            minibatch = order[batch_start : batch_start + batch_chunks]
            count = sum(stop - start for _, start, stop in minibatch)
            optimizer.zero_grad(set_to_none=True)
            for ep_index, start, stop in minibatch:
                ep, inputs = episodes[ep_index], encoded[ep_index]
                tick = perf_counter()
                hidden = prefix_hidden(policy, inputs, start, device)
                prefix_seconds += perf_counter() - tick
                tick = perf_counter()
                logits, value, _ = policy(
                    batch_inputs(inputs[start:stop], device), hidden, sequence=True
                )
                index = torch.tensor(ep["chosen"][start:stop], device=device)
                old_logp = torch.tensor(ep["old_logp"][start:stop], device=device)
                returns = torch.full(
                    (stop - start,), float(ep["result"]["reward"]), device=device
                )
                adv = torch.tensor(advantages[ep_index][start:stop], device=device)
                loss, stats = objective(
                    logits, value, index, old_logp, returns, adv, entropy_coefficient
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                (loss * ((stop - start) / count)).backward()
                backward_seconds += perf_counter() - tick
                for key, val in stats.items():
                    totals[key] = totals.get(key, 0) + val.item() * (stop - start)
                tokens += stop - start
            norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                config["ppo"]["max_gradient_norm"],
                error_if_nonfinite=True,
            )
            gradients.append(norm.item())
            optimizer.step()
        tick = perf_counter()
        kl, _ = rollout_kl(policy, episodes, encoded, device)
        kl_seconds += perf_counter() - tick
        if monitor:
            monitor.mark("ppo", ppo_epoch=epoch + 1)
        if kl > config["ppo"]["stop_kl"]:
            break
    values = np.concatenate([e["old_value"] for e in episodes])
    returns = np.concatenate(
        [[e["result"]["reward"]] * len(e["chosen"]) for e in episodes]
    )
    variance = np.var(returns)
    return {
        **{k: v / tokens for k, v in totals.items()},
        "lr": lr,
        "entropy_coefficient": entropy_coefficient,
        "epochs": epoch + 1,
        "kl": kl,
        "pre_update_kl": pre_kl,
        "pre_update_max_logp_error": pre_error,
        "gradient_norm_mean": float(np.mean(gradients)),
        "gradient_norm_max": max(gradients),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "value_mean": float(values.mean()),
        "value_std": float(values.std()),
        "raw_advantage_mean": float((returns - values).mean()),
        "raw_advantage_std": float((returns - values).std()),
        "explained_variance": None
        if variance == 0
        else float(1 - np.var(returns - values) / variance),
        "prefix_seconds": prefix_seconds,
        "backward_seconds": backward_seconds,
        "kl_seconds": kl_seconds,
        "ppo_seconds": perf_counter() - start_update,
    }
