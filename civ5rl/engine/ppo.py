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


class RolloutBackend(Protocol):
    """Backend-specific replay and minibatch operations for a PPO update."""

    prefix_seconds: float
    backward_seconds: float

    def precheck(self) -> tuple[float, float]: ...
    def backward(self, minibatch, count, entropy_coefficient) -> int: ...
    def kl(self) -> float: ...
    def record_gradient(self, norm) -> None: ...
    def statistics(self, tokens) -> tuple[dict, float, float]: ...
    def extras(self, optimizer_seconds) -> dict: ...


class ReferenceBackend:
    def __init__(self, policy, episodes, config, device, monitor=None):
        self.policy, self.episodes, self.config, self.device = (
            policy,
            episodes,
            config,
            device,
        )
        self.encoded = [encode_episode(e, policy.variant) for e in episodes]
        self.advantages = targets(episodes)
        self.totals, self.gradients = {}, []
        self.prefix_seconds = self.backward_seconds = 0.0

    def precheck(self):
        pre_kl, pre_error = rollout_kl(
            self.policy, self.episodes, self.encoded, self.device
        )
        if pre_error > 0.001:
            raise RuntimeError(f"Behavior/replay mismatch before PPO: {pre_error}")
        return pre_kl, pre_error

    def backward(self, minibatch, count, entropy_coefficient):
        tokens = 0
        for ep_index, start, stop in minibatch:
            ep, inputs = self.episodes[ep_index], self.encoded[ep_index]
            tick = perf_counter()
            hidden = prefix_hidden(self.policy, inputs, start, self.device)
            self.prefix_seconds += perf_counter() - tick
            tick = perf_counter()
            logits, value, _ = self.policy(
                batch_inputs(inputs[start:stop], self.device), hidden, sequence=True
            )
            index = torch.tensor(ep["chosen"][start:stop], device=self.device)
            old_logp = torch.tensor(ep["old_logp"][start:stop], device=self.device)
            returns = torch.full(
                (stop - start,), float(ep["result"]["reward"]), device=self.device
            )
            adv = torch.tensor(
                self.advantages[ep_index][start:stop], device=self.device
            )
            loss, stats = objective(
                logits,
                value,
                index,
                old_logp,
                returns,
                adv,
                entropy_coefficient,
                clip_ratio=self.config["ppo"]["clip_ratio"],
                value_coefficient=self.config["ppo"]["value_coefficient"],
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite PPO loss")
            (loss * ((stop - start) / count)).backward()
            self.backward_seconds += perf_counter() - tick
            for key, val in stats.items():
                self.totals[key] = self.totals.get(key, 0) + val.item() * (stop - start)
            tokens += stop - start
        return tokens

    def kl(self):
        return rollout_kl(self.policy, self.episodes, self.encoded, self.device)[0]

    def record_gradient(self, norm):
        self.gradients.append(norm.item())

    def statistics(self, tokens):
        return (
            {k: v / tokens for k, v in self.totals.items()},
            float(np.mean(self.gradients)),
            max(self.gradients),
        )

    def extras(self, optimizer_seconds):
        return {}


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
):
    policy.train()
    start_update = perf_counter()
    if _backend is None:
        if config.get("performance", {}).get("batched_ppo", False):
            from civ5rl.engine.ppo_fast import BatchedBackend

            backend: RolloutBackend = BatchedBackend(
                policy, episodes, config, device, monitor
            )
        else:
            backend = ReferenceBackend(policy, episodes, config, device, monitor)
    else:
        backend = _backend(policy, episodes, config, device, monitor)
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
    pre_kl, pre_error = backend.precheck()
    tokens = steps = 0
    kl_seconds = optimizer_seconds = 0.0
    kl = pre_kl
    batch_chunks = config["ppo"]["sequences_per_minibatch"]
    for epoch in range(config["ppo"]["epochs"]):
        order = list(chunks)
        shuffle_rng.shuffle(order)
        for first in range(0, len(order), batch_chunks):
            minibatch = order[first : first + batch_chunks]
            count = sum(stop - start for _, start, stop in minibatch)
            optimizer.zero_grad(set_to_none=True)
            tokens += backend.backward(minibatch, count, entropy_coefficient)
            tick = perf_counter()
            norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                config["ppo"]["max_gradient_norm"],
                error_if_nonfinite=True,
            )
            backend.record_gradient(norm)
            optimizer.step()
            optimizer_seconds += perf_counter() - tick
            steps += 1
        tick = perf_counter()
        kl = backend.kl()
        kl_seconds += perf_counter() - tick
        if monitor:
            monitor.mark("ppo", ppo_epoch=epoch + 1)
        if kl > config["ppo"]["stop_kl"]:
            break
    return _summarize_ppo_update(
        backend,
        episodes,
        tokens,
        steps,
        epoch + 1,
        lr,
        entropy_coefficient,
        kl,
        pre_kl,
        pre_error,
        kl_seconds,
        optimizer_seconds,
        start_update,
    )


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
