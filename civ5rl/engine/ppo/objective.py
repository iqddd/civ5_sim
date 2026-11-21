"""Rollout encoding, return targets and the recurrent PPO objective."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from civ5rl.core.context import context_from_spec
from civ5rl.core.distributions import policy_logits
from civ5rl.core.model import distribution_stats
from civ5rl.core.observation import ObservationEncoder

if TYPE_CHECKING:
    from civ5rl.core.contracts import Episode


def encode_episode(episode: Episode, variant, *, reuse_cached=True):
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
    for obs, actions, chosen in zip(episode["observations"], episode["actions"], episode["chosen"]):
        result.append(encoder.encode(obs, actions))
        encoder.remember_action(actions[chosen], obs)
    return result


def episode_returns(episodes, gamma=1.0):
    gamma = float(gamma)
    result = []
    for e in episodes:
        t = len(e["chosen"])
        if t == 0:
            result.append(np.empty(0, dtype=np.float32))
            continue
        reward = float(e["result"]["reward"])
        if reward == 0.0:
            result.append(np.zeros(t, dtype=np.float32))
        elif gamma == 1.0:
            result.append(np.full(t, reward, dtype=np.float32))
        else:
            powers = np.arange(t - 1, -1, -1, dtype=np.float64)
            result.append((reward * (gamma**powers)).astype(np.float32))
    return result


def raw_gae_advantages(episodes, gamma=1.0, gae_lambda=1.0):
    gamma = float(gamma)
    gae_lambda = float(gae_lambda)
    if gamma == 1.0 and gae_lambda == 1.0:
        return [
            np.asarray([e["result"]["reward"]] * len(e["chosen"]), np.float32)
            - np.asarray(e["old_value"], np.float32)
            for e in episodes
        ]
    discount = gamma * gae_lambda
    advantages = []
    for e in episodes:
        t = len(e["chosen"])
        if t == 0:
            advantages.append(np.empty(0, dtype=np.float32))
            continue
        reward = float(e["result"]["reward"])
        values = np.asarray(e["old_value"], dtype=np.float64)
        adv = np.empty(t, dtype=np.float64)
        gae = 0.0
        next_val = 0.0
        for idx in range(t - 1, -1, -1):
            r_t = reward if idx == t - 1 else 0.0
            delta = r_t + gamma * next_val - values[idx]
            gae = delta + discount * gae
            adv[idx] = gae
            next_val = values[idx]
        advantages.append(adv.astype(np.float32))
    return advantages


def targets(episodes, *, gamma=1.0, gae_lambda=1.0):
    advantages = raw_gae_advantages(episodes, gamma=gamma, gae_lambda=gae_lambda)
    all_advantages = np.concatenate(advantages) if advantages else np.empty(0, np.float32)
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
    temperature=1.0,
):
    use_entropy = bool(entropy_coefficient)
    logp, entropy, normalized_entropy = distribution_stats(
        policy_logits(logits, temperature),
        indices,
        entropy_grad=use_entropy,
    )
    ratio = (logp - old_logp).exp()
    policy = -torch.minimum(
        ratio * advantages, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages
    ).mean()
    value = (values - returns).square().mean()
    loss = policy + value_coefficient * value
    if use_entropy:
        loss = loss - entropy_coefficient * normalized_entropy.mean()
    return loss, {
        "policy_loss": policy.detach(),
        "value_loss": value.detach(),
        "entropy": entropy.mean().detach(),
        "normalized_entropy": normalized_entropy.mean().detach(),
        "clip_fraction": ((ratio - 1).abs() > clip_ratio).float().mean().detach(),
        "logp": logp.detach(),
    }
