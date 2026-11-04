"""GPU-resident rollout and batched recurrent chunks; same PPO minibatch/targets."""

from time import perf_counter

import numpy as np
import torch

from .model import batch_inputs, distribution_stats
from .ppo import encode_episode, objective, schedule, targets


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


class RolloutCache:
    def __init__(self, episodes, variant, device, *, resident=True):
        self.episodes, self.device = episodes, device
        self.encoded = [encode_episode(e, variant) for e in episodes]
        self.offsets = np.cumsum([0, *[len(e) for e in self.encoded]])
        flat = [x for episode in self.encoded for x in episode]
        self.action_counts = np.asarray([len(x.sources) for x in flat])
        self.resident = resident
        self.data = batch_inputs(flat, device, non_blocking=True) if resident else None
        advantages = targets(episodes)
        self.labels = {
            "indices": torch.tensor(
                [v for ep in episodes for v in ep["chosen"]], device=device
            ),
            "old_logp": torch.tensor(
                [v for ep in episodes for v in ep["old_logp"]], device=device
            ),
            "returns": torch.tensor(
                [
                    float(ep["result"]["reward"])
                    for ep in episodes
                    for _ in ep["chosen"]
                ],
                device=device,
            ),
            "advantages": torch.tensor(np.concatenate(advantages), device=device),
        }
        has_full = ["old_log_probs" in ep for ep in episodes]
        if any(has_full) and not all(has_full):
            raise ValueError("Mixed complete/incomplete behavior distributions")
        self.has_behavior_distribution = all(has_full)
        if self.has_behavior_distribution:
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

    def batch(self, spans):
        # Spans are independent, internally ordered episode intervals.
        indices = torch.cat(
            [
                torch.arange(
                    int(self.offsets[ep]) + start,
                    int(self.offsets[ep]) + stop,
                    device=self.device,
                )
                for ep, start, stop in spans
            ]
        )
        if self.data is None:
            data = batch_inputs(
                [x for ep, start, stop in spans for x in self.encoded[ep][start:stop]],
                self.device,
                non_blocking=True,
            )
        else:
            data = {k: v.index_select(0, indices) for k, v in self.data.items()}
            max_actions = max(
                int(
                    self.action_counts[
                        self.offsets[ep] + start : self.offsets[ep] + stop
                    ].max()
                )
                for ep, start, stop in spans
            )
            for key in ("sources", "targets", "action_features", "action_mask"):
                data[key] = data[key][:, :max_actions].contiguous()
        labels = {k: v.index_select(0, indices) for k, v in self.labels.items()}
        return data, labels, [stop - start for _, start, stop in spans]


@torch.no_grad()
def prefixes(policy, cache, spans, block_size=128):
    if policy.variant != "C":
        return None
    hidden = torch.zeros((1, len(spans), 256), device=cache.device)
    for offset in range(0, max(start for _, start, _ in spans), block_size):
        active = [
            (j, ep, start) for j, (ep, start, _) in enumerate(spans) if start > offset
        ]
        subspans = [
            (ep, offset, min(start, offset + block_size)) for _, ep, start in active
        ]
        batch, _, lengths = cache.batch(subspans)
        slots = torch.tensor([j for j, _, _ in active], device=cache.device)
        last = policy.forward_sequences(
            batch, lengths, hidden.index_select(1, slots), memory_only=True
        )
        hidden.index_copy_(1, slots, last)
    return hidden


@torch.no_grad()
def measure_replay(policy, cache, sequences_per_batch=8, *, full_distribution=False):
    device = cache.device
    total, maximum = torch.zeros((), device=device), torch.zeros((), device=device)
    count = 0
    full_total = torch.zeros((), device=device, dtype=torch.float64)
    max_tv = torch.zeros_like(full_total)
    for first in range(0, len(cache.episodes), sequences_per_batch):
        episodes = list(
            range(first, min(first + sequences_per_batch, len(cache.episodes)))
        )
        hidden = (
            torch.zeros((1, len(episodes), 256), device=device)
            if policy.variant == "C"
            else None
        )
        for offset in range(0, max(len(cache.encoded[ep]) for ep in episodes), 128):
            active = [
                (j, ep)
                for j, ep in enumerate(episodes)
                if len(cache.encoded[ep]) > offset
            ]
            spans = [
                (ep, offset, min(offset + 128, len(cache.encoded[ep])))
                for _, ep in active
            ]
            batch, labels, lengths = cache.batch(spans)
            slots = torch.tensor([j for j, _ in active], device=device)
            logits, _, last = policy.forward_sequences(
                batch,
                lengths,
                None if hidden is None else hidden.index_select(1, slots),
            )
            if hidden is not None:
                hidden.index_copy_(1, slots, last)
            logp, _, _ = distribution_stats(logits, labels["indices"])
            delta = logp - labels["old_logp"]
            total += (delta.exp() - 1 - delta).sum()
            maximum = torch.maximum(maximum, delta.abs().max())
            if full_distribution and cache.has_behavior_distribution:
                kl, tv = categorical_distance(
                    labels["old_log_probs"][:, : logits.shape[1]],
                    logits,
                    batch["action_mask"],
                )
                full_total += kl.sum()
                max_tv = torch.maximum(max_tv, tv.max())
            count += len(delta)
    return {
        "sampled_kl": total.item() / count,
        "max_logp_error": maximum.item(),
        "full_kl": full_total.item() / count
        if full_distribution and cache.has_behavior_distribution
        else None,
        "max_total_variation": max_tv.item()
        if full_distribution and cache.has_behavior_distribution
        else None,
    }


def measure_kl(policy, cache, sequences_per_batch=8):
    result = measure_replay(policy, cache, sequences_per_batch)
    return result["sampled_kl"], result["max_logp_error"]


def update_fast(
    policy, optimizer, episodes, config, decisions, device, shuffle_rng, monitor=None
):
    policy.train()
    start_update = perf_counter()
    options = config.get("performance", {})
    tick = perf_counter()
    cache = RolloutCache(
        episodes, policy.variant, device, resident=options.get("resident_rollout", True)
    )
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    cache_seconds = perf_counter() - tick
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
    kl_batch = options.get("kl_sequences_per_batch", 8)
    if monitor:
        monitor.mark("behavior_replay")
    replay = measure_replay(policy, cache, kl_batch, full_distribution=True)
    pre_kl, pre_error = replay["sampled_kl"], replay["max_logp_error"]
    replay_tolerance = validate_behavior_replay(
        pre_kl,
        pre_error,
        amp_enabled=policy.amp_enabled,
        numerics=config["numerics"],
        full_kl=replay["full_kl"],
        max_total_variation=replay["max_total_variation"],
    )
    if monitor:
        monitor.mark("ppo", ppo_epoch=0)
    names = (
        "policy_loss",
        "value_loss",
        "entropy",
        "normalized_entropy",
        "clip_fraction",
    )
    totals = torch.zeros(len(names), device=device)
    tokens, gradients = 0, []
    prefix_seconds = backward_seconds = kl_seconds = optimizer_seconds = 0.0
    batch_chunks = config["ppo"]["sequences_per_minibatch"]
    micro_chunks = config["ppo"]["sequences_per_microbatch"]
    if not 1 <= micro_chunks <= batch_chunks:
        raise ValueError("Microbatch must contain 1..minibatch chunks")
    for epoch in range(config["ppo"]["epochs"]):
        order = list(chunks)
        shuffle_rng.shuffle(order)
        for first in range(0, len(order), batch_chunks):
            minibatch = order[first : first + batch_chunks]
            count = sum(stop - start for _, start, stop in minibatch)
            optimizer.zero_grad(set_to_none=True)
            tick = perf_counter()
            hidden = prefixes(policy, cache, minibatch)
            prefix_seconds += perf_counter() - tick
            finite = torch.ones((), dtype=torch.bool, device=device)
            for micro_start in range(0, len(minibatch), micro_chunks):
                spans = minibatch[micro_start : micro_start + micro_chunks]
                tick = perf_counter()
                batch, labels, lengths = cache.batch(spans)
                logits, value, _ = policy.forward_sequences(
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
                    clip_ratio=config["ppo"]["clip_ratio"],
                    value_coefficient=config["ppo"]["value_coefficient"],
                )
                finite &= torch.isfinite(loss.detach())
                valid_tokens = sum(lengths)
                (loss * (valid_tokens / count)).backward()
                backward_seconds += perf_counter() - tick
                totals += torch.stack([stats[key] for key in names]) * valid_tokens
                tokens += valid_tokens
            if not finite.item():
                raise FloatingPointError("Non-finite batched PPO loss")
            tick = perf_counter()
            norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(),
                config["ppo"]["max_gradient_norm"],
                error_if_nonfinite=True,
            )
            gradients.append(norm.detach())
            optimizer.step()
            optimizer_seconds += perf_counter() - tick
        tick = perf_counter()
        kl, _ = measure_kl(policy, cache, kl_batch)
        kl_seconds += perf_counter() - tick
        if monitor:
            monitor.mark("ppo", ppo_epoch=epoch + 1)
        if kl > config["ppo"]["stop_kl"]:
            break
    stats = (totals / tokens).cpu().tolist()
    gradient_values = torch.stack(gradients).cpu().numpy()
    values = np.concatenate([e["old_value"] for e in episodes])
    returns = np.concatenate(
        [[e["result"]["reward"]] * len(e["chosen"]) for e in episodes]
    )
    variance = np.var(returns)
    return {
        **dict(zip(names, stats)),
        "lr": lr,
        "entropy_coefficient": entropy_coefficient,
        "epochs": epoch + 1,
        "optimizer_steps": len(gradients),
        "objective_valid_tokens": tokens,
        "kl": kl,
        "pre_update_kl": pre_kl,
        "pre_update_max_logp_error": pre_error,
        "pre_update_full_kl": replay["full_kl"],
        "pre_update_max_total_variation": replay["max_total_variation"],
        "behavior_check": "full_distribution"
        if replay_tolerance is None
        else "sampled_logp",
        "behavior_logp_tolerance": replay_tolerance,
        "gradient_norm_mean": float(gradient_values.mean()),
        "gradient_norm_max": float(gradient_values.max()),
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
        "optimizer_seconds": optimizer_seconds,
        "kl_seconds": kl_seconds,
        "cache_seconds": cache_seconds,
        "rollout_device_bytes": cache.resident_bytes,
        "microbatch_sequences": micro_chunks,
        "ppo_seconds": perf_counter() - start_update,
    }
