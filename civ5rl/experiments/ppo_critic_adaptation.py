"""Recoverable value-only adaptation without rebuilding the parent's PPO Adam."""

import os
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from civ5rl.engine.runtime.budget import deadline_reached
from civ5rl.engine.runtime.transactions import cpu_snapshot
from civ5rl.engine.ppo.adam import expected_counters
from civ5rl.engine.ppo.core import episode_returns, raw_gae_advantages
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import seed_for


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def adapt_critic(policy, optimizer, episodes, config, device, out, state, shuffle,
                 identity, clock, deadline, *, features_for, behavior_hash,
                 statistics, policy_hash):
    """Each completed batch commits only the value head, its Adam and RNG."""
    started = perf_counter()
    out = Path(out)
    settings = config["critic"]
    behavior = behavior_hash(episodes)
    original_policy = policy_hash(policy.state_dict())
    value_group = next(g for g in optimizer.param_groups if g["name"] == "value")
    capsule_path = out / "critic-adaptation.pt"
    progress = dict(steps=0, cursor=0, fit_seconds=0.0, positions_processed=0)
    if capsule_path.exists():
        capsule = torch.load(capsule_path, map_location="cpu", weights_only=False)
        if capsule["identity"] != identity or capsule["behavior"] != behavior:
            raise ValueError("Critic adaptation capsule belongs to another rollout/run")
        policy.value.load_state_dict(capsule["value"])
        for parameter, item in zip(value_group["params"], capsule["adam"], strict=True):
            optimizer.state.pop(parameter, None)
            if item:
                optimizer.state[parameter] = {k: v.to("cpu" if k == "step" and not (value_group.get("fused") or value_group.get("capturable")) else device) if isinstance(v, torch.Tensor) else v
                                              for k, v in item.items()}
        progress = capsule["progress"]
        checkpoint.restore_rng(capsule["rng"], shuffle)
        state["active_seconds"] = max(state["active_seconds"], capsule["active_seconds"])
    # Heartbeats may include interrupted extraction/compilation since the last batch.
    progress["fit_seconds"] = max(progress["fit_seconds"], state.get("critic_fit_seconds", 0.0))
    state["critic_progress"] = progress
    elapsed_before = progress["fit_seconds"]

    def tick(*, persist=True):
        progress["fit_seconds"] = elapsed_before + perf_counter() - started
        state["critic_fit_seconds"] = progress["fit_seconds"]
        state["Adam_counters"] = expected_counters(state)
        if persist:
            clock.progress()
        else:
            clock.sync()

    def commit():
        tick(persist=False)
        model_adam = cpu_snapshot({
            "value": policy.value.state_dict(),
            "adam": [optimizer.state.get(p, {}) for p in value_group["params"]],
        })
        atomic_save(dict(identity=identity, behavior=behavior, **model_adam,
                         progress=dict(progress), rng=checkpoint.capture_rng(shuffle),
                         active_seconds=state["active_seconds"]), capsule_path)
        clock.progress()

    fit_finished = False
    try:
        gamma = float(config.get("ppo", {}).get("gamma", 1.0))
        returns = np.concatenate(episode_returns(episodes, gamma=gamma))
        before = statistics(np.concatenate([ep["old_value"] for ep in episodes]), returns)
        cache_path = out / "critic-features.pt"
        if cache_path.exists():
            saved = torch.load(cache_path, map_location="cpu", weights_only=False)
            if (saved["behavior"] != behavior or saved["policy_hash"] != original_policy
                    or saved["features"].shape != (len(returns), policy.hidden_size)):
                raise ValueError("Critic feature cache differs from frozen rollout/encoder")
            features = saved["features"]
        else:
            features = torch.empty((len(returns), policy.hidden_size), dtype=torch.float32, device=device)
            offsets = np.cumsum([0, *[len(ep["chosen"]) for ep in episodes]])
            cached = 0
            last_progress = perf_counter()
            extraction_rng = checkpoint.capture_rng(shuffle)
            try:
                for episode, start, part in features_for(policy, episodes, device):
                    offset = int(offsets[episode]) + start
                    features[offset:offset + len(part)].copy_(part, non_blocking=True)
                    cached += len(part)
                    # Keep budget recovery current without writing JSON for every block.
                    now = perf_counter()
                    if now - last_progress >= 1.0:
                        tick()
                        last_progress = now
            finally:
                checkpoint.restore_rng(extraction_rng, shuffle)
            if cached != len(returns):
                raise ValueError("Incomplete critic feature cache")
            atomic_save(dict(behavior=behavior, policy_hash=original_policy, features=features.cpu()), cache_path)
        # Small [positions, hidden] cache and labels move once, rather than per epoch/batch.
        features = features.to(device)
        labels = torch.from_numpy(returns).to(device)
        flags = [p.requires_grad for p in policy.parameters()]
        for p in policy.parameters():
            p.requires_grad_(False)
        for p in policy.value.parameters():
            p.requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        old_lr = value_group["lr"]
        value_group["lr"] = settings["lr"]
        batches = (len(returns) + settings["batch_size"] - 1) // settings["batch_size"]
        shuffle_epochs = bool(
            settings.get("shuffle", bool(config.get("self_play", {}).get("enabled")))
        )
        critic_namespace = config.get("experiment", {}).get("critic_namespace", "critic")
        base_seed = int(config.get("seed", 17))
        cached_epoch = None
        epoch_order = None
        try:
            while progress["cursor"] < batches * settings["epochs"]:
                tick(persist=False)
                if progress["fit_seconds"] >= settings["active_seconds"] or deadline_reached(deadline):
                    break
                epoch_idx, batch_idx = divmod(progress["cursor"], batches)
                first = batch_idx * settings["batch_size"]
                stop = min(first + settings["batch_size"], len(returns))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(torch.device(device).type, enabled=False):
                    if shuffle_epochs:
                        if cached_epoch != epoch_idx:
                            perm = list(range(len(returns)))
                            random.Random(
                                seed_for(base_seed, epoch_idx, "critic_shuffle", namespace=critic_namespace)
                            ).shuffle(perm)
                            epoch_order = torch.tensor(perm, device=device, dtype=torch.long)
                            cached_epoch = epoch_idx
                        sel = epoch_order[first:stop]
                        loss = F.mse_loss(policy.value(features[sel]).squeeze(-1), labels[sel])
                    else:
                        loss = F.mse_loss(policy.value(features[first:stop]).squeeze(-1), labels[first:stop])
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(policy.value.parameters(), settings["max_gradient_norm"])
                finite_loss, finite_norm = torch.isfinite(torch.stack((loss.detach(), norm))).tolist()
                if not finite_loss:
                    raise FloatingPointError("Non-finite critic adaptation loss")
                if not finite_norm:
                    raise FloatingPointError("Non-finite critic adaptation gradient norm")
                optimizer.step()
                progress["steps"] += 1
                progress["cursor"] += 1
                progress["positions_processed"] += stop - first
                commit()
            if not capsule_path.exists():
                commit()
        finally:
            value_group["lr"] = old_lr
            optimizer.zero_grad(set_to_none=True)
            for p, enabled in zip(policy.parameters(), flags):
                p.requires_grad_(enabled)
        tick()
        fit_finished = True
        refresh_started = perf_counter()
        with torch.no_grad(), torch.autocast(torch.device(device).type, enabled=False):
            # One CPU transfer after bounded value-head batches; no repeated spatial/GRU pass.
            refreshed = torch.cat([policy.value(part).squeeze(-1)
                                   for part in features.split(settings["batch_size"])]).cpu().numpy()
        offset = 0
        for episode in episodes:
            size = len(episode["old_value"])
            episode["old_value"][:] = refreshed[offset:offset + size].tolist()
            offset += size
        clock.progress()
        if policy_hash(policy.state_dict()) != original_policy or behavior_hash(episodes) != behavior:
            raise RuntimeError("Critic adaptation changed policy or recorded behavior")
        advantages = np.concatenate(raw_gae_advantages(episodes, gamma=gamma,
                                                      gae_lambda=config["ppo"]["gae_lambda"]))
        remaining = np.concatenate([np.arange(len(ep["chosen"]) - 1, -1, -1) for ep in episodes])
        horizon_statistics = {}
        for label, mask in (("0_99", remaining < 100),
                            ("100_299", (remaining >= 100) & (remaining < 300)),
                            ("300_plus", remaining >= 300)):
            if mask.any():
                horizon_statistics[label] = dict(positions=int(mask.sum()),
                                                 **statistics(refreshed[mask], returns[mask]),
                                                 raw_advantage_mean=float(advantages[mask].mean()))
        return dict(steps=progress["steps"], positions_processed=progress["positions_processed"],
                    positions_cached=len(returns), epochs_completed=progress["cursor"] // batches,
                    fit_seconds=progress["fit_seconds"], refresh_seconds=perf_counter() - refresh_started,
                    before=before, after=statistics(refreshed, returns),
                    raw_advantage_std=float(advantages.std()),
                    return_min=float(returns.min()), return_max=float(returns.max()),
                    remaining_decisions=horizon_statistics,
                    behavior_parameters_unchanged=True, recorded_behavior_unchanged=True)
    finally:
        if not fit_finished:
            tick()
