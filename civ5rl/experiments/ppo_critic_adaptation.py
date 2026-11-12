"""Recoverable value-only adaptation without rebuilding the parent's PPO Adam."""

import os
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from civ5rl.engine.budget import deadline_reached
from civ5rl.engine.ppo_adam import expected_counters
from civ5rl.engine.ppo import raw_gae_advantages
from civ5rl.provenance import checkpoint


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

    def tick():
        progress["fit_seconds"] = elapsed_before + perf_counter() - started
        state["critic_fit_seconds"] = progress["fit_seconds"]
        state["Adam_counters"] = expected_counters(state)
        clock.progress()

    def commit():
        tick()
        atomic_save(dict(identity=identity, behavior=behavior,
                         value={k: v.detach().cpu() for k, v in policy.value.state_dict().items()},
                         adam=[{k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                                for k, v in optimizer.state.get(p, {}).items()}
                               for p in value_group["params"]],
                         progress=dict(progress), rng=checkpoint.capture_rng(shuffle),
                         active_seconds=state["active_seconds"]), capsule_path)

    fit_finished = False
    try:
        returns = np.concatenate([np.full(len(ep["chosen"]), ep["result"]["reward"], np.float32)
                                  for ep in episodes])
        before = statistics(np.concatenate([ep["old_value"] for ep in episodes]), returns)
        cache_path = out / "critic-features.pt"
        if cache_path.exists():
            saved = torch.load(cache_path, map_location="cpu", weights_only=False)
            if (saved["behavior"] != behavior or saved["policy_hash"] != original_policy
                    or saved["features"].shape != (len(returns), policy.hidden_size)):
                raise ValueError("Critic feature cache differs from frozen rollout/encoder")
            features = saved["features"]
        else:
            features = torch.empty((len(returns), policy.hidden_size), dtype=torch.float32)
            offset = 0
            extraction_rng = checkpoint.capture_rng(shuffle)
            try:
                for _, _, part in features_for(policy, episodes, device):
                    features[offset:offset + len(part)] = part
                    offset += len(part)
                    tick()
            finally:
                checkpoint.restore_rng(extraction_rng, shuffle)
            if offset != len(returns):
                raise ValueError("Incomplete critic feature cache")
            atomic_save(dict(behavior=behavior, policy_hash=original_policy, features=features), cache_path)
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
        try:
            while progress["cursor"] < batches * settings["epochs"]:
                tick()
                if progress["fit_seconds"] >= settings["active_seconds"] or deadline_reached(deadline):
                    break
                first = (progress["cursor"] % batches) * settings["batch_size"]
                stop = min(first + settings["batch_size"], len(returns))
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(torch.device(device).type, enabled=False):
                    loss = F.mse_loss(policy.value(features[first:stop]).squeeze(-1), labels[first:stop])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite critic adaptation loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.value.parameters(), settings["max_gradient_norm"],
                                               error_if_nonfinite=True)
                optimizer.step()
                progress["steps"] += 1
                progress["cursor"] += 1
                progress["positions_processed"] += stop - first
                commit()
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
        advantages = np.concatenate(raw_gae_advantages(episodes, gamma=1.0,
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
