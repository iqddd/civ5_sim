"""Bounded local PPO from BC, with a short value-only first-rollout fit."""

import gc
import json
import random
import traceback
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from civ5rl.core.model import batch_inputs
from civ5rl.engine.collector import Collector
from civ5rl.engine.constants import DEFAULT_PINNED_REPLAY_BYTES
from civ5rl.engine.control import (
    RunLock,
    assert_not_closed,
    pause_due,
    target_bc_evaluation_due,
)
from civ5rl.engine.performance import performance_markdown, summarize_performance
from civ5rl.engine.ppo import encode_episode, targets, update
from civ5rl.engine.ppo_fast import BatchedBackend
from civ5rl.engine.resources import ResourceMonitor
from civ5rl.experiments.evaluation import paired_world_bootstrap
from civ5rl.experiments.imitation_helpers import (
    CompilationClock,
    isolated_rng,
    load_policy,
)
from civ5rl.experiments.ppo_after_imitation import (
    optimizer_for,
    policy_hash,
    summarize_rollout,
)
from civ5rl.experiments.ppo_artifacts import (
    load_rollout,
    prune_checkpoints,
    save_rollout,
)
from civ5rl.experiments.ppo_target_scale import evaluate
from civ5rl.protocols.ppo_after_target_bc import (
    KIND,
    check_development_gain,
    record_collection,
    record_optimization,
    validate_profile,
)
from civ5rl.protocols.ppo_target_scale import TargetFactory
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import (
    ROOT,
    configure,
    file_sha256,
    read_json,
    seed_for,
    write_json,
)
from civ5sim.engine import digest


def critic_statistics(predictions, returns):
    predictions = np.asarray(predictions, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    residual = returns - predictions
    variance = returns.var()
    return {
        "prediction_mean": float(predictions.mean()),
        "return_mean": float(returns.mean()),
        "mse": float(np.mean(residual**2)),
        "constant_return_mean_mse": float(np.mean((returns - returns.mean())**2)),
        "explained_variance": float(1 - residual.var() / variance)
        if variance > 0
        else None,
    }


def behavior_digest(episodes):
    """The actual sampled choices and their probabilities are immutable."""
    import hashlib

    result = hashlib.sha256()
    for episode in episodes:
        result.update(np.asarray(episode["chosen"], dtype=np.int64).tobytes())
        result.update(np.asarray(episode["old_logp"], dtype=np.float64).tobytes())
        for row in episode["old_log_probs"]:
            result.update(row.tobytes())
    return result.hexdigest()


@torch.no_grad()
def recurrent_features(policy, episodes, device):
    """Yield bounded FP32 GRU features, preserving complete episode prefixes."""
    for episode_index, episode in enumerate(episodes):
        encoded = encode_episode(episode, policy.variant)
        hidden = None
        for start in range(0, len(encoded), 128):
            if policy.spatial_cuda_graphs:
                torch.compiler.cudagraph_mark_step_begin()
            part = encoded[start : start + 128]
            batch = batch_inputs(part, device, non_blocking=True)
            _, context = policy.encode_spatial(batch)
            features, hidden = policy.sequence_memory(context, [len(part)], hidden)
            # Own the host data: compiled/CUDA runners may reuse their buffers.
            yield episode_index, start, features.detach().cpu().clone()


def prepare_critic(
    policy,
    episodes,
    settings,
    device,
    *,
    deadline=None,
    progress=None,
    preparation_started=None,
):
    """Fit only value parameters, then refresh old_value without rewriting behavior."""
    started = preparation_started if preparation_started is not None else perf_counter()
    fit_deadline = min(started + settings["active_seconds"], deadline or float("inf"))
    original_policy_hash = policy_hash(policy.state_dict())
    original_behavior = behavior_digest(episodes)
    returns = np.concatenate(
        [
            np.full(len(ep["chosen"]), ep["result"]["reward"], dtype=np.float32)
            for ep in episodes
        ]
    )
    before = critic_statistics(
        np.concatenate([ep["old_value"] for ep in episodes]), returns
    )
    flags = [parameter.requires_grad for parameter in policy.parameters()]
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.value.parameters():
        parameter.requires_grad_(True)
    temporary = torch.optim.Adam(
        policy.value.parameters(),
        lr=settings["lr"],
        betas=tuple(settings["betas"]),
        eps=settings["eps"],
        weight_decay=settings["weight_decay"],
        fused=str(device).startswith("cuda"),
    )
    cache = torch.empty((len(returns), policy.hidden_size), dtype=torch.float32)
    cached = processed = steps = completed_epochs = 0
    batch_size = settings["batch_size"]

    def fit_batch(first, stop):
        nonlocal processed, steps
        temporary.zero_grad(set_to_none=True)
        features = cache[first:stop].to(device)
        labels = torch.from_numpy(returns[first:stop]).to(device)
        with torch.autocast(torch.device(device).type, enabled=False):
            prediction = policy.value(features).squeeze(-1)
            loss = F.mse_loss(prediction, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite critic preparation loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            policy.value.parameters(),
            settings["max_gradient_norm"],
            error_if_nonfinite=True,
        )
        temporary.step()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        processed += stop - first
        steps += 1
        if progress:
            progress()

    try:
        portions = iter(recurrent_features(policy, episodes, device))
        trained = 0
        while cached < len(returns) and perf_counter() < fit_deadline:
            _, _, features = next(portions)
            cache[cached : cached + len(features)] = features
            cached += len(features)
            if progress:
                progress()
            while cached - trained >= batch_size and perf_counter() < fit_deadline:
                fit_batch(trained, trained + batch_size)
                trained += batch_size
        if cached == len(returns):
            if trained < cached and perf_counter() < fit_deadline:
                fit_batch(trained, cached)
                trained = cached
            if trained == cached:
                completed_epochs = 1
                for _ in range(1, settings["epochs"]):
                    finished = True
                    for first in range(0, cached, batch_size):
                        if perf_counter() >= fit_deadline:
                            finished = False
                            break
                        fit_batch(first, min(first + batch_size, cached))
                    if not finished:
                        break
                    completed_epochs += 1
        fit_seconds = perf_counter() - started
    finally:
        for parameter, enabled in zip(policy.parameters(), flags):
            parameter.requires_grad_(enabled)
        temporary = None

    # Required full refresh is outside the 120-second fitting limit.
    refresh_started = perf_counter()
    flat_offset = 0
    after_values = []
    with torch.no_grad():
        for episode_index, start, features in recurrent_features(
            policy, episodes, device
        ):
            values = policy.value(features.to(device)).squeeze(-1).cpu().tolist()
            episodes[episode_index]["old_value"][start : start + len(values)] = values
            after_values.extend(values)
            flat_offset += len(values)
            if progress:
                progress()
    if flat_offset != len(returns):
        raise ValueError("Incomplete first-rollout old_value refresh")
    if policy_hash(policy.state_dict()) != original_policy_hash:
        raise RuntimeError("Critic preparation changed encoder/GRU/policy parameters")
    if behavior_digest(episodes) != original_behavior:
        raise RuntimeError("Critic preparation changed recorded behavior")
    advantages = targets(episodes)
    return {
        "steps": steps,
        "positions_processed": processed,
        "positions_cached": cached,
        "epochs_completed": completed_epochs,
        "fit_seconds": fit_seconds,
        "fit_time_overshoot_seconds": max(0, fit_seconds - settings["active_seconds"]),
        "refresh_seconds": perf_counter() - refresh_started,
        "before": before,
        "after": critic_statistics(after_values, returns),
        "raw_advantage_std": float(np.std(returns - np.asarray(after_values))),
        "normalized_advantage_std": float(np.concatenate(advantages).std()),
        "behavior_parameters_unchanged": True,
        "recorded_behavior_unchanged": True,
    }


class ActiveClock:
    """One elapsed-time counter; evaluations explicitly pause it."""

    def __init__(self, state, out):
        self.state = state
        self.out = out
        self.started = perf_counter()

    def sync(self):
        if self.started is not None:
            now = perf_counter()
            self.state["active_seconds"] += now - self.started
            self.started = now

    def progress(self):
        self.sync()
        write_json(self.out / "state.json", self.state)

    def pause(self):
        self.sync()
        self.started = None

    def resume(self):
        self.started = perf_counter()

    def deadline(self, config):
        self.sync()
        return perf_counter() + max(
            0, config["experiment"]["active_seconds"] - self.state["active_seconds"]
        )


def make_policy(config, device):
    policy = load_policy(config, ROOT / config["experiment"]["parent"], device)
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    return policy


def save_boundary(out, policy, optimizer, state, shuffle, config, identity, clock=None):
    if clock:
        clock.sync()
    checkpoint.save(
        out / state["latest"], policy, optimizer, state, shuffle, config, identity
    )
    if clock:
        clock.sync()
    write_json(out / "state.json", state)


def initialize(config, out):
    experiment = config["experiment"]
    parent = ROOT / experiment["parent"]
    if file_sha256(parent) != experiment["parent_sha256"]:
        raise ValueError("BC epoch-ten checkpoint SHA256 differs")
    plan = read_json(ROOT / experiment["parent_manifest"])
    if digest({k: v for k, v in plan.items() if k != "hash"}) != plan["hash"]:
        raise ValueError("Target manifest hash differs")
    if len(plan["development"]) != 128 or len(plan["confirmation"]) != 256:
        raise ValueError("Unexpected evaluation cases")
    write_json(out / "manifest.json", plan)
    configure(17, "cpu")
    policy = make_policy(config, "cpu")
    saved = torch.load(parent, map_location="cpu", weights_only=False)
    if not all(
        torch.equal(value, saved["model"][name])
        for name, value in policy.state_dict().items()
    ):
        raise ValueError("Initial model differs from BC epoch ten")
    if policy_hash(policy.state_dict()) != experiment["parent_policy_hash"]:
        raise ValueError("Initial behavior hash differs")
    optimizer = optimizer_for(policy, config, "cpu")
    if optimizer.state:
        raise ValueError("New PPO Adam must be empty")
    shuffle_seed = seed_for(17, 0, "shuffle", namespace=experiment["shuffle_namespace"])
    record = {
        "kind": KIND,
        "config": config,
        "parent_sha256": experiment["parent_sha256"],
        "parent_model_equal": True,
        "new_Adam_empty": True,
        "behavior_hash": policy_hash(policy.state_dict()),
        "manifest_hash": plan["hash"],
        "seed": 17,
        "shuffle_seed": shuffle_seed,
        "GPU": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "implementation_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "civ5rl/experiments/ppo_after_target_bc.py",
                "civ5rl/protocols/ppo_after_target_bc.py",
                "tools/run_ppo.py",
            )
        },
        "source_commit": __import__("subprocess")
        .check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
        .strip(),
    }
    record["run_id"] = digest(record)
    state = {
        "rollouts": 0,
        "updates": 0,
        "decisions": 0,
        "next_episode": 0,
        "optimizer_steps": 0,
        "attempted_steps": 0,
        "rejected_steps": 0,
        "winless_streak": 0,
        "zero_accepted_streak": 0,
        "active_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "critic_complete": False,
        "phase": "collect",
        "latest": "checkpoints/initial.pt",
        "stop_reason": None,
        "evaluated": {},
        "development_points": {},
        "baseline_wins": None,
        "best": {"path": "checkpoints/initial.pt", "wins": -1, "update": 0},
    }
    write_json(out / "run.json", record)
    save_boundary(
        out,
        policy,
        optimizer,
        state,
        random.Random(shuffle_seed),
        config,
        {"run_id": record["run_id"]},
    )
    return plan


def development(config, plan, out, state, shuffle, clock=None):
    if clock:
        clock.pause()
    started = perf_counter()
    try:
        with isolated_rng(shuffle):
            result = evaluate(config, plan, out, state["latest"], "development")
        state["evaluation_seconds"] += perf_counter() - started
        state["evaluated"][result["policy_hash"]] = {
            **result,
            "decisions": state["decisions"],
        }
        wins = result["runs"]["argmax"]["cp"]["summary"]["W"]
        if state["baseline_wins"] is None:
            state["baseline_wins"] = wins
        if wins > state["best"]["wins"]:
            state["best"] = {
                "path": result["checkpoint"],
                "wins": wins,
                "update": state["updates"],
                "policy_hash": result["policy_hash"],
            }
        for point in config["experiment"]["milestones"]:
            if point <= state["decisions"]:
                state["development_points"].setdefault(
                    str(point), result["policy_hash"]
                )
        check_development_gain(state, config)
        write_json(out / "state.json", state)
        report(config, out)
    finally:
        if clock:
            clock.resume()


def skipped_metrics(reason):
    return {
        "PPO_skipped": True,
        "stop_reason": reason,
        "optimizer_steps": 0,
        "attempted_optimizer_steps": 0,
        "rejected_optimizer_steps": 0,
    }


def train(config, plan, out, device, runtime_path=None, max_updates=None):
    state = read_json(out / "state.json")
    configure(17, device)
    record = read_json(out / "run.json")
    identity = {"run_id": record["run_id"]}
    shuffle = random.Random(record["shuffle_seed"])
    clock = ActiveClock(state, out)
    try:
        policy = make_policy(config, device)
        optimizer = optimizer_for(policy, config, device)
        restored = checkpoint.restore(
            out / state["latest"], policy, optimizer, shuffle, config, identity
        )
        for key in ("phase", "latest", "rollouts", "updates", "decisions", "optimizer_steps", "attempted_steps", "rejected_steps"):
            if restored.get(key) != state.get(key):
                raise ValueError(f"Saved checkpoint and state disagree on {key}")
        due = target_bc_evaluation_due(config, state, current_policy_hash=policy_hash(policy.state_dict()))
        if state["phase"] != "collect":
            ledger = out / "updates.jsonl"
            committed = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()] if ledger.exists() else []
            previous = {**state, "decisions": committed[-1]["total_decisions"] if committed else 0}
            if target_bc_evaluation_due(config, previous, current_policy_hash=policy_hash(policy.state_dict())):
                raise ValueError("Cannot recover earlier development evaluation from a pending PPO boundary")
            # A milestone crossed by the pending rollout is evaluated only after PPO.
            due = False
        if due:
            if state["rollouts"]:
                rows = [json.loads(line) for line in (out / "updates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
                if (not rows or rows[-1].get("rollout") != state["rollouts"]
                        or rows[-1].get("total_decisions") != state["decisions"]):
                    raise ValueError("Cannot recover development evaluation: update ledger and checkpoint disagree")
                if not state["evaluated"]:
                    raise ValueError("Cannot recover initial development evaluation from a later checkpoint")
                pending = [point for point in config["experiment"]["milestones"]
                           if point <= state["decisions"] and str(point) not in state["development_points"]]
                if any(row.get("total_decisions", 0) >= point
                       for point in pending for row in rows[:-1]):
                    raise ValueError("Cannot recover development evaluation: milestone was crossed before the latest checkpoint")
            development(config, plan, out, state, shuffle, clock)
        if pause_due(out, "update", state["updates"], at_boundary=state["phase"] == "collect"):
            write_json(out / "status.json", {"status": "paused", "checkpoint": state["latest"]})
            return
        if state["stop_reason"]:
            return
        factory = TargetFactory(config, plan["base"])
        # Runtime capacity can change rollout boundaries; record it separately.
        runtime_path = Path(
            runtime_path or ROOT / "configs/rl-ppo-after-target-bc-runtime.json"
        )
        runtime = read_json(runtime_path)
        if not {"cpu_workers", "worker_reply_size"} <= set(runtime) or set(runtime) - {
            "cpu_workers",
            "worker_reply_size",
            "num_envs",
            "collection_scheduler",
            "inference_min_batch",
            "inference_wait_seconds",
        }:
            raise ValueError("Unsupported Collection runtime option")
        collection_workers = runtime["cpu_workers"]
        reply_size = runtime["worker_reply_size"]
        num_envs = runtime.get("num_envs", config["collection"]["num_envs"])
        scheduler = runtime.get("collection_scheduler", "ready")
        if (
            type(collection_workers) is not int
            or collection_workers < 1
            or type(num_envs) is not int
            or num_envs < 1
            or collection_workers > num_envs
            or (scheduler != "streaming" and num_envs % collection_workers)
            or type(reply_size) is not int
            or not 1
            <= reply_size
            <= (num_envs + collection_workers - 1) // collection_workers
        ):
            raise ValueError(
                "Runtime workers/reply size must fit the environment count"
            )
        write_json(
            out / "runtime.json",
            {
                "collection_workers": collection_workers,
                "num_envs": num_envs,
                "envs_per_worker": (num_envs + collection_workers - 1)
                // collection_workers,
                "worker_reply_size": reply_size,
                "runtime_config": str(runtime_path),
                "runtime_config_sha256": file_sha256(runtime_path),
                "ppo_pinned_replay": True,
                "pinned_replay_limit_bytes": DEFAULT_PINNED_REPLAY_BYTES,
                "collection_scheduler": scheduler,
                "inference_min_batch": runtime.get("inference_min_batch", 0),
                "inference_wait_seconds": runtime.get("inference_wait_seconds", 0),
                "inference_coalescing": runtime.get("inference_wait_seconds", 0) > 0,
            },
        )
        with (
            ResourceMonitor(out / "telemetry") as monitor,
            Collector(
                config,
                factory,
                device,
                out / "errors",
                cpu_workers=collection_workers,
                reply_size=reply_size,
                num_envs=num_envs,
                collection_scheduler=scheduler,
                inference_min_batch=runtime.get("inference_min_batch", 0),
                inference_wait_seconds=runtime.get("inference_wait_seconds", 0),
            ) as collector,
        ):
            while not state["stop_reason"]:
                critic_started = perf_counter() if state["phase"] == "critic" else None
                if max_updates is not None and state["updates"] >= max_updates:
                    write_json(out / "status.json", {"status": "paused", "reason": "max_updates"})
                    return
                deadline = clock.deadline(config)
                if perf_counter() >= deadline:
                    state["stop_reason"] = "active_time_limit"
                    break
                if pause_due(out, "update", state["updates"], at_boundary=state["phase"] == "collect"):
                    write_json(out / "status.json", {"status": "paused"})
                    return
                if state["phase"] == "collect":
                    if state["decisions"] >= config["experiment"]["decisions"]:
                        state["stop_reason"] = "decision_limit"
                        break
                    monitor.mark("collection", rollout=state["rollouts"] + 1)
                    episodes, following, timing = collector.collect(
                        policy,
                        None,
                        state["next_episode"],
                        monitor,
                        deadline=deadline,
                        max_decisions=config["experiment"]["decisions"]
                        - state["decisions"],
                        episode_length_hint=(
                            state["rollout_summary"]["decisions"]
                            / max(1, state["rollout_summary"]["episodes"])
                            if state.get("rollout_summary")
                            else None
                        ),
                    )
                    if not state["critic_complete"]:
                        critic_started = perf_counter()
                    if not episodes:
                        state["stop_reason"] = "active_time_limit"
                        break
                    summary = summarize_rollout(episodes)
                    record_collection(state, summary, config)
                    state["next_episode"] = following
                    state["collection"] = timing
                    state["rollout_summary"] = summary
                    artifact = save_rollout(
                        episodes, out / "rollouts", state["rollouts"]
                    )
                    state["pending_rollout"] = "rollouts/" + artifact["path"]
                    state["phase"] = "ppo" if state["critic_complete"] else "critic"
                    state["latest"] = "checkpoints/pending.pt"
                    save_boundary(
                        out, policy, optimizer, state, shuffle, config, identity, clock
                    )
                    print(
                        f"Sampling rollout {state['rollouts']}: {summary}; total={state['decisions']}",
                        flush=True,
                    )
                else:
                    episodes = load_rollout(out / state["pending_rollout"])

                metrics = skipped_metrics(state["stop_reason"] or "active_time_limit")
                if not state["stop_reason"] and state["phase"] == "critic":
                    monitor.mark("critic_preparation")
                    with CompilationClock() as compiler:
                        compiler.monitor = monitor
                        state["critic"] = prepare_critic(
                            policy,
                            episodes,
                            config["critic"],
                            device,
                            deadline=deadline,
                            progress=clock.progress,
                            preparation_started=critic_started,
                        )
                    state["critic"]["runtime_compilation_seconds"] = sum(
                        b - a for a, b in compiler.intervals
                    )
                    state["critic_complete"] = True
                    state["phase"] = "ppo"
                    artifact = save_rollout(episodes, out / "critic-rollout", 1)
                    state["pending_rollout"] = "critic-rollout/" + artifact["path"]
                    optimizer = optimizer_for(policy, config, device)
                    assert not optimizer.state
                    state["latest"] = "checkpoints/critic-ready.pt"
                    save_boundary(
                        out, policy, optimizer, state, shuffle, config, identity, clock
                    )
                    write_json(out / "critic-preparation.json", state["critic"])
                    print(
                        "Critic preparation: " + json.dumps(state["critic"]), flush=True
                    )

                if not state["stop_reason"] and perf_counter() < deadline:
                    monitor.mark("ppo", rollout=state["rollouts"])

                    def inspect_step(step, optimizer=optimizer):
                        if (
                            step["accepted"]
                            and not (out / "first-accepted-step.json").exists()
                        ):
                            counters = {
                                int(value["step"]) for value in optimizer.state.values()
                            }
                            if counters != {1}:
                                raise RuntimeError(
                                    f"First PPO Adam counter differs: {counters}"
                                )
                            write_json(
                                out / "first-accepted-step.json",
                                {"Adam_counter": 1, "step": step},
                            )

                    with CompilationClock() as compiler:
                        compiler.monitor = monitor
                        operation_started = perf_counter()
                        metrics = update(
                            policy,
                            optimizer,
                            episodes,
                            config,
                            state["decisions"],
                            device,
                            shuffle,
                            deadline=deadline,
                            on_step=inspect_step,
                            _backend=partial(BatchedBackend, pinned_replay=True),
                        )
                        operation_ended = perf_counter()
                    metrics["runtime_compilation_seconds"] = sum(
                        b - a for a, b in compiler.intervals
                    )
                    metrics["operation_seconds"] = operation_ended - operation_started
                    metrics["operation_without_compilation_seconds"] = (
                        metrics["operation_seconds"]
                        - compiler.seconds(operation_started, operation_ended)
                    )
                    metrics["PPO_skipped"] = False
                    record_optimization(state, metrics, config)
                elif not state["stop_reason"]:
                    state["stop_reason"] = "active_time_limit"
                del episodes
                gc.collect()
                clock.sync()
                if state["active_seconds"] >= config["experiment"]["active_seconds"]:
                    state["stop_reason"] = state["stop_reason"] or "active_time_limit"
                elif state["decisions"] >= config["experiment"]["decisions"]:
                    state["stop_reason"] = state["stop_reason"] or "decision_limit"
                state["phase"] = "collect"
                state["latest"] = f"checkpoints/update-{state['rollouts']:04}.pt"
                monitor.mark("checkpoint")
                save_boundary(
                    out, policy, optimizer, state, shuffle, config, identity, clock
                )
                row = {
                    "rollout": state["rollouts"],
                    **state["rollout_summary"],
                    "total_decisions": state["decisions"],
                    "active_seconds": state["active_seconds"],
                    "winless_streak": state["winless_streak"],
                    "zero_accepted_streak": state["zero_accepted_streak"],
                    "decision_overshoot": max(
                        0, state["decisions"] - config["experiment"]["decisions"]
                    ),
                    "active_time_overshoot_seconds": max(
                        0,
                        state["active_seconds"]
                        - config["experiment"]["active_seconds"],
                    ),
                    "collection": state["collection"],
                    "ppo": metrics,
                }
                with (out / "updates.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                write_json(out / "coverage.json", factory.summary())
                print(
                    f"Rollout {state['rollouts']}: W/D/L={row['W']}/{row['D']}/{row['L']}; decisions={state['decisions']}; Adam={state['optimizer_steps']}; active={state['active_seconds']:.1f}s; {metrics['stop_reason']}",
                    flush=True,
                )
                due = any(
                    str(point) not in state["development_points"]
                    and point <= state["decisions"]
                    for point in config["experiment"]["milestones"]
                )
                changed = policy_hash(policy.state_dict()) not in state["evaluated"]
                if due or (state["stop_reason"] and changed):
                    monitor.mark("evaluation")
                    development(config, plan, out, state, shuffle, clock)
                if pause_due(out, "update", state["updates"]):
                    write_json(out / "status.json", {"status": "paused", "checkpoint": state["latest"]})
                    return
                prune_checkpoints(out, state)
                clock.progress()
                if max_updates is not None and state["updates"] >= max_updates and not state["stop_reason"]:
                    write_json(out / "status.json", {"status": "paused", "reason": "max_updates"})
                    return
            if policy_hash(policy.state_dict()) not in state["evaluated"]:
                development(config, plan, out, state, shuffle, clock)
            clock.progress()
    finally:
        clock.progress()


def finish(config, plan, out):
    state = read_json(out / "state.json")
    write_json(out / "selected.json", state["best"])
    if state["best"]["path"] == "checkpoints/initial.pt":
        write_json(out / "confirmation.json", {"status": "not_run_parent_selected"})
        return
    baseline = evaluate(config, plan, out, "checkpoints/initial.pt", "confirmation")
    candidate = evaluate(config, plan, out, state["best"]["path"], "confirmation")

    def outcomes(point):
        directory = out / point["runs"]["argmax"]["cp"]["directory"]
        results, cases = {}, {}
        for path in sorted(directory.glob("*/metrics.json")):
            row = read_json(path)
            case = read_json(path.parent / "case.json")
            key = (case["parent_world"], case["case_id"], case["learner_side"])
            results[key] = int(row["winner"] == row["side"])
            cases[key] = case
        return results, cases

    before, cases = outcomes(baseline)
    after, paired = outcomes(candidate)
    if cases != paired:
        raise ValueError("Additional cases are not exactly paired")
    comparison = paired_world_bootstrap(
        before, after, samples=10000, seed=20261103, cases=cases, cases_per_world=16
    )
    write_json(
        out / "confirmation.json",
        {"baseline": baseline, "candidate": candidate, "comparison": comparison},
    )


def report(config, out):
    state = read_json(out / "state.json")
    lines = [
        "# PPO после target BC epoch 10",
        "",
        f"Статус: {state['stop_reason'] or state['phase']}; решения {state['decisions']}; active {state['active_seconds']:.2f} с.",
        f"Sampling rollouts {state['rollouts']}; PPO updates {state['updates']}; accepted/rejected Adam {state['optimizer_steps']}/{state['rejected_steps']}.",
        f"GPU {read_json(out / 'run.json')['GPU']}; T={config.get('policy_temperature', 1)}; microbatch {config['ppo']['sequences_per_microbatch']}; обычный CP; reward 1/0/0.",
        "",
        "| Решения | Checkpoint | W/D/L development |",
        "|---:|---|---|",
    ]
    for point in state["evaluated"].values():
        summary = point["runs"]["argmax"]["cp"]["summary"]
        lines.append(
            f"| {point['decisions']} | {point['checkpoint']} | {summary['W']}/{summary['D']}/{summary['L']} |"
        )
    lines += ["", f"Выбор: {state['best']}."]
    if (out / "runtime.json").exists():
        lines += ["", "Runtime: " + json.dumps(read_json(out / "runtime.json"), ensure_ascii=False)]
    if (out / "updates.jsonl").exists():
        for name, last in (("performance", None), ("performance-last10", 10)):
            performance = summarize_performance(out, last=last)
            write_json(out / f"{name}.json", performance)
            (out / f"{name}.md").write_text(
                performance_markdown(performance), encoding="utf-8"
            )
        lines += ["", "Performance: [all cycles](performance.md), [last 10](performance-last10.md)."]
        rows = [
            json.loads(line)
            for line in (out / "updates.jsonl").read_text().splitlines()
        ]
        lines += [
            "",
            "| Rollout | Sampling W/D/L | Raw advantage std | Critic MSE / EV / mean | Entropy | Full KL | Adam accept/reject | PPO skipped |",
            "|---:|---|---:|---|---:|---:|---|---|",
        ]
        for row in rows:
            metrics = row["ppo"]
            lines.append(
                f"| {row['rollout']} | {row['W']}/{row['D']}/{row['L']} | {metrics.get('raw_advantage_std')} | "
                f"{metrics.get('critic_mse')}/{metrics.get('critic_explained_variance')}/{metrics.get('critic_mean')} | "
                f"{metrics.get('entropy')} | {metrics.get('final_accepted_kl')} | "
                f"{metrics['optimizer_steps']}/{metrics['rejected_optimizer_steps']} | {metrics['PPO_skipped']} |"
            )
        lines += [
            "",
            f"Rollout без побед: {sum(row['W'] == 0 for row in rows)}/{len(rows)}.",
        ]
    for point in state["evaluated"].values():
        lines += [
            "",
            f"Срезы development {point['checkpoint']}",
            "",
            json.dumps(point["slices"], ensure_ascii=False),
        ]
    if state["stop_reason"] == "no_development_gain":
        lines += ["", "Прироста в установленном раннем бюджете не получено."]
    lines += [
        "",
        "Одна новая development-победа разрешает продолжение; полезность оценивается по парному additional-приросту и CI.",
    ]
    for name in ("critic-preparation.json", "confirmation.json"):
        if (out / name).exists():
            lines += [
                "",
                name,
                "",
                json.dumps(read_json(out / name), ensure_ascii=False),
            ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(phase, config, out, device, runtime_path=None, max_updates=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    validate_profile(config)
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be a positive operational limit")
    if phase == "pause":
        (out / "pause.request").write_text("pause requested\n", encoding="utf-8")
        return
    with RunLock(out):
        if phase in ("train", "run", "resume"):
            assert_not_closed(out)
        try:
            if phase == "resume":
                (out / "pause.request").unlink(missing_ok=True)
            if not (out / "run.json").exists():
                initialize(config, out)
            if read_json(out / "run.json")["config"] != config:
                raise ValueError("Run configuration changed")
            plan = read_json(out / "manifest.json")
            if phase in ("train", "run", "resume"):
                write_json(out / "status.json", {"status": "training"})
                train(config, plan, out, device, runtime_path=runtime_path, max_updates=max_updates)
                if read_json(out / "status.json").get("status") == "paused":
                    report(config, out)
                    return
            if phase in ("evaluate", "run", "resume"):
                if not read_json(out / "state.json")["stop_reason"]:
                    raise ValueError("Training has not stopped")
                finish(config, plan, out)
                write_json(out / "status.json", {"status": "complete"})
            report(config, out)
        except BaseException:
            (out / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            write_json(
                out / "status.json", {"status": "technical_error", "phase": phase}
            )
            raise
