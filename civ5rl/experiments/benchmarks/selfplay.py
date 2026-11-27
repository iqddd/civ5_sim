"""Profile fixed mixed self-play collection and both PPO KL schedules."""

from __future__ import annotations

import gzip
import random
import shutil
from copy import deepcopy
from functools import partial
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import torch

from civ5rl.engine.collection.collector import Collector
from civ5rl.engine.collection.opponents import OpponentPool, select_opponent
from civ5rl.engine.ppo import core
from civ5rl.engine.ppo.adam import grouped_optimizer
from civ5rl.engine.ppo.storage import load_rollout
from civ5rl.engine.runtime.telemetry import CompilationClock, ResourceMonitor
from civ5rl.experiments.benchmarks.profiling import summarize_kineto_trace
from civ5rl.experiments.imitation.helpers import load_policy
from civ5rl.experiments.ppo.after_target_bc.optimizer import load_initial_optimizer
from civ5rl.provenance import bound_cases, case_manifest
from civ5rl.provenance.common import ROOT, configure, file_sha256, write_json
from civ5sim.engine import digest


class TraceWindow:
    def __init__(self, first, last, path):
        self.first, self.last, self.path = first, last, path
        self.index = 0
        self.active = False
        self.seconds = None
        self.profile = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        )

    def call(self, function, *args, **kwargs):
        self.index += 1
        if self.index == self.last + 1:
            self.stop()
        if self.index == self.first:
            torch.cuda.synchronize()
            self.profile.start()
            self.started = perf_counter()
            self.active = True
        if self.active:
            with torch.profiler.record_function(function.__name__):
                return function(*args, **kwargs)
        return function(*args, **kwargs)

    def stop(self):
        if self.active:
            torch.cuda.synchronize()
            self.seconds = perf_counter() - self.started
            self.profile.stop()
            self.active = False

    def save(self):
        self.stop()
        if self.index < self.last or self.seconds is None:
            raise RuntimeError(f"Incomplete profiling window for {self.path.name}")
        raw = self.path.with_suffix("")
        self.profile.export_chrome_trace(str(raw))
        with raw.open("rb") as source, gzip.open(self.path, "wb", compresslevel=3) as target:
            shutil.copyfileobj(source, target)
        raw.unlink()
        return summarize_kineto_trace(self.path, self.seconds)


def _measured(action, monitor, phase):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    monitor.mark(phase)
    start = perf_counter()
    with CompilationClock() as compiler:
        result = action()
        torch.cuda.synchronize()
    timing = {
        "seconds": perf_counter() - start,
        "compilation_seconds": sum(b - a for a, b in compiler.intervals),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    monitor.mark("between_trials")
    return result, timing


def _collection(args, config, parent, carrier, policy, monitor):
    snapshots = [{"id": name} for name in ("archive:10", "archive:20")]
    specs = [deepcopy(ep["spec"]) for ep in carrier[: args.episode_limit]]
    for spec in specs:
        opponent, identity = select_opponent(spec["episode_id"], config, snapshots)
        spec["opponent"] = opponent
        spec.pop("opponent_id", None)
        spec.pop("opponent_search", None)
        if opponent == "policy":
            spec["opponent_id"] = identity
        elif opponent == "cp":
            spec["opponent_search"] = config["teacher_search"]
    manifest = case_manifest(specs, source="selfplay_profile", split="train", namespace="fixed")
    pool = OpponentPool(config, "cuda", args.out, parent_checkpoint=parent)
    for entry in snapshots:
        pool.add_archive_snapshot(entry, policy)
    windows = TraceWindow(33, 56, args.out / "collection.kineto.json.gz")
    with Collector(
        config, None, "cuda", args.out / "errors", cpu_workers=args.workers, num_envs=args.envs,
        reply_size=args.reply_size, collection_scheduler="streaming", opponent_pool=pool, record_hidden=False,
    ) as collector:
        def run():
            return collector.collect(policy, None, 0, fixed_specs=bound_cases(manifest), trusted_manifest=manifest)

        with torch.no_grad():
            for _ in range(args.warmup):
                _measured(run, monitor, "warmup_collection")
            trials = []
            for _ in range(args.repeats):
                (episodes, _, counters), timing = _measured(run, monitor, "collection")
                trials.append({**timing, **counters, "learner_decisions": sum(len(ep["chosen"]) for ep in episodes)})
            with patch.object(collector, "_infer_live", partial(windows.call, collector._infer_live)):
                _measured(run, monitor, "trace_collection")
                windows.stop()
    return episodes, {
        "trials": trials, "trace": windows.save(),
        "command_hash": digest([ep["joint_commands"] for ep in episodes]),
    }


def _ppo(args, config, mode, episodes, policy, optimizer, model_state, adam_state, monitor):
    trial_config = deepcopy(config)
    production = mode == "production"
    if not production:
        trial_config["ppo"]["epochs"] = 1
        trial_config["ppo"]["minibatch_kl_spike_threshold"] = 1.0 if mode == "epoch" else -1.0
    decisions = sum(len(ep["chosen"]) for ep in episodes)
    expected = (decisions + config["ppo"]["valid_tokens_per_minibatch"] - 1) // config["ppo"]["valid_tokens_per_minibatch"]
    backend = None
    if config["experiment"].get("three_opponent_evaluation"):
        backend = partial(core.BatchedBackend, pinned_replay=True)

    def reset():
        policy.load_state_dict(model_state)
        optimizer.load_state_dict(deepcopy(adam_state))
        optimizer.zero_grad(set_to_none=True)

    def run():
        return core.update(
            policy, optimizer, episodes, trial_config, decisions, "cuda", random.Random(args.seed), _backend=backend,
        )

    for _ in range(args.warmup):
        reset()
        _measured(run, monitor, f"warmup_ppo_{mode}")
    trials = []
    for _ in range(args.repeats):
        reset()
        result, timing = _measured(run, monitor, f"ppo_{mode}")
        trials.append({
            **timing, **{k: result[k] for k in (
                "prefix_seconds", "backward_seconds", "optimizer_seconds", "kl_seconds", "cache_seconds",
                "transaction_seconds", "accepted_optimizer_steps", "attempted_optimizer_steps", "stop_reason",
                "pre_update_full_kl", "pre_update_max_total_variation", "full_kl", "replay_mode",
            )},
            "full_checks": sum(row["full_kl_checked"] for row in result["optimizer_step_records"]),
        })
    if production:
        return {"trials": trials, "decisions": decisions}
    if config["experiment"].get("three_opponent_evaluation"):
        trace_steps = min(3, *(trial["attempted_optimizer_steps"] for trial in trials))
        if trace_steps < 1:
            raise ValueError("No PPO optimizer steps available for profiling")
        trace = TraceWindow(1, trace_steps, args.out / f"ppo-{mode}.kineto.json.gz")
    else:
        trace = TraceWindow(max(1, expected - 2), expected, args.out / f"ppo-{mode}.kineto.json.gz")
    reset()
    with patch.object(core, "_optimizer_step", partial(trace.call, core._optimizer_step)):
        try:
            _measured(run, monitor, f"trace_ppo_{mode}")
        finally:
            trace.stop()
    return {"trials": trials, "trace": trace.save(), "trace_steps": [trace.first, trace.last]}


def profile_selfplay(args, parser, config, inputs):
    if args.device != "cuda" or not config.get("self_play", {}).get("enabled") or min(args.warmup, args.repeats) < 1:
        parser.error("selfplay-profile requires CUDA, self-play, warmup, and measured repeats")
    out = args.out.resolve()
    # Remote transport may stage the runner's lock before creating any results.
    populated = out.exists() and any(path.name != "run.lock" for path in out.iterdir())
    if any(p == out or p.is_relative_to(out) for p in inputs) or populated:
        parser.error("selfplay-profile output must be separate and empty")
    if min(args.workers, args.envs, args.reply_size) < 1 or args.workers > args.envs:
        parser.error("Invalid selfplay-profile worker, environment or reply counts")
    hashes = [file_sha256(path) for path in inputs]
    parent = ROOT / config["experiment"]["parent"]
    if file_sha256(parent) != config["experiment"]["parent_sha256"]:
        raise ValueError("Wrong self-play parent checkpoint")
    carrier = load_rollout(inputs[2])
    if len(carrier) < 8 or (args.episode_limit is not None and args.episode_limit < 8):
        parser.error("selfplay-profile requires at least eight fixed cases")
    if inputs[1] != parent.resolve():
        parser.error("selfplay-profile uses the configured parent checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    configure(args.seed, "cuda")
    saved = torch.load(parent, map_location="cpu", weights_only=False)
    policy = load_policy(config, parent, "cpu").requires_grad_(True)
    initial_optimizer = load_initial_optimizer(policy, saved, config)
    model_state, adam_state = deepcopy(policy.state_dict()), initial_optimizer.state_dict()
    policy = load_policy(config, parent, "cuda").requires_grad_(True)
    policy.load_state_dict(model_state)
    optimizer = grouped_optimizer(policy, config, "cuda")
    with ResourceMonitor(out / "telemetry", interval=0.5) as monitor:
        print("Profiling fixed mixed collection", flush=True)
        episodes, collection = _collection(args, config, parent, carrier, policy.eval(), monitor)
        result = {"collection": collection}
        modes = ("production", "epoch", "step") if config["experiment"].get("three_opponent_evaluation") else ("epoch", "step")
        for mode in modes:
            print(f"Profiling PPO with {mode} KL", flush=True)
            result["ppo_" + mode] = _ppo(args, config, mode, episodes, policy.train(), optimizer, model_state, adam_state, monitor)
    result.update(
        gpu=torch.cuda.get_device_name(0), variant=policy.variant,
        runtime={"workers": args.workers, "envs": args.envs, "reply_size": args.reply_size},
        episodes=len(episodes), inputs_sha256=hashes,
        protected_inputs_unchanged=hashes == [file_sha256(path) for path in inputs],
    )
    write_json(out / "summary.json", result)
    if not result["protected_inputs_unchanged"]:
        raise RuntimeError("Profiling changed protected inputs")
