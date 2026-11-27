"""Bounded RL benchmarks for saved replay, updates, collection and evaluation."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def resolve_inputs(args, parser):
    from civ5rl.config import load_config
    from civ5rl.provenance.common import read_json

    run = args.run.resolve() if args.run else None
    config_path = args.config
    if config_path is None and run:
        config_path = run / "resolved-config.json"
        if not config_path.exists():
            config_path = run / "run.json"
    if config_path is None:
        parser.error("provide --run RUN, or --config, --checkpoint and --rollout")
    config_path = config_path.resolve()
    config = load_config(config_path)
    config = config.get("config", config)
    checkpoint = args.checkpoint
    if checkpoint is None and run:
        checkpoint = run / read_json(run / "state.json")["latest"]
    rollout = args.rollout
    if rollout is None and run:
        for name in ("latest.pt", "first.pt"):
            if (run / "rollouts" / name).is_file():
                rollout = run / "rollouts" / name
                break
        if rollout is None:
            files = sorted((run / "rollouts").glob("*.pt"))
            rollout = files[-1] if files else None
    if checkpoint is None or rollout is None:
        parser.error(
            "checkpoint and rollout are required; specify their paths or --run"
        )
    inputs = [config_path, checkpoint.resolve(), rollout.resolve()]
    if any(not path.is_file() for path in inputs):
        parser.error("config, checkpoint and rollout must be existing files")
    return config, inputs


def replay(args, parser, config, inputs):
    from civ5rl.engine.ppo import backend
    from civ5rl.engine.ppo.storage import load_rollout
    from civ5rl.engine.runtime.telemetry import ResourceMonitor
    from civ5rl.experiments.benchmarks.ppo import run_replay_benchmark
    from civ5rl.experiments.imitation.helpers import load_policy
    from civ5rl.provenance.common import (
        configure,
        file_sha256,
        write_json,
    )
    from civ5sim.engine import digest

    output = args.out.resolve()
    if output.suffix.lower() != ".json":
        output /= "benchmark.json"
    if output.exists():
        parser.error("benchmark output already exists; choose a new result path")
    if output in inputs:
        parser.error("output must not overwrite an input")
    configure(args.seed, args.device)
    episodes = load_rollout(inputs[2])
    total = len(episodes)
    if args.episode_limit:
        episodes = episodes[: args.episode_limit]
    backend_path = Path(backend.__file__).resolve()
    shared = sorted(
        {
            Path(module.__file__).resolve()
            for name, module in tuple(sys.modules.items())
            if name.startswith(("civ5rl.", "civ5sim.", "civ5map."))
            and getattr(module, "__file__", None)
            and Path(module.__file__).is_file()
        }
        - {backend_path}
    )
    protected = list(dict.fromkeys([*inputs, backend_path, *shared]))
    policy = load_policy(config, inputs[1], args.device)
    telemetry = output.parent / (output.stem + "-telemetry")
    if telemetry.exists():
        parser.error("benchmark telemetry directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ResourceMonitor(telemetry, interval=0.25) as monitor:
        result = run_replay_benchmark(
            policy=policy,
            episodes=episodes,
            config=config,
            device=args.device,
            repeats=args.repeats,
            warmup=args.warmup,
            protected_inputs=protected,
            monitor=monitor,
        )
        monitor.mark("finished")
    result.update(
        current_source_sha256=file_sha256(backend_path),
        shared_source_sha256={
            p.relative_to(ROOT).as_posix(): file_sha256(p) for p in shared
        },
        config=str(inputs[0]),
        checkpoint=str(inputs[1]),
        rollout=str(inputs[2]),
        resolved_config_sha256=digest(config),
        available_episodes=total,
        selected_episode_indices=list(range(len(episodes))),
        device=args.device,
        telemetry_directory=str(telemetry),
        scientific_training_extended=False,
    )
    write_json(output, result)
    print(
        {
            "output": str(output),
            "order": result["order"],
            "results": result["results"],
        }
    )


def full_cycle(args, parser, config, inputs):
    import torch

    from civ5rl.engine.collection.collector import Collector
    from civ5rl.engine.ppo.adam import optimizer_for
    from civ5rl.engine.ppo.storage import load_rollout
    from civ5rl.engine.runtime.telemetry import ResourceMonitor
    from civ5rl.experiments.benchmarks.ppo import run_full_cycle_benchmark
    from civ5rl.experiments.imitation.helpers import load_policy
    from civ5rl.provenance import bound_cases, case_manifest
    from civ5rl.provenance.common import configure, write_json
    from civ5rl.provenance.rng import restore_rng
    from civ5sim.engine import digest

    out = args.out.resolve()
    if any(p == out or p.is_relative_to(out) for p in inputs):
        parser.error("output must not contain an input")
    if out.exists() and any(out.iterdir()):
        parser.error("output directory must be empty")
    configure(args.seed, args.device)
    saved = torch.load(inputs[1], map_location="cpu", weights_only=False)
    if "optimizer" not in saved:
        parser.error(
            "full-cycle requires a PPO boundary checkpoint containing compatible Adam state"
        )
    shuffle = random.Random(args.seed)
    episodes = load_rollout(inputs[2])
    manifest = case_manifest(
        [e["spec"] for e in episodes],
        source="rl01.benchmark",
        split="train",
        namespace="fixed_rollout",
    )
    policy = load_policy(config, inputs[1], args.device)
    for parameter in policy.value.parameters():
        parameter.requires_grad_(True)
    optimizer = optimizer_for(policy, config, args.device)
    optimizer.load_state_dict(saved["optimizer"])
    keys = ("python_rng", "numpy_rng", "torch_rng", "cuda_rng", "shuffle_rng")
    if all(k in saved for k in keys):
        restore_rng(saved, shuffle)
        rng_source = "checkpoint"
    else:
        rng_source = "seed"
    out.mkdir(parents=True, exist_ok=True)
    with ResourceMonitor(out / "telemetry", interval=0.25) as monitor:
        with Collector(config, None, args.device, out / "errors") as collector:
            result = run_full_cycle_benchmark(
                policy=policy,
                optimizer=optimizer,
                shuffle_rng=shuffle,
                collector=collector,
                config=config,
                specs=bound_cases(manifest),
                manifest=manifest,
                output=out,
                device=args.device,
                repeats=args.repeats,
                protected_inputs=inputs,
                monitor=monitor,
            )
        monitor.mark("finished")
    result.update(
        resolved_config_sha256=digest(config),
        rng_source=rng_source,
        device=args.device,
        scientific_training_extended=False,
    )
    write_json(out / "benchmark.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        nargs="?",
        choices=(
            "full-cycle",
            "replay",
            "ppo-update",
            "selfplay-profile",
            "bc",
            "evaluation",
            "comparison",
            "comparison-collection",
        ),
        default="full-cycle",
    )
    parser.add_argument(
        "--run",
        type=Path,
        help="Resolve saved config, latest checkpoint and rollout from RUN",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument(
        "--history-microbatch",
        type=int,
        default=2,
        help="Whole-turn fit histories per GPU microbatch",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whole-turn fit CUDA compilation",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--rollout", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Measured replay passes or full-cycle/PPO update trials",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Replay/PPO update warmup passes (or BC warmup minibatches)",
    )
    parser.add_argument(
        "--episode-limit",
        type=int,
        help="Replay/PPO update: first N saved episodes",
    )
    ppo_update = parser.add_argument_group("ppo-update")
    ppo_update.add_argument(
        "--baseline-result", type=Path, help="Saved ppo-update benchmark.json"
    )
    ppo_update.add_argument(
        "--minibatch-tokens", type=int, help="Override valid tokens per PPO minibatch"
    )
    ppo_update.add_argument(
        "--check-rollback",
        action="store_true",
        help="Also force and check a KL rejection",
    )
    ppo_update.add_argument("--no-resident-replay-indices", action="store_true")
    ppo_update.add_argument("--replay-action-bucket", type=int, help="Split replay scoring rows at this legal-action count")
    parser.add_argument(
        "--steps",
        type=int,
        help="BC and comparison: number of measured Adam steps",
    )
    parser.add_argument(
        "--trace-steps",
        type=int,
        default=0,
        help="BC only: post-measurement Kineto trace steps",
    )
    parser.add_argument(
        "--scratch-dir",
        type=Path,
        help="BC only: scratch directory outside scientific run directory",
    )
    parser.add_argument(
        "--prep-workers",
        type=int,
        help="BC only: worker processes for one-time prepared-inputs",
    )
    parser.add_argument(
        "--cold-cache",
        action="store_true",
        help="BC only: clear CPU LRU episode cache before measured window",
    )
    parser.add_argument(
        "--legacy-loop",
        action="store_true",
        help="BC only: force synchronous per-step baseline loop",
    )
    parser.add_argument(
        "--verify-parity",
        action="store_true",
        help="BC only: verify CE, grad norm, and parameter parity vs legacy loop",
    )
    parser.add_argument(
        "--parity-steps",
        type=int,
        default=8,
        help="BC only: number of steps for parity verification",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--cases", type=int, help="Evaluation and rollout-comparison saved case count"
    )
    parser.add_argument(
        "--workers", type=int, help="Evaluation, comparison, or self-play workers"
    )
    parser.add_argument("--envs", type=int, default=80, help="Self-play profile environments")
    parser.add_argument("--reply-size", type=int, default=1, help="Self-play profile reply size")
    parser.add_argument(
        "--max-batch",
        type=int,
        default=32,
        help="Evaluation and comparison inference batch",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=0.001,
        help="Evaluation and comparison batch wait",
    )
    collection = parser.add_argument_group("comparison-collection")
    collection.add_argument(
        "--worker-counts", default="6,8,12,16", help="Comma-separated W candidates"
    )
    collection.add_argument(
        "--inference-services",
        type=int,
        default=0,
        help="Isolated GPU inference processes; 0 uses the in-process thread",
    )
    collection.add_argument(
        "--prefix-states",
        action="store_true",
        help="Restore branch GRU state at reset and skip prefix observe RPCs",
    )
    collection.add_argument(
        "--forks", type=int, default=64, help="Saved forks in the measured work"
    )
    collection.add_argument(
        "--seconds", type=float, default=300, help="Measured window per mode"
    )
    collection.add_argument("--warmup-forks", type=int, default=4)
    collection.add_argument(
        "--warmup-measured-work",
        action="store_true",
        help="Warm up with one complete pass of the measured forks before timing the same work",
    )
    collection.add_argument("--source-cases", type=int, default=8)
    collection.add_argument(
        "--barrier-workers",
        type=int,
        default=8,
        help="Former per-fork barrier control; 0 skips",
    )
    args = parser.parse_args()
    if args.phase != "ppo-update" and (
        args.baseline_result or args.minibatch_tokens is not None or args.check_rollback
    ):
        parser.error(
            "--baseline-result, --minibatch-tokens and --check-rollback apply to ppo-update only"
        )
    if args.phase not in ("ppo-update", "selfplay-profile") and (
        args.no_resident_replay_indices or args.replay_action_bucket is not None
    ):
        parser.error("Replay performance overrides require ppo-update or selfplay-profile")
    if args.replay_action_bucket is not None and (
        args.replay_action_bucket < 1 or args.no_resident_replay_indices
    ):
        parser.error("Replay action bucketing needs a positive width and resident replay indices")
    if args.phase == "comparison-collection":
        from civ5rl.experiments.benchmarks.comparison_collection import run_benchmark

        return run_benchmark(args, parser)
    if args.phase == "comparison":
        config = (
            json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
        )
        if config.get("kind") == "rl01.turn_comparison_fit":
            if args.steps is None:
                args.steps = 1
            from civ5rl.experiments.benchmarks.turn_fit import benchmark

            return benchmark(args, parser)
        if config.get("training", {}).get("mode") in (
            "full_corpus_step_kl",
            "frozen_heads_step_kl",
        ):
            if args.steps is None:
                args.steps = 1
            from civ5rl.experiments.benchmarks.comparison_control import benchmark

            return benchmark(args, parser)
        if args.steps is None:
            args.steps = 3
        from civ5rl.experiments.benchmarks.comparison_performance import run_benchmark

        return run_benchmark(args, parser)
    if args.phase == "evaluation":
        from tools._evaluation_benchmark import run_evaluation_benchmark

        if args.cases is None:
            args.cases = 3
        if args.workers is None:
            args.workers = 4
        run_evaluation_benchmark(args, parser)
        return
    if args.phase == "bc":
        if args.steps is None:
            args.steps = 450
        if args.config is None or args.checkpoint is None:
            parser.error("bc benchmark requires --config and --checkpoint")
        if args.steps < 1 or args.warmup < 0 or args.trace_steps < 0:
            parser.error("bc --steps must be >= 1 and --warmup/--trace-steps >= 0")
        from tools._bc_benchmark import run_bc_benchmark

        run_bc_benchmark(args, parser)
        return
    if (
        args.repeats < 1
        or args.warmup < 1
        or (args.episode_limit is not None and args.episode_limit < 1)
    ):
        parser.error("repeats, warmup and episode limit must be positive")
    if args.episode_limit and args.phase not in ("replay", "ppo-update", "selfplay-profile"):
        parser.error("--episode-limit applies to replay, ppo-update and selfplay-profile only")
    config, inputs = resolve_inputs(args, parser)
    if args.no_resident_replay_indices:
        config["performance"]["ppo_resident_replay_indices"] = False
    if args.replay_action_bucket is not None:
        config["performance"]["ppo_replay_action_bucket"] = args.replay_action_bucket
    if args.phase == "replay":
        replay(args, parser, config, inputs)
    elif args.phase == "ppo-update":
        from civ5rl.experiments.benchmarks.ppo_update import run_benchmark

        if not run_benchmark(args, parser, config, inputs):
            raise SystemExit(1)
    elif args.phase == "selfplay-profile":
        from civ5rl.experiments.benchmarks.selfplay import profile_selfplay
        from civ5rl.engine.runtime.hardware import hardware

        if args.workers is None:
            args.workers = min(args.envs, max(1, int(hardware()["effective_cpu_budget"])))
        profile_selfplay(args, parser, config, inputs)
    else:
        full_cycle(args, parser, config, inputs)


if __name__ == "__main__":
    main()
