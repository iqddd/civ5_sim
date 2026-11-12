"""Bounded full-cycle PPO or saved-replay benchmark, optionally against Git."""

import argparse
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
    import civ5rl.engine.ppo_fast as current
    from civ5rl.engine.benchmark import run_replay_benchmark
    from civ5rl.engine.resources import ResourceMonitor
    from civ5rl.experiments.imitation_helpers import load_policy
    from civ5rl.experiments.ppo_artifacts import load_rollout
    from civ5rl.provenance.checkpoint import artifact_sources
    from civ5rl.provenance.common import (
        configure,
        file_sha256,
        load_module_from_git,
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
    modules = {}
    if args.compare_git_rev:
        modules["git"] = load_module_from_git(
            "civ5rl.engine._git_replay_baseline",
            "civ5rl/engine/ppo_fast.py",
            args.compare_git_rev,
        )
    modules["working"] = current
    configure(args.seed, args.device)
    episodes = load_rollout(inputs[2])
    total = len(episodes)
    if args.episode_limit:
        episodes = episodes[: args.episode_limit]
    # The baseline is intentionally one module. Record and protect all shared
    # execution sources to make the comparison's dependency scope explicit.
    shared = [
        p
        for p in artifact_sources(config)
        if p.resolve() != Path(current.__file__).resolve()
    ]
    protected = list(dict.fromkeys([*inputs, Path(current.__file__), *shared]))
    policy = load_policy(config, inputs[1], args.device)
    telemetry = output.parent / (output.stem + "-telemetry")
    if telemetry.exists():
        parser.error("benchmark telemetry directory already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    with ResourceMonitor(telemetry, interval=0.25) as monitor:
        result = run_replay_benchmark(
            policy=policy,
            episodes=episodes,
            implementations=modules,
            config=config,
            device=args.device,
            repeats=args.repeats,
            warmup=args.warmup,
            protected_inputs=protected,
            monitor=monitor,
        )
        monitor.mark("finished")
    result.update(
        comparison_scope="selected ppo_fast.py module; shared current model/dependencies/native libraries",
        git_baseline=getattr(modules.get("git"), "__git_baseline__", None),
        current_source_sha256=file_sha256(current.__file__),
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
    if "git" in result["results"]:
        result["working_over_git_time"] = (
            result["results"]["working"]["mean_seconds"]
            / result["results"]["git"]["mean_seconds"]
        )
    write_json(output, result)
    print(
        {
            "output": str(output),
            "order": result["order"],
            "parity_verified": result["parity_verified"],
            "results": result["results"],
        }
    )


def full_cycle(args, parser, config, inputs):
    import torch

    from civ5rl.engine.benchmark import run_full_cycle_benchmark
    from civ5rl.engine.collector import Collector
    from civ5rl.engine.resources import ResourceMonitor
    from civ5rl.experiments.imitation_helpers import load_policy
    from civ5rl.experiments.ppo_after_imitation import optimizer_for
    from civ5rl.experiments.ppo_artifacts import load_rollout
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
        choices=("full-cycle", "replay", "bc", "evaluation"),
        default="full-cycle",
    )
    parser.add_argument(
        "--run",
        type=Path,
        help="Resolve saved config, latest checkpoint and rollout from RUN",
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--rollout", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Replay ABBA blocks or full-cycle measured trials",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Replay warmup passes per implementation (or BC warmup minibatches)",
    )
    parser.add_argument(
        "--compare-git-rev", help="Replay baseline: ppo_fast.py from this Git revision"
    )
    parser.add_argument(
        "--episode-limit",
        type=int,
        help="Replay only: first N saved episodes, recorded in result",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=450,
        help="BC only: number of measured Adam steps",
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
    parser.add_argument("--cases", type=int, default=3, help="Evaluation benchmark only")
    parser.add_argument("--workers", type=int, default=4, help="Evaluation benchmark only")
    parser.add_argument("--max-batch", type=int, default=32, help="Evaluation benchmark only")
    parser.add_argument("--wait-seconds", type=float, default=0.001, help="Evaluation benchmark only")
    args = parser.parse_args()
    if args.phase == "evaluation":
        from tools._evaluation_benchmark import run_evaluation_benchmark
        run_evaluation_benchmark(args, parser)
        return
    if args.phase == "bc":
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
    if args.phase != "replay" and (args.compare_git_rev or args.episode_limit):
        parser.error("--compare-git-rev/--episode-limit apply to replay only")
    config, inputs = resolve_inputs(args, parser)
    if args.phase == "replay":
        replay(args, parser, config, inputs)
    else:
        full_cycle(args, parser, config, inputs)


if __name__ == "__main__":
    main()
