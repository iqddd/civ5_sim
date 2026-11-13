"""Local RL experiment control and artifact-only analysis."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from civ5rl.engine.control import close_run, request_pause
from civ5rl.engine.performance import performance_markdown, summarize_performance
from civ5rl.provenance.common import write_json
from civ5rl.reporting.operations import audit_run, compare, extend_budget, wait_and_finalize
from civ5rl.reporting.snapshot import write_snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cmd = commands.add_parser("play-cases")
    cmd.add_argument("--manifest", type=Path, required=True)
    cmd.add_argument("--checkpoint", type=Path, required=True)
    cmd.add_argument("--out", type=Path, required=True)
    for name in (
        "snapshot",
        "performance",
        "overhead",
        "audit",
        "request-pause",
        "close",
        "extend-budget",
        "wait-and-finalize",
    ):
        cmd = commands.add_parser(name)
        cmd.add_argument("--run", type=Path, required=True)
        if name in ("snapshot", "audit", "wait-and-finalize", "overhead"):
            cmd.add_argument("--out", type=Path, required=True)
        if name == "overhead":
            cmd.add_argument("--reference-start", type=int, required=True)
            cmd.add_argument("--reference-stop", type=int, required=True)
            cmd.add_argument("--through-update", type=int)
            cmd.add_argument("--deployment", type=Path, action="append", default=[])
        if name in ("snapshot", "performance", "wait-and-finalize"):
            cmd.add_argument("--last", type=int)
        if name in ("snapshot", "wait-and-finalize"):
            cmd.add_argument("--block-size", type=int, default=10)
        if name == "performance":
            cmd.add_argument("--json", action="store_true")
        if name == "request-pause":
            boundary = cmd.add_mutually_exclusive_group(required=True)
            boundary.add_argument("--after-epoch", type=int)
            boundary.add_argument("--after-update", type=int)
        if name in ("close", "extend-budget"):
            cmd.add_argument("--reason", required=True)
        if name == "extend-budget":
            cmd.add_argument("--additional-seconds", type=int, required=True)
            cmd.add_argument("--request-id")
        if name == "wait-and-finalize":
            cmd.add_argument("--timeout", type=float, required=True)
    cmd = commands.add_parser("compare")
    for name in ("before", "after", "out"):
        cmd.add_argument(f"--{name}", type=Path, required=True)
    cmd.add_argument("--expected-n", type=int, required=True)
    cmd.add_argument("--clusters", type=int, default=16)
    cmd.add_argument("--samples", type=int, default=10000)
    cmd.add_argument("--seed", type=int, default=20261103)
    cmd = commands.add_parser("progress")
    source = cmd.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path)
    source.add_argument("--deployment", type=Path)
    cmd.add_argument("--watch", action="store_true")
    cmd.add_argument("--interval", type=float, default=900)
    cmd.add_argument("--alpha", type=float, default=0.1)
    cmd.add_argument("--exclude-compilation", action="store_true")
    cmd.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "play-cases":
        from civ5rl.reporting.showcase import play_cases
        result = play_cases(args.manifest, args.checkpoint, args.out)
    elif args.command == "progress":
        from civ5rl.reporting.progress_process import watch_progress
        return watch_progress(run=args.run, deployment=args.deployment, watch=args.watch,
                              interval=args.interval, alpha=args.alpha,
                              exclude_compilation=args.exclude_compilation, json_output=args.json)
    elif args.command == "snapshot":
        result = write_snapshot(
            args.run, args.out, last=args.last, block_size=args.block_size
        )
    elif args.command == "performance":
        result = summarize_performance(args.run, last=args.last)
        if not args.json:
            print(performance_markdown(result))
            return result
    elif args.command == "audit":
        result = audit_run(args.run, args.out)
    elif args.command == "overhead":
        from civ5rl.reporting.overhead import write_overhead
        result = write_overhead(
            args.run, args.out, reference_start=args.reference_start,
            reference_stop=args.reference_stop, through_update=args.through_update, deployments=args.deployment,
        )
    elif args.command == "compare":
        result = compare(
            args.before,
            args.after,
            args.expected_n,
            clusters=args.clusters,
            samples=args.samples,
            seed=args.seed,
        )
        args.out.mkdir(parents=True, exist_ok=True)
        write_json(args.out / "comparison.json", result)
    elif args.command == "request-pause":
        result = request_pause(
            args.run, after_epoch=args.after_epoch, after_update=args.after_update
        )
    elif args.command == "close":
        result = close_run(args.run, args.reason)
    elif args.command == "extend-budget":
        result = extend_budget(args.run, args.additional_seconds, args.reason, request_id=args.request_id)
    else:
        result = wait_and_finalize(
            args.run,
            args.out,
            timeout=args.timeout,
            last=args.last,
            block_size=args.block_size,
        )
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
