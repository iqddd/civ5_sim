"""Local RL experiment control and artifact-only analysis."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from civ5rl.engine.control import close_run, request_pause
from civ5rl.engine.performance import performance_markdown, summarize_performance
from civ5rl.provenance.common import write_json
from civ5rl.reporting.operations import audit_run, compare, wait_and_finalize
from civ5rl.reporting.snapshot import write_snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in (
        "snapshot",
        "performance",
        "audit",
        "request-pause",
        "close",
        "wait-and-finalize",
    ):
        cmd = commands.add_parser(name)
        cmd.add_argument("--run", type=Path, required=True)
        if name in ("snapshot", "audit", "wait-and-finalize"):
            cmd.add_argument("--out", type=Path, required=True)
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
        if name == "close":
            cmd.add_argument("--reason", required=True)
        if name == "wait-and-finalize":
            cmd.add_argument("--timeout", type=float, required=True)
    cmd = commands.add_parser("compare")
    for name in ("before", "after", "out"):
        cmd.add_argument(f"--{name}", type=Path, required=True)
    cmd.add_argument("--expected-n", type=int, required=True)
    cmd.add_argument("--clusters", type=int, default=16)
    cmd.add_argument("--samples", type=int, default=10000)
    cmd.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args(argv)
    if args.command == "snapshot":
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
