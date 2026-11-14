"""Package, launch, inspect and retrieve experiments on an existing instance."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "rent":
        from civ5rl.deploy.vast_cpu import rent
        code = rent(argv[1:])
        if code:
            raise SystemExit(code)
        return code
    from civ5rl.deploy.operations import launch, package, pull, status, extend_remote_budget

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("rent", help="stock Vast create options plus optional --cpu-score-min")
    cmd = commands.add_parser("package")
    cmd.add_argument("--config", required=True)
    cmd.add_argument("--out", required=True)
    cmd.add_argument("--instance")
    cmd.add_argument("--extra-input", action="append", default=[])
    for name in ("launch", "status", "pull", "extend-budget"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--deployment", required=True)
        if name == "extend-budget":
            cmd.add_argument("--additional-seconds", type=int, required=True)
            cmd.add_argument("--reason", required=True)
            cmd.add_argument("--request-id")
        if name == "launch":
            cmd.add_argument("--host-resume-reason")
            cmd.add_argument("--mode", choices=("tune", "run", "resume"), required=True)
            cmd.add_argument("--instance")
            cmd.add_argument("--evaluation-device", choices=("auto", "cpu", "cuda"))
            cmd.add_argument("--evaluation-workers", type=int)
            cmd.add_argument("--evaluation-max-batch", type=int)
            cmd.add_argument("--evaluation-wait-seconds", type=float)
        if name == "pull":
            flags = cmd.add_mutually_exclusive_group()
            flags.add_argument("--delete-instance", action="store_true")
            flags.add_argument("--keep-instance", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "package":
        result = package(
            args.config, args.out, instance=args.instance, extra_inputs=args.extra_input
        )
    elif args.command == "launch":
        result = launch(args.deployment, args.mode, instance=args.instance,
                        host_resume_reason=args.host_resume_reason,
                        evaluation_runtime={k: getattr(args, "evaluation_" + k) for k in ("device", "workers", "max_batch", "wait_seconds")
                                            if getattr(args, "evaluation_" + k) is not None})
    elif args.command == "status":
        result = status(args.deployment)
    elif args.command == "extend-budget":
        result = extend_remote_budget(args.deployment, args.additional_seconds, args.reason, request_id=args.request_id)
    else:
        result = pull(args.deployment, delete_instance=args.delete_instance)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
