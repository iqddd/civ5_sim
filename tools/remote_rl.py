"""Package, launch, inspect and retrieve experiments on an existing instance."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from civ5rl.deploy.operations import launch, package, pull, status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cmd = commands.add_parser("package")
    cmd.add_argument("--config", required=True)
    cmd.add_argument("--out", required=True)
    cmd.add_argument("--instance")
    cmd.add_argument("--extra-input", action="append", default=[])
    for name in ("launch", "status", "pull"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--deployment", required=True)
        if name == "launch":
            cmd.add_argument("--mode", choices=("tune", "run", "resume"), required=True)
            cmd.add_argument("--instance")
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
        result = launch(args.deployment, args.mode, instance=args.instance)
    elif args.command == "status":
        result = status(args.deployment)
    else:
        result = pull(args.deployment, delete_instance=args.delete_instance)
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
