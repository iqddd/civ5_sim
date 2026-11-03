"""Run agent matches, execute command scripts or verify a recorded replay."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .engine import Battle, digest, load_scenario
from .match import load_match_config, run_match


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    match = commands.add_parser("match", help="Run a finite match between two agents")
    match.add_argument("config", type=Path)
    match.add_argument(
        "--out", type=Path, required=True, help="Match artifact directory"
    )
    match.add_argument("--diagnostics", choices=("off", "compact", "full"), default="off")
    run = commands.add_parser("run")
    run.add_argument("scenario", type=Path)
    run.add_argument(
        "--commands", type=Path, required=True, help="JSON array of actions"
    )
    run.add_argument("--out", type=Path, required=True, help="Final snapshot")
    run.add_argument(
        "--replay", type=Path, help="Optional full-information replay journal"
    )
    replay = commands.add_parser("replay")
    replay.add_argument("source", type=Path)
    replay.add_argument("--out", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "match":
            config, agents = load_match_config(args.config)
            match = run_match(config, agents, diagnostics=args.diagnostics)
            match.save(args.out)
            print(
                json.dumps(
                    {
                        k: match.result[k]
                        for k in (
                            "status",
                            "outcome",
                            "winner",
                            "reason",
                            "commands",
                            "alive",
                        )
                    }
                )
            )
            if match.result["status"] == "error":
                parser.exit(1, f"Agent error: {match.result['error']}\n")
        elif args.command == "run":
            scenario = load_scenario(args.scenario)
            actions = json.loads(args.commands.read_text(encoding="utf-8"))
            if not isinstance(actions, list):
                raise ValueError("Commands must be a JSON array")
            with Battle(scenario, log=args.replay is not None) as battle:
                for i, action in enumerate(actions):
                    result = battle.step(action)
                    if not result.ok:
                        raise ValueError(f"Command {i}: {result.error}")
                write(args.out, battle.snapshot())
                if args.replay:
                    write(args.replay, battle.replay())
                print(json.dumps({"commands": len(actions), **battle.army_status()}))
        else:
            value = json.loads(args.source.read_text(encoding="utf-8"))
            if value.get("schema") != 2 or value.get("kind") != "civ5sim.replay":
                raise ValueError("Expected civ5sim.replay schema 2")
            with Battle(value["initial"]) as battle:
                for i, entry in enumerate(value["steps"]):
                    result = battle.step(entry["action"])
                    if (
                        asdict(result) != entry["result"]
                        or digest(battle.snapshot()) != entry["state_sha256"]
                    ):
                        raise ValueError(f"Replay diverged at command {i}")
                if args.out:
                    write(args.out, battle.snapshot())
                print(f"Verified {len(value['steps'])} commands")
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
