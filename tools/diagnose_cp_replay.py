"""Replay a saved episode and record the first divergence and optional CP traces."""

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def diagnose(episode, *, trace_from=None):
    from civ5rl.core.env import DecisionEnv
    from civ5sim.combat_ai import Planner, PlanningState
    from civ5sim.engine import digest

    env = DecisionEnv(episode["spec"])
    records = []
    previous = Planner._plan

    def capture(planner, request, mode=None, **kwargs):
        record = trace_from is not None and len(env.history) >= trace_from
        if record:
            planner.set_trace(True)
        try:
            result = previous(planner, request, mode, **kwargs)
            if record:
                records.append(
                    {
                        "command_index": len(env.history),
                        "request": asdict(request),
                        "config": asdict(planner.config),
                        "mode": int(mode) if mode is not None else None,
                        "candidates_only": kwargs.get("candidates_only", False),
                        "result": asdict(result) if is_dataclass(result) else result,
                        "trace": planner.read_trace(),
                    }
                )
            return result
        finally:
            if record:
                planner.set_trace(False)

    Planner._plan = capture
    mismatch = None
    root_checks = 0
    state = {}
    try:
        for index, chosen in enumerate(episode["chosen"]):
            state = env.request()
            before = digest(env.battle.snapshot())
            for side in (0, 1):
                PlanningState.from_battle(env.battle, side)
                root_checks += 1
            if digest(env.battle.snapshot()) != before:
                raise AssertionError("PlanningState.from_battle changed the root state")
            if (
                state.get("actions") != episode["actions"][index]
                or state.get("observation") != episode["observations"][index]
            ):
                mismatch = index
                break
            action = state["actions"][chosen]
            env.step(action)
        else:
            state = env.request()
        actual = state.get("joint_commands", env.history)
        expected = episode["joint_commands"]
        command_mismatch = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b), None)
        if command_mismatch is None and mismatch is None and len(expected) != len(actual):
            command_mismatch = min(len(expected), len(actual))
        expected_result = {k: v for k, v in episode["result"].items() if k != "timings"}
        actual_result = {k: v for k, v in state.get("result", {}).items() if k != "timings"}
        return {
            "matched": (
                mismatch is None
                and command_mismatch is None
                and state.get("done", False)
                and actual_result == expected_result
            ),
            "public_mismatch_decision": mismatch,
            "first_command_mismatch": command_mismatch,
            "expected_command_count": len(expected),
            "actual_command_count": len(actual),
            "expected_commands": expected[max(0, command_mismatch - 2) : command_mismatch + 5]
            if command_mismatch is not None
            else [],
            "actual_commands": actual[max(0, command_mismatch - 2) : command_mismatch + 5]
            if command_mismatch is not None
            else [],
            "done": state.get("done", False),
            "root_checks": root_checks,
            "expected_result": expected_result,
            "actual_result": actual_result,
            "planner_traces": records,
        }
    finally:
        Planner._plan = previous
        env.close()


def main():
    from civ5rl.experiments.ppo_artifacts import load_rollout
    from civ5rl.provenance.common import file_sha256

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--trace-from", type=int, help="Trace CP plans from this command index")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.episode < 0 or (args.trace_from is not None and args.trace_from < 0):
        parser.error("indices must be nonnegative")
    if args.out.resolve() == args.rollout.resolve():
        parser.error("output must not overwrite the input rollout")
    checksum = file_sha256(args.rollout)
    result = diagnose(load_rollout(args.rollout)[args.episode], trace_from=args.trace_from)
    if checksum != file_sha256(args.rollout):
        raise RuntimeError("Input rollout changed")
    result.update(rollout_sha256=checksum, episode=args.episode)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
