"""Dispatch configured PPO experiment phases through one explicit entrypoint."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train", "evaluate", "report", "run", "pause", "resume",
                                          "prepare-carrier", "pilot-report"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--evaluation-device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--evaluation-workers", type=int)
    parser.add_argument("--evaluation-max-batch", type=int)
    parser.add_argument("--evaluation-wait-seconds", type=float)
    parser.add_argument("--diagnostic-cpv2", action="store_true")
    parser.add_argument("--continue-after-dual-target", action="store_true")
    args = parser.parse_args(argv)
    from civ5rl.config import load_config

    config = load_config(ROOT / args.config)
    kind = config.get("kind")
    modules = {
        "rl01.ppo_after_imitation": "after_imitation",
        "rl01.ppo_controlled_updates": "after_imitation",
        "rl01.ppo_scale_up": "scale_up",
        "rl01.ppo_target_scale": "target_scale",
        "rl01.ppo_after_target_bc": "after_target_bc",
    }
    if kind not in modules:
        parser.error(f"Unsupported PPO kind: {kind!r}")
    if any(
        v is not None
        for v in (
            args.evaluation_device,
            args.evaluation_workers,
            args.evaluation_max_batch,
            args.evaluation_wait_seconds,
        )
    ) and kind not in ("rl01.ppo_target_scale", "rl01.ppo_after_target_bc"):
        parser.error("evaluation runtime options apply only to target-scale or after-target-BC PPO")
    target = kind in ("rl01.ppo_target_scale", "rl01.ppo_after_target_bc")
    if args.phase in ("pause", "resume") and not target:
        parser.error("pause/resume apply only to target-scale or after-target-BC PPO")
    if (
        args.runtime_config is not None or args.max_updates is not None or args.out is not None
    ) and kind != "rl01.ppo_after_target_bc":
        parser.error("--runtime-config/--max-updates/--out apply only to PPO after target BC")
    name = modules[kind]
    if name != "after_imitation":
        protocol = importlib.import_module("civ5rl.protocols.ppo_" + name)
        config = protocol.load_config(ROOT / args.config)
    if args.phase in ("prepare-carrier", "pilot-report"):
        if not config.get("experiment", {}).get("three_opponent_evaluation") or args.out is None:
            parser.error("CPv2 pilot tools require the CPv2 configuration and --out")
        from civ5rl.experiments.ppo.after_target_bc.diagnostic import pilot_report, prepare_carrier

        result = prepare_carrier(config, args.out) if args.phase == "prepare-carrier" else pilot_report(args.out)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return
    if args.diagnostic_cpv2 or args.continue_after_dual_target:
        if kind != "rl01.ppo_after_target_bc" or not config["experiment"].get("three_opponent_evaluation"):
            parser.error("These options require three-opponent after-target-BC PPO")
    if args.diagnostic_cpv2:
        if args.phase not in ("run", "resume") or args.max_updates is None or args.max_updates > 3:
            parser.error("CPv2 diagnostic requires run/resume with --max-updates at most 3")
        if not config["experiment"].get("pilot_only") and (
            args.out is None or args.out.resolve() == (ROOT / config["output"]).resolve()
        ):
            parser.error("Use the separate pilot configuration or an explicit isolated --out")
        config["experiment"]["diagnostic_training_only"] = True
        for key, value in config["experiment"].items():
            if key.endswith("_namespace"):
                config["experiment"][key] = value + "_diagnostic"
        config["self_play"]["namespace"] += "_diagnostic"
    elif config.get("experiment", {}).get("pilot_only") and args.phase in ("run", "train", "resume", "evaluate"):
        parser.error("The pilot configuration requires --diagnostic-cpv2 for execution")
    if args.continue_after_dual_target and (args.phase != "resume" or args.diagnostic_cpv2):
        parser.error("--continue-after-dual-target requires a production resume")
    runner = importlib.import_module("civ5rl.experiments.ppo." + name).run
    out = args.out or ROOT / config["output"]
    from civ5rl.provenance.execution import execution_session

    execution_session(out, args.phase, reopening=args.continue_after_dual_target)
    from civ5rl.engine.evaluation.executor import evaluation_options
    from civ5rl.reporting.progress_process import ProcessProgress

    progress = (
        ProcessProgress(out, config)
        if args.phase in ("train", "run", "resume", "evaluate")
        else nullcontext()
    )
    with (
        evaluation_options(
            device=args.evaluation_device,
            workers=args.evaluation_workers,
            max_batch=args.evaluation_max_batch,
            wait_seconds=args.evaluation_wait_seconds,
        ),
        progress,
    ):
        return _execute(args, config, kind, name, runner, out)


def _execute(args, config, kind, name, runner, out):
    if name == "after_imitation" and args.phase == "run" and not (out / "passport.json").exists():
        runner("prepare", config, out, args.device)
    if kind == "rl01.ppo_after_target_bc":
        return runner(
            args.phase,
            config,
            out,
            args.device,
            runtime_path=args.runtime_config,
            max_updates=args.max_updates,
            continue_dual_target=args.continue_after_dual_target,
        )
    return runner(args.phase, config, out, args.device)


if __name__ == "__main__":
    main()
