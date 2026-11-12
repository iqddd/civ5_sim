"""Dispatch configured PPO experiment phases through one explicit entrypoint."""

import argparse
import importlib
import os
import sys
from contextlib import nullcontext
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("prepare", "train", "evaluate", "report", "run", "pause", "resume")
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-updates", type=int)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--runtime-config", type=Path)
    parser.add_argument("--evaluation-device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--evaluation-workers", type=int)
    parser.add_argument("--evaluation-max-batch", type=int)
    parser.add_argument("--evaluation-wait-seconds", type=float)
    args = parser.parse_args(argv)
    from civ5rl.config import load_config

    config = load_config(ROOT / args.config)
    kind = config.get("kind")
    modules = {
        "rl01.ppo_after_imitation": "ppo_after_imitation",
        "rl01.ppo_controlled_updates": "ppo_after_imitation",
        "rl01.ppo_scale_up": "ppo_scale_up",
        "rl01.ppo_target_scale": "ppo_target_scale",
        "rl01.ppo_after_target_bc": "ppo_after_target_bc",
    }
    if kind not in modules:
        parser.error(f"Unsupported PPO kind: {kind!r}")
    if any(v is not None for v in (args.evaluation_device, args.evaluation_workers, args.evaluation_max_batch, args.evaluation_wait_seconds)) and kind not in ("rl01.ppo_target_scale", "rl01.ppo_after_target_bc"):
        parser.error("evaluation runtime options apply only to target-scale or after-target-BC PPO")
    target = kind in ("rl01.ppo_target_scale", "rl01.ppo_after_target_bc")
    if args.phase in ("pause", "resume") and not target:
        parser.error("pause/resume apply only to target-scale or after-target-BC PPO")
    if (
        args.runtime_config is not None or args.max_updates is not None or args.out is not None
    ) and kind != "rl01.ppo_after_target_bc":
        parser.error("--runtime-config/--max-updates/--out apply only to PPO after target BC")
    name = modules[kind]
    if name != "ppo_after_imitation":
        protocol = importlib.import_module("civ5rl.protocols." + name)
        config = protocol.load_config(ROOT / args.config)
    runner = importlib.import_module("civ5rl.experiments." + name).run
    out = args.out or ROOT / config["output"]
    from civ5rl.reporting.progress_process import ProcessProgress
    from civ5rl.engine.evaluation import evaluation_options

    progress = ProcessProgress(out, config) if args.phase in ("train", "run", "resume", "evaluate") else nullcontext()
    with evaluation_options(device=args.evaluation_device, workers=args.evaluation_workers,
                            max_batch=args.evaluation_max_batch, wait_seconds=args.evaluation_wait_seconds), progress:
        return _execute(args, config, kind, name, runner, out)


def _execute(args, config, kind, name, runner, out):
    if (
        name == "ppo_after_imitation" and args.phase == "run"
        and not (out / "passport.json").exists()
    ):
        runner("prepare", config, out, args.device)
    if kind == "rl01.ppo_after_target_bc":
        return runner(
            args.phase, config, out, args.device,
            runtime_path=args.runtime_config, max_updates=args.max_updates,
        )
    return runner(args.phase, config, out, args.device)


if __name__ == "__main__":
    main()
