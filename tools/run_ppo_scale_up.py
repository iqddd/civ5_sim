"""Continue model/Adam with bounded PPO; use run.json instead of environment certification."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("prepare", "train", "evaluate", "report", "run", "pause", "resume"),
    )
    parser.add_argument("--config", default="configs/rl-ppo-scale-up.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-updates",
        type=int,
        help="Pause after this many completed PPO updates; does not change the scientific budget",
    )
    parser.add_argument(
        "--out", type=Path, help="Isolated output for a bounded local verification run"
    )
    parser.add_argument(
        "--runtime-config", type=Path, help="Machine settings for PPO after target BC"
    )
    args = parser.parse_args()
    import json

    kind = json.loads((ROOT / args.config).read_text())["kind"]
    if kind == "rl01.ppo_after_target_bc":
        from civ5rl.experiments.ppo_after_target_bc import run
        from civ5rl.protocols.ppo_after_target_bc import load_config
    elif kind == "rl01.ppo_target_scale":
        from civ5rl.experiments.ppo_target_scale import run
        from civ5rl.protocols.ppo_target_scale import load_config
    else:
        if args.phase in ("pause", "resume"):
            parser.error(
                "pause/resume entrypoints apply to the target-scale experiment"
            )
        from civ5rl.experiments.ppo_scale_up import run
        from civ5rl.protocols.ppo_scale_up import load_config
    config = load_config(ROOT / args.config)
    if args.runtime_config is not None and kind != "rl01.ppo_after_target_bc":
        parser.error("--runtime-config applies to PPO after target BC")
    if (
        args.max_updates is not None or args.out is not None
    ) and kind != "rl01.ppo_after_target_bc":
        parser.error("--max-updates/--out apply to PPO after target BC")
    if kind == "rl01.ppo_after_target_bc":
        run(
            args.phase,
            config,
            args.out or ROOT / config["output"],
            args.device,
            runtime_path=args.runtime_config,
            max_updates=args.max_updates,
        )
    else:
        run(args.phase, config, ROOT / config["output"], args.device)


if __name__ == "__main__":
    main()
