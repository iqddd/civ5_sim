"""Prepare and run the bounded seed-17 PPO-from-imitation protocol."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("prepare", "train", "evaluate", "report", "run")
    )
    parser.add_argument("--config", default="configs/rl-ppo-after-imitation.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from civ5rl.config import load_config
    from civ5rl.experiments.ppo_after_imitation import run

    config = load_config(ROOT / args.config)
    if args.phase == "run" and not (ROOT / config["output"] / "passport.json").exists():
        run("prepare", config, ROOT / config["output"], args.device)
    run(args.phase, config, ROOT / config["output"], args.device)


if __name__ == "__main__":
    main()
