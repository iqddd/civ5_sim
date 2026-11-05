"""Run bounded PPO with exact categorical KL acceptance and rollback."""

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
    parser.add_argument("--config", default="configs/rl-ppo-controlled-updates.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    from civ5rl.provenance.common import read_json
    from civ5rl.experiments.ppo_after_imitation import run

    config = read_json(ROOT / args.config)
    if args.phase == "run" and not (ROOT / config["output"] / "passport.json").exists():
        run("prepare", config, ROOT / config["output"], args.device)
    run(args.phase, config, ROOT / config["output"], args.device)


if __name__ == "__main__":
    main()
