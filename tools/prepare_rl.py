"""Prepare RL-01 parent-world corpus without model evaluation."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from civ5rl.config import load_config
from civ5rl.corpus import prepare_corpus


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rl-v1.json")
    p.add_argument("--out", default="artifacts/rl/corpus")
    p.add_argument("--game", default="E:/civ5/Sid Meier's Civilization V")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    prepare_corpus(load_config(args.config), args.out, args.game, args.workers)


if __name__ == "__main__":
    main()
