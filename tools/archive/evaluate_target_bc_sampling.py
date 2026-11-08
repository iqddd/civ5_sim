"""Evaluate the fixed BC epoch-ten policy at argmax, T=1 and T=0.5 on CPU."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("check", "run", "report"), nargs="?", default="run"
    )
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/rl-target-bc-sampling.json"
    )
    args = parser.parse_args()
    from civ5rl.experiments.target_bc_sampling import run
    from civ5rl.provenance.common import read_json

    run(read_json(args.config), args.phase)


if __name__ == "__main__":
    main()
