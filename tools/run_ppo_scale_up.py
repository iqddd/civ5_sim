"""Continue model/Adam with bounded PPO; use run.json instead of environment certification."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "train", "evaluate", "report", "run", "pause", "resume"))
    parser.add_argument("--config", default="configs/rl-ppo-scale-up.json")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    import json
    kind = json.loads((ROOT / args.config).read_text())['kind']
    if kind == 'rl01.ppo_after_target_bc':
        from civ5rl.protocols.ppo_after_target_bc import load_config
        from civ5rl.experiments.ppo_after_target_bc import run
    elif kind == 'rl01.ppo_target_scale':
        from civ5rl.protocols.ppo_target_scale import load_config
        from civ5rl.experiments.ppo_target_scale import run
    else:
        if args.phase in ('pause','resume'):
            parser.error('pause/resume entrypoints apply to the target-scale experiment')
        from civ5rl.protocols.ppo_scale_up import load_config
        from civ5rl.experiments.ppo_scale_up import run
    config = load_config(ROOT / args.config)
    run(args.phase, config, ROOT / config["output"], args.device)


if __name__ == "__main__":
    main()
