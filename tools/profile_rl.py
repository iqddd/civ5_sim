"""Bounded fixed-policy rollout, CPU profiles and a reusable public rollout."""

import argparse
import copy
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def main():
    import torch

    from civ5rl.config import load_config
    from civ5rl.collector import Collector
    from civ5rl.common import configure, read_json, write_json
    from civ5rl.corpus import ScenarioFactory
    from civ5rl.model import Policy
    from civ5rl.resources import ResourceMonitor

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--profile-workers", action="store_true")
    p.add_argument("--stage", type=int, default=2)
    p.add_argument("--config", default="configs/rl-v1.json")
    p.add_argument(
        "--checkpoint",
        default="artifacts/rl/training-result/checkpoints/initial.pt",
    )
    args = p.parse_args()
    directory = Path(args.out).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(load_config(args.config))
    config["collection"]["min_learner_decisions_per_update"] = 1
    if args.profile_workers:
        config.setdefault("performance", {})["worker_profile_dir"] = str(directory)
    configure(config["seed"], "cuda")
    policy = Policy().cuda()
    policy.load_state_dict(
        torch.load(args.checkpoint, weights_only=False, map_location="cuda")["model"]
    )
    manifest = read_json("artifacts/rl/corpus/manifest.json")
    with (
        ResourceMonitor(directory) as monitor,
        Collector(
            config, ScenarioFactory(config, manifest), "cuda", directory / "errors"
        ) as collector,
    ):
        monitor.mark("fixed_rollout")
        episodes, next_episode, times = collector.collect(
            policy, config["curriculum"][args.stage], 0, monitor
        )
    torch.save(episodes, directory / "rollout.pt")
    write_json(
        directory / "summary.json",
        {
            "timings": times,
            "decisions": sum(len(e["chosen"]) for e in episodes),
            "next_episode": next_episode,
            "opponent_seconds_sum": sum(
                e["result"]["timings"]["opponent_seconds"] for e in episodes
            ),
        },
    )


if __name__ == "__main__":
    main()
