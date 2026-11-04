"""Evaluate an RL checkpoint on pilot-permitted splits only."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from civ5rl.common import configure, read_json, write_json
    from civ5rl.evaluation import require_split

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("--split", default="validation")
    p.add_argument("--config", default="configs/rl-v1.json")
    p.add_argument("--corpus", default="artifacts/rl/corpus/manifest.json")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    config = read_json(args.config)
    require_split(config, args.split)  # Fail before loading the model/corpus.
    configure(config["seed"], args.device)
    from civ5rl.agent import RLAgent
    from civ5rl.resources import ResourceMonitor

    agent = RLAgent.from_checkpoint(args.checkpoint, args.device)
    with ResourceMonitor(args.out) as monitor:
        if args.split == "memory_dev":
            from civ5rl.memory import build_dev, evaluate_dev

            cases = build_dev()
            write_json(
                Path(args.out) / "manifest.json",
                [
                    {k: v for k, v in case.items() if k not in ("actions", "history")}
                    for case in cases
                ],
            )
            monitor.mark("memory_dev")
            evaluate_dev(agent.policy, cases, args.device, args.out, config)
        else:
            from civ5rl.collector import Collector
            from civ5rl.corpus import ScenarioFactory
            from civ5rl.evaluation import evaluate, validation_specs

            factory = ScenarioFactory(config, read_json(args.corpus))
            specs = validation_specs(factory, config)
            with Collector(
                config, factory, args.device, Path(args.out) / "errors"
            ) as collector:
                evaluate(agent.policy, collector, specs, config, args.out, monitor)


if __name__ == "__main__":
    main()
