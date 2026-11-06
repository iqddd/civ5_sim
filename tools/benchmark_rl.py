"""Three independent 2-warmup/10-measured full PPO cycle benchmarks."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    from civ5rl.config import load_config
    from civ5rl.common import read_json, write_json
    from tools.train_rl import run

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rl-v1.json")
    p.add_argument("--corpus", default="artifacts/rl/corpus/manifest.json")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    config, manifest = load_config(args.config), read_json(args.corpus)
    target = config["resource_targets"]
    rows = []
    for repetition in range(target["benchmark_repetitions"]):
        directory = Path(args.out) / f"repeat-{repetition}"
        run(
            config,
            manifest,
            directory,
            device=args.device,
            budget=2**63,
            max_updates=target["benchmark_warmup_updates"]
            + target["benchmark_measured_updates"],
            evaluate_every=False,
            replays=False,
            stage_cycle=[0, 1, 2],
        )
        metrics = [
            __import__("json").loads(line)
            for line in (directory / "metrics.jsonl").read_text().splitlines()
        ]
        measured = metrics[target["benchmark_warmup_updates"] :]
        decisions = sum(x["rollout_decisions"] for x in measured)
        seconds = sum(x["update_seconds"] for x in measured)
        rows.append(
            {
                "repetition": repetition,
                "decisions": decisions,
                "seconds": seconds,
                "decisions_per_second": decisions / seconds,
                "stages": {
                    str(stage): {
                        "decisions": sum(
                            x["rollout_decisions"]
                            for x in measured
                            if x["stage"] == stage
                        ),
                        "seconds": sum(
                            x["update_seconds"] for x in measured if x["stage"] == stage
                        ),
                    }
                    for stage in (0, 1, 2)
                },
            }
        )
        write_json(
            Path(args.out) / "summary.json",
            {
                "repetitions": rows,
                "decisions_per_second": sum(r["decisions"] for r in rows)
                / sum(r["seconds"] for r in rows),
            },
        )


if __name__ == "__main__":
    main()
