"""Bounded, fixed-weight PPO replay benchmark; never runs Adam or a match."""

import argparse
import sys
from pathlib import Path
from statistics import mean, median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch

    from civ5rl.engine.ppo_fast import RolloutCache, measure_replay
    from civ5rl.engine.resources import ResourceMonitor
    from civ5rl.experiments.imitation_helpers import load_policy
    from civ5rl.experiments.ppo_artifacts import load_rollout
    from civ5rl.provenance.common import configure, file_sha256, read_json, write_json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=65)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    config = read_json(args.config)
    configure(17, "cuda")
    episodes = load_rollout(args.rollout)
    policy = load_policy(config, args.checkpoint, "cuda")
    policy.eval()
    cache = RolloutCache(episodes, "C", "cuda")
    cache.enable_compiled_gather()
    cache.enable_async_transfers()
    records = []
    with ResourceMonitor(out / "telemetry", interval=0.25) as monitor:
        for mode in ("uncached", "cached"):
            if mode == "cached":
                cache.enable_cached_replay()
            monitor.mark("warmup_" + mode)
            for _ in range(3):
                result = measure_replay(policy, cache, 8)
            monitor.mark(mode)
            times = []
            torch.cuda.synchronize()
            start = perf_counter()
            while perf_counter() - start < args.seconds:
                tick = perf_counter()
                result = measure_replay(policy, cache, 8)
                torch.cuda.synchronize()
                times.append(perf_counter() - tick)
            duration = perf_counter() - start
            record = {
                "mode": mode, "seconds": duration, "passes": len(times),
                "mean_seconds": mean(times), "median_seconds": median(times),
                "min_seconds": min(times), "max_seconds": max(times),
                "metrics": result, "cached_inputs_enabled": cache.replay_inputs,
            }
            records.append(record)
            print(record, flush=True)
    write_json(out / "benchmark.json", {
        "config_sha256": file_sha256(args.config),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "rollout_sha256": file_sha256(args.rollout),
        "source_sha256": file_sha256(ROOT / "civ5rl/engine/ppo_fast.py"),
        "warmup_excluded": True, "Adam_steps": 0, "results": records,
    })


if __name__ == "__main__":
    main()
