"""Measurements for the initial BC protocol; invoked by run_cp_imitation CLI.

The benchmark writes demonstrations reused by collection, so this mixed helper
remains part of execution provenance despite living under tools/.
"""

from time import perf_counter

from civ5rl.experiments.imitation_helpers import pool
from civ5rl.protocols.imitation import load_episode, match_task, task_for
from civ5rl.provenance.common import read_json, write_json

def benchmark(config, manifest, out, workers):
    from civ5rl.engine.resources import ResourceMonitor

    destination = out / "benchmark/collection.json"
    if destination.exists():
        return read_json(destination)
    jobs = [
        task_for(config, manifest, "training", i, out / f"data/training/match-{i:06}")
        for i in range(8)
    ]
    # Freeze every case before launching any CP decisions.
    write_json(out / "benchmark/cases.json", [job[2] for job in jobs])
    with ResourceMonitor(out / "benchmark/telemetry") as monitor:
        monitor.mark("CP_generation", workers=workers)
        start = perf_counter()
        with pool(workers) as executor:
            rows = list(executor.map(match_task, jobs))
        seconds = perf_counter() - start
    result = {
        "workers": workers,
        "pool_wall_seconds": seconds,
        "matches": rows,
        "labels": sum(r["labels"] for r in rows),
        "forced": sum(r["forced"] for r in rows),
        "labels_per_second": sum(r["labels"] for r in rows) / seconds,
        "closed_test_evaluated": False,
    }
    write_json(destination, result)
    return result

def latency(config, out):
    import numpy as np
    import torch

    from civ5rl.core.agent import RLAgent
    from civ5rl.provenance.common import configure
    from civ5sim import Action

    episodes = [
        load_episode(out / f"data/training/match-{i:06}/demonstration.json.gz")
        for i in range(8)
    ]
    result = {}
    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            result[device] = {"unavailable": True}
            continue
        configure(17, device)
        agent = RLAgent.from_checkpoint(out / "checkpoints/initial.pt", device=device)
        synchronize = torch.cuda.synchronize if device == "cuda" else lambda: None
        warm_start = perf_counter()
        agent.reset(episodes[0]["context"])
        for row in episodes[0]["decisions"][:8]:
            agent.act(row["observation"], tuple(Action(**a) for a in row["actions"]))
        synchronize()
        warm = perf_counter() - warm_start
        samples = {"forced": [], "free": [], "all": []}
        for episode in episodes:
            agent.reset(episode["context"])
            for row in episode["decisions"]:
                legal = tuple(Action(**a) for a in row["actions"])
                synchronize()
                start = perf_counter()
                agent.act(row["observation"], legal)
                synchronize()
                seconds = perf_counter() - start
                # Replay the teacher's previous command into the encoder; timing
                # includes the real Agent API, not CP or counterfactual simulation.
                agent.encoder.remember_action(legal[row["chosen"]], row["observation"])
                agent.drain_diagnostics()
                samples["free" if row["supervised"] else "forced"].append(seconds)
                samples["all"].append(seconds)
        result[device] = {
            "warmup_seconds": warm,
            "compiled": False,
            "history": "fixed_public_teacher_inputs_with_teacher_previous_action",
            "precision": "mixed_bfloat16" if device == "cuda" else "float32",
            "samples": {
                key: {
                    "count": len(values),
                    "mean_ms": float(np.mean(values) * 1000),
                    "median_ms": float(np.median(values) * 1000),
                    "p95_ms": float(np.quantile(values, 0.95) * 1000),
                }
                if values
                else None
                for key, values in samples.items()
            },
        }
        del agent
    write_json(out / "benchmark/latency.json", result)
    return result
