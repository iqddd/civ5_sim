"""Discardable whole-turn smoke; short horizon is never a scientific collection."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from torch.profiler import ProfilerActivity, profile

from civ5rl.engine.evaluation.executor import EvaluationExecutor, EvaluationRuntime
from civ5rl.engine.runtime.telemetry import ResourceMonitor
from civ5rl.experiments.benchmarks.profiling import top_cuda_operators
from civ5rl.experiments.comparison.executor import ProposalExecutor
from civ5rl.experiments.comparison.match import match_task
from civ5rl.experiments.comparison.turn_collection import branch_spec, position_candidates
from civ5rl.experiments.comparison.turn_confirmation import summarize_confirmation
from civ5rl.experiments.comparison.turn_executor import TurnExecutor
from civ5rl.experiments.comparison.turn_protocol import SCREENING_REPETITIONS, SCREENING_VARIANTS
from civ5rl.provenance.common import ROOT, read_json, write_json
from civ5sim.engine import digest


def smoke(cfg, out, *, workers=None):
    out = Path(out)
    if any(p.name not in ("run.lock", "derived") for p in out.iterdir()):
        raise ValueError("Use empty smoke output")
    config = read_json(ROOT / cfg["collection_input"]["run"] / "match-config.json")
    spec = _smoke_scenario()
    case = {
        "scenario": spec,
        "armies": [["warrior"], ["warrior"]],
        "learner_side": 0,
        "max_rounds": 2,
        "max_calls_per_turn": 200,
        "agent_seeds": [17, 18],
        "parent_world": 0,
    }
    runtime = EvaluationRuntime(
        "cuda", cfg.get("workers", 2) if workers is None else workers, 8, 0.001
    ).resolve()
    parent = str(ROOT / cfg["parent"])
    source = out / "source"
    started = perf_counter()
    with ProposalExecutor([parent, parent], runtime) as executor:
        list(executor.run(match_task, [(config, case, str(source), parent, None)]))
    positions = position_candidates(
        0, case, read_json(source / "history.json"), read_json(source / "replay.json")
    )
    if not positions:
        raise AssertionError("Smoke needs a nonempty reconstructed prefix")
    position = positions[0]
    tasks = []
    variants = (0, 2) if "confirmation" in cfg else SCREENING_VARIANTS
    repetitions = cfg["confirmation"]["repetitions"] if "confirmation" in cfg else SCREENING_REPETITIONS
    for variant in variants:
        for repetition in repetitions:
            branch = branch_spec(position, variant, repetition, source)
            tasks.append((config, case, str(out / "branches" / f"{variant}-{repetition}"), parent, branch))
    parity_tasks = tasks[:: max(1, len(tasks) // 4)]
    with ResourceMonitor(out / "telemetry") as monitor, EvaluationExecutor(parent, runtime) as executor:
        monitor.mark("turn_smoke")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            folders = list(executor.run(match_task, tasks))
        trace.export_chrome_trace(str(out / "trace.json"))
        repeat = [
            (original_history, repeated_history, str(out / "repeated" / Path(c).name), d, e)
            for original_history, repeated_history, c, d, e in reversed(parity_tasks)
        ]
        list(executor.run(match_task, repeat))
    with TurnExecutor(parent, runtime) as executor:
        prefixed = [
            (original_history, repeated_history, str(out / "prefixed" / Path(c).name), d, e)
            for original_history, repeated_history, c, d, e in parity_tasks
        ]
        list(executor.run(match_task, prefixed))
        assert executor.prefixes.prefills == 1 and executor.prefixes.hits >= len(prefixed) - 1
    for task in parity_tasks:
        original_history = read_json(Path(task[2]) / "turn-history.json")
        repeated_history = read_json(out / "repeated" / Path(task[2]).name / "turn-history.json")
        assert original_history == repeated_history
        assert original_history == read_json(out / "prefixed" / Path(task[2]).name / "turn-history.json")
    rows = [read_json(Path(folder) / "completed.json") for folder in folders]
    assert len(rows) == len(tasks) and all(r["prefix_verified"] for r in rows)
    for folder in folders:
        folder = Path(folder)
        row = read_json(folder / "completed.json")
        replay = read_json(folder / "replay.json")
        history = read_json(folder / "turn-history.json")

        step = replay["steps"][row["boundary_replay_index"]]
        assert step["state_sha256"] == row["boundary_state_sha256"]
        assert step["action"] == history[-1]["actions"][history[-1]["chosen"]]
        assert row["turn_sequence_sha256"] == digest([r["actions"][r["chosen"]] for r in history])
    if "confirmation" in cfg:
        manifest = {
            "positions": [{**position, "selected_variant": 2}],
            "tasks": tasks,
            "repetitions": list(repetitions),
        }
        confirmation = summarize_confirmation(cfg, out, manifest, rows, perf_counter() - started)
        assert confirmation["continuations"] == len(tasks) and confirmation["primary"]["worlds"] == 1
    result = {
        "status": "passed",
        "discarded": True,
        "smoke_rounds": case["max_rounds"],
        "continuations": len(tasks),
        "workers": runtime.workers,
        "task_order_reproducible": True,
        "replay_verified": True,
        "seconds": perf_counter() - started,
        "top_kernels": top_cuda_operators(trace, 12),
    }
    write_json(out / "precheck.json", result)
    print(result, flush=True)
    return result


def _smoke_scenario():
    """Two warriors on an open field, using the simulator's public scenario format."""
    width, height = 12, 8
    terrain = {
        "schema": 1,
        "width": width,
        "height": height,
        "row_parity": 0,
        "wrap_x": False,
        "metadata": {},
        "tiles": [
            {"terrain": "grass", "elevation": "flat", "feature": None, "rivers": [-1, -1, -1]}
            for _ in range(width * height)
        ],
    }
    return {
        "schema": 1,
        "kind": "civ5sim.scenario",
        "map": terrain,
        "units": [
            {"type": "warrior", "side": 0, "tile": [2, 2]},
            {"type": "warrior", "side": 1, "tile": [5, 2]},
        ],
        "fog": False,
        "random_damage": True,
        "seed": 42,
        "turn_reset_mode": "end_turn",
    }
