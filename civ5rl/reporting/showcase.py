"""Play an existing public showcase manifest with a new frozen checkpoint."""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

from civ5rl.engine.evaluation.matches import run_saved_match
from civ5rl.engine.evaluation.proxy import evaluation_agent
from civ5rl.engine.evaluation.replay import verify_replay
from civ5rl.protocols.imitation import search_config
from civ5rl.protocols.turn_reset import mode
from civ5rl.provenance.common import configure, file_sha256, read_json, write_json
from civ5rl.reporting.evaluation_audit import classify_outcome
from civ5rl.reporting.markdown import format_wdl
from civ5sim.combat_ai import LIBRARY_PATH, Planner
from civ5sim.cp_agent import CPAgent
from civ5sim.engine import digest


def write_viewer_passport(out, checkpoint, checkpoint_sha256):
    planner = Planner()
    try:
        version = planner.version
    finally:
        planner.close()
    write_json(
        Path(out) / "passport.json",
        {
            "schema": 1,
            "kind": "civ5rl.showcase_viewer",
            "port": version,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha256,
            "cp_library_sha256": file_sha256(LIBRARY_PATH),
            "purpose": "viewer provenance; not a training execution passport",
        },
    )


def play(task):
    case, teacher, folder, checkpoint = task
    folder = Path(folder)
    learner = evaluation_agent(checkpoint, "argmax")
    cp = CPAgent(search_config({"teacher_search": teacher}))
    agents = [None, None]
    side = case["learner_side"]
    agents[side], agents[1 - side] = learner, cp
    started = time.perf_counter()
    try:
        match = run_saved_match(
            case,
            agents,
            folder,
            max_calls_per_turn=case["max_calls_per_turn"],
            seed=case["streams"]["battle"],
        )
    finally:
        cp.close()
    if match.result["status"] != "completed":
        raise RuntimeError(f"Match {folder.name} failed: {match.result.get('error')}")
    verification = verify_replay(folder, project_relative=False)
    if not verification["verified"] or verification["commands"] != len(match.replay["steps"]):
        raise ValueError(f"Showcase replay failed verification: {folder}")
    wdl = classify_outcome(match.result["winner"], side)
    row = {
        "case_id": case["case_id"],
        "directory": folder.name,
        "side": side,
        "army_size": len(case["armies"][0]),
        "roster": case["armies"][0],
        "parent_world": case["parent_world"],
        "map_hash": digest(case["scenario"]["map"]),
        "outcome": wdl,
        "winner": match.result["winner"],
        "reason": match.result["reason"],
        "rounds": match.result["completed_rounds"],
        "commands": match.result["commands"],
        "seconds": time.perf_counter() - started,
        "replay_verified": True,
    }
    write_json(folder / "metrics.json", row)
    write_json(
        folder / "checked.json",
        {
            "replay_verified": True,
            "commands": match.result["commands"],
            "replay_sha256": file_sha256(folder / "replay.json"),
        },
    )
    return row


def play_cases(manifest_path, checkpoint, out, *, turn_reset_mode=None):
    # Spawned match workers import this module without initializing Torch.
    from civ5rl.engine.evaluation.executor import (
        EvaluationExecutor,
        EvaluationRuntime,
        evaluation_precision,
    )

    manifest_path, checkpoint, out = (
        Path(manifest_path).resolve(),
        Path(checkpoint).resolve(),
        Path(out).resolve(),
    )
    source = read_json(manifest_path)
    if source.get("kind") != "civ5rl.showcase" or source.get("closed_test_used") is not False:
        raise ValueError("Expected an explicitly public showcase manifest")
    cases = source["cases"]
    if any(case.get("split") not in ("train", "development", "dev") for case in cases):
        raise ValueError("Only open cases are allowed")
    if turn_reset_mode is not None:
        mode({"environment": {"turn_reset_mode": turn_reset_mode}})
        for case in cases:
            scenario = case["scenario"]
            scenario["turn_reset_mode"] = turn_reset_mode
            case["scenario_hash"] = digest(scenario)
            if "case_hash" in case:
                case["case_hash"] = digest({k: v for k, v in case.items() if k != "case_hash"})
        source["turn_reset_mode"] = turn_reset_mode
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty output directory")
    checkpoint_sha256 = file_sha256(checkpoint)
    configure(source["seed"], "cuda")
    out.mkdir(parents=True, exist_ok=True)
    write_viewer_passport(out, checkpoint, checkpoint_sha256)
    manifest = {
        **source,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": file_sha256(manifest_path),
        "selected_update": None,
    }
    write_json(out / "manifest.json", manifest)
    tasks = []
    for case in cases:
        name = f"battle-{case['case_id'] + 1:02d}-{len(case['armies'][0])}v{len(case['armies'][0])}-RL-{'A' if case['learner_side'] == 0 else 'B'}-map{case['parent_world']}"
        folder = out / name
        write_json(folder / "case.json", case)
        tasks.append((case, source["teacher_search"], str(folder), str(checkpoint)))
    runtime = EvaluationRuntime("cuda", 8, 32, 0.001).resolve()
    rows = []
    started = time.perf_counter()
    with evaluation_precision():
        executor = EvaluationExecutor(str(checkpoint), runtime)
        write_json(out / "runtime.json", runtime.identity())
        for row in executor.run(play, tasks):
            rows.append(row)
            print(
                f"{len(rows)}/{len(cases)} {row['directory']}: {row['outcome']}; verified",
                flush=True,
            )
        inference = executor.statistics()
    if len(rows) != len(cases) or not all(row["replay_verified"] for row in rows):
        raise ValueError("Showcase results are incomplete")
    if file_sha256(checkpoint) != checkpoint_sha256:
        raise ValueError("Showcase checkpoint changed during execution")
    wdl = Counter(r["outcome"] for r in rows)
    result = {
        "schema": 1,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "mode": "argmax",
        "seconds": time.perf_counter() - started,
        "summary": {"matches": len(rows), **{k: wdl[k] for k in ("W", "D", "L")}},
        "matches": sorted(rows, key=lambda r: r["case_id"]),
        "runtime": runtime.identity(),
        "inference": inference,
        "all_replays_verified": True,
        "closed_test_used": False,
        "viewer_summary": (
            f"{checkpoint.parent.parent.name} / {checkpoint.stem}: argmax vs CP, "
            f"W/D/L {format_wdl(wdl)}; "
            f"{', '.join(sorted({case['scenario'].get('turn_reset_mode', 'legacy') for case in cases}))}"
        ),
    }
    write_json(out / "summary.json", result)
    return result
