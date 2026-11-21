"""Execute saved validation cases through bounded match workers."""

from __future__ import annotations

from time import perf_counter
from typing import Any

from civ5rl.engine.evaluation.matches import run_policy_match
from civ5rl.engine.runtime.shared import pool
from civ5rl.experiments.evaluation.paired import run_chunked_pool
from civ5rl.protocols.imitation import MatchTask, match_task
from civ5rl.protocols.turn_reset import validate_case
from civ5rl.provenance.common import read_json, write_json
from civ5rl.reporting.evaluation_audit import MatchRow, match_summary, summarize_wdl
from civ5sim.engine import digest


def target_scale_evaluation_task(task):
    config, case, path, directory = task
    if (
        case["split"] != "validation"
        or digest({key: value for key, value in case.items() if key != "case_hash"}) != case["case_hash"]
    ):
        raise ValueError("Invalid validation case")
    return run_policy_match(config, case, path, "argmax", "cp", directory, include_combat_metrics=True)


def run_validation_cases(
    config,
    manifest,
    checkpoint,
    mode,
    directories,
    workers,
    *,
    batch_size=None,
    progress=None,
):
    """Run validation cases in their recorded world, side, opponent order."""
    # Workers load the simulator without Torch until they require a local policy.
    jobs = []
    for original in manifest["validation_cases"]:
        for side in (0, 1):
            case = {**original, "learner_side": side}
            case["case_hash"] = digest({k: v for k, v in case.items() if k != "case_hash"})
            for enemy, directory in directories.items():
                jobs.append(
                    MatchTask(
                        config=config,
                        manifest=manifest,
                        case=case,
                        output_dir=str(directory / f"case-{case['case_id']:03}-side-{side}"),
                        contender={"checkpoint": str(checkpoint), "mode": mode},
                        opponent=enemy,
                        device="cpu",
                    )
                )
    with pool(workers) as executor:
        if batch_size is None:
            list(executor.map(match_task, jobs))
        else:
            run_chunked_pool(executor, match_task, jobs, batch_size, on_chunk=progress)
    return len(jobs)


def _compute_evaluation_slices(
    rows: list[MatchRow],
) -> dict[str, dict[str, dict[str, Any]]]:
    slices: dict[str, dict[str, dict[str, Any]]] = {}
    for field in ("map_size", "army_bin", "army_size", "placement", "side"):
        slices[field] = {}
        for value in sorted({row[field] for row in rows}, key=str):
            selected = [row for row in rows if row[field] == value]
            summary = summarize_wdl(selected, extra_metrics=True)
            slices[field][str(value)] = {
                "matches": summary["matches"],
                "W": summary["W"],
                "D": summary["D"],
                "L": summary["L"],
                "seconds": summary["seconds"],
                "mean_rounds": summary["mean_rounds"],
                "draw_reasons": summary["draw_reasons"],
            }
    return slices


def evaluate_target_scale(config, plan, out, path, purpose):
    from civ5rl.engine.evaluation.executor import evaluation_precision

    with evaluation_precision():
        return _evaluate_target_scale(config, plan, out, path, purpose)


def _evaluate_target_scale(config, plan, out, path, purpose):
    # Spawned simulator workers can import this module without loading Torch.
    from civ5rl.engine.evaluation import executor as evaluation_executor
    from civ5rl.engine.evaluation.executor import evaluation_key

    for case in plan[purpose]:
        validate_case(case, config)

    runtime = evaluation_executor.runtime_for(out, config, cp_only=path == "cp")
    runtime_identity = runtime.identity()
    fingerprint, key = evaluation_key(config, plan, out, path, purpose, runtime_identity)
    cache = read_json(out / "evaluations.json") if (out / "evaluations.json").exists() else {}
    if key in cache:
        return cache[key]
    cases = plan[purpose]
    started = perf_counter()
    relative = f"evaluation/{purpose}/{fingerprint[:16]}/{key[:16]}/argmax/cp"
    directory = out / relative
    workers = runtime.workers
    checkpoint_arg = "cp" if path == "cp" else str(out / path)
    tasks = []
    for case in cases:
        case_dir = directory / f"case-{case['case_id']:03}-side-{case['learner_side']}"
        metrics_path = case_dir / "metrics.json"
        if metrics_path.is_file() and "units_lost" in read_json(metrics_path):
            continue
        # Older workers could publish ordinary metrics before adding combat totals.
        tasks.append((config, case, checkpoint_arg, str(case_dir)))
    inference_statistics = None
    if runtime.device == "cuda" and tasks:
        executor = evaluation_executor.EvaluationExecutor(checkpoint_arg, runtime)
        for count, _ in enumerate(executor.run(target_scale_evaluation_task, tasks), 1):
            if count % workers == 0 or count == len(tasks):
                print(
                    f"Evaluation {purpose}: {count}/{len(tasks)} new matches",
                    flush=True,
                )
        inference_statistics = executor.statistics()
    elif tasks:
        with evaluation_executor.evaluation_worker_pool(workers) as executor:
            run_chunked_pool(
                executor,
                target_scale_evaluation_task,
                tasks,
                workers,
                lambda count: print(
                    f"Evaluation {purpose}: {count}/{len(tasks)} new matches",
                    flush=True,
                ),
            )
    summary = match_summary(directory, len(cases))
    slices = _compute_evaluation_slices([read_json(path) for path in directory.glob("*/metrics.json")])
    result = {
        "checkpoint": path,
        "policy_hash": fingerprint,
        "purpose": purpose,
        "seconds": perf_counter() - started,
        "runtime": runtime_identity,
        "inference": inference_statistics,
        "runs": {"argmax": {"cp": {"directory": relative, "summary": summary}}},
        "slices": slices,
    }
    cache[key] = result
    write_json(out / "evaluations.json", cache)
    print(f"Evaluation {purpose}: {summary}; wall={result['seconds']:.1f}s", flush=True)
    return result
