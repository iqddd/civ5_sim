"""Match evaluation and checkpoint selection for target-scale imitation."""

from __future__ import annotations

import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from civ5rl.engine.evaluation.executor import evaluation_key, runtime_for
from civ5rl.experiments.evaluation.matches import evaluate_target_scale
from civ5rl.experiments.imitation.selection import (
    c2_gate_criteria,
    check_c2_gate,
    select_best_c2_candidate,
    select_best_candidate,
)
from civ5rl.protocols.imitation_target_scale import C2_KIND, MORE_DATA_KIND
from civ5rl.provenance.common import ROOT, read_json, write_json
from civ5rl.reporting.evaluation_audit import (
    CaseKey,
    collect_paired_outcomes,
    load_eval_rows,
    paired_world_bootstrap,
    verify_match_artifacts,
)

EvaluationData = dict[str, Any]


def complete_eval_dir(directory: str | Path, expected_matches: int) -> bool:
    directory = Path(directory)
    if not directory.is_dir():
        return False
    paths = list(directory.glob("*/metrics.json"))
    if len(paths) != expected_matches:
        return False
    required = ("case.json", "result.json", "final.json", "replay.json")
    return all(verify_match_artifacts(path.parent, required) for path in paths)


def _try_reuse_saved_evaluation(
    config: EvaluationData,
    plan: EvaluationData,
    out: Path,
    source_run_dir: str | Path,
    checkpoint_rel: str,
    fingerprint: str,
    purpose: str,
) -> EvaluationData | None:
    source_run_dir = Path(source_run_dir)
    evals_path = source_run_dir / "evaluations.json"
    if not evals_path.is_file():
        return None
    source_cfg = (
        read_json(source_run_dir / "run.json").get("config", {})
        if (source_run_dir / "run.json").is_file()
        else {}
    )
    runtime = runtime_for(out, config).identity()
    actual_fingerprint, key = evaluation_key(config, plan, out, checkpoint_rel, purpose, runtime)
    if actual_fingerprint != fingerprint:
        raise ValueError("Reference checkpoint policy hash changed")
    saved_evals = read_json(evals_path)
    cases = plan[purpose]
    point = saved_evals.get(key)
    if point is None:
        # A previous run may have published the same case set under an older key.
        if source_cfg.get("teacher_search") != config["teacher_search"]:
            return None
        point = next(
            (
                item
                for item in saved_evals.values()
                if item.get("policy_hash") == fingerprint
                and item.get("purpose") == purpose
                and item.get("runtime") == runtime
            ),
            None,
        )
    if point is None or point.get("runtime") != runtime:
        return None
    rel_dir = point["runs"]["argmax"]["cp"]["directory"]
    src_dir = (source_run_dir / rel_dir).resolve()
    if not src_dir.is_relative_to(source_run_dir.resolve()) or not complete_eval_dir(src_dir, len(cases)):
        return None
    rows = load_eval_rows(src_dir, len(cases), plan=cases)
    expected = {(case["parent_world"], case["case_id"], case["learner_side"]): case for case in cases}
    if any(case != expected[key] for key, (_, case, _) in rows.items()):
        return None
    dst_dir = out / rel_dir
    if src_dir.resolve() != dst_dir.resolve():
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                target = dst_dir / path.relative_to(src_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    return {
        **deepcopy(point),
        "checkpoint": checkpoint_rel,
        "reused_saved_evaluation": True,
        "reused_from": source_run_dir.relative_to(ROOT).as_posix()
        if source_run_dir.resolve().is_relative_to(ROOT.resolve())
        else str(source_run_dir),
    }


def _evaluate_reference(
    config: EvaluationData,
    plan: EvaluationData,
    out: Path,
    checkpoint_path: str,
    fingerprint: str,
    purpose: str,
    *source_runs: Path,
) -> EvaluationData:
    for source_run in source_runs:
        point = _try_reuse_saved_evaluation(
            config, plan, out, source_run, checkpoint_path, fingerprint, purpose
        )
        if point is not None:
            return point
    result: EvaluationData = evaluate_target_scale(config, plan, out, checkpoint_path, purpose)
    result["reused_saved_evaluation"] = False
    return result


def baseline(config: EvaluationData, plan: EvaluationData, out: Path) -> None:
    if config["kind"] == C2_KIND or (out / "selection.json").exists():
        return
    if config["kind"] == MORE_DATA_KIND:
        ref_run = ROOT / config["reference_run"]
        old_run = ROOT / config["old_corpus_run"]
        bc_hash = config["parent_policy_hash"]
        ppo_hash = config["evaluation_reference_policy_hash"]
        bc_point = _evaluate_reference(
            config,
            plan,
            out,
            "checkpoints/initial.pt",
            bc_hash,
            "development",
            ref_run,
            old_run,
        )
        bc_point.update(
            checkpoint="checkpoints/initial.pt",
            epoch=0,
            Adam_steps=0,
            baseline_role="training_parent",
        )
        ppo_point = _evaluate_reference(
            config,
            plan,
            out,
            "checkpoints/reference-update-0037.pt",
            ppo_hash,
            "development",
            ref_run,
        )
        ppo_point.update(
            checkpoint="checkpoints/reference-update-0037.pt",
            epoch=0,
            Adam_steps=0,
            baseline_role="evaluation_reference",
        )
        evaluations = read_json(out / "evaluations.json") if (out / "evaluations.json").exists() else {}
        evaluations["development:" + bc_point["policy_hash"]] = bc_point
        evaluations["development:" + ppo_point["policy_hash"]] = ppo_point
        write_json(out / "evaluations.json", evaluations)
        selection = select_best_candidate(bc_point, ppo_point, [])
        write_json(out / "selection.json", selection)
        print(
            f"Development baselines: BC epoch 10={bc_point['runs']['argmax']['cp']['summary']}, "
            f"PPO update 37={ppo_point['runs']['argmax']['cp']['summary']}",
            flush=True,
        )
        return
    point = evaluate_target_scale(config, plan, out, "checkpoints/initial.pt", "development")
    point.update(epoch=0, Adam_steps=0)
    evaluations = read_json(out / "evaluations.json")
    evaluations["development:" + point["policy_hash"]] = point
    write_json(out / "evaluations.json", evaluations)
    wins = point["runs"]["argmax"]["cp"]["summary"]["W"]
    write_json(out / "selection.json", {"baseline": point, "best": point, "best_wins": wins})
    print(
        "Development baseline: " + str(point["runs"]["argmax"]["cp"]["summary"]),
        flush=True,
    )


def _outcomes_from_dir(
    directory: str | Path,
) -> tuple[dict[CaseKey, int], dict[CaseKey, EvaluationData]]:
    outcomes, cases, _ = collect_paired_outcomes(directory)
    return outcomes, cases


def _evaluate_c2(
    config: EvaluationData,
    plan: EvaluationData,
    out: str | Path,
    training: EvaluationData,
) -> None:
    out = Path(out)
    baseline_offline = (
        read_json(out / "baseline-offline-metrics.json")
        if (out / "baseline-offline-metrics.json").exists()
        else {}
    )
    epoch_by_num = {int(e["epoch"]): e for e in training.get("epochs", [])}

    def _run_dev512(path, role, *, epoch=0, steps=0, fixed_tr=None, holdout_tr=None):
        point = evaluate_target_scale(config, plan, out, path, "development_512")
        point = {
            **point,
            "checkpoint": path,
            "epoch": epoch,
            "Adam_steps": steps,
            "candidate_role": role,
        }
        if fixed_tr is not None:
            point["fixed_training"] = fixed_tr
        if holdout_tr is not None:
            point["holdout"] = holdout_tr
        evals = read_json(out / "evaluations.json") if (out / "evaluations.json").exists() else {}
        evals["development_512:" + point["policy_hash"]] = point
        write_json(out / "evaluations.json", evals)
        return point

    cp_point = _run_dev512("cp", "cp_mirror")
    bc10_off = baseline_offline.get("bc10", {})
    bc10_point = _run_dev512(
        "checkpoints/reference-bc-epoch-10.pt",
        "bc10_reference",
        fixed_tr=bc10_off.get("fixed_training"),
        holdout_tr=bc10_off.get("holdout"),
    )
    ppo37_point = _run_dev512(
        "checkpoints/reference-update-0037.pt",
        "ppo37_reference",
    )
    bc1m_off = baseline_offline.get("bc_more_data_epoch_10", {})
    bc1m_point = _run_dev512(
        "checkpoints/reference-bc-more-data-epoch-10.pt",
        "bc_more_data_baseline",
        fixed_tr=bc1m_off.get("fixed_training"),
        holdout_tr=bc1m_off.get("holdout"),
    )

    c2_candidates = []
    seen_paths = set()
    milestone_paths = []
    for e_num in config["evaluate_epochs"]:
        rel = f"checkpoints/epoch-{int(e_num):02d}.pt"
        if (out / rel).is_file():
            milestone_paths.append((int(e_num), rel))
    latest_rel = training.get("latest")
    if latest_rel and (out / latest_rel).is_file():
        last_epoch_num = int(training["epochs"][-1]["epoch"]) if training.get("epochs") else 0
        if all(rel != latest_rel for _, rel in milestone_paths):
            milestone_paths.append((last_epoch_num, latest_rel))

    cum_steps = 0
    steps_by_epoch = {}
    for e_row in training.get("epochs", []):
        cum_steps += int(e_row["steps"])
        steps_by_epoch[int(e_row["epoch"])] = cum_steps

    for e_num, rel in milestone_paths:
        if rel in seen_paths:
            continue
        seen_paths.add(rel)
        saved_ckpt = torch.load(out / rel, map_location="cpu", weights_only=False)
        steps = int(saved_ckpt.get("state", {}).get("Adam_steps", steps_by_epoch.get(e_num, 0)))
        e_info = epoch_by_num.get(e_num, {})
        point = _run_dev512(
            rel,
            "c2_candidate",
            epoch=e_num,
            steps=steps,
            fixed_tr=e_info.get("fixed_training"),
            holdout_tr=e_info.get("holdout"),
        )
        c2_candidates.append(point)

    best_c2 = select_best_c2_candidate(c2_candidates)
    best_c2_wins = int(best_c2["runs"]["argmax"]["cp"]["summary"]["W"])

    best_c2_outcomes, cases_512 = _outcomes_from_dir(out / best_c2["runs"]["argmax"]["cp"]["directory"])
    bc1m_outcomes, bc1m_cases = _outcomes_from_dir(out / bc1m_point["runs"]["argmax"]["cp"]["directory"])
    ppo37_outcomes, ppo37_cases = _outcomes_from_dir(out / ppo37_point["runs"]["argmax"]["cp"]["directory"])
    bc10_outcomes, bc10_cases = _outcomes_from_dir(out / bc10_point["runs"]["argmax"]["cp"]["directory"])
    if cases_512 != bc1m_cases or cases_512 != ppo37_cases or cases_512 != bc10_cases:
        raise ValueError("Unpaired Development-512 evaluations")

    vs_bc1m = paired_world_bootstrap(
        bc1m_outcomes,
        best_c2_outcomes,
        samples=10000,
        seed=20261103,
        cases=cases_512,
        cases_per_world=32,
    )
    vs_ppo37 = paired_world_bootstrap(
        ppo37_outcomes,
        best_c2_outcomes,
        samples=10000,
        seed=20261103,
        cases=cases_512,
        cases_per_world=32,
    )
    vs_bc10 = paired_world_bootstrap(
        bc10_outcomes,
        best_c2_outcomes,
        samples=10000,
        seed=20261103,
        cases=cases_512,
        cases_per_world=32,
    )
    bc1m_wins = int(bc1m_point["runs"]["argmax"]["cp"]["summary"]["W"])
    gate_passed = check_c2_gate(vs_bc1m)

    selection = {
        "purpose": "development_512",
        "best": best_c2,
        "best_wins": best_c2_wins,
        "c2_candidates": c2_candidates,
        "bc_more_data_baseline": bc1m_point,
        "bc_more_data_wins": bc1m_wins,
        "ppo37_reference": ppo37_point,
        "ppo37_wins": int(ppo37_point["runs"]["argmax"]["cp"]["summary"]["W"]),
        "bc10_reference": bc10_point,
        "bc10_wins": int(bc10_point["runs"]["argmax"]["cp"]["summary"]["W"]),
        "cp_mirror": cp_point,
        "cp_mirror_wins": int(cp_point["runs"]["argmax"]["cp"]["summary"]["W"]),
        "comparison_vs_bc_more_data": vs_bc1m,
        "comparison_vs_ppo37": vs_ppo37,
        "comparison_vs_bc10": vs_bc10,
        "gate_criteria": c2_gate_criteria(vs_bc1m),
    }
    write_json(out / "selection.json", selection)

    if not gate_passed:
        write_json(
            out / "confirmation.json",
            {
                "status": "not_run_development_512_gate_not_met",
                "selected": best_c2,
                "bc_more_data_baseline": bc1m_point,
                "ppo37_reference": ppo37_point,
                "comparison_vs_bc_more_data": vs_bc1m,
                "comparison_vs_ppo37": vs_ppo37,
                "gate_criteria": selection["gate_criteria"],
            },
        )
        return

    _confirm_c2(config, plan, out, best_c2)


def _confirm_c2(config: EvaluationData, plan: EvaluationData, out: Path, best_c2: EvaluationData) -> None:
    ref_run = ROOT / config["reference_run"]
    old_run = ROOT / config["old_corpus_run"]
    bc_hash = config["parent_policy_hash"]
    ppo_hash = config["evaluation_reference_policy_hash"]

    bc1m_conf = _evaluate_reference(
        config,
        plan,
        out,
        "checkpoints/reference-bc-more-data-epoch-10.pt",
        bc_hash,
        "confirmation",
        ref_run,
        old_run,
    )
    ppo37_conf = _evaluate_reference(
        config,
        plan,
        out,
        "checkpoints/reference-update-0037.pt",
        ppo_hash,
        "confirmation",
        ref_run,
    )

    c2_conf = evaluate_target_scale(config, plan, out, best_c2["checkpoint"], "confirmation")
    bc1m_c_out, conf_cases = _outcomes_from_dir(out / bc1m_conf["runs"]["argmax"]["cp"]["directory"])
    ppo37_c_out, ppo_conf_cases = _outcomes_from_dir(out / ppo37_conf["runs"]["argmax"]["cp"]["directory"])
    c2_c_out, c2_conf_cases = _outcomes_from_dir(out / c2_conf["runs"]["argmax"]["cp"]["directory"])
    if conf_cases != c2_conf_cases or ppo_conf_cases != c2_conf_cases:
        raise ValueError("Unpaired Additional-256 confirmation")
    conf_vs_bc1m = paired_world_bootstrap(
        bc1m_c_out,
        c2_c_out,
        samples=10000,
        seed=20261103,
        cases=conf_cases,
        cases_per_world=16,
    )
    conf_vs_ppo37 = paired_world_bootstrap(
        ppo37_c_out,
        c2_c_out,
        samples=10000,
        seed=20261103,
        cases=conf_cases,
        cases_per_world=16,
    )
    write_json(
        out / "confirmation.json",
        {
            "status": "complete",
            "baseline": bc1m_conf,
            "ppo37_reference": ppo37_conf,
            "candidate": c2_conf,
            "comparison": conf_vs_bc1m,
            "comparison_vs_ppo37": conf_vs_ppo37,
        },
    )


def _confirm_more_data(
    config: EvaluationData,
    plan: EvaluationData,
    out: Path,
    selection: EvaluationData,
) -> None:
    best = selection["best"]
    if selection.get("baseline_retained", False) or (selection["best_wins"] <= selection["baseline_wins"]):
        write_json(
            out / "confirmation.json",
            {
                "status": "not_run_no_development_improvement",
                "selected": best,
                "baseline": selection["baseline"],
                "baseline_wins": selection["baseline_wins"],
                "required_wins": selection["required_new_candidate_wins"],
            },
        )
        return
    ref_run = ROOT / config["reference_run"]
    old_run = ROOT / config["old_corpus_run"]
    ppo_hash = config["evaluation_reference_policy_hash"]
    bc_hash = config["parent_policy_hash"]
    before = _evaluate_reference(
        config,
        plan,
        out,
        "checkpoints/reference-update-0037.pt",
        ppo_hash,
        "confirmation",
        ref_run,
    )
    after = evaluate_target_scale(config, plan, out, best["checkpoint"], "confirmation")
    first, cases = _outcomes_from_dir(out / before["runs"]["argmax"]["cp"]["directory"])
    second, paired = _outcomes_from_dir(out / after["runs"]["argmax"]["cp"]["directory"])
    if cases != paired:
        raise ValueError("Unpaired target BC confirmation")
    interval = paired_world_bootstrap(
        first,
        second,
        samples=10000,
        seed=20261103,
        cases=cases,
        cases_per_world=16,
    )
    # Secondary comparison against already available BC epoch 10 confirmation results (no 3rd 256-match run).
    secondary = None
    for candidate_root in (ref_run, old_run):
        evals_path = candidate_root / "evaluations.json"
        if not evals_path.is_file():
            continue
        saved_evals = read_json(evals_path)
        bc_conf = next(
            (
                v
                for v in saved_evals.values()
                if isinstance(v, dict)
                and v.get("purpose") == "confirmation"
                and v.get("policy_hash") == bc_hash
            ),
            None,
        )
        if bc_conf is not None:
            bc_dir = candidate_root / bc_conf["runs"]["argmax"]["cp"]["directory"]
            if (bc_dir).is_dir() and len(list(bc_dir.glob("*/metrics.json"))) == 256:
                bc_outcomes, bc_cases = _outcomes_from_dir(bc_dir)
                if bc_cases == paired:
                    secondary = {
                        "reference": "bc_epoch_10",
                        "policy_hash": bc_hash,
                        "source_run": candidate_root.relative_to(ROOT).as_posix(),
                        "summary": bc_conf["runs"]["argmax"]["cp"]["summary"],
                        "comparison": paired_world_bootstrap(
                            bc_outcomes,
                            second,
                            samples=10000,
                            seed=20261103,
                            cases=cases,
                            cases_per_world=16,
                        ),
                    }
                    break
    write_json(
        out / "confirmation.json",
        {
            "status": "complete",
            "baseline": before,
            "candidate": after,
            "comparison": interval,
            "secondary_bc_epoch_10": secondary,
        },
    )


def evaluate(config: EvaluationData, plan: EvaluationData, out: Path) -> None:
    if (out / "confirmation.json").exists():
        return
    training = read_json(out / "training.json")
    if training["status"] != "complete":
        raise ValueError("Training is incomplete")
    if config["kind"] == C2_KIND:
        _evaluate_c2(config, plan, out, training)
        return
    selection = read_json(out / "selection.json")
    best = selection["best"]
    if config["kind"] == MORE_DATA_KIND:
        _confirm_more_data(config, plan, out, selection)
        return
    base = selection["baseline"]
    if selection["best_wins"] <= base["runs"]["argmax"]["cp"]["summary"]["W"]:
        write_json(
            out / "confirmation.json",
            {"status": "not_run_no_development_improvement", "selected": best},
        )
        return
    before = evaluate_target_scale(config, plan, out, "checkpoints/initial.pt", "confirmation")
    after = evaluate_target_scale(config, plan, out, best["checkpoint"], "confirmation")
    first, cases = _outcomes_from_dir(out / before["runs"]["argmax"]["cp"]["directory"])
    second, paired = _outcomes_from_dir(out / after["runs"]["argmax"]["cp"]["directory"])
    if cases != paired:
        raise ValueError("Unpaired target BC confirmation")
    interval = paired_world_bootstrap(
        first, second, samples=10000, seed=20261103, cases=cases, cases_per_world=16
    )
    write_json(
        out / "confirmation.json",
        {
            "status": "complete",
            "baseline": before,
            "candidate": after,
            "comparison": interval,
        },
    )
