"""Artifact-only operations, independent of the current execution sources."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
import zipfile
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from uuid import uuid4

import numpy as np

from civ5rl.engine.evaluation.storage import readable_evaluations
from civ5rl.engine.runtime.budget import budget_base, read_extension
from civ5rl.engine.runtime.control import RunLock, target_bc_evaluation_due
from civ5rl.provenance.audit import (
    verify_bc_ledger,
    verify_checkpoint_chain,
    verify_ppo_ledger,
)
from civ5rl.provenance.checkpoint import verify_frozen_passport
from civ5rl.provenance.common import write_json
from civ5rl.provenance.host_resume import verify_revision
from civ5rl.reporting.evaluation_audit import (
    clustered_paired_bootstrap,
    collect_paired_outcomes,
    dual_promotion_decision,
    eval_wdl_summary,
    h2h_world_bootstrap,
    load_eval_rows,
    select_dual_milestone,
)
from civ5rl.reporting.output_paths import derived_output
from civ5rl.reporting.snapshot import write_snapshot

ARTIFACT_ERRORS = (
    ValueError,
    KeyError,
    TypeError,
    OSError,
    RuntimeError,
    EOFError,
    pickle.UnpicklingError,
    zipfile.BadZipFile,
)


def _read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def extend_budget(directory, additional_seconds, reason, *, request_id=None):
    """Record an authorized operational extension without rewriting saved config."""
    if type(additional_seconds) is not int or additional_seconds <= 0 or not reason.strip():
        raise ValueError("A positive integer extension and reason are required")
    directory = Path(directory)
    request_id = request_id or uuid4().hex
    if not isinstance(request_id, str) or not request_id.strip():
        raise ValueError("Request ID must be a nonempty string")
    with RunLock(directory, "budget.lock"):
        record = _read(directory / "run.json")
        state = _read(directory / "state.json")
        status = _read(directory / "status.json", {})
        base = budget_base(record)
        previous = read_extension(directory, record)
        existing = next((r for r in previous.get("requests", []) if r.get("request_id") == request_id), None)
        if existing:
            if existing["additional_seconds"] != additional_seconds or existing["reason"] != reason:
                raise ValueError("Request ID already used for a different extension")
            value = previous
        else:
            if (
                status.get("status") not in ("training", "running", "evaluating", "paused", "technical_error")
                or state.get("stop_reason")
                or (directory / "closure.json").exists()
                or _read(directory / "training.json", {}).get("status") == "complete"
            ):
                raise ValueError("Cannot extend a completed, closed or stopped training run")
            value = {
                "schema": 1,
                "run_id": record["run_id"],
                "base_active_seconds": base,
                "additional_active_seconds": previous.get("additional_active_seconds", 0)
                + additional_seconds,
                "requests": [
                    *previous.get("requests", []),
                    {
                        "at_utc": datetime.now(UTC).isoformat(),
                        "additional_seconds": additional_seconds,
                        "request_id": request_id,
                        "reason": reason,
                        "update": state.get("updates"),
                        "epoch": state.get("epoch"),
                        "active_seconds": state.get("active_seconds", 0),
                    },
                ],
            }
            write_json(directory / "active-budget.json", value)
        receipt = _read(directory / "budget-applied.json", {})
        revision = next(
            i for i, request in enumerate(value["requests"], 1) if request.get("request_id") == request_id
        )
        return {
            **value,
            "request_id": request_id,
            "requested_revision": revision,
            "applied_revision": receipt.get("revision"),
            "applied": receipt.get("run_id") == record["run_id"] and receipt.get("revision", -1) >= revision,
        }


def compare(before, after, expected_n, *, clusters=16, samples=10000, seed=20261103):
    return clustered_paired_bootstrap(
        load_eval_rows(before, expected_n),
        load_eval_rows(after, expected_n),
        clusters=clusters,
        samples=samples,
        seed=seed,
    )


def _bc_evaluation_due(config, state, evaluations):
    dev_points = [
        v
        for v in evaluations.values()
        if isinstance(v, dict) and v.get("purpose") in ("development", "development_512")
    ]
    if not dev_points:
        return True
    milestones = config.get("evaluate_epochs", [2, 5, 10])
    epochs_history = state.get("epochs_history", [])
    completed_epochs = {int(e["epoch"]) for e in epochs_history if e.get("completed")}
    evaluated_epochs = {int(v.get("epoch", 0)) for v in dev_points if int(v.get("Adam_steps", 0)) > 0}
    for m in milestones:
        if m in completed_epochs and m not in evaluated_epochs:
            return True
    if state.get("stop_reason") and int(state.get("Adam_steps", 0)) > 0:
        latest = state.get("latest")
        if not any(v.get("checkpoint") == latest for v in dev_points):
            return True
    return False


def _dual_candidates(state):
    candidates = []
    for p_hash, item in state.get("evaluated", {}).items():
        h2h = item.get("h2h")
        if not isinstance(h2h, dict) or not isinstance(h2h.get("summary"), dict):
            raise ValueError("Missing H2H evaluation on Development milestone")
        cp_sum = item["runs"]["argmax"]["cp"]["summary"]
        h2h_sum = h2h["summary"]
        candidates.append(
            {
                "path": item["checkpoint"],
                "update": int(item.get("update", 0)),
                "decisions": int(item.get("decisions", 0)),
                "policy_hash": p_hash,
                "cp_wins": int(cp_sum["W"]),
                "h2h_wins": int(h2h_sum["W"]),
                "h2h_losses": int(h2h_sum["L"]),
                "cp_summary": cp_sum,
                "h2h_summary": h2h_sum,
            }
        )
    return candidates


def _check_parent_selected(directory, state, *, dual):
    selected = _read(directory / "selected.json", {})
    points = list(state.get("evaluated", {}).values())
    baseline = next((p for p in points if p.get("checkpoint") == "checkpoints/initial.pt"), None)
    if (
        not state.get("stop_reason")
        or selected != state.get("best")
        or selected.get("path") != "checkpoints/initial.pt"
        or baseline is None
    ):
        raise ValueError("Skipped Additional does not have a selected parent boundary")
    wins = baseline["runs"]["argmax"]["cp"]["summary"]["W"]
    if selected.get("wins") != wins or selected.get("policy_hash") != baseline.get("policy_hash"):
        raise ValueError("Skipped Additional despite a better Development candidate")
    if dual:
        expected = select_dual_milestone(_dual_candidates(state), wins)
        h2h_base = baseline["h2h"]["summary"]
        if h2h_base["W"] != h2h_base["L"]:
            raise ValueError("Baseline H2H zero-point is asymmetric")
        if expected["path"] != "checkpoints/initial.pt":
            raise ValueError("Skipped Additional despite an admissible dual Development candidate")
        if selected != expected:
            raise ValueError("Saved dual selection differs from recomputation")
        return {
            "status": "not_run_parent_selected",
            "bootstrap_required": False,
            "selection_rule": "dual_nonregression_max_J",
            "best_development_wins": wins,
            "evaluated_checkpoints": len(points),
        }
    if any(p["runs"]["argmax"]["cp"]["summary"]["W"] > wins for p in points):
        raise ValueError("Skipped Additional despite a better Development candidate")
    return {
        "status": "not_run_parent_selected",
        "bootstrap_required": False,
        "best_development_wins": wins,
        "evaluated_checkpoints": len(points),
    }


def _check_cp_confirmation(directory, confirmation):
    before = confirmation["baseline"]["runs"]["argmax"]["cp"]
    after = confirmation["candidate"]["runs"]["argmax"]["cp"]
    settings = confirmation.get("comparison", {})
    result = compare(
        directory / before["directory"],
        directory / after["directory"],
        before["summary"]["matches"],
        samples=settings.get("bootstrap_samples", 10000),
        seed=settings.get("bootstrap_seed", 20261103),
        clusters=settings.get("bootstrap_clusters", 16),
    )
    for name in ("delta", "CI95"):
        if name in settings and not np.allclose(result[name], settings[name], rtol=0, atol=1e-12):
            raise ValueError(f"Saved confirmation {name} differs from recomputation")
    return result


def _check_h2h_confirmation(directory, state, confirmation, result, *, dual):
    h2h_conf = confirmation.get("h2h_confirmation")
    saved_promotion = confirmation.get("promotion")
    if not isinstance(h2h_conf, dict) or not isinstance(saved_promotion, dict):
        raise ValueError("Missing H2H confirmation or dual promotion decision")
    if dual and state.get("evaluated"):
        points = list(state["evaluated"].values())
        baseline_dev = next(
            (p for p in points if p.get("checkpoint") == "checkpoints/initial.pt"),
            None,
        )
        if baseline_dev is None:
            raise ValueError("Missing baseline Development evaluation")
        expected_best = select_dual_milestone(
            _dual_candidates(state),
            baseline_dev["runs"]["argmax"]["cp"]["summary"]["W"],
        )
        selected = _read(directory / "selected.json", {})
        if selected != expected_best or confirmation["candidate"].get("checkpoint") != expected_best["path"]:
            raise ValueError("Saved dual selection or confirmed candidate differs from recomputation")
    h2h_settings = h2h_conf.get("bootstrap", {})
    h2h_result = h2h_world_bootstrap(
        directory / h2h_conf["directory"],
        h2h_conf["summary"]["matches"],
        samples=h2h_settings.get("bootstrap_samples", 10000),
        seed=h2h_settings.get("bootstrap_seed", 20261103),
        clusters=h2h_settings.get("bootstrap_clusters", 16),
    )
    for key in ("W", "D", "L", "delta_wins"):
        if h2h_result[key] != h2h_conf["summary"][key]:
            raise ValueError(f"Saved H2H confirmation {key} differs from recomputation")
    for name in ("delta", "CI95"):
        if name in h2h_settings and not np.allclose(h2h_result[name], h2h_settings[name], rtol=0, atol=1e-12):
            raise ValueError(f"Saved H2H confirmation {name} differs from recomputation")
    promotion = dual_promotion_decision(result, h2h_result)
    for flag in ("promoted", "both_nonnegative", "significant_ci95"):
        if saved_promotion.get(flag) != promotion[flag]:
            raise ValueError(f"Saved dual promotion {flag} differs from recomputation")
    for name in ("cp_delta", "cp_ci95", "h2h_delta", "h2h_ci95"):
        if name not in saved_promotion or not np.allclose(
            promotion[name], saved_promotion[name], rtol=0, atol=1e-12
        ):
            raise ValueError(f"Saved dual promotion {name} differs from recomputation")
    return {"h2h": h2h_result, "promotion": promotion}


SKIPPED_CONFIRMATION = (
    "not_run_parent_selected",
    "not_run_no_development_improvement",
    "not_run_development_512_gate_not_met",
)


def _confirmation_check(directory, state, confirmation, config=None):
    if config is None:
        config = _read(directory / "run.json", {}).get("config", {})
    if config.get("experiment", {}).get("three_opponent_evaluation"):
        return _three_opponent_confirmation_check(directory, state, confirmation, config["experiment"])
    dual = bool(config.get("self_play", {}).get("enabled"))
    status = confirmation.get("status")
    if status == "not_run_parent_selected":
        return _check_parent_selected(directory, state, dual=dual)
    if status in SKIPPED_CONFIRMATION:
        return {"status": status}
    cp_result = _check_cp_confirmation(directory, confirmation)
    if dual or "h2h_confirmation" in confirmation:
        return {**cp_result, **_check_h2h_confirmation(directory, state, confirmation, cp_result, dual=dual)}
    return cp_result


def _three_opponent_confirmation_check(directory, state, confirmation, experiment):
    from civ5rl.experiments.ppo.after_target_bc.evaluation import bot_comparison
    from civ5rl.protocols import three_opponents as triple

    points = state["three_opponent_points"]
    manifest = _read(directory / "manifest.json")
    native_u0 = "checkpoints/reference-u0.pt"
    continuation = state.get("dual_target_continuation")
    for point, fingerprint in state["development_points"].items():
        entry = points.get(point, {})
        if entry.get("policy_hash") != fingerprint or entry.get("decisions", 0) < int(point):
            raise ValueError("Three-opponent milestone differs from the saved decision boundary")
    boundary = state.get("dual_target_boundary")
    if boundary:
        until_stop = {point: item for point, item in points.items() if int(point) <= boundary["milestone"]}
        if triple.stop_reason(until_stop, experiment["milestones"], gain=experiment["early_gain_min_wins"],
                              gain_boundary=experiment["early_gain_decisions"]) != triple.EARLY_GOAL:
            raise ValueError("Invalid early goal boundary")
        if (until_stop[str(boundary["milestone"])]["checkpoint"] != boundary["checkpoint"]
                or not (directory / boundary["checkpoint"]).is_file()):
            raise ValueError("Early goal checkpoint is missing or differs")
    elif state.get("stop_reason") == triple.EARLY_GOAL:
        raise ValueError("Missing early goal boundary")

    def check_cases(relative, purpose):
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Evaluation path escapes saved run")
        _, cases, _ = collect_paired_outcomes(path, check_duplicates=True)
        expected = {case["case_hash"] for case in manifest[purpose]}
        if len(cases) != len(expected) or {case["case_hash"] for case in cases.values()} != expected:
            raise ValueError(f"Evaluation cases differ from {purpose}")

    def check_series(series, candidate_path, purpose):
        candidate, baseline = series["candidate"], series["baseline"]
        if (candidate["checkpoint"] != candidate_path or candidate["cpv2"]["checkpoint"] != candidate_path
                or baseline["checkpoint"] != native_u0 or baseline["cpv2"]["checkpoint"] != native_u0):
            raise ValueError("Confirmed checkpoint differs from selected checkpoint")
        for enemy in ("cp", "cpv2"):
            old, new = (baseline, candidate) if enemy == "cp" else (baseline[enemy], candidate[enemy])
            for item in (old, new):
                if item["purpose"] != purpose:
                    raise ValueError(f"Saved {enemy} series belongs to another split")
                check_cases(item["runs"]["argmax"][enemy]["directory"], purpose)
            actual = bot_comparison(directory, old, new, enemy)
            for key in ("delta", "CI95"):
                if not np.allclose(actual[key], series[enemy][key], atol=1e-12, rtol=0):
                    raise ValueError(f"Saved {enemy} bootstrap differs")
        h2h = candidate["h2h"]
        if (h2h["checkpoint"] != candidate_path or h2h["reference"] != native_u0 or h2h["purpose"] != purpose
                or h2h["policy_hash"] != candidate["policy_hash"] or h2h["reference_policy_hash"] != baseline["policy_hash"]):
            raise ValueError("Wrong H2H candidate, reference or split")
        check_cases(h2h["directory"], purpose)
        actual = h2h_world_bootstrap(directory / h2h["directory"], 256, samples=10000, seed=20261103, clusters=16)
        for key in ("W", "D", "L", "delta", "CI95"):
            if not np.allclose(actual[key], h2h["bootstrap"][key], atol=1e-12, rtol=0):
                raise ValueError("Saved H2H confirmation differs")

    if continuation:
        if continuation != boundary:
            raise ValueError("Continuation boundary differs from original stop")
        first = _read(directory / "confirmation-first.json", {})
        first_path = triple.select(until_stop)["path"]
        if first.get("attempt") != 1 or not triple.allows_continuation(first["additional"]["candidate"]):
            raise ValueError("First Additional does not allow continuation")
        check_series(first["additional"], first_path, "confirmation")
    selection = triple.select(points, continuation=continuation)
    if _read(directory / "selected.json", {}) != selection or state["best"] != selection:
        raise ValueError("Three-opponent selection differs from recomputation")
    if confirmation["attempt"] != (2 if continuation else 1):
        raise ValueError("Wrong confirmation attempt")
    path = selection["path"]
    if path == "checkpoints/initial.pt":
        if confirmation.get("status") != "not_run_parent_selected":
            raise ValueError("Confirmation exists without an admissible new candidate")
        return {"status": confirmation["status"]}
    additional = confirmation["additional"]
    check_series(additional, path, "confirmation")
    if continuation:
        reserve = confirmation["reserve"]
        check_series(reserve, path, "reserve_256")
        confirmed = all(triple.meets(item["candidate"], triple.ADDITIONAL_TARGET) for item in (additional, reserve))
        promotion = {"promoted": confirmed, "dual_target_confirmed": confirmed}
    else:
        promotion = triple.initial_promotion(additional["cpv2"], additional["cp"], additional["candidate"])
    if confirmation["promotion"] != promotion:
        raise ValueError("Three-opponent promotion differs from recomputation")
    return promotion


def _audit_section(sections, name, operation, missing=None):
    if missing:
        sections[name] = {"status": "unavailable", "reason": missing}
        return
    try:
        sections[name] = {"status": "passed", "result": operation()}
    except ARTIFACT_ERRORS as exc:
        unsupported = any(
            text in str(exc)
            for text in (
                "Legacy non-transactional",
                "Optimizer step records unavailable",
            )
        )
        sections[name] = {
            "status": "unsupported" if unsupported else "failed",
            "reason": str(exc),
        }


def _passport_check(directory, config):
    if (directory / "passport.json").exists():
        return {
            "source_files": len(verify_frozen_passport(directory)["source_hashes"]),
            "scope": "frozen_passport",
            "working_tree_compared": False,
        }
    passport = _read(directory / "deployment-passport.json")
    expected = passport["source_hashes"]
    with zipfile.ZipFile(directory / "deployment-sources.zip") as archive:
        actual = {
            info.filename: hashlib.sha256(archive.read(info)).hexdigest()
            for info in archive.infolist()
            if not info.is_dir()
        }
    if actual != expected:
        raise ValueError("ZIP member hashes differ: deployment-sources.zip")
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if config_hash != passport["config_hash"]:
        raise ValueError("Deployment configuration differs from saved run")
    return {
        "source_files": len(passport["source_hashes"]),
        "scope": "frozen_deployment",
        "working_tree_compared": False,
        "original_runner_passport": False,
    }


def _evaluation_check(directory, evaluations):
    results = []
    for entry in {v["directory"]: v for v in evaluations}.values():
        path = (directory / entry["directory"]).resolve()
        if not path.is_relative_to(directory.resolve()):
            raise ValueError("Evaluation path escapes saved run")
        summary = entry["summary"]
        rows = load_eval_rows(path, summary["matches"])
        counts = eval_wdl_summary(rows)
        if any(counts[k] != summary[k] for k in ("W", "D", "L")):
            raise ValueError(f"W/D/L differs: {path}")
        results.append(
            {
                "directory": entry["directory"],
                **{k: counts[k] for k in ("W", "D", "L")},
            }
        )
    return results


def _evaluation_entries(node):
    if isinstance(node, dict):
        if isinstance(node.get("directory"), str) and isinstance(node.get("summary"), dict):
            yield node
        else:
            for value in node.values():
                yield from _evaluation_entries(value)
    elif isinstance(node, list):
        for value in node:
            yield from _evaluation_entries(value)


def _evaluation_pending(check, *args):
    return {"pending": check(*args)}


def _audit_scheduling(sections, kind, supported_bc, config, state, evals_dict):
    if kind == "rl01.ppo_after_target_bc":
        _audit_section(
            sections,
            "scheduled_evaluations",
            partial(_evaluation_pending, target_bc_evaluation_due, config, state),
            None,
        )
    elif supported_bc:
        _audit_section(
            sections,
            "scheduled_evaluations",
            partial(_evaluation_pending, _bc_evaluation_due, config, state, evals_dict),
            None,
        )
    else:
        _audit_section(
            sections,
            "scheduled_evaluations",
            None,
            "Historical scheduling protocol not covered; only referenced match artifacts were checked",
        )
    scheduled = sections["scheduled_evaluations"]
    if scheduled.get("result", {}).get("pending"):
        scheduled.update(
            status="incomplete",
            reason="A required development evaluation is still pending",
        )


def _host_resume_check(directory, revisions):
    before = _read(directory / "passport.json")
    return [verify_revision(path, before, _read(path.parent / "identity-after.json")) for path in revisions]


def _audit_training(sections, directory, run, config):
    kind = run.get("kind", "")
    supported_ppo = kind in {
        "rl01.ppo_after_target_bc",
        "rl01.ppo_target_scale",
        "rl01.ppo_scale_up",
    }
    supported_bc = kind in {
        "rl01.cp_imitation_target_more_data",
        "rl01.cp_imitation_target_c2",
        "rl01.cp_imitation_target_scale",
    }
    if supported_bc:
        _audit_section(
            sections,
            "ppo_ledger",
            lambda: verify_bc_ledger(directory),
            None
            if (directory / "state.json").exists() and (directory / "collection.json").exists()
            else "No saved BC state/collection ledger",
        )
    else:
        _audit_section(
            sections,
            "ppo_ledger",
            lambda: verify_ppo_ledger(directory),
            None
            if supported_ppo and (directory / "updates.jsonl").exists()
            else "No supported PPO ledger schema; historical/BC accounting is not covered",
        )
    # Continuations use saved optimizer offsets.
    offset = run.get(
        "parent_Adam_counter",
        config.get("experiment", {}).get("parent_optimizer_steps", 0),
    )
    _audit_section(
        sections,
        "checkpoints_adam_rng",
        lambda: verify_checkpoint_chain(
            directory,
            adam_offset=offset,
            parent_checkpoint=(Path(__file__).resolve().parents[2] / config["experiment"]["parent"])
            if config.get("experiment", {}).get("parent_optimizer") == "preserve"
            else None,
        ),
        None
        if (supported_ppo or supported_bc) and (directory / "checkpoints").exists()
        else "No supported checkpoint schema",
    )
    return kind, supported_bc


def _audit_stable(directory):
    sections = {}
    run = _read(directory / "run.json", {})
    config = run.get("config", _read(directory / "resolved-config.json", {}))
    _audit_section(
        sections,
        "passport",
        partial(_passport_check, directory, config),
        None
        if any((directory / name).exists() for name in ("passport.json", "deployment-passport.json"))
        else "No saved passport or frozen deployment archive",
    )

    kind, supported_bc = _audit_training(sections, directory, run, config)
    state = _read(directory / "state.json", {})
    evaluations = _read(directory / "evaluations.json", {})
    confirmation = _read(directory / "confirmation.json", {})
    first_confirmation = _read(directory / "confirmation-first.json", {})
    entries = list(_evaluation_entries([state.get("evaluated", {}), evaluations, confirmation, first_confirmation]))
    _audit_section(
        sections,
        "evaluation_completeness_wdl",
        partial(_evaluation_check, directory, entries),
        None if entries else "No recognized saved evaluation directory references",
    )
    _audit_scheduling(sections, kind, supported_bc, config, state, evaluations)
    _audit_section(
        sections,
        "confirmation_bootstrap",
        partial(_confirmation_check, directory, state, confirmation, config),
        None
        if {"baseline", "candidate"} <= confirmation.keys()
        or "additional" in confirmation
        or confirmation.get("status") in SKIPPED_CONFIRMATION
        else "No recognized paired confirmation; special historical analyses are not covered",
    )
    revisions = sorted((directory / "source-revisions").glob("host-*/revision.json"))
    if revisions:
        _audit_section(sections, "host_resume_revisions", partial(_host_resume_check, directory, revisions))
    statuses = {value["status"] for value in sections.values()}
    return {
        "status": "failed"
        if "failed" in statuses
        else "incomplete"
        if statuses != {"passed"} or state.get("phase") in {"critic", "ppo"}
        else "passed",
        "sections": sections,
    }


def audit_run(directory, output):
    directory = Path(directory).resolve()
    output = derived_output(directory, output)
    lock_path = directory / "run.lock"
    if not lock_path.is_file() or not lock_path.stat().st_size:
        result = {
            "status": "incomplete",
            "sections": {},
            "reason": "No existing writer lock; cannot establish a stable saved boundary",
        }
    else:
        lock = RunLock(directory)
        try:
            lock.__enter__()
        except (RuntimeError, OSError) as exc:
            result = {
                "status": "incomplete",
                "sections": {},
                "reason": f"Writer lock unavailable; use snapshot for intermediate observations: {exc}",
            }
        else:
            try:
                try:
                    with readable_evaluations(directory):
                        result = _audit_stable(directory)
                except ARTIFACT_ERRORS as exc:
                    result = {
                        "status": "failed",
                        "sections": {},
                        "reason": f"Saved metadata cannot be parsed: {exc}",
                    }
            finally:
                lock.__exit__(None, None, None)
    for name in (
        "passport",
        "ppo_ledger",
        "checkpoints_adam_rng",
        "evaluation_completeness_wdl",
        "confirmation_bootstrap",
        "scheduled_evaluations",
    ):
        result["sections"].setdefault(
            name,
            {
                "status": "unavailable",
                "reason": result.get("reason", "Check not supported"),
            },
        )
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "audit.json", result)
    lines = [f"# Saved-run audit: {result['status']}", "", result.get("reason", "")]
    for name, section in result["sections"].items():
        lines += [f"- {name}: {section['status']}. {section.get('reason', '')}"]
    (output / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def wait_and_finalize(directory, output, *, timeout=60, poll=1, block_size=10, last=None):
    if timeout < 0 or poll <= 0:
        raise ValueError("timeout must be nonnegative and poll must be positive")
    directory = Path(directory)
    if not (directory / "run.lock").is_file() or not (directory / "run.lock").stat().st_size:
        raise ValueError("Existing run lock file is required to wait safely")
    deadline = time.monotonic() + timeout
    terminal = {
        "paused",
        "complete",
        "closed_by_user",
        "technical_error",
        "invalid_dataset",
        "training_complete",
    }
    while True:
        lock = RunLock(directory)
        try:
            lock.__enter__()
        except RuntimeError:
            pass
        else:
            try:
                status = _read(directory / "status.json", {})
                if status.get("status") in terminal:
                    write_snapshot(directory, output, block_size=block_size, last=last)
                    return {
                        "status": status["status"],
                        "snapshot": str(Path(output).resolve()),
                    }
            finally:
                lock.__exit__(None, None, None)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"Run did not reach an inactive terminal boundary in {timeout}s")
        time.sleep(min(poll, remaining))
