"""Artifact-only operations, independent of the current execution sources."""

import json
import pickle
import time
import zipfile
from datetime import UTC, datetime
from collections import Counter
from pathlib import Path
from uuid import uuid4

from civ5rl.engine.control import RunLock, target_bc_evaluation_due
from civ5rl.provenance.audit import (
    verify_bc_ledger,
    verify_checkpoint_chain,
    verify_ppo_ledger,
)
from civ5rl.provenance.checkpoint import verify_frozen_passport
from civ5rl.provenance.common import write_json
from civ5rl.reporting.evaluation_audit import (
    clustered_paired_bootstrap,
    load_eval_rows,
    outcome,
)
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
    from civ5rl.engine.budget import budget_base, read_extension
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
            if (status.get("status") not in ("training", "running", "evaluating", "paused", "technical_error")
                    or state.get("stop_reason") or (directory / "closure.json").exists()
                    or _read(directory / "training.json", {}).get("status") == "complete"):
                raise ValueError("Cannot extend a completed, closed or stopped training run")
            value = {
                "schema": 1, "run_id": record["run_id"], "base_active_seconds": base,
                "additional_active_seconds": previous.get("additional_active_seconds", 0) + additional_seconds,
                "requests": [*previous.get("requests", []), {
                    "at_utc": datetime.now(UTC).isoformat(), "additional_seconds": additional_seconds,
                    "request_id": request_id, "reason": reason,
                    "update": state.get("updates"), "epoch": state.get("epoch"),
                    "active_seconds": state.get("active_seconds", 0),
                }],
            }
            write_json(directory / "active-budget.json", value)
        receipt = _read(directory / "budget-applied.json", {})
        revision = next(i for i, request in enumerate(value["requests"], 1) if request.get("request_id") == request_id)
        return {**value, "request_id": request_id, "requested_revision": revision,
                "applied_revision": receipt.get("revision"),
                "applied": receipt.get("run_id") == record["run_id"] and receipt.get("revision", -1) >= revision}


def compare(before, after, expected_n, *, clusters=16, samples=10000, seed=20261103):
    return clustered_paired_bootstrap(
        load_eval_rows(before, expected_n),
        load_eval_rows(after, expected_n),
        clusters=clusters,
        samples=samples,
        seed=seed,
    )


def derived_output(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if directory == output or directory.is_relative_to(output):
        raise ValueError("Derived output must be separate from the scientific run root")
    if output.is_relative_to(directory) and output.relative_to(directory).parts[
        0
    ] not in (
        "derived",
        "snapshots",
        "interim-snapshot",
    ):
        raise ValueError(
            "In-run derived output must use derived/, snapshots/ or interim-snapshot/"
        )
    return output


def _bc_evaluation_due(config, state, evaluations):
    dev_points = [
        v
        for v in evaluations.values()
        if isinstance(v, dict)
        and v.get("purpose") in ("development", "development_512")
    ]
    if not dev_points:
        return True
    milestones = config.get("evaluate_epochs", [2, 5, 10])
    epochs_history = state.get("epochs_history", [])
    completed_epochs = {
        int(e["epoch"]) for e in epochs_history if e.get("completed")
    }
    evaluated_epochs = {
        int(v.get("epoch", 0))
        for v in dev_points
        if int(v.get("Adam_steps", 0)) > 0
    }
    for m in milestones:
        if m in completed_epochs and m not in evaluated_epochs:
            return True
    if state.get("stop_reason") and int(state.get("Adam_steps", 0)) > 0:
        latest = state.get("latest")
        if not any(v.get("checkpoint") == latest for v in dev_points):
            return True
    return False


def _audit_stable(directory):
    sections = {}

    def check(name, operation, missing=None):
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

    run = _read(directory / "run.json", {})
    config = run.get("config", _read(directory / "resolved-config.json", {}))

    def passport_check():
        if (directory / "passport.json").exists():
            return {
                "source_files": len(verify_frozen_passport(directory)["source_hashes"]),
                "scope": "frozen_passport",
                "working_tree_compared": False,
            }
        import hashlib
        import zipfile

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
        config_hash = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()
        if config_hash != passport["config_hash"]:
            raise ValueError("Deployment configuration differs from saved run")
        return {
            "source_files": len(passport["source_hashes"]),
            "scope": "frozen_deployment",
            "working_tree_compared": False,
            "original_runner_passport": False,
        }

    check(
        "passport",
        passport_check,
        None
        if any(
            (directory / name).exists()
            for name in ("passport.json", "deployment-passport.json")
        )
        else "No saved passport or frozen deployment archive",
    )
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
        check(
            "ppo_ledger",
            lambda: verify_bc_ledger(directory),
            None
            if (directory / "state.json").exists()
            and (directory / "collection.json").exists()
            else "No saved BC state/collection ledger",
        )
    else:
        check(
            "ppo_ledger",
            lambda: verify_ppo_ledger(directory),
            None
            if supported_ppo and (directory / "updates.jsonl").exists()
            else "No supported PPO ledger schema; historical/BC accounting is not covered",
        )
    # Continuations need independently established offsets; never guess them.
    offset = run.get(
        "parent_Adam_counter",
        config.get("experiment", {}).get("parent_optimizer_steps", 0),
    )
    check(
        "checkpoints_adam_rng",
        lambda: verify_checkpoint_chain(
            directory, adam_offset=offset,
            parent_checkpoint=(Path(__file__).resolve().parents[2] / config["experiment"]["parent"])
            if config.get("experiment", {}).get("parent_optimizer") == "preserve" else None,
        ),
        None
        if (supported_ppo or supported_bc) and (directory / "checkpoints").exists()
        else "No supported checkpoint schema",
    )

    evaluations = []
    state = _read(directory / "state.json", {})
    evals_dict = _read(directory / "evaluations.json", {})
    candidates = [state.get("evaluated", {}), evals_dict]
    confirmation = _read(directory / "confirmation.json", {})
    candidates.append(confirmation)

    def visit(node):
        if isinstance(node, dict):
            if isinstance(node.get("directory"), str) and isinstance(
                node.get("summary"), dict
            ):
                evaluations.append(node)
            else:
                for value in node.values():
                    visit(value)
        elif isinstance(node, list):
            for value in node:
                visit(value)

    for candidate in candidates:
        visit(candidate)

    def evaluation_check():
        results = []
        for entry in {v["directory"]: v for v in evaluations}.values():
            path = (directory / entry["directory"]).resolve()
            if not path.is_relative_to(directory.resolve()):
                raise ValueError("Evaluation path escapes saved run")
            summary = entry["summary"]
            rows = load_eval_rows(path, summary["matches"])
            counts = Counter(outcome(row[0]) for row in rows.values())
            if any(counts[k] != summary[k] for k in ("W", "D", "L")):
                raise ValueError(f"W/D/L differs: {path}")
            results.append(
                {
                    "directory": entry["directory"],
                    **{k: counts[k] for k in ("W", "D", "L")},
                }
            )
        return results

    check(
        "evaluation_completeness_wdl",
        evaluation_check,
        None if evaluations else "No recognized saved evaluation directory references",
    )
    if kind == "rl01.ppo_after_target_bc":
        check(
            "scheduled_evaluations",
            lambda: {"pending": target_bc_evaluation_due(config, state)},
            None,
        )
    elif supported_bc:
        check(
            "scheduled_evaluations",
            lambda: {"pending": _bc_evaluation_due(config, state, evals_dict)},
            None,
        )
    else:
        check(
            "scheduled_evaluations",
            lambda: None,
            "Historical scheduling protocol not covered; only referenced match artifacts were checked",
        )
    scheduled = sections["scheduled_evaluations"]
    if scheduled.get("result", {}).get("pending"):
        scheduled.update(
            status="incomplete",
            reason="A required development evaluation is still pending",
        )

    def confirmation_check():
        if confirmation.get("status") == "not_run_parent_selected":
            selected = _read(directory / "selected.json", {})
            points = list(state.get("evaluated", {}).values())
            baseline = next((p for p in points if p.get("checkpoint") == "checkpoints/initial.pt"), None)
            if (not state.get("stop_reason") or selected != state.get("best")
                    or selected.get("path") != "checkpoints/initial.pt" or baseline is None):
                raise ValueError("Skipped Additional does not have a selected parent boundary")
            wins = baseline["runs"]["argmax"]["cp"]["summary"]["W"]
            if (selected.get("wins") != wins or selected.get("policy_hash") != baseline.get("policy_hash")
                    or any(p["runs"]["argmax"]["cp"]["summary"]["W"] > wins for p in points)):
                raise ValueError("Skipped Additional despite a better Development candidate")
            return {"status": "not_run_parent_selected", "bootstrap_required": False,
                    "best_development_wins": wins, "evaluated_checkpoints": len(points)}
        if confirmation.get("status") in (
            "not_run_no_development_improvement",
            "not_run_development_512_gate_not_met",
        ):
            return {"status": confirmation.get("status")}
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
        import numpy as np

        for name in ("delta", "CI95"):
            if name in settings and not np.allclose(
                result[name], settings[name], rtol=0, atol=1e-12
            ):
                raise ValueError(
                    f"Saved confirmation {name} differs from recomputation"
                )
        return result

    check(
        "confirmation_bootstrap",
        confirmation_check,
        None
        if {"baseline", "candidate"} <= confirmation.keys()
        or confirmation.get("status")
        in (
            "not_run_parent_selected",
            "not_run_no_development_improvement",
            "not_run_development_512_gate_not_met",
        )
        else "No recognized paired confirmation; special historical analyses are not covered",
    )
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


def wait_and_finalize(
    directory, output, *, timeout=60, poll=1, block_size=10, last=None
):
    if timeout < 0 or poll <= 0:
        raise ValueError("timeout must be nonnegative and poll must be positive")
    directory = Path(directory)
    if (
        not (directory / "run.lock").is_file()
        or not (directory / "run.lock").stat().st_size
    ):
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
            raise TimeoutError(
                f"Run did not reach an inactive terminal boundary in {timeout}s"
            )
        time.sleep(min(poll, remaining))
