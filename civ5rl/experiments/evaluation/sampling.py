"""One CPU-only, paired evaluation of the frozen target BC policy."""

from __future__ import annotations

import math
import platform
import random
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from civ5rl.config import load_config
from civ5rl.core.agent import RLAgent, action_probabilities
from civ5rl.core.context import context_from_spec
from civ5rl.core.observation import ordered_actions
from civ5rl.engine.evaluation.matches import run_policy_match
from civ5rl.engine.ppo.storage import load_rollout
from civ5rl.engine.runtime.control import RunLock
from civ5rl.engine.runtime.shared import pool
from civ5rl.experiments._shared.lifecycle import record_experiment_errors
from civ5rl.experiments._shared.provenance_checks import verify_manifest_digest
from civ5rl.experiments.evaluation.paired import run_chunked_pool
from civ5rl.provenance.common import (
    ROOT,
    choose,
    configure,
    file_sha256,
    read_json,
    write_json,
)
from civ5rl.reporting.evaluation_audit import (
    classify_outcome,
    paired_world_bootstrap,
    summarize_wdl,
    verify_match_artifacts,
)
from civ5rl.reporting.target_bc_sampling import render_sampling_report, write_incomplete_report
from civ5sim import combat_ai
from civ5sim.engine import digest

REQUIRED_MODES = {
    "argmax": ("argmax", 1.0),
    "sample-t1": ("sample", 1.0),
    "sample-t0.5": ("sample", 0.5),
}


def execution_identity():
    kernel = ROOT / "build" / ("combat.dll" if sys.platform == "win32" else "libcombat.so")
    return {
        "version": 2,
        "kernel": file_sha256(kernel),
        "planner": file_sha256(combat_ai.LIBRARY_PATH),
        "rules": file_sha256(ROOT / "build/combat_rules.json"),
        "platform": platform.system(),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "device": "cpu",
        "torch_threads_per_worker": 1,
    }


def mode_key(identity, setting):
    return digest(
        {
            "experiment": identity,
            "mode": setting["mode"],
            "temperature": setting["temperature"],
        }
    )


def inputs(settings):
    for name in ("workers", "bootstrap_samples"):
        if type(settings[name]) is not int or settings[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(settings["bootstrap_seed"]) is not int or settings["bootstrap_seed"] < 0:
        raise ValueError("bootstrap_seed must be a nonnegative integer")
    modes = {}
    for setting in settings["modes"]:
        label, mode, temperature = setting["label"], setting["mode"], setting["temperature"]
        if label in modes or not label or Path(label).name != label or label in (".", ".."):
            raise ValueError(f"Invalid or repeated mode label: {label}")
        if mode not in ("argmax", "sample") or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError(f"Invalid action-selection settings: {label}")
        modes[label] = (mode, temperature)
    if any(modes.get(label) != mode for label, mode in REQUIRED_MODES.items()):
        raise ValueError("Include argmax and sampling at temperatures 1 and 0.5 for the paired comparison")
    parent = ROOT / settings["checkpoint"]
    checkpoint_sha256 = file_sha256(parent)
    if checkpoint_sha256 != settings["checkpoint_sha256"]:
        raise ValueError("Configured checkpoint SHA256 differs")
    manifest = ROOT / settings["manifest"]
    manifest_sha256 = file_sha256(manifest)
    if settings.get("manifest_sha256", manifest_sha256) != manifest_sha256:
        raise ValueError("Configured manifest SHA256 differs")
    plan = read_json(manifest)
    verify_manifest_digest(plan, error_msg="Invalid target manifest digest")
    cases = plan["development"]
    config = load_config(ROOT / settings["evaluator_config"])
    if len(cases) != 128 or len({c["case_hash"] for c in cases}) != 128:
        raise ValueError("Expected exactly 128 unique development cases")
    worlds = Counter(c["parent_world"] for c in cases)
    if len(worlds) != 16 or set(worlds.values()) != {8}:
        raise ValueError("Expected 16 parent worlds, eight paired cases per world")
    for case in cases:
        if (
            case["split"] != "validation"
            or case["max_rounds"] != 30
            or case["opponent"] != "cp"
            or case["max_calls_per_turn"] != config["environment"]["max_calls_per_turn"]
            or case["opponent_search"] != config["teacher_search"]
            or digest({k: v for k, v in case.items() if k != "case_hash"}) != case["case_hash"]
        ):
            raise ValueError("Development conditions changed")
    identity = {
        "checkpoint_sha256": checkpoint_sha256,
        "manifest_sha256": manifest_sha256,
        "ordered_case_hashes": [c["case_hash"] for c in cases],
        "evaluator_config": config,
        "execution": execution_identity(),
    }
    return config, cases, identity


def preflight(settings, out):
    """Real BC logits on saved public states; no new battle or model training."""
    configure(17, "cpu")
    rollout = settings.get("preflight_rollout", "artifacts/rl/ppo-after-target-bc/rollouts/latest.pt")
    episode = load_rollout(ROOT / rollout)[0]
    agent = RLAgent.from_checkpoint(ROOT / settings["checkpoint"], "cpu")
    agent.reset(context_from_spec(episode["spec"]))
    tensors, checks = [], []
    with torch.no_grad():
        for index in range(3):
            observation = episode["observations"][index]
            actions = ordered_actions(episode["actions"][index])
            logits = agent._infer(observation, actions)
            old = logits.softmax(-1)[0].cpu().numpy()
            new = action_probabilities(logits, 1)[0].cpu().numpy()
            np.testing.assert_array_equal(old, new)
            before, after = random.Random(20261103), random.Random(20261103)
            assert [choose(old, before) for _ in range(64)] == [choose(new, after) for _ in range(64)]
            assert before.getstate() == after.getstate()
            colder = action_probabilities(logits, 0.5)
            assert int(colder[0].argmax()) == int(logits[0].argmax())
            assert torch.all(colder[torch.isneginf(logits)] == 0)
            tensors.append(logits.cpu())
            checks.append(
                {
                    "saved_observation": index,
                    "legal_actions": len(actions),
                    "T1_probabilities_and_seeded_choices_exact": True,
                    "argmax_and_mask_preserved": True,
                }
            )
            agent.encoder.remember_action(episode["actions"][index][episode["chosen"][index]], observation)
    torch.save(tensors, out / "preflight-logits.pt")
    write_json(
        out / "preflight.json",
        {
            "checkpoint_sha256": settings["checkpoint_sha256"],
            "checks": checks,
            "new_matches": 0,
            "device": "cpu",
        },
    )


def completed_case(directory, case, key):
    names = (
        "metrics.json",
        "case.json",
        "result.json",
        "replay.json",
        "final.json",
        "selection-metrics.json",
    )
    if not verify_match_artifacts(directory, names):
        return False
    metric = read_json(directory / "metrics.json")
    return (
        metric.get("status") == "completed"
        and metric.get("evaluation_key") == key
        and metric.get("case_hash") == case["case_hash"]
        and read_json(directory / "case.json") == case
        and read_json(directory / "result.json").get("status") == "completed"
    )


@dataclass(frozen=True)
class SamplingMatchTask:
    config: dict[str, Any]
    case: dict[str, Any]
    parent: str
    setting: dict[str, Any]
    directory: str
    key: str


def match_task(task):
    if not isinstance(task, SamplingMatchTask):
        task = SamplingMatchTask(*task)
    directory = Path(task.directory)
    return run_policy_match(
        task.config,
        task.case,
        task.parent,
        task.setting["mode"],
        "cp",
        directory,
        temperature=task.setting["temperature"],
        collect_selection_metrics=True,
        metric_fields={
            "evaluation_key": task.key,
            "mode": task.setting["mode"],
            "temperature": task.setting["temperature"],
        },
    )


def case_directory(out, setting, case):
    return out / setting["label"] / f"case-{case['case_id']:03}-side-{case['learner_side']}"


def wdl(rows):
    rounds = [row["rounds"] for row in rows]
    return {
        **summarize_wdl(rows),
        "mean_rounds": float(np.mean(rounds)),
        "median_rounds": float(np.median(rounds)),
        "turn_limit_draw_rate": sum(row["winner"] is None and row["reason"] == "turn_limit" for row in rows)
        / len(rows),
    }


def paired(cases, rows):
    return {
        (case["parent_world"], case["case_id"], case["learner_side"]): int(
            classify_outcome(row["winner"], row["side"]) == "W"
        )
        for case, row in zip(cases, rows, strict=True)
    }


def _load_mode_results(cases, setting, key, out):
    rows, decisions = [], []
    for case in cases:
        directory = case_directory(out, setting, case)
        if not completed_case(directory, case, key):
            continue
        row = read_json(directory / "metrics.json")
        # Independently verify W/D/L against the match completion record.
        raw = read_json(directory / "result.json")
        if row["winner"] != raw["winner"] or row["reason"] != raw["reason"]:
            raise ValueError("Metrics differ from saved match result")
        rows.append(row)
        decisions.extend(read_json(directory / "selection-metrics.json"))
    return rows, decisions


def _sampling_slices(cases, rows):
    slices = {}
    for field in ("map_size", "army_bin", "placement", "learner_side"):
        slices[field] = {
            str(value): wdl([row for case, row in zip(cases, rows, strict=True) if case[field] == value])
            for value in sorted({case[field] for case in cases}, key=str)
        }
    return slices


def _sampling_behavior(decisions):
    behavior = {"free_decisions": len(decisions)}
    for field in (
        "p_max",
        "argmax_selected",
        "normalized_entropy",
        "legal_actions",
    ):
        values = [decision[field] for decision in decisions]
        behavior[field] = (
            {"mean": float(np.mean(values)), "median": float(np.median(values))}
            if values
            else {"mean": None, "median": None}
        )
    return behavior


def _summarize_modes(settings, cases, identity, out):
    modes, outcomes = {}, {}
    wall_times = read_json(out / "run.json").get("mode_wall_seconds", {})
    for setting in settings["modes"]:
        key = mode_key(identity, setting)
        rows, decisions = _load_mode_results(cases, setting, key, out)
        if len(rows) != len(cases):
            write_incomplete_report(out, setting, rows)
            return None
        slices = _sampling_slices(cases, rows)
        behavior = _sampling_behavior(decisions)
        modes[setting["label"]] = {
            **wdl(rows),
            "match_seconds_sum": sum(row["seconds"] for row in rows),
            "wall_seconds": wall_times.get(setting["label"]),
            "behavior": behavior,
            "slices": slices,
            "evaluation_key": key,
            "mode": setting["mode"],
            "temperature": setting["temperature"],
        }
        outcomes[setting["label"]] = paired(cases, rows)
    return modes, outcomes


def _build_sampling_summary(settings, cases, modes, outcomes):
    metadata = {(case["parent_world"], case["case_id"], case["learner_side"]): case for case in cases}
    comparison = paired_world_bootstrap(
        outcomes["sample-t1"],
        outcomes["sample-t0.5"],
        samples=settings["bootstrap_samples"],
        seed=settings["bootstrap_seed"],
        clusters=16,
        cases=metadata,
        cases_per_world=8,
    )
    result = {
        "status": "complete",
        "modes": modes,
        "T05_minus_T1": comparison,
        "versus_argmax": {
            label: modes[label]["win_rate"] - modes["argmax"]["win_rate"]
            for label in modes
            if label != "argmax"
        },
        "T05_at_least_13_wins": modes["sample-t0.5"]["W"] >= 13,
        "positive_lower_CI": comparison["CI95"][0] > 0,
        "paired_case_hashes_verified": True,
        "clusters": 16,
        "cases_per_world": 8,
    }
    return result


def report(settings, cases, identity, out):
    results = _summarize_modes(settings, cases, identity, out)
    if results is None:
        return False
    modes, outcomes = results
    result = _build_sampling_summary(settings, cases, modes, outcomes)
    write_json(out / "summary.json", result)
    render_sampling_report(out, result, workers=settings["workers"])
    return True


def run(settings, phase="run"):
    if phase not in ("run", "report", "check"):
        raise ValueError(f"Unsupported sampling phase: {phase}")
    config, cases, identity = inputs(settings)
    out = ROOT / settings["output"]
    out.mkdir(parents=True, exist_ok=True)
    with RunLock(out):
        if phase == "check":
            preflight(settings, out)
            print(
                "Preflight passed: three saved BC logits, exact T=1 choices; no matches run.",
                flush=True,
            )
            return True
        if (out / "run.json").exists():
            record = read_json(out / "run.json")
            if record["identity"] != identity or record["settings"] != settings:
                raise ValueError("Existing evaluation has different inputs or evaluator; use a new output")
        else:
            record = {
                "identity": identity,
                "settings": settings,
                "created_utc": datetime.now(UTC).isoformat(),
                "case_seeds": [
                    {
                        "case_hash": c["case_hash"],
                        "battle": c["scenario"]["seed"],
                        "agent_seeds": c["agent_seeds"],
                    }
                    for c in cases
                ],
            }
            write_json(out / "run.json", record)
            write_json(out / "cases.json", cases)
        if phase == "report":
            return report(settings, cases, identity, out)
        with record_experiment_errors(out, phase=None):
            preflight(settings, out)
            write_json(out / "status.json", {"status": "running"})
            started = perf_counter()
            new_matches = 0
            pending = {}
            for setting in settings["modes"]:
                key = mode_key(identity, setting)
                pending[setting["label"]] = [
                    SamplingMatchTask(
                        config,
                        case,
                        str(ROOT / settings["checkpoint"]),
                        setting,
                        str(case_directory(out, setting, case)),
                        key,
                    )
                    for case in cases
                    if not completed_case(case_directory(out, setting, case), case, key)
                ]
            remaining = sum(map(len, pending.values()))
            with pool(settings["workers"]) as executor:
                for setting in settings["modes"]:
                    tasks = pending[setting["label"]]
                    tick = perf_counter()
                    total_tasks = len(tasks)
                    previous = 0

                    def report_progress(completed, label=setting["label"], total=total_tasks):
                        nonlocal new_matches, remaining, previous
                        completed_in_chunk = completed - previous
                        previous = completed
                        new_matches += completed_in_chunk
                        remaining -= completed_in_chunk
                        elapsed = perf_counter() - started
                        print(
                            f"{label}: {completed}/{total} new matches; elapsed={elapsed:.1f}s; ETA~{elapsed / new_matches * remaining:.0f}s",
                            flush=True,
                        )

                    run_chunked_pool(
                        executor, match_task, tasks, settings["workers"], on_chunk=report_progress
                    )
                    if tasks:
                        record.setdefault("mode_wall_seconds", {})[setting["label"]] = perf_counter() - tick
                    write_json(out / "run.json", record)
            if (
                execution_identity() != identity["execution"]
                or file_sha256(ROOT / settings["checkpoint"]) != identity["checkpoint_sha256"]
                or file_sha256(ROOT / settings["manifest"]) != identity["manifest_sha256"]
            ):
                raise ValueError("Evaluator or parent changed during evaluation")
            if not report(settings, cases, identity, out):
                raise ValueError("Incomplete comparison")
            record["finished_utc"] = datetime.now(UTC).isoformat()
            write_json(out / "run.json", record)
            write_json(out / "status.json", {"status": "complete", "new_matches": new_matches})
            print((out / "report.md").read_text(encoding="utf-8"), flush=True)
            return True
