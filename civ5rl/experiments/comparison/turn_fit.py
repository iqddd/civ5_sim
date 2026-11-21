"""Prepare and fit a policy from complete, recorded own-turn comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from civ5rl.engine.comparison.turn_fit import TurnTrainingState, publish, train
from civ5rl.engine.comparison.turn_inputs import TurnHistories
from civ5rl.engine.evaluation.storage import archive_evaluations
from civ5rl.engine.runtime import is_int_at_least
from civ5rl.engine.runtime.budget import PhaseBudget
from civ5rl.engine.runtime.control import RunLock
from civ5rl.engine.runtime.telemetry import ResourceMonitor
from civ5rl.engine.runtime.transactions import atomic_torch_save
from civ5rl.experiments.comparison.data import validate_fit_options
from civ5rl.experiments.comparison.turn_data import (
    build_dataset,
    history_inputs,
    verify_inputs,
)
from civ5rl.experiments.evaluation.paired import (
    evaluate_paired_dev512,
    prepare_paired_evaluation_parent,
)
from civ5rl.experiments.ppo.target_scale import evaluate as evaluator
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import ROOT, configure, file_sha256, read_json, write_json
from civ5rl.provenance.identity import policy_hash
from civ5sim.engine import digest

KIND = "rl01.turn_comparison_fit"


def load_config(path):
    cfg = read_json(path)
    if cfg.get("kind") != KIND:
        raise ValueError("Expected whole-turn fit config")
    if not is_int_at_least(cfg.get("seed"), 0):
        raise ValueError("seed must be a nonnegative integer")
    if not is_int_at_least(cfg.get("training_seconds")):
        raise ValueError("training_seconds must be a positive integer")
    if len(cfg["parents"]) != 1 or len(cfg["parent_sha256"]) != 1:
        raise ValueError("Whole-turn fitting requires one parent and its hash")
    training = cfg["training"]
    validate_fit_options(training)
    for key in ("positions_per_step", "max_epochs"):
        if not is_int_at_least(training.get(key)):
            raise ValueError(f"training.{key} must be a positive integer")
    return cfg


def prepare(cfg, out):
    verify_inputs(cfg)
    if file_sha256(ROOT / cfg["parents"][0]) != cfg["parent_sha256"][0]:
        raise ValueError("Parent changed")
    lineage = {
        "input_sha256": cfg["input_sha256"],
        "parent_sha256": cfg["parent_sha256"],
    }
    if (out / "passport.json").exists():
        checkpoint.verify_execution_passport(out, cfg, lineage)
    else:
        identity = checkpoint.passport(out, cfg, lineage)
        write_json(out / "run.json", {"kind": KIND, "config": cfg, "identity": identity})
    if not (out / "dataset.json").exists():
        dataset = build_dataset(cfg)
        write_json(out / "dataset.json", dataset)
    else:
        dataset = read_json(out / "dataset.json")
    if dataset["input_sha256"] != cfg["input_sha256"]:
        raise ValueError("Prepared input changed")
    if cfg["training"]["positions_per_step"] > len(dataset["positions"]):
        raise ValueError("training.positions_per_step exceeds the number of prepared positions")
    if not (out / "training.json").exists():
        write_json(out / "status.json", {"status": "prepared", "phase": "training"})
    return dataset


def _write_identical_fit_evaluation(out: Path) -> None:
    skipped = {"status": "skipped_identical_policy"}
    write_json(
        out / "result.json",
        {
            "status": "complete",
            "development": skipped,
            "parent_development": skipped,
            "comparison": None,
            "candidate_selection": "last_accepted",
            "additional": {"status": "disabled_by_protocol"},
        },
    )
    write_json(
        out / "status.json",
        {
            "status": "complete",
            "phase": "complete",
            "development": skipped["status"],
        },
    )


def evaluate(cfg, out):
    """Evaluate Development explicitly, independently of the fit loop."""
    training = read_json(out / "training.json")
    if training["status"] != "complete":
        raise ValueError("Finish fit before evaluation")
    candidate = out / "candidate.pt"
    if file_sha256(candidate) != training["candidate_sha256"]:
        raise ValueError("Candidate changed")
    original = ROOT / cfg["parents"][0]
    if file_sha256(original) != cfg["parent_sha256"][0]:
        raise ValueError("Parent changed")
    candidate_hash = policy_hash(torch.load(candidate, map_location="cpu", weights_only=False)["model"])
    parent_hash = policy_hash(torch.load(original, map_location="cpu", weights_only=False)["model"])
    if candidate_hash == parent_hash:
        _write_identical_fit_evaluation(out)
        return
    prepare_paired_evaluation_parent(out, original, cfg["parent_sha256"][0], error_message="Parent changed")
    source = ROOT / cfg["source_run"]
    config = read_json(source / "run.json")["config"]
    plan = {"development_512": read_json(source / "manifest.json")["development_512"]}
    with RunLock(out):
        write_json(out / "status.json", {"status": "evaluating", "phase": "development_512"})
        result = evaluate_paired_dev512(cfg, out, config, plan, evaluator, parent_prepared=True)
        write_json(out / "result.json", result)
    archive_evaluations(out)
    with RunLock(out):
        write_json(out / "status.json", {"status": "complete", "phase": "complete"})


@dataclass(frozen=True)
class _TurnFitRuntime:
    microbatch: int
    compile_model: bool

    def record(self) -> dict[str, Any]:
        return {
            "history_microbatch": self.microbatch,
            "compile": self.compile_model,
            "precision": "deterministic_cuda_fp32_no_amp_no_tf32",
        }


def _record_training_runtime(out: Path, settings: _TurnFitRuntime) -> None:
    runtime_path = out / "training-runtime.json"
    runtime = settings.record()
    if runtime_path.exists() and read_json(runtime_path) != runtime:
        raise ValueError("Resume runtime changed")
    write_json(runtime_path, runtime)


def _publish_fit_timeout(cfg, out, dataset, budget):
    identity = {"dataset": digest(dataset), "config": digest(cfg)}
    checkpoint_path = out / "training.pt"
    payload = (
        torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint_path.exists()
        else TurnTrainingState().to_checkpoint_dict()
    )
    payload.update({"stop_reason": "time_limit", "seconds": budget.elapsed(), "identity": identity})
    publish(cfg, out, payload, identity)
    atomic_torch_save(payload, checkpoint_path)


def _run_fit_training(cfg, out, dataset, budget, settings: _TurnFitRuntime):
    write_json(out / "status.json", {"status": "training", "phase": "preparation"})
    with ResourceMonitor(out / "telemetry") as monitor:
        monitor.mark("preparation")
        try:
            with history_inputs(cfg, dataset) as (sources, roots, branches):
                histories = TurnHistories(out, sources, roots, branches, dataset["anchors"])
                train(
                    cfg,
                    out,
                    dataset,
                    histories,
                    budget,
                    microbatch=settings.microbatch,
                    compile_model=settings.compile_model,
                    monitor=monitor,
                )
        except TimeoutError:
            _publish_fit_timeout(cfg, out, dataset, budget)


def _finish_fit_accounting(cfg, out, budget):
    # Refresh accounting after extraction cleanup and the final checkpoint.
    report = read_json(out / "training.json")
    report["seconds"] = budget.elapsed()
    report["overshoot_seconds"] = max(0.0, report["seconds"] - cfg["training_seconds"])
    write_json(out / "training.json", report)
    write_json(
        out / "status.json",
        {"status": "complete", "phase": "fit_complete", "development": "not_requested"},
    )


def run(phase, cfg, out, *, microbatch=2, compile_model=True, preparation_seconds=0.0):
    out = Path(out).resolve()
    configure(cfg["seed"], "cuda")
    if phase == "evaluate":
        return evaluate(cfg, out)
    with RunLock(out):
        # Preparation shares the training budget, including explicit prepare.
        with PhaseBudget(
            out,
            "training",
            cfg["training_seconds"],
            initial_seconds=preparation_seconds,
        ) as budget:
            dataset = prepare(cfg, out)
            if phase == "prepare":
                return
            settings = _TurnFitRuntime(microbatch, compile_model)
            _record_training_runtime(out, settings)
            _run_fit_training(cfg, out, dataset, budget, settings)
        _finish_fit_accounting(cfg, out, budget)
