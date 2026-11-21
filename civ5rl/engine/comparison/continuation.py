"""Explicit continuation of a completed head fit, preserving its optimizer clock.

The predecessor is frozen evidence, not an executable resume target. Its verified
state is imported once under the new passport; all cumulative counters, shuffle
order, diagnostics and RNG survive. Derived caches can be discarded and rebuilt.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import torch

from civ5rl.provenance.checkpoint import verify_frozen_passport
from civ5rl.provenance.common import ROOT, file_sha256, read_json, write_json

from civ5rl.deploy.archive import _inventory
from civ5rl.engine.evaluation.executor import (
    EvaluationRuntime,
    evaluation_key,
    evaluation_precision,
)
from civ5rl.engine.evaluation.storage import readable_evaluations
from civ5rl.engine.runtime.control import RunLock


PINNED = (
    "candidate.pt",
    "training.json",
    "status.json",
    "training.pt",
    "resolved-config.json",
    "dataset.json",
    "phase-budgets/training.json",
    "manifest.json",
    "match-config.json",
)


def continuation_seconds(cfg):
    spec = cfg.get("fit_continuation")
    if not spec:
        return 0.0
    source = ROOT / spec["run"]
    return max(
        read_json(source / "phase-budgets/training.json")["seconds"],
        read_json(source / "training.json")["seconds"],
    )


def validate_continuation(cfg):
    spec = cfg["fit_continuation"]
    source = ROOT / spec["run"]
    if source.resolve() == (ROOT / cfg["output"]).resolve():
        raise ValueError("Use a separate continuation directory")
    if not set(PINNED).issubset(spec["sha256"]):
        raise ValueError("Incomplete fit continuation lineage")
    for name in PINNED:
        sha = spec["sha256"][name]
        if file_sha256(source / name) != sha:
            raise ValueError(f"Fit continuation source changed: {name}")
    verify_frozen_passport(source, sources=False)
    predecessor = read_json(source / "resolved-config.json")
    _verify_continuation_options(cfg, predecessor)
    state = torch.load(source / "training.pt", map_location="cpu", weights_only=False)
    report = read_json(source / "training.json")
    dataset = read_json(source / "dataset.json")
    _verify_progress(source, state, report, predecessor["training"], len(dataset["anchors"]))
    _verify_inherited_model_and_adam(cfg, source, state)
    spent = max(continuation_seconds(cfg), state["seconds"])
    if not math.isfinite(spent) or not 0 <= spent < cfg["training_seconds"]:
        raise ValueError("No active budget remains for continuation")
    if report["candidate_sha256"] != file_sha256(source / "candidate.pt"):
        raise ValueError("Source candidate report differs")
    return state


def _verify_continuation_options(config, predecessor):
    training = config["training"]
    old_training = predecessor["training"]
    expected = {
        **predecessor,
        "output": config["output"],
        "fit_continuation": config["fit_continuation"],
        "training_seconds": config["training_seconds"],
        "training": {
            **old_training,
            "max_accepted_steps": training["max_accepted_steps"],
            "cycle_anchors": True,
        },
    }
    if (
        config != expected
        or old_training["mode"] != "frozen_heads_step_kl"
        or training["max_accepted_steps"] <= old_training["max_accepted_steps"]
    ):
        raise ValueError("Continuation must extend a frozen-head fit under the same scientific settings")


def _verify_progress(source, state, report, options, anchor_count):
    steps = options["max_accepted_steps"]
    history = state["history"]
    if (
        read_json(source / "status.json")["status"] != "complete"
        or report["status"] != "complete"
        or state["stop_reason"] != "accepted_step_limit"
        or state["steps"] != steps
        or state["attempts"] != len(history)
        or sum(attempt["accepted"] for attempt in history) != steps
        or state["anchor_cursor"] != steps * options["anchors_per_step"]
        or state["pending"] is not None
    ):
        raise ValueError("Expected a verified predecessor at its accepted-step limit")
    if sorted(state["anchor_order"]) != list(range(anchor_count)):
        raise ValueError("Invalid inherited anchor permutation")


def _verify_inherited_model_and_adam(cfg, source, state):
    candidate = torch.load(source / "candidate.pt", map_location="cpu", weights_only=False)["model"]
    parent = torch.load(ROOT / cfg["parents"][0], map_location="cpu", weights_only=False)["model"]
    model = state["model"]
    if model.keys() != candidate.keys() or any(
        not torch.equal(value, candidate[name]) for name, value in model.items()
    ):
        raise ValueError("Source candidate differs from its accepted checkpoint")
    if model.keys() != parent.keys() or any(
        not torch.equal(value, parent[name])
        for name, value in model.items()
        if not name.startswith(("action.", "attack_preview."))
    ):
        raise ValueError("Source fit changed frozen representations")
    trainable = [value for name, value in model.items() if name.startswith(("action.", "attack_preview."))]
    groups = state["optimizer"]["param_groups"]
    adam_state = state["optimizer"]["state"]
    if len(groups) != 1 or groups[0]["lr"] != state["lr"] or not trainable:
        raise ValueError("Invalid inherited head Adam groups")
    parameter_ids = groups[0]["params"]
    if len(parameter_ids) != len(trainable) or set(parameter_ids) != adam_state.keys():
        raise ValueError("Invalid inherited head Adam parameters")
    for parameter_id, parameter in zip(parameter_ids, trainable, strict=True):
        moments = adam_state[parameter_id]
        if (
            int(moments["step"]) != state["steps"]
            or moments["exp_avg"].shape != parameter.shape
            or moments["exp_avg_sq"].shape != parameter.shape
        ):
            raise ValueError("Invalid inherited head Adam moments or clock")


def inherited_state(cfg, identity):
    state = validate_continuation(cfg)
    if state["identity"]["dataset_hash"] != identity["dataset_hash"]:
        raise ValueError("Continuation dataset changed")
    state.update(
        identity=identity,
        stop_reason=None,
        seconds=continuation_seconds(cfg),
        inherited_seconds=continuation_seconds(cfg),
        inherited_steps=state["steps"],
    )
    return state


def reuse_parent_development(cfg, out, config, plan):
    """Import only a cache entry whose ordinary evaluator key still matches.

    The key includes policy, cases/seeds, runtime/GPU, selection and evaluator
    implementation/binaries. No compatible key means an explicit error, never
    a fallback GPU evaluation. The original archive is restored after reading.
    """

    source = ROOT / cfg["fit_continuation"]["run"]
    out = Path(out)
    if read_json(source / "match-config.json") != config or read_json(source / "manifest.json") != plan:
        raise ValueError("Parent evaluation cases/rules changed")
    with evaluation_precision():
        runtime = EvaluationRuntime("cuda", 8, 32, 0.001).resolve().identity()
        _, key = evaluation_key(config, plan, out, "checkpoints/parent.pt", "development_512", runtime)
    saved = read_json(source / "evaluations.json")
    if key not in saved:
        raise ValueError("Parent Development evaluator/runtime/GPU is incompatible")
    parent = saved[key]
    if parent["checkpoint"] != "checkpoints/parent.pt" or parent["runtime"] != runtime:
        raise ValueError("Parent evaluation policy/runtime changed")
    directory = Path(parent["runs"]["argmax"]["cp"]["directory"])
    if directory.is_absolute() or ".." in directory.parts or directory.parts[0] != "evaluation":
        raise ValueError("Unsafe parent evaluation directory")
    receipt = out / "reused-parent.json"
    if not receipt.exists():
        with RunLock(source), readable_evaluations(source):
            shutil.copytree(source / directory, out / directory, dirs_exist_ok=True)
            if _inventory(source / directory) != _inventory(out / directory):
                raise ValueError("Reused parent evaluation inventory differs")
        write_json(receipt, dict(source=cfg["fit_continuation"], cache_key=key))
    elif read_json(receipt) != dict(source=cfg["fit_continuation"], cache_key=key):
        raise ValueError("Parent evaluation reuse identity changed")
    cache = read_json(out / "evaluations.json") if (out / "evaluations.json").exists() else {}
    cache[key] = parent
    write_json(out / "evaluations.json", cache)
    return parent
