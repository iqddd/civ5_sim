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

from civ5rl.engine.comparison.heads import MODE
from civ5rl.engine.comparison.policy import TRAINABLE_HEADS
from civ5rl.engine.evaluation.executor import EvaluationRuntime, evaluation_key, evaluation_precision
from civ5rl.engine.evaluation.storage import readable_evaluations
from civ5rl.engine.runtime.archive import inventory
from civ5rl.engine.runtime.control import RunLock
from civ5rl.provenance.checkpoint import verify_frozen_passport
from civ5rl.provenance.common import ROOT, file_sha256, read_json, write_json
from civ5sim.engine import digest


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
    """Verify frozen inputs and the scientific state needed to keep fitting."""
    spec = cfg["fit_continuation"]
    source = ROOT / spec["run"]
    if source.resolve() == (ROOT / cfg["output"]).resolve():
        raise ValueError("Use a separate continuation directory")
    _verify_source_files(
        source,
        spec["sha256"],
        "training.pt",
        "candidate.pt",
        "training.json",
        "status.json",
        "resolved-config.json",
        "dataset.json",
        "phase-budgets/training.json",
    )
    passport = verify_frozen_passport(source, sources=False)
    predecessor = read_json(source / "resolved-config.json")
    state = torch.load(source / "training.pt", map_location="cpu", weights_only=False)
    report = read_json(source / "training.json")
    dataset = read_json(source / "dataset.json")
    identity = state["identity"]
    if (
        identity["dataset_hash"] != digest(dataset)
        or identity["dataset_hash"] != _input_dataset_hash(cfg)
        or identity["passport"]["corpus_hash"] != passport["corpus_hash"]
        or identity["passport"].get("config_hash") != passport.get("config_hash")
    ):
        raise ValueError("Continuation dataset or predecessor identity changed")
    _verify_continuation_options(cfg, predecessor, len(dataset["anchors"]))
    _verify_progress(source, state, report, predecessor["training"], len(dataset["anchors"]))
    _verify_inherited_model_and_adam(cfg, source, state)
    spent = max(continuation_seconds(cfg), state["seconds"])
    if not math.isfinite(spent) or not 0 <= spent < cfg["training_seconds"]:
        raise ValueError("No active budget remains for continuation")
    if report["candidate_sha256"] != file_sha256(source / "candidate.pt"):
        raise ValueError("Source candidate report differs")
    return state


def _verify_source_files(source, hashes, *names):
    for name in names:
        if name not in hashes:
            raise ValueError(f"Missing fit continuation input hash: {name}")
        if file_sha256(source / name) != hashes[name]:
            raise ValueError(f"Fit continuation source changed: {name}")


def _input_dataset_hash(config):
    source = config["collection_input"]
    path = ROOT / source["run"]
    _verify_source_files(path, source["sha256"], "dataset.json")
    return digest(read_json(path / "dataset.json"))


def _verify_continuation_options(config, predecessor, anchor_count):
    training = config["training"]
    old_training = predecessor["training"]
    if (
        config["kind"] != "rl01.rollout_comparison"
        or predecessor["kind"] != config["kind"]
        or config["evaluation_protocol"] != "paired_development_only"
        or predecessor["evaluation_protocol"] != config["evaluation_protocol"]
        or old_training["mode"] != MODE
        or training["mode"] != MODE
        or predecessor["parent_sha256"] != config["parent_sha256"]
        or training["max_accepted_steps"] <= old_training["max_accepted_steps"]
        or any(
            training[key] != old_training[key]
            for key in ("anchors_per_step", "initial_lr", "kl_limit", "max_retries")
        )
    ):
        raise ValueError("Continuation must extend a frozen-head fit under the same scientific settings")
    required = training["anchors_per_step"] * (
        1 if training.get("cycle_anchors") else training["max_accepted_steps"]
    )
    if anchor_count < required:
        raise ValueError("Anchor pool does not cover the continuation step limit")


def _verify_progress(source, state, report, options, anchor_count):
    steps = options["max_accepted_steps"]
    history = state["history"]
    if (
        read_json(source / "status.json")["status"] != "complete"
        or report["status"] != "complete"
        or state["mode"] != MODE
        or state["stop_reason"] != "accepted_step_limit"
        or state["steps"] != steps
        or report["training_steps"] != steps
        or state["attempts"] != len(history)
        or sum(attempt["accepted"] for attempt in history) != steps
        or state["anchor_cursor"] != steps * options["anchors_per_step"]
        or state["pending"] is not None
    ):
        raise ValueError("Expected a verified predecessor at its accepted-step limit")
    if sorted(state["anchor_order"]) != list(range(anchor_count)):
        raise ValueError("Invalid inherited anchor permutation")


def _verify_inherited_model_and_adam(cfg, source, state):
    candidate = torch.load(source / "candidate.pt", map_location="cpu", weights_only=False)
    parent_path = ROOT / cfg["parents"][0]
    if file_sha256(parent_path) != cfg["parent_sha256"][0]:
        raise ValueError("Fit continuation parent checkpoint changed")
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    model = state["model"]
    if candidate["variant"] != parent["variant"] or parent["variant"] != "C2P":
        raise ValueError("Incompatible fit continuation policy variant")
    reference = parent["model"]
    candidate_model = candidate["model"]
    if model.keys() != candidate_model.keys() or any(
        value.dtype != candidate_model[name].dtype or not torch.equal(value, candidate_model[name])
        for name, value in model.items()
    ):
        raise ValueError("Source candidate differs from its accepted checkpoint")
    if model.keys() != reference.keys():
        raise ValueError("Source fit changed model parameters")
    trainable = []
    for name, value in reference.items():
        inherited = model[name]
        if inherited.shape != value.shape or inherited.dtype != value.dtype:
            raise ValueError(f"Incompatible inherited model parameter: {name}")
        if name.split(".", 1)[0] in TRAINABLE_HEADS:
            trainable.append((name, inherited))
        elif not torch.equal(inherited, value):
            raise ValueError("Source fit changed frozen representations")

    groups = state["optimizer"]["param_groups"]
    adam_state = state["optimizer"]["state"]
    if len(groups) != 1 or groups[0]["lr"] != state["lr"] or not trainable:
        raise ValueError("Invalid inherited head Adam groups")
    group = groups[0]
    names = [name for name, _ in trainable]
    parameter_ids = group["params"]
    if (
        group.get("param_names", names) != names
        or len(parameter_ids) != len(trainable)
        or len(set(parameter_ids)) != len(trainable)
        or set(parameter_ids) != adam_state.keys()
    ):
        raise ValueError("Invalid inherited head Adam parameters")
    for parameter_id, (name, parameter) in zip(parameter_ids, trainable, strict=True):
        moments = adam_state[parameter_id]
        if float(moments["step"]) != state["steps"] or any(
            moments[key].shape != parameter.shape or moments[key].dtype != parameter.dtype
            for key in ("exp_avg", "exp_avg_sq")
        ):
            raise ValueError(f"Invalid inherited head Adam moments or clock: {name}")


def inherited_state(cfg, identity):
    state = validate_continuation(cfg)
    if state["identity"]["dataset_hash"] != identity["dataset_hash"]:
        raise ValueError("Continuation dataset changed")
    spent = max(continuation_seconds(cfg), state["seconds"])
    state.update(
        identity=identity,
        stop_reason=None,
        seconds=spent,
        inherited_seconds=spent,
        inherited_steps=state["steps"],
    )
    return state


def reuse_parent_development(cfg, out, config, plan):
    """Import only a cache entry whose ordinary evaluator key still matches.

    The key includes policy, cases/seeds, runtime/GPU, selection and evaluator
    implementation/binaries. No compatible key means an explicit error, never
    a fallback GPU evaluation. The original archive is restored after reading.
    """

    spec = cfg["fit_continuation"]
    source = ROOT / spec["run"]
    out = Path(out)
    _verify_source_files(source, spec["sha256"], "evaluations.json")
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
            if inventory(source / directory) != inventory(out / directory):
                raise ValueError("Reused parent evaluation inventory differs")
        write_json(receipt, {"source": cfg["fit_continuation"], "cache_key": key})
    elif read_json(receipt) != {"source": cfg["fit_continuation"], "cache_key": key}:
        raise ValueError("Parent evaluation reuse identity changed")
    cache = read_json(out / "evaluations.json") if (out / "evaluations.json").exists() else {}
    cache[key] = parent
    write_json(out / "evaluations.json", cache)
    return parent
