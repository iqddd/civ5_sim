"""Orchestrate bounded rollout-comparison fitting and checkpoint publication."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from time import perf_counter

import torch

from civ5rl.engine.comparison.continuation import inherited_state
from civ5rl.engine.comparison.control import ComparisonTrainingState, initial_state, run_updates
from civ5rl.engine.comparison.heads import HeadFeatures
from civ5rl.engine.comparison.inputs import Histories
from civ5rl.engine.comparison.objectives import (
    batched_target_logits,
    categorical_kl,
    control_runner,
    full_kl,
    full_kl_rows,
    logical_step,
    loss_runner,
    pair_loss,
    preference_and_kl,
    prefix_states,
)
from civ5rl.engine.comparison.policy import (
    ATTENTION_BACKEND,
    TRAINABLE_HEADS,
    configure_heads,
    configure_policy,
    head_parameter_count,
)
from civ5rl.engine.comparison.reference import (
    HISTORY_BATCH,
    DeviceReferences,
    encode_histories,
    prepare_references,
    reference_distributions,
    reference_logits_batches,
    runtime_references,
    target_logits,
)
from civ5rl.engine.comparison.storage import archive_tree, restore_tree
from civ5rl.engine.runtime import is_int_at_least
from civ5rl.engine.runtime.budget import PhaseBudget
from civ5rl.engine.runtime.transactions import atomic_torch_save
from civ5rl.provenance.common import ROOT, atomic_write_jsonl, file_sha256, read_json, seed_for, write_json
from civ5rl.provenance.rng import capture_rng, restore_rng
from civ5sim.engine import digest

# Numerical helpers exposed here for fit diagnostics and benchmark callers.
_encode_histories = encode_histories
__all__ = [
    "ATTENTION_BACKEND",
    "HISTORY_BATCH",
    "DeviceReferences",
    "Histories",
    "_encode_histories",
    "batched_target_logits",
    "categorical_kl",
    "configure_policy",
    "control_runner",
    "epoch_fit_options",
    "fit",
    "fit_full_corpus",
    "fit_identity",
    "fit_with_storage",
    "full_kl",
    "full_kl_rows",
    "logical_step",
    "loss_runner",
    "pair_loss",
    "preference_and_kl",
    "prefix_states",
    "prepare_fit",
    "prepare_histories",
    "prepare_references",
    "publish_candidate",
    "reference_distributions",
    "reference_logits_batches",
    "runtime_references",
    "target_logits",
    "training_runtime",
]


def epoch_fit_options(cfg):
    """Resolve the bounded rollout fitter's logical step and control settings."""
    training = cfg.get("training", {})
    options = {
        key: training.get(key, default)
        for key, default in {
            "max_epochs": 2,
            "pairs_per_step": 32,
            "initial_lr": 1e-5,
            "kl_limit": 0.015,
        }.items()
    }
    for key in ("max_epochs", "pairs_per_step"):
        if not is_int_at_least(options[key]):
            raise ValueError(f"training.{key} must be a positive integer")
    for key in ("initial_lr", "kl_limit"):
        value = options[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{key} must be positive and finite")
    return options


def training_runtime(cfg=None):
    if cfg and cfg.get("training", {}).get("mode") == "frozen_heads_step_kl":
        return {
            "trainable": list(TRAINABLE_HEADS),
            "trainable_parameters": head_parameter_count(),
            "precision": "deterministic_fp32_no_amp_no_tf32",
            "feature_cache": "frozen_public_histories_fp32",
            "full_kl_backend": "fp32_logsoftmax_fp64_cuda_reduction",
            "compile_mode": "default",
            "cuda_graphs": False,
        }
    return {
        "attention_backend": ATTENTION_BACKEND,
        "history_batch": HISTORY_BATCH,
        "spatial_batch": 32,
        "bptt": 128,
        "compile_mode": "default",
        "cuda_graphs": False,
        "coalesced_spatial_gradients": True,
        "activation_checkpointing": False,
        "prefix_reuse": "per_step_episode",
        "spatial_window_reuse": "per_microbatch_union",
        "input_storage": "verified_mmap_with_bounded_spatial_residency",
        "loss_backend": "batched_compiled",
        "full_kl_backend": "fp32_logsoftmax_fp64_cuda_reduction",
        "precision": "mixed_bf16_spatial_fp32_gru_heads_loss",
    }


def fit(cfg, out, dataset, identity, budget=None):
    if budget is not None:
        return fit_with_storage(cfg, out, dataset, identity, budget)
    with PhaseBudget(out, "training", cfg["training_seconds"]) as budget:
        return fit_with_storage(cfg, out, dataset, identity, budget)


def fit_with_storage(cfg, out, dataset, identity, budget):
    out = Path(out)
    state = read_json(out / "training.json") if (out / "training.json").exists() else {}
    if state.get("status") in ("complete", "no_accepted_pairs"):
        return _fit(cfg, out, dataset, identity, budget)
    result = _fit(cfg, out, dataset, identity, budget)
    if (out / "sources").exists():
        archive_tree(out, "sources")
    state = read_json(out / "training.json")
    state.update(
        seconds=budget.elapsed(),
        overshoot_seconds=max(0, budget.elapsed() - cfg["training_seconds"]),
    )
    write_json(out / "training.json", state)
    return result


def publish_candidate(cfg, out, state, identity):
    original = torch.load(cfg["parents"][0], map_location="cpu", weights_only=False)
    changed = any(
        not torch.equal(value.cpu(), original["model"][name]) for name, value in state["model"].items()
    )
    atomic_torch_save(
        {
            "variant": "C2P",
            "model": state["model"],
            "optimizer": state["optimizer"],
            "state": {k: v for k, v in state.items() if k not in ("model", "optimizer")},
            "config": original["config"],
            "identity": identity,
        },
        out / "candidate.pt",
    )
    report = {
        "status": "complete",
        "training_steps": state["steps"],
        "epoch": state["epoch"],
        "cursor": state["cursor"],
        "seconds": state["seconds"],
        "overshoot_seconds": max(0, state["seconds"] - cfg["training_seconds"]),
        "stop_reason": state["stop_reason"],
        "changed": changed,
        "attempted_final_kl": state.get("attempted_final_kl", state.get("last_full_kl")),
        "input_cache": state.get("input_cache"),
        "candidate_sha256": file_sha256(out / "candidate.pt"),
    }
    if state.get("mode") in ("full_corpus_step_kl", "frozen_heads_step_kl"):
        report.update(
            {
                k: state[k]
                for k in (
                    "mode",
                    "attempts",
                    "rejected_attempts",
                    "lr",
                    "anchor_cursor",
                    "last_full_kl",
                )
            }
        )
        for key in (
            "learning",
            "baseline_learning",
            "feature_parity",
            "inherited_steps",
            "inherited_seconds",
        ):
            if key in state:
                report[key] = state[key]
    write_json(out / "training.json", report)
    return out / "candidate.pt" if changed else None


def fit_identity(passport, dataset):
    return {"passport": passport, "dataset_hash": digest(dataset)}


def prepare_fit(cfg, out, state, identity, *, empty_dataset=None):
    """Validate saved state and publish or verify a completed candidate."""
    if state.get("identity", identity) != identity:
        raise ValueError("Training dataset/identity changed")
    if state.get("stop_reason"):
        report_path, candidate = out / "training.json", out / "candidate.pt"
        report = read_json(report_path) if report_path.exists() else {}
        if candidate.exists() and report.get("status") == "complete":
            if file_sha256(candidate) != report["candidate_sha256"]:
                raise ValueError("Candidate checkpoint changed")
            return True, candidate if report["changed"] else None
        return True, publish_candidate(cfg, out, state, identity)
    if empty_dataset is not None and not empty_dataset["pairs"]:
        write_json(
            out / "training.json",
            dict(
                status="no_accepted_pairs",
                training_steps=0,
                **{
                    k: empty_dataset.get(k)
                    for k in (
                        "selected_forks",
                        "completed_forks",
                        "accepted_pairs",
                        "rejections",
                    )
                },
            ),
        )
        return True, None
    restore_tree(out, "sources")
    return False, None


def prepare_histories(out, dataset, policy, budget, *, cache_dir=None):
    pairs, anchors = dataset["pairs"], dataset["anchors"]
    keys = [(p["episode"], p["index"]) for p in pairs] + [tuple(k) for k in anchors]
    limits = {}
    for episode, index in keys:
        limits[episode] = max(limits.get(episode, 0), index + 1)
    histories = Histories(
        out,
        source_hashes=dataset.get("source_hashes"),
        limits=limits,
        cache_dir=cache_dir,
    )
    histories.prepare(keys, device=next(policy.parameters()).device, budget=budget)
    return histories


def fit_full_corpus(cfg, out, dataset, identity, budget):
    out = Path(out)
    options = cfg["training"]
    anchors, pairs = dataset["anchors"], dataset["pairs"]
    if len({tuple(key) for key in anchors}) != len(anchors):
        raise ValueError("Full-corpus control requires distinct anchors")
    required = options["anchors_per_step"] * (
        1 if options.get("cycle_anchors") else options["max_accepted_steps"]
    )
    if len(anchors) < required:
        raise ValueError("Anchor pool does not cover the accepted-step limit")
    identity = fit_identity(identity, dataset)
    path = out / "training.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
    elif cfg.get("fit_continuation"):
        state = inherited_state(cfg, identity)
        atomic_torch_save(state, path)
        write_json(out / "fit-lineage.json", cfg["fit_continuation"])
    else:
        state = initial_state(
            anchors,
            options,
            seed_for(cfg["seed"], 0, "anchors", namespace=cfg["namespace"]),
        )
    state = (
        state
        if isinstance(state, ComparisonTrainingState)
        else ComparisonTrainingState.from_checkpoint_dict(state)
    )
    done, result = prepare_fit(cfg, out, state.to_checkpoint_dict(), identity)
    if done:
        return result
    head_only = options["mode"] == "frozen_heads_step_kl"
    if head_only:
        policy = configure_heads(cfg["parents"][0])
    else:
        policy = configure_policy(cfg["parents"][0])
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad],
        lr=options["initial_lr"],
        betas=(0.9, 0.999),
        eps=1e-5,
        weight_decay=0,
    )
    cache_dir = None
    if cfg.get("fit_continuation"):
        previous = Histories(ROOT / cfg["fit_continuation"]["run"])
        if previous.directory.exists():
            cache_dir = previous.directory
    histories = prepare_histories(out, dataset, policy, budget, cache_dir=cache_dir)
    features = None
    if head_only:
        features = HeadFeatures(policy, histories, dataset)
        try:
            features.prepare(cfg["parents"][0], budget=budget)
            _, baseline = features.measure()
            state.metadata.setdefault("baseline_learning", baseline)
            state.metadata.setdefault("learning", baseline)
            state.metadata["feature_parity"] = features.parity
        except TimeoutError:
            if not budget.expired():
                raise
        state.mode = options["mode"]
    else:
        references = prepare_references(policy, histories, anchors, out / "references.pt", budget=budget)
    if "model" in state.metadata:
        policy.load_state_dict(state.metadata["model"])
        optimizer.load_state_dict(state.metadata["optimizer"])
        restore_rng(state.metadata["rng"])

    def save():
        state.metadata.update(
            model=policy.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=capture_rng(),
            identity=identity,
            input_cache=histories.statistics(),
        )
        if features is not None:
            state.metadata["input_cache"]["head_features"] = features.statistics
        state.seconds = budget.elapsed()
        data = state.to_checkpoint_dict()
        atomic_torch_save(data, path)
        cycle = 0.0
        accepted = []
        for row in state.history:
            cycle += row["seconds"]
            if row["accepted"]:
                accepted.append(dict(row, cycle_seconds=cycle))
                cycle = 0.0
        atomic_write_jsonl(out / "attempts.jsonl", state.history)
        atomic_write_jsonl(out / "steps.jsonl", accepted)
        if features is not None:
            learning_rows = (
                [{"step": 0, "learning": state.metadata["baseline_learning"]}]
                if "baseline_learning" in state.metadata
                else []
            )
            learning_rows += [
                {"step": row["step"], "learning": row["learning"]} for row in state.history if row["accepted"]
            ]
            atomic_write_jsonl(out / "learning.jsonl", learning_rows)
        write_json(
            out / "training.json",
            {
                k: data[k]
                for k in (
                    "mode",
                    "steps",
                    "attempts",
                    "rejected_attempts",
                    "lr",
                    "anchor_cursor",
                    "pending",
                    "last_full_kl",
                    "seconds",
                    "stop_reason",
                    "input_cache",
                )
            }
            | {
                "learning": state.metadata.get("learning"),
                "inherited_steps": state.metadata.get("inherited_steps", 0),
                "inherited_seconds": state.metadata.get("inherited_seconds", 0.0),
            },
        )

    def update(selected):
        if features is not None:
            return features.update(optimizer, pairs, selected)
        return logical_step(policy, optimizer, histories, pairs, selected, references["mixed"])

    run_updates(
        policy,
        optimizer,
        state,
        anchors,
        options,
        budget,
        save,
        update,
        features.measure
        if features is not None
        else lambda: full_kl(policy, histories, anchors, references["fp32"]),
    )
    frozen = torch.load(path, map_location="cpu", weights_only=False)
    return publish_candidate(cfg, out, frozen, identity)


def _fit(cfg, out, dataset, identity, budget):
    if cfg.get("training", {}).get("mode") in (
        "full_corpus_step_kl",
        "frozen_heads_step_kl",
    ):
        return fit_full_corpus(cfg, out, dataset, identity, budget)
    options = epoch_fit_options(cfg)
    out = Path(out)
    state_path = out / "training.pt"
    state = (
        ComparisonTrainingState.from_checkpoint_dict(
            torch.load(state_path, map_location="cpu", weights_only=False)
        )
        if state_path.exists()
        else ComparisonTrainingState(order=[])
    )
    pairs = dataset["pairs"]
    anchors = dataset["anchors"]
    identity = fit_identity(identity, dataset)
    done, result = prepare_fit(cfg, out, state.to_checkpoint_dict(), identity, empty_dataset=dataset)
    if done:
        return result
    policy = configure_policy(cfg["parents"][0])
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad],
        lr=options["initial_lr"],
        betas=(0.9, 0.999),
        eps=1e-5,
        weight_decay=0,
    )
    shuffle = random.Random(cfg["seed"] + 101)
    histories = prepare_histories(out, dataset, policy, budget)
    # Policy still has parent weights. Reuse it for frozen references before
    # installing the student's resumed state; a second CUDA model is unnecessary.
    references = prepare_references(policy, histories, anchors, out / "references.pt", budget=budget)
    if "model" in state.metadata:
        policy.load_state_dict(state.metadata["model"])
        optimizer.load_state_dict(state.metadata["optimizer"])
        shuffle.setstate(state.metadata["shuffle"])
    if "rng" in state.metadata:
        restore_rng(state.metadata["rng"])

    def save(path, *copies):
        state.metadata.update(
            model=policy.state_dict(),
            optimizer=optimizer.state_dict(),
            shuffle=shuffle.getstate(),
            rng=capture_rng(),
            identity=identity,
            input_cache=histories.statistics(),
        )
        state.seconds = budget.elapsed()
        frozen = atomic_torch_save(state.to_checkpoint_dict(), path, *copies)
        write_json(
            out / "training.json",
            {k: frozen[k] for k in ("epoch", "cursor", "steps", "seconds", "input_cache")},
        )
        return frozen

    if not (out / "verified.pt").exists():
        save(out / "verified.pt")
    stop = False
    while state.epoch < options["max_epochs"] and not stop:
        if state.cursor >= len(pairs):
            kl = full_kl(policy, histories, anchors, references["fp32"])
            if kl > options["kl_limit"]:
                stop = True
                break
            state.epoch += 1
            state.cursor = 0
            state.order = []
            state.last_kl = kl
            save(out / "verified.pt", state_path)
            continue
        if budget.expired():
            break
        if not state.order:
            state.order = list(range(len(pairs)))
            shuffle.shuffle(state.order)
        selected = state.order[state.cursor : state.cursor + options["pairs_per_step"]]
        anchor_batch = []
        for _ in range(2 * len(selected)):
            if state.anchor_cursor >= len(state.anchor_order):
                state.anchor_order = list(range(len(anchors)))
                shuffle.shuffle(state.anchor_order)
                state.anchor_cursor = 0
            anchor_batch.append(anchors[state.anchor_order[state.anchor_cursor]])
            state.anchor_cursor += 1
        step_started = perf_counter()
        metrics = logical_step(
            policy,
            optimizer,
            histories,
            [pairs[i] for i in selected],
            anchor_batch,
            references["mixed"],
        )
        state.record_epoch_step(len(selected))
        save(state_path)
        with (out / "steps.jsonl").open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    dict(
                        step=state.steps,
                        seconds=state.seconds,
                        step_seconds=perf_counter() - step_started,
                        **metrics,
                    )
                )
                + "\n"
            )
    if not stop:
        kl = (
            state.last_kl
            if state.cursor == 0 and state.last_kl is not None
            else full_kl(policy, histories, anchors, references["fp32"])
        )
    if stop or kl > options["kl_limit"]:
        state = ComparisonTrainingState.from_checkpoint_dict(
            torch.load(out / "verified.pt", map_location="cpu", weights_only=False)
        )
        policy.load_state_dict(state.metadata["model"])
        optimizer.load_state_dict(state.metadata["optimizer"])
        shuffle.setstate(state.metadata["shuffle"])
        restore_rng(state.metadata["rng"])
        state.stop_reason = "kl_limit"
    else:
        state.stop_reason = "epochs_completed" if state.epoch == options["max_epochs"] else "time_limit"
    state.attempted_final_kl = kl
    frozen = save(state_path)
    return publish_candidate(cfg, out, frozen, identity)
