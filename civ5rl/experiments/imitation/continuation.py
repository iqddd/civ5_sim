"""Bounded, explicitly parented continuation of the fixed CP imitation experiment."""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from civ5rl.engine.imitation.core import (
    DemonstrationCache,
    backward_minibatch,
    label_minibatches,
    teacher_metrics,
)
from civ5rl.engine.runtime.telemetry import CompilationClock, ResourceMonitor, analyze_telemetry
from civ5rl.engine.runtime.transactions import FrozenParamsCheck, bc_optimizer_step
from civ5rl.experiments.evaluation.matches import run_validation_cases
from civ5rl.experiments.imitation.helpers import (
    adam,
    assert_equal,
    demonstrations,
    isolated_rng,
    load_policy,
    restore_rng,
    update_selection,
)
from civ5rl.protocols.imitation import prepare_manifest
from civ5rl.protocols.imitation_invariants import (
    CONTINUATION_BUDGET,
    CONTINUATION_CHANGES,
    require_changes,
    require_values,
)
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import ROOT, configure, file_sha256, read_json, write_json
from civ5rl.provenance.lineage import current_parent_lineage, verify_existing_run
from civ5rl.reporting.evaluation_audit import collect_summary
from civ5rl.reporting.imitation_continuation import report as write_report


def initialize(config, out):
    settings = config["continuation"]
    parent = ROOT / settings["parent_run"]
    parent_config = read_json(parent / "resolved-config.json")
    require_changes(config, parent_config, CONTINUATION_CHANGES)
    require_values(config, CONTINUATION_BUDGET, "continuation")
    path = parent / settings["parent_checkpoint"]
    if file_sha256(path) != settings["parent_checkpoint_sha256"]:
        raise ValueError("Parent checkpoint hash mismatch")
    if file_sha256(parent / "training-inputs.json") != settings["parent_training_inputs_sha256"]:
        raise ValueError("Parent training-inputs hash mismatch")
    manifest = prepare_manifest(config)
    if manifest != read_json(parent / "manifest.json"):
        raise ValueError("Continuation corpus differs from parent")
    current, changes = current_parent_lineage(parent, config, manifest, same_corpus=True)
    if (out / "passport.json").exists():
        identity = verify_existing_run(out, config, current)
    else:
        identity = checkpoint.passport(out, config, manifest)
        write_json(out / "manifest.json", manifest)
        write_json(
            out / "lineage.json",
            {
                "parent_checkpoint": str(path.relative_to(ROOT)),
                "parent_checkpoint_sha256": file_sha256(path),
                "parent_training_inputs_sha256": file_sha256(parent / "training-inputs.json"),
                "parent_passport_sha256": file_sha256(parent / "passport.json"),
                "changed_execution_sources": changes,
                "parent_epochs": 3,
                "parent_Adam_steps": 60,
                "new_optimizer_work_limit_seconds": 1800,
                "total_epoch_limit": 100,
            },
        )
    return parent, manifest, identity


def evaluate_candidate(config, manifest, out, parent, candidate, workers, mode, label):
    path = Path(candidate["path"])
    key = file_sha256(path) + ":" + mode
    recorded = (
        read_json(out / "evaluations.json")
        if (out / "evaluations.json").exists()
        else {"runs": {}, "labels": {}}
    )
    if key in recorded["runs"]:
        recorded["labels"][label] = key
        write_json(out / "evaluations.json", recorded)
        return recorded["runs"][key]
    started = perf_counter()
    if candidate["epoch"] == 3 and mode == "sample":
        directories = {enemy: parent / "evaluation" / ("final-" + enemy) for enemy in ("cp", "greedy")}
        new_matches = 0
    else:
        base = out / "evaluation" / mode / f"epoch-{candidate['epoch']:03}"
        directories = {enemy: base / enemy for enemy in ("cp", "greedy")}
        new_matches = run_validation_cases(config, manifest, path.resolve(), mode, directories, workers)
    result = {
        "epoch": candidate["epoch"],
        "checkpoint_sha256": file_sha256(path),
        "mode": mode,
        "new_matches": new_matches,
        "wall_seconds": perf_counter() - started,
        "directories": {k: str(v.relative_to(ROOT)) for k, v in directories.items()},
        "opponents": {enemy: collect_summary(directory) for enemy, directory in directories.items()},
    }
    recorded["runs"][key], recorded["labels"][label] = result, key
    write_json(out / "evaluations.json", recorded)
    print(json.dumps({"evaluation": label, **result}), flush=True)
    return result


def train(config, out, parent, manifest, identity, workers, device):

    if (out / "training.json").exists():
        raise ValueError("Continuation training already recorded; do not repeat optimization")
    if device != "cuda":
        raise ValueError("This continuation protocol requires CUDA/BF16")
    configure(17, device)
    settings = config["continuation"]
    preparation = perf_counter()
    episodes, train_files = demonstrations(config, manifest, parent, "training")
    holdout, holdout_files = demonstrations(config, manifest, parent, "holdout")
    inputs = {"training": train_files, "holdout": holdout_files}
    if inputs != read_json(parent / "training-inputs.json"):
        raise ValueError("Continuation demonstration set changed")
    counts = {
        split: sum(r["supervised"] for ep in data for r in ep["decisions"])
        for split, data in [("training", episodes), ("holdout", holdout)]
    }
    if counts != {"training": 20018, "holdout": 2485}:
        raise ValueError(f"Unexpected continuation label counts: {counts}")
    write_json(out / "training-inputs.json", inputs)
    subset, subset_files, subset_labels = [], {}, 0
    for ep, (name, sha) in zip(episodes, train_files.items()):
        subset.append(ep)
        subset_files[name] = sha
        subset_labels += sum(r["supervised"] for r in ep["decisions"])
        if subset_labels >= settings["fixed_training_labels_minimum"]:
            break
    write_json(
        out / "fixed-training-subset.json",
        {"files": subset_files, "labels": subset_labels, "episodes": len(subset)},
    )
    source = parent / settings["parent_checkpoint"]
    saved = torch.load(source, weights_only=False, map_location="cpu")
    if saved["config"] != read_json(
        parent / "resolved-config.json"
    ) or not checkpoint.same_execution_identity(saved["identity"], read_json(parent / "passport.json")):
        raise ValueError("Parent checkpoint does not match parent experiment identity")
    if saved["state"]["epoch"] != 3 or saved["state"]["Adam_steps"] != 60:
        raise ValueError("Unexpected parent optimization boundary")
    policy = load_policy(config, source, device)
    optimizer = adam(policy, config, device)
    optimizer.load_state_dict(saved["optimizer"])
    shuffle = random.Random()
    cache, dev_cache, fixed_cache = [DemonstrationCache(data, device) for data in (episodes, holdout, subset)]
    frozen_value = FrozenParamsCheck(policy.value)
    clock = CompilationClock()

    preparation_seconds = perf_counter() - preparation
    warm_start = perf_counter()
    with clock, ResourceMonitor(out / "telemetry") as monitor:
        clock.monitor = monitor
        monitor.mark("warmup")
        # Compile representative gradient and no-grad paths without an optimizer step.
        spans, count = next(label_minibatches(cache.chunks(), cache.masks))
        optimizer.zero_grad(set_to_none=True)
        backward_minibatch(policy, cache, spans, count)
        optimizer.zero_grad(set_to_none=True)
        parent_metrics = {
            "holdout": teacher_metrics(policy, dev_cache),
            "fixed_training": teacher_metrics(policy, fixed_cache),
        }
        torch.cuda.synchronize()
        warmup_seconds = perf_counter() - warm_start
        restore_rng(saved, shuffle)
        assert_equal(policy.state_dict(), saved["model"])
        assert_equal(optimizer.state_dict(), saved["optimizer"])
        assert_equal(random.getstate(), saved["python_rng"])
        assert_equal(np.random.get_state(), saved["numpy_rng"])
        assert_equal(torch.get_rng_state(), saved["torch_rng"])
        assert_equal(torch.cuda.get_rng_state_all(), saved["cuda_rng"])
        assert_equal(shuffle.getstate(), saved["shuffle_rng"])
        write_json(
            out / "initialization-check.json",
            {
                "model_bitwise_equal": True,
                "Adam_moments_and_groups_equal": True,
                "all_RNG_and_shuffle_equal": True,
                "Adam_steps_before": 60,
                "warmup_Adam_steps": 0,
                "parent_checkpoint_sha256": file_sha256(source),
                "parent_metrics": parent_metrics,
            },
        )
        historical_ce = read_json(parent / "training.json")["epochs"][-1]["holdout"]["ce"]
        state = {
            "best_epoch": 3,
            "best_ce": historical_ce,
            "anchor_ce": historical_ce,
            "plateau_count": 0,
            "plateau": False,
        }
        best = {"epoch": 3, "ce": historical_ce, "path": str(source.resolve())}
        history, metrics, steps, elapsed, evaluation_seconds = (
            [],
            [{"epoch": 3, **parent_metrics}],
            60,
            0.0,
            0.0,
        )
        reason, last_complete = None, 3
        utilization_justification = None
        checkpoint_dir = out / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        shutil.copy2(source, checkpoint_dir / "best.pt")

        def persist():
            write_json(
                out / "training.json",
                {
                    "status": "running" if reason is None else "complete",
                    "stop_reason": reason,
                    "epochs": history,
                    "offline_metrics": metrics,
                    "selection": state,
                    "best": best,
                    "last_completed_epoch": last_complete,
                    "Adam_steps_total": steps,
                    "new_Adam_steps": steps - 60,
                    "new_optimizer_seconds": elapsed,
                    "preparation_seconds": preparation_seconds,
                    "warmup_and_parent_metrics_seconds": warmup_seconds,
                    "compile_seconds": sum(b - a for a, b in clock.intervals),
                    "battle_evaluation_seconds": evaluation_seconds,
                    "GPU_below_target_justification": utilization_justification,
                },
            )

        def measure(epoch, candidate_path):
            nonlocal state, best
            monitor.mark("offline_metrics", epoch=epoch)
            start = perf_counter()
            with isolated_rng(shuffle):
                result = {
                    "epoch": epoch,
                    "holdout": teacher_metrics(policy, dev_cache),
                    "fixed_training": teacher_metrics(policy, fixed_cache),
                }
            result["wall_seconds"] = perf_counter() - start
            metrics.append(result)
            state = update_selection(state, epoch, result["holdout"]["ce"], settings)
            if state["best_epoch"] == epoch:
                best = {
                    "epoch": epoch,
                    "ce": state["best_ce"],
                    "path": str(candidate_path.resolve()),
                }
                shutil.copy2(candidate_path, checkpoint_dir / "best.pt")

        for epoch in range(4, config["epochs"] + 1):
            if elapsed >= config["training_seconds"]:
                reason = "optimizer_time_limit"
                break
            monitor.mark("iterative_training", epoch=epoch, Adam_steps=steps)
            order = cache.chunks()
            shuffle.shuffle(order)
            records, loss_sum, labels, complete = [], 0.0, 0, True
            for spans, count in label_minibatches(order, cache.masks):
                if elapsed >= config["training_seconds"]:
                    complete = False
                    break
                torch.cuda.synchronize()
                start = perf_counter()
                ce, norm = bc_optimizer_step(
                    policy,
                    optimizer,
                    backward_minibatch,
                    cache,
                    spans,
                    count,
                    max_gradient_norm=config["max_gradient_norm"],
                )
                torch.cuda.synchronize()
                stop = perf_counter()
                compile_seconds = clock.seconds(start, stop)
                elapsed += stop - start - compile_seconds
                steps += 1
                if elapsed >= 600 and utilization_justification is None:
                    recent_windows = analyze_telemetry(out / "telemetry/resources.jsonl")["windows"]
                    utilization_justification = {
                        "reason": "Fixed-condition continuation: CPU batch assembly, synchronous host/device checks and small spatial blocks constrain throughput; do not change minibatch or numerics mid-experiment.",
                        "investigation": "Inspected DemonstrationCache.batch, backward_minibatch and fixed spatial 32; candidate optimizations are resident inputs and fewer host synchronizations, deferred to preserve this experiment.",
                        "optimizer_seconds_at_review": elapsed,
                        "last_60_seconds_core_average": recent_windows[-1]["core_average"]
                        if recent_windows
                        else None,
                    }
                    print(
                        json.dumps({"GPU_utilization_review": utilization_justification}),
                        flush=True,
                    )
                if steps == 61:
                    assert {int(v["step"]) for v in optimizer.state.values()} == {61}
                    write_json(
                        out / "first-step-check.json",
                        {
                            "Adam_steps_before": 60,
                            "Adam_steps_after": 61,
                            "all_Adam_counters_equal": True,
                            "labels": count,
                            "spans": spans,
                            "pre_step_state": "initialization-check.json",
                            "ce": ce,
                        },
                    )
                records.append(
                    {
                        "Adam_step": steps,
                        "labels": count,
                        "ce": ce,
                        "gradient_norm_pre_clip": float(norm),
                        "wall_seconds": stop - start,
                        "compile_seconds": compile_seconds,
                    }
                )
                loss_sum += ce * count
                labels += count
            frozen_value.verify()
            row = {
                "epoch": epoch,
                "completed": complete,
                "labels": labels,
                "training_ce": loss_sum / labels if labels else None,
                "Adam_steps_total": steps,
                "optimizer_steps": records,
            }
            history.append(row)
            if not complete:
                checkpoint.save(
                    checkpoint_dir / "partial.pt",
                    policy,
                    optimizer,
                    {
                        "epoch": last_complete,
                        "partial_epoch": epoch,
                        "Adam_steps": steps,
                    },
                    shuffle,
                    config,
                    identity,
                )
                reason = "optimizer_time_limit"
                break
            last_complete = epoch
            monitor.mark("checkpoint", epoch=epoch)
            checkpoint.save(
                checkpoint_dir / "last-complete.pt",
                policy,
                optimizer,
                {"epoch": epoch, "Adam_steps": steps, "new_optimizer_seconds": elapsed},
                shuffle,
                config,
                identity,
            )
            if epoch % settings["holdout_interval"] == 0:
                candidate_path = checkpoint_dir / f"epoch-{epoch:03}.pt"
                shutil.copy2(checkpoint_dir / "last-complete.pt", candidate_path)
                measure(epoch, candidate_path)
            persist()
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "Adam_steps": steps,
                        "new_optimizer_seconds": elapsed,
                        "training_ce": row["training_ce"],
                        "best_epoch": best["epoch"],
                        "best_holdout_ce": best["ce"],
                        "plateau_count": state["plateau_count"],
                    }
                ),
                flush=True,
            )
            if epoch == settings["battle_epoch"]:
                monitor.mark("battle_evaluation", epoch=epoch)
                with isolated_rng(shuffle):
                    result = evaluate_candidate(
                        config,
                        manifest,
                        out,
                        parent,
                        best,
                        workers,
                        "sample",
                        "epoch30",
                    )
                evaluation_seconds += result["wall_seconds"]
            if state["plateau"]:
                reason = "holdout_plateau"
                break
            if elapsed >= config["training_seconds"]:
                reason = "optimizer_time_limit"
                break
        if reason is None:
            reason = "epoch_limit"
        if metrics[-1]["epoch"] != last_complete and last_complete > 3:
            # A partial next epoch must not contaminate measurements of the last full one.
            last = torch.load(
                checkpoint_dir / "last-complete.pt",
                weights_only=False,
                map_location="cpu",
            )
            policy.load_state_dict(last["model"])
            candidate_path = checkpoint_dir / f"epoch-{last_complete:03}.pt"
            shutil.copy2(checkpoint_dir / "last-complete.pt", candidate_path)
            measure(last_complete, candidate_path)
        # Freeze final selection before any final battle results are available.
        best["path"] = str((checkpoint_dir / "best.pt").resolve())
        best["checkpoint_sha256"] = file_sha256(checkpoint_dir / "best.pt")
        persist()


def evaluate(config, out, parent, manifest, workers):
    training = read_json(out / "training.json")
    if training["status"] != "complete":
        raise ValueError("Cannot evaluate an unfinished continuation")
    best = training["best"]
    if file_sha256(best["path"]) != best["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed after selection")
    evaluate_candidate(config, manifest, out, parent, best, workers, "sample", "final_sample")
    if config["continuation"]["final_argmax"]:
        evaluate_candidate(config, manifest, out, parent, best, workers, "argmax", "final_argmax")


def run(phase, config, out, workers, device):
    if phase not in ("prepare", "train", "evaluate", "report"):
        raise ValueError("Continuation does not allow collection or benchmark")
    if phase == "report" and (out / "passport.json").exists():
        checkpoint.verify_frozen_passport(out, config=config, manifest=read_json(out / "manifest.json"))
        return report(config, out, ROOT / config["continuation"]["parent_run"])
    parent, manifest, identity = initialize(config, out)
    started = perf_counter()
    try:
        if phase == "train":
            train(config, out, parent, manifest, identity, workers, device)
        elif phase == "evaluate":
            evaluate(config, out, parent, manifest, workers)
        elif phase == "report":
            report(config, out, parent)
        write_json(
            out / f"{phase}-execution.json",
            {"wall_seconds": perf_counter() - started, "status": "complete"},
        )
    except Exception as exc:
        write_json(
            out / "status.json",
            {
                "status": "technical_error",
                "phase": phase,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


def _verify_report_checkpoint(config, out, parent):
    saved = torch.load(out / "checkpoints/best.pt", weights_only=False, map_location="cpu")
    parent_saved = torch.load(
        parent / config["continuation"]["parent_checkpoint"],
        weights_only=False,
        map_location="cpu",
    )
    for key, value in parent_saved["model"].items():
        if key.startswith("value."):
            assert_equal(value, saved["model"][key])


def report(config, out, parent):
    write_report(config, out, parent, verify_checkpoint=_verify_report_checkpoint)
