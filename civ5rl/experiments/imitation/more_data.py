"""Fixed expanded CP corpus, early throughput estimate and bounded BC training."""

from __future__ import annotations

import json
import random
import shutil
from time import perf_counter

import numpy as np
import torch

from civ5rl.engine.imitation.core import (
    DemonstrationCache,
    backward_minibatch,
    label_minibatches,
    teacher_metrics,
)
from civ5rl.engine.imitation.pipeline import (
    ResidentDemonstrations,
    backward_minibatch_fast,
)
from civ5rl.engine.runtime.shared import pool
from civ5rl.engine.runtime.telemetry import CompilationClock, ResourceMonitor
from civ5rl.engine.runtime.telemetry_analysis import analyze_telemetry
from civ5rl.engine.runtime.transactions import FrozenParamsCheck, bc_optimizer_step
from civ5rl.experiments.evaluation.matches import run_validation_cases
from civ5rl.experiments.imitation.helpers import (
    adam,
    assert_equal,
    isolated_rng,
    load_policy,
    update_selection,
)
from civ5rl.protocols.imitation import (
    load_episode,
    match_task,
    prepare_manifest,
    task_for,
    validate_case,
)
from civ5rl.protocols.imitation_invariants import (
    MORE_DATA_BUDGET,
    MORE_DATA_CHANGES,
    require_changes,
    require_values,
)
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import ROOT, configure, file_sha256, read_json, write_json
from civ5rl.provenance.lineage import current_parent_lineage, verify_existing_run
from civ5rl.reporting.evaluation_audit import collect_summary
from civ5rl.reporting.imitation_more_data import report
from civ5sim.engine import digest


def initialize(config, out):
    settings = config["more_data"]
    parent, data = ROOT / settings["parent_run"], ROOT / settings["data_run"]
    original = read_json(parent / "resolved-config.json")
    require_changes(config, original, MORE_DATA_CHANGES)
    require_values(
        config,
        MORE_DATA_BUDGET,
        "more_data",
    )
    source = parent / settings["parent_checkpoint"]
    if (
        file_sha256(source) != settings["parent_checkpoint_sha256"]
        or file_sha256(data / "training-inputs.json") != settings["old_inputs_sha256"]
    ):
        raise ValueError("Parent checkpoint/data hashes differ")
    base = prepare_manifest(config)
    if base != read_json(data / "manifest.json") or base != read_json(parent / "manifest.json"):
        raise ValueError("Expanded-data world split or validation cases changed")
    inputs = read_json(data / "training-inputs.json")
    planned = {
        "schema": 1,
        "kind": "rl01.cp_imitation_more_data",
        "base": base,
        "parent_checkpoint": {
            "path": source.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(source),
        },
        "old_inputs": {
            "path": (data / "training-inputs.json").relative_to(ROOT).as_posix(),
            "sha256": file_sha256(data / "training-inputs.json"),
        },
        "config_sha256": digest(config),
        "first_new_case_id": 860,
        "inputs": {
            split: {(data / name).relative_to(ROOT).as_posix(): sha for name, sha in files.items()}
            for split, files in inputs.items()
        },
    }
    manifest = read_json(out / "manifest.json") if (out / "manifest.json").exists() else planned
    if {k: v for k, v in manifest.items() if k != "inputs"} != {
        k: v for k, v in planned.items() if k != "inputs"
    }:
        raise ValueError("Expanded-data manifest plan mismatch")
    if manifest["inputs"]["holdout"] != planned["inputs"]["holdout"] or any(
        manifest["inputs"]["training"].get(k) != v for k, v in planned["inputs"]["training"].items()
    ):
        raise ValueError("Parent demonstrations changed in manifest")
    current, changes = current_parent_lineage(parent, config, manifest, same_corpus=False)
    if (out / "passport.json").exists():
        identity = verify_existing_run(out, config, current)
    else:
        identity = checkpoint.passport(out, config, manifest)
        write_json(out / "manifest.json", manifest)
        write_json(
            out / "lineage.json",
            {
                "parent_run": settings["parent_run"],
                "parent_checkpoint_sha256": file_sha256(source),
                "parent_Adam_steps": 600,
                "changed_execution_sources": changes,
                "old_inputs_sha256": file_sha256(data / "training-inputs.json"),
                "new_shuffle_seed": 17,
            },
        )
    return parent, data, manifest, identity


def freeze_dataset(config, out, manifest):
    directory = out / "collection-passport"
    directory.mkdir(exist_ok=False)
    for name in (
        "passport.json",
        "sources.zip",
        "resolved-config.json",
        "corpus-manifest.json",
        "dirty.diff",
        "environment-lock.txt",
        "manifest.json",
    ):
        if (out / name).exists():
            shutil.copy2(out / name, directory / name)
    write_json(out / "manifest.json", manifest)
    checkpoint.passport(out, config, manifest)


def collect(config, out, parent, data, manifest, workers):

    if (out / "collection.json").exists():
        raise ValueError("New collection already recorded; do not automatically repeat it")
    old_files = read_json(data / "training-inputs.json")["training"]
    for name, sha in old_files.items():
        if file_sha256(data / name) != sha:
            raise ValueError("Old demonstration changed before collection")
    old_rows = [read_json((data / name).parent / "metrics.json") for name in old_files]
    labels = sum(r["labels"] for r in old_rows)
    if labels != 20018 or len(old_rows) != 860:
        raise ValueError("Unexpected original training corpus")
    index, start = 860, perf_counter()
    with (
        ResourceMonitor(out / "collection-telemetry") as monitor,
        pool(workers) as executor,
    ):
        monitor.mark("CP_collection", workers=workers)
        while labels < config["training_labels"] and perf_counter() - start < config["collection_seconds"]:
            jobs = [
                task_for(
                    config,
                    manifest["base"],
                    "training",
                    i,
                    out / f"data/training/match-{i:06}",
                )
                for i in range(index, index + workers)
            ]
            rows = list(executor.map(match_task, jobs))
            labels += sum(r["labels"] for r in rows)
            for i, row in zip(range(index, index + workers), rows):
                path = out / f"data/training/match-{i:06}/demonstration.json.gz"
                manifest["inputs"]["training"][path.relative_to(ROOT).as_posix()] = row[
                    "demonstration_sha256"
                ]
            index += workers
            print(
                json.dumps(
                    {
                        "total_training_labels": labels,
                        "target": 100000,
                        "last_new_case_id": index - 1,
                    }
                ),
                flush=True,
            )
    result = {
        "status": "complete" if labels >= 100000 else "collection_budget_exhausted",
        "training_labels": labels,
        "old_training_labels": 20018,
        "new_training_labels": labels - 20018,
        "new_matches": index - 860,
        "last_case_id": index - 1,
        "overflow": max(0, labels - 100000),
        "wall_seconds": perf_counter() - start,
    }
    write_json(out / "collection.json", result)
    freeze_dataset(config, out, manifest)
    write_json(out / "status.json", {"status": result["status"], "phase": "collect"})


def load_data(manifest, split, config):
    episodes = []
    for name, sha in manifest["inputs"][split].items():
        path = ROOT / name
        if file_sha256(path) != sha:
            raise ValueError(f"Demonstration hash mismatch: {name}")
        episode = load_episode(path)
        case = read_json(path.parent / "case.json")
        validate_case(case, config, manifest["base"])
        if (
            episode["case_hash"] != case["case_hash"]
            or episode["split"] != split
            or episode["parent_world"] not in manifest["base"]["worlds"][split]
        ):
            raise ValueError("Demonstration provenance/split mismatch")
        if any(
            set(row) != {"observation", "actions", "chosen", "supervised"} for row in episode["decisions"]
        ):
            raise ValueError("Non-public fields in demonstration")
        episodes.append(episode)
    return episodes


def fixed_subset(episodes, manifest, config, out):
    groups = {"old": [], "new": []}
    for i, episode in enumerate(episodes):
        groups["old" if episode["case_id"] < 860 else "new"].append(i)
    counts = {
        key: sum(sum(r["supervised"] for r in episodes[i]["decisions"]) for i in indices)
        for key, indices in groups.items()
    }
    chosen = []
    for key, indices in groups.items():
        random.Random(int(digest({"seed": 17, "namespace": "fixed_subset", "group": key})[:16], 16)).shuffle(
            indices
        )
        target = round(2000 * counts[key] / sum(counts.values()))
        labels = 0
        for i in indices:
            chosen.append(i)
            labels += sum(r["supervised"] for r in episodes[i]["decisions"])
            if labels >= target:
                break
    entries = list(manifest["inputs"]["training"].items())
    write_json(
        out / "fixed-training-subset.json",
        {
            "indices": chosen,
            "files": {entries[i][0]: entries[i][1] for i in chosen},
            "labels_by_group": {
                key: sum(
                    sum(r["supervised"] for r in episodes[i]["decisions"])
                    for i in chosen
                    if (episodes[i]["case_id"] < 860) == (key == "old")
                )
                for key in groups
            },
        },
    )
    return [episodes[i] for i in chosen]


def optimizer_step(policy, optimizer, cache, spans, count, backward, clock):
    torch.cuda.synchronize()
    started = perf_counter()
    ce, norm = bc_optimizer_step(policy, optimizer, backward, cache, spans, count)
    torch.cuda.synchronize()
    stopped = perf_counter()
    compilation = clock.seconds(started, stopped)
    return {
        "ce": ce,
        "labels": count,
        "gradient_norm_pre_clip": float(norm),
        "wall_seconds": stopped - started,
        "compile_seconds": compilation,
        "optimizer_seconds": stopped - started - compilation,
    }


def preflight(policy, optimizer, saved, cache, batches, config, out, clock, monitor):
    settings = config["more_data"]

    def reset():
        policy.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        optimizer.zero_grad(set_to_none=True)
        configure(17, "cuda")

    monitor.mark("warmup")
    spans, count = batches[0]
    backward_minibatch(policy, cache, spans, count)
    optimizer.zero_grad(set_to_none=True)
    reset()
    monitor.mark("throughput_probe")
    reference = [
        optimizer_step(policy, optimizer, cache, spans, count, backward_minibatch, clock)
        for spans, count in batches[:8]
    ]
    mean = float(np.mean([r["optimizer_seconds"] for r in reference]))
    eta = mean * len(batches) * 10
    result = {
        "reference_steps": reference,
        "mean_step_seconds": mean,
        "std_step_seconds": float(np.std([r["optimizer_seconds"] for r in reference])),
        "epoch_eta_seconds": mean * len(batches),
        "ten_epoch_eta_seconds": eta,
        "Adam_steps_per_epoch": len(batches),
        "selected": "reference",
        "probe_Adam_steps_discarded": 8,
    }
    chosen, backward = cache, backward_minibatch
    if eta >= settings["optimization_eta_threshold_seconds"]:
        start = perf_counter()
        deadline = start + settings["optimization_budget_seconds"]
        reset()
        monitor.mark("performance_investigation")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            optimizer_step(policy, optimizer, cache, *batches[0], backward_minibatch, clock)
        profiler.export_chrome_trace(str(out / "performance-trace.json"))
        reset()
        try:
            candidate = ResidentDemonstrations(cache, deadline=deadline)
            result["resident_input_bytes"] = candidate.required_bytes
            parity = compare_math(policy, optimizer, saved, cache, candidate, batches[0])
            result["candidate_math"] = parity
            reset()
            accelerated = []
            for spans, count in batches[:8]:
                if perf_counter() >= deadline:
                    raise TimeoutError("Optimization comparison budget exhausted")
                accelerated.append(
                    optimizer_step(
                        policy,
                        optimizer,
                        candidate,
                        spans,
                        count,
                        backward_minibatch_fast,
                        clock,
                    )
                )
            fast_mean = float(np.mean([r["optimizer_seconds"] for r in accelerated]))
            result.update(
                candidate_steps=accelerated,
                candidate_mean_step_seconds=fast_mean,
                speedup_fraction=1 - fast_mean / mean,
            )
            if fast_mean <= mean * (1 - settings["minimum_speedup"]):
                chosen, backward = candidate, backward_minibatch_fast
                result["selected"] = "resident_reduced_sync"
                result["selected_ten_epoch_eta_seconds"] = fast_mean * len(batches) * 10
        except (
            MemoryError,
            torch.cuda.OutOfMemoryError,
            TimeoutError,
            AssertionError,
        ) as exc:
            result["candidate_rejected"] = f"{type(exc).__name__}: {exc}"
        result["optimization_investigation_seconds"] = perf_counter() - start
    else:
        result["optimization_skipped"] = "Predicted iterative training is shorter than ten minutes"
    reset()
    assert_equal(policy.state_dict(), saved["model"])
    assert_equal(optimizer.state_dict(), saved["optimizer"])
    write_json(out / "performance-preflight.json", result)
    print(json.dumps({"performance_preflight": result}), flush=True)
    return chosen, backward


def compare_math(policy, optimizer, saved, reference, candidate, batch):
    spans, count = batch
    for key, value in reference.batch(spans)[0].items():
        assert_equal(value, candidate.batch(spans)[0][key])
    outputs = []
    for cache, backward in (
        (reference, backward_minibatch),
        (candidate, backward_minibatch_fast),
    ):
        policy.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        optimizer.zero_grad(set_to_none=True)
        ce = backward(policy, cache, spans, count)
        gradients = {
            k: v.grad.detach().cpu().clone() for k, v in policy.named_parameters() if v.grad is not None
        }
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5, error_if_nonfinite=True)
        optimizer.step()
        outputs.append(
            (
                ce,
                gradients,
                {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()},
            )
        )
    assert abs(outputs[0][0] - outputs[1][0]) <= 1e-6
    for a, b in zip(outputs[0][1:], outputs[1][1:]):
        assert a.keys() == b.keys()
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=3e-5, atol=2e-7)
    return {
        "inputs_equal": True,
        "CE_gradients_Adam_equal_within_existing_tolerances": True,
    }


def train(config, out, parent, data, manifest, identity):

    if read_json(out / "collection.json")["status"] != "complete":
        raise ValueError("Corpus target not reached; refusing smaller experiment")
    if (out / "training.json").exists():
        raise ValueError("Training already recorded; do not repeat optimization")
    configure(17, "cuda")
    started = perf_counter()
    episodes, dev = (
        load_data(manifest, "training", config),
        load_data(manifest, "holdout", config),
    )
    if sum(r["supervised"] for e in dev for r in e["decisions"]) != 2485:
        raise ValueError("Holdout label count changed")
    subset = fixed_subset(episodes, manifest, config, out)
    cache, dev_cache, fixed_cache = [DemonstrationCache(e, "cuda") for e in (episodes, dev, subset)]
    source = parent / config["more_data"]["parent_checkpoint"]
    saved = torch.load(source, weights_only=False, map_location="cpu")
    if saved["state"]["epoch"] != 30 or saved["state"]["Adam_steps"] != 600:
        raise ValueError("Wrong parent optimization boundary")
    if saved["config"] != read_json(
        parent / "resolved-config.json"
    ) or not checkpoint.same_execution_identity(saved["identity"], read_json(parent / "passport.json")):
        raise ValueError("Parent checkpoint identity differs")
    policy = load_policy(config, source, "cuda")
    optimizer = adam(policy, config, "cuda")
    optimizer.load_state_dict(saved["optimizer"])
    clock = CompilationClock()
    order = cache.chunks()
    random.Random(17).shuffle(order)
    batches = list(label_minibatches(order, cache.masks))
    preparation_seconds = perf_counter() - started
    with clock, ResourceMonitor(out / "training-telemetry") as monitor:
        clock.monitor = monitor
        chosen_cache, backward = preflight(
            policy, optimizer, saved, cache, batches, config, out, clock, monitor
        )
        configure(17, "cuda")
        shuffle = random.Random(17)
        assert_equal(policy.state_dict(), saved["model"])
        assert_equal(optimizer.state_dict(), saved["optimizer"])
        frozen_value = FrozenParamsCheck(policy.value)
        write_json(
            out / "initialization-check.json",
            {
                "model_equal": True,
                "Adam_moments_groups_equal": True,
                "Adam_steps": 600,
                "new_shuffle_seed": 17,
                "probe_updates_discarded": True,
            },
        )
        monitor.mark("offline_metrics", epoch=0)
        start = perf_counter()
        with isolated_rng(shuffle):
            baseline = {
                "epoch": 0,
                "holdout": teacher_metrics(policy, dev_cache, move_details=True),
                "fixed_training": teacher_metrics(policy, fixed_cache, move_details=True),
            }
        baseline["wall_seconds"] = perf_counter() - start
        metrics, history = [baseline], []
        state = {
            "best_epoch": 0,
            "best_ce": baseline["holdout"]["ce"],
            "anchor_ce": baseline["holdout"]["ce"],
            "plateau_count": 0,
            "plateau": False,
        }
        (out / "checkpoints").mkdir(exist_ok=True)
        shutil.copy2(source, out / "checkpoints/best.pt")
        steps, elapsed, last_complete, reason = 600, 0.0, 0, None
        utilization_review = None
        profile_pending = False

        def persist():
            write_json(
                out / "training.json",
                {
                    "status": "running" if reason is None else "complete",
                    "stop_reason": reason,
                    "epochs": history,
                    "offline_metrics": metrics,
                    "selection": state,
                    "best_epoch": state["best_epoch"],
                    "best_ce": state["best_ce"],
                    "last_completed_epoch": last_complete,
                    "Adam_steps_total": steps,
                    "new_Adam_steps": steps - 600,
                    "new_optimizer_seconds": elapsed,
                    "preparation_seconds": preparation_seconds,
                    "compile_seconds": sum(b - a for a, b in clock.intervals),
                    "backend": read_json(out / "performance-preflight.json")["selected"],
                    "GPU_utilization_review": utilization_review,
                },
            )

        for epoch in range(1, 11):
            if elapsed >= 1800:
                reason = "optimizer_time_limit"
                break
            monitor.mark("iterative_training", epoch=epoch)
            order = cache.chunks()
            shuffle.shuffle(order)
            records = []
            ce_sum = 0.0
            labels = 0
            complete = True
            for spans, count in label_minibatches(order, cache.masks):
                if elapsed >= 1800:
                    complete = False
                    break
                if profile_pending:
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as profiler:
                        row = optimizer_step(
                            policy,
                            optimizer,
                            chosen_cache,
                            spans,
                            count,
                            backward,
                            clock,
                        )
                    monitor.mark("performance_investigation", epoch=epoch)
                    profiler.export_chrome_trace(str(out / "late-performance-trace.json"))
                    monitor.mark("iterative_training", epoch=epoch)
                    profile_pending = False
                else:
                    row = optimizer_step(policy, optimizer, chosen_cache, spans, count, backward, clock)
                steps += 1
                elapsed += row["optimizer_seconds"]
                if elapsed >= 600 and utilization_review is None:
                    analysis = analyze_telemetry(out / "training-telemetry/resources.jsonl")
                    windows = analysis["windows"]
                    average = windows[-1]["core_average"] if windows else None
                    utilization_review = {
                        "optimizer_seconds": elapsed,
                        "last_60_seconds_core_average": average,
                        "backend": read_json(out / "performance-preflight.json")["selected"],
                        "continuation_reason": "Preserve validated loss/minibatch/numerics and bounded experiment; host batching, scalar synchronizations and spatial block 32 constrain the selected backend. See measured preflight comparison and trace. No artificial GPU load or extra training.",
                    }
                    profile_pending = average is not None and average < 80
                    print(
                        json.dumps({"GPU_utilization_review": utilization_review}),
                        flush=True,
                    )
                row["Adam_step"] = steps
                records.append(row)
                ce_sum += row["ce"] * count
                labels += count
                if steps == 601:
                    assert {int(s["step"]) for s in optimizer.state.values()} == {601}
                    write_json(
                        out / "first-step-check.json",
                        {
                            "before": 600,
                            "after": 601,
                            "all_counters_equal": True,
                            "first_spans": spans,
                            "labels": count,
                        },
                    )
            history.append(
                {
                    "epoch": epoch,
                    "completed": complete,
                    "training_ce": ce_sum / labels if labels else None,
                    "labels": labels,
                    "optimizer_steps": records,
                }
            )
            frozen_value.verify()
            monitor.mark("checkpoint", epoch=epoch)
            if not complete:
                checkpoint.save(
                    out / "checkpoints/partial.pt",
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
            path = out / f"checkpoints/epoch-{epoch:02}.pt"
            checkpoint.save(
                path,
                policy,
                optimizer,
                {"epoch": epoch, "Adam_steps": steps, "new_optimizer_seconds": elapsed},
                shuffle,
                config,
                identity,
            )
            monitor.mark("offline_metrics", epoch=epoch)
            start = perf_counter()
            with isolated_rng(shuffle):
                measured = {
                    "epoch": epoch,
                    "holdout": teacher_metrics(policy, dev_cache, move_details=True),
                    "fixed_training": teacher_metrics(policy, fixed_cache, move_details=True),
                }
            measured["wall_seconds"] = perf_counter() - start
            metrics.append(measured)
            state = update_selection(state, epoch, measured["holdout"]["ce"], config["more_data"])
            if state["best_epoch"] == epoch:
                shutil.copy2(path, out / "checkpoints/best.pt")
            persist()
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "Adam_steps": steps,
                        "optimizer_seconds": elapsed,
                        "holdout_ce": measured["holdout"]["ce"],
                        "best_epoch": state["best_epoch"],
                        "plateau_count": state["plateau_count"],
                    }
                ),
                flush=True,
            )
            if state["plateau"]:
                reason = "holdout_plateau"
                break
        if reason is None:
            reason = "epoch_limit"
        persist()
        result = read_json(out / "training.json")
        result["best_checkpoint_sha256"] = file_sha256(out / "checkpoints/best.pt")
        write_json(out / "training.json", result)


def evaluate(config, out, parent, data, manifest, workers):

    training = read_json(out / "training.json")
    if (
        training["status"] != "complete"
        or file_sha256(out / "checkpoints/best.pt") != training["best_checkpoint_sha256"]
    ):
        raise ValueError("Final selection not frozen")
    inherited = read_json(parent / "evaluations.json")
    runs = {}
    started = perf_counter()
    for mode in ("sample", "argmax"):
        original = inherited["runs"][inherited["labels"]["final_" + mode]]
        if training["best_epoch"] == 0:
            runs[mode] = {**original, "new_matches": 0}
            continue
        directories = {enemy: out / "evaluation" / mode / enemy for enemy in ("cp", "greedy")}
        new_matches = run_validation_cases(
            config,
            manifest["base"],
            (out / "checkpoints/best.pt").resolve(),
            mode,
            directories,
            workers,
        )
        runs[mode] = {
            "new_matches": new_matches,
            "mode": mode,
            "checkpoint_sha256": file_sha256(out / "checkpoints/best.pt"),
            "directories": {k: v.relative_to(ROOT).as_posix() for k, v in directories.items()},
            "opponents": {enemy: collect_summary(path) for enemy, path in directories.items()},
        }
        write_json(
            out / "evaluations.json",
            {"runs": runs, "wall_seconds": perf_counter() - started},
        )
    write_json(
        out / "evaluations.json",
        {"runs": runs, "wall_seconds": perf_counter() - started},
    )


def run(phase, config, out, workers, device):
    if phase not in ("prepare", "collect", "train", "evaluate", "report"):
        raise ValueError("Unsupported expanded-data phase")
    if phase == "report" and (out / "passport.json").exists():
        manifest = read_json(out / "manifest.json")
        checkpoint.verify_frozen_passport(out, config=config, manifest=manifest)
        return report(
            config,
            out,
            ROOT / config["more_data"]["parent_run"],
            ROOT / config["more_data"]["data_run"],
            manifest,
        )
    parent, data, manifest, identity = initialize(config, out)
    start = perf_counter()
    try:
        if phase == "collect":
            collect(config, out, parent, data, manifest, workers)
        elif phase == "train":
            if device != "cuda":
                raise ValueError("Expanded-data training requires CUDA/BF16")
            train(config, out, parent, data, manifest, identity)
        elif phase == "evaluate":
            evaluate(config, out, parent, data, manifest, workers)
        elif phase == "report":
            report(config, out, parent, data, manifest)
        write_json(
            out / f"{phase}-execution.json",
            {"status": "complete", "wall_seconds": perf_counter() - start},
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
