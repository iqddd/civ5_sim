"""One local target-scale PPO run, with bounded CPU-backed rollout storage."""

from __future__ import annotations

import gc
import platform
import random
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import psutil
import torch

from civ5rl.core.model import batch_inputs
from civ5rl.engine.collection.collector import Collector
from civ5rl.engine.evaluation.replay import verify_replay
from civ5rl.engine.ppo.backend import BatchedBackend
from civ5rl.engine.ppo.core import encode_episode, token_minibatches
from civ5rl.engine.ppo.storage import load_rollout, save_rollout
from civ5rl.engine.runtime.control import RunLock, TrainingCounters, assert_not_closed
from civ5rl.experiments._shared.lifecycle import record_experiment_errors
from civ5rl.experiments.evaluation.matches import evaluate_target_scale as evaluate
from civ5rl.experiments.imitation.helpers import assert_equal, isolated_rng
from civ5rl.experiments.ppo import scale_up as shared
from civ5rl.protocols.corpus import validate_corpus
from civ5rl.protocols.ppo_target_scale import (
    KIND,
    SIZES,
    InvalidDataset,
    TargetFactory,
    evaluation_cases,
    validate_profile,
)
from civ5rl.provenance import bound_cases, case_manifest, checkpoint
from civ5rl.provenance.common import (
    ROOT,
    configure,
    file_sha256,
    read_json,
    seed_for,
    write_json,
)
from civ5rl.reporting.evaluation_audit import (
    collect_paired_outcomes,
    paired_world_bootstrap,
    select_outcome_representatives,
)
from civ5rl.reporting.ppo_target_scale import report, write_invalid_dataset_report
from civ5sim.engine import UNIT_TYPES, digest


def math_check(config, out, rollout):
    """Compare CPU-backed and resident PPO on a supplied public fixture."""
    episodes = load_rollout(rollout)[:2]
    if len(episodes) != 2 or any(
        len(episode["chosen"]) < length for episode, length in zip(episodes, (7, 11), strict=True)
    ):
        raise ValueError("Math check needs two episodes of at least seven and eleven decisions")
    for ep, length in zip(episodes, (7, 11)):
        for key in (
            "observations",
            "actions",
            "chosen",
            "old_logp",
            "old_log_probs",
            "old_value",
            "old_hidden",
            "decision_metrics",
        ):
            if key in ep:
                ep[key] = ep[key][:length]
    config = deepcopy(config)
    config["performance"]["compile"] = False
    # Re-label the public fixture with this parent's own FP32 distributions;
    # the historical rollout was collected by a different policy.
    configure(config.get("seed", 17), "cpu")
    reference, _ = shared.load_parent(config)
    with torch.no_grad():
        for episode in episodes:
            inputs = encode_episode(episode, reference.variant)
            logits, values, _ = reference.forward_sequences(batch_inputs(inputs, "cpu"), [len(inputs)])
            logp = (logits / config.get("policy_temperature", 1.0)).log_softmax(-1).numpy()
            episode["old_log_probs"] = [
                row[: len(actions)].copy() for row, actions in zip(logp, episode["actions"])
            ]
            episode["old_logp"] = [
                float(row[chosen]) for row, chosen in zip(episode["old_log_probs"], episode["chosen"])
            ]
            episode["old_value"] = values.numpy().tolist()
    del reference
    outputs = []
    for streamed in (False, True):
        configure(config.get("seed", 17), "cpu")
        policy, optimizer = shared.load_parent(config)
        options = deepcopy(config)
        options["performance"].update(cpu_rollout_labels=streamed, resident_rollout=not streamed)
        backend = BatchedBackend(policy, episodes, options, "cpu")
        replay = backend.precheck()
        spans = [(0, 2, 7), (1, 1, 11)]
        count = sum(b - a for _, a, b in spans)
        optimizer.zero_grad(set_to_none=True)
        backend.backward(spans, count, 0)
        gradients = {name: parameter.grad.clone() for name, parameter in policy.named_parameters()}
        backend.record_gradient(torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5))
        optimizer.step()
        stats = backend.statistics(count)[0]
        outputs.append((policy.state_dict(), optimizer.state_dict(), gradients, stats, replay))
    for before, after in zip(outputs[0], outputs[1]):
        assert_equal(before, after)
    write_json(
        out / "math-check.json",
        {
            "loss_gradients_Adam_equal": True,
            "valid_tokens": count,
            "nonzero_GRU_prefixes": [2, 1],
            "cpu_ragged_labels": True,
            "stats": outputs[1][3],
        },
    )


def prepare(config, out):
    started = perf_counter()
    experiment = config["experiment"]
    if file_sha256(ROOT / experiment["parent"]) != experiment["parent_sha256"]:
        raise ValueError("Parent checkpoint SHA256 differs")
    previous = read_json(ROOT / experiment["parent_manifest"])
    corpus = read_json(ROOT / experiment.get("corpus_manifest", "artifacts/rl/corpus/manifest.json"))
    worlds = previous["base"]["worlds"]
    training_worlds = set(worlds["training"])
    validation_worlds = set(worlds["validation"])
    if not training_worlds or (training_worlds | validation_worlds) & set(worlds.get("holdout", ())):
        raise ValueError("Target-scale corpus includes closed worlds")
    if training_worlds & validation_worlds:
        raise ValueError("Invalid parent-world split")
    maps = [
        m
        for m in corpus["maps"]
        if (m["split"] == "train" and m["parent_seed"] in worlds["training"])
        or (m["split"] == "validation" and m["parent_seed"] in worlds["validation"])
    ]
    for split, expected in (("train", training_worlds), ("validation", validation_worlds)):
        if {(entry["parent_seed"], entry["size"]) for entry in maps if entry["split"] == split} != {
            (world, size) for world in expected for size in SIZES
        }:
            raise ValueError(f"Target-scale {split} crops do not cover the world/size grid")
    validate_corpus({"maps": maps})
    base = {"worlds": worlds, "maps": maps, "hash": digest(maps)}
    factory = TargetFactory(config, base)
    capacity = []
    for index, entry in enumerate(maps):
        options = sorted(
            (
                m
                for m in maps
                if m["split"] == entry["split"]
                and m["parent_seed"] == entry["parent_seed"]
                and m["size"] == entry["size"]
            ),
            key=lambda m: m["path"],
        )
        for contact in (False, True):
            try:
                case = factory.case(
                    index,
                    entry["parent_seed"],
                    entry["size"],
                    2,
                    contact,
                    0,
                    "ppo_target_scale_capacity_v1",
                    split=entry["split"],
                    crop=options.index(entry),
                    n=13,
                )
            except InvalidDataset:
                write_json(out / "capacity-checks.json", capacity)
                raise
            capacity.append(
                {
                    "map": entry["path"],
                    "contact": contact,
                    "attempts": case["placement_attempts"],
                    "algorithm": case["placement_algorithm"],
                    "method": case["placement_method"],
                    "spawn_zones_revised": case["spawn_zones_revised"],
                }
            )
        if (index + 1) % 50 == 0:
            print(f"Placement capacity: {index + 1}/{len(maps)} crops", flush=True)
    write_json(out / "capacity-checks.json", capacity)
    plan = {
        "schema": 1,
        "kind": KIND,
        "base": base,
        **evaluation_cases(config, base),
        "small_control": previous["development"],
    }
    for purpose in ("development", "confirmation"):
        cases = plan[purpose]
        assert len(cases) == (128 if purpose == "development" else 256)
        counts = Counter((c["map_size"], c["army_bin"]) for c in cases if c["learner_side"] == 0)
        assert max(counts.values()) - min(counts.values()) <= 1
        assert {k for c in cases for k in c["armies"][0]} == set(UNIT_TYPES)
        for world in worlds["validation"]:
            assert len([c for c in cases if c["parent_world"] == world]) == (
                8 if purpose == "development" else 16
            )
    plan["hash"] = digest(plan)
    write_json(out / "manifest.json", plan)
    if fixture := experiment.get("math_check_rollout"):
        math_check(config, out, ROOT / fixture)
    write_json(
        out / "preparation.json",
        {
            "seconds": perf_counter() - started,
            "capacity_cases": len(capacity),
            "development": len(plan["development"]),
            "confirmation": len(plan["confirmation"]),
            "training_crops": sum(entry["split"] == "train" for entry in maps),
            "test_used": False,
        },
    )
    print(
        "Cases and CPU-backed PPO math verified; starting bounded memory calibration",
        flush=True,
    )
    return plan


@dataclass(slots=True)
class _CalibrationProbe:
    """Keep probe state and closed collectors' staging buffers across retries."""

    policy: Any = None
    optimizer: Any = None
    episodes: Any = None
    backend: BatchedBackend | None = None
    collector: Collector | None = None


def _probe_collection_capacity(
    config: dict[str, Any],
    out: Path,
    device: str,
    factory: TargetFactory,
    specs: list[dict[str, Any]],
    probe: _CalibrationProbe,
) -> tuple[int, dict[str, Any]]:
    configure(config.get("seed", 17), device)
    probe.policy, probe.optimizer = shared.load_parent(config, device)
    torch.cuda.reset_peak_memory_stats()
    trusted = case_manifest(specs, source=KIND, split="train", namespace="ppo_target_scale_memory_v1")
    with Collector(config, factory, device, out / "preparation-errors") as probe.collector:
        probe.episodes, _, timings = probe.collector.collect(
            probe.policy,
            None,
            0,
            fixed_specs=bound_cases(trusted),
            trusted_manifest=trusted,
        )
    free, total = torch.cuda.mem_get_info()
    collection_peak = torch.cuda.max_memory_reserved() + max(0, total - free - torch.cuda.memory_reserved())
    if total - collection_peak < 2 * 1024**3:
        raise MemoryError("Less than 2 GiB VRAM reserve during collection")
    if psutil.virtual_memory().available < 4 * 1024**3:
        raise MemoryError("Less than 4 GiB available RAM after collection")
    save_rollout(probe.episodes, out / "preparation-rollout", 1)
    return collection_peak, timings


def _probe_microbatch_vram(
    config: dict[str, Any],
    device: str,
    probe: _CalibrationProbe,
    collection_peak: int,
    timings: dict[str, Any],
    envs: int,
    microbatch: int,
) -> dict[str, Any]:
    # One real Adam attempt on a disposable copy, including full KL.
    probe.backend = BatchedBackend(probe.policy, probe.episodes, config, device)
    probe.backend.precheck()
    sequence_length = config["ppo"]["sequence_length"]
    spans = [
        (index, start, min(start + sequence_length, len(episode["chosen"])))
        for index, episode in enumerate(probe.episodes)
        for start in range(0, len(episode["chosen"]), sequence_length)
    ]
    token_count = config["ppo"].get("valid_tokens_per_minibatch")
    batch = (
        next(token_minibatches(spans, token_count))
        if token_count
        else spans[: config["ppo"]["sequences_per_minibatch"]]
    )
    probe.optimizer.zero_grad(set_to_none=True)
    probe.backend.backward(batch, sum(end - start for _, start, end in batch), 0)
    torch.nn.utils.clip_grad_norm_(probe.policy.parameters(), 0.5, error_if_nonfinite=True)
    probe.optimizer.step()
    kl = probe.backend.kl()
    probe.backend = None
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    peak = max(
        collection_peak,
        torch.cuda.max_memory_reserved() + max(0, total - free - torch.cuda.memory_reserved()),
    )
    return {
        "envs": envs,
        "microbatch": microbatch,
        "peak_reserved_bytes": peak,
        "total_vram": total,
        "available_ram": psutil.virtual_memory().available,
        "decisions": sum(len(episode["chosen"]) for episode in probe.episodes),
        "collection": timings,
        "temporary_poststep_KL": kl,
    }


def _verify_mixed_rollout_replay(
    config: dict[str, Any],
    plan: dict[str, Any],
    out: Path,
    device: str,
    factory: TargetFactory,
    probe: _CalibrationProbe,
    envs: int,
) -> None:
    mixed = [
        factory.case(
            910000 + index,
            plan["base"]["worlds"]["training"][index % len(plan["base"]["worlds"]["training"])],
            (10, 12, 16)[index % 3],
            index % 3,
            index % 2,
            index % 2,
            "ppo_target_scale_mixed_v1",
        )
        for index in range(envs)
    ]
    trusted = case_manifest(mixed, source=KIND, split="train", namespace="ppo_target_scale_mixed_v1")
    with Collector(config, factory, device, out / "preparation-errors") as probe.collector:
        probe.episodes, _, _ = probe.collector.collect(
            probe.policy,
            None,
            0,
            fixed_specs=bound_cases(trusted),
            trusted_manifest=trusted,
        )
    save_rollout(probe.episodes, out / "mixed-preparation-rollout", 1)
    probe.backend = BatchedBackend(probe.policy, probe.episodes, config, device)
    probe.backend.precheck()
    write_json(out / "mixed-behavior-check.json", probe.backend.replay)
    probe.backend = None


def _calibration_sizes(size):
    """Start at the requested capacity and halve until the minimum."""
    while size > 1:
        yield size
        size //= 2
    yield 1


def calibrate(config: dict[str, Any], plan: dict[str, Any], out: Path, device: str) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    probe = _CalibrationProbe()
    requested = config
    workers = requested["collection"]["cpu_workers"]
    envs_per_worker = requested["collection"]["envs_per_worker"]
    requested_microbatch = requested["ppo"]["sequences_per_microbatch"]
    for slots in _calibration_sizes(envs_per_worker):
        config = deepcopy(requested)
        envs = min(requested["collection"]["num_envs"], slots * workers)
        config["collection"].update(num_envs=envs, envs_per_worker=slots)
        if "worker_reply_size" in config["performance"]:
            config["performance"]["worker_reply_size"] = min(
                config["performance"]["worker_reply_size"], slots
            )
        factory = TargetFactory(config, plan["base"])
        specs = [
            factory.case(
                900000 + index,
                plan["base"]["worlds"]["training"][index % len(plan["base"]["worlds"]["training"])],
                16,
                2,
                index % 2,
                index % 2,
                "ppo_target_scale_memory_v1",
                n=13,
            )
            for index in range(envs)
        ]
        probe.episodes = None
        try:
            collection_peak, timings = _probe_collection_capacity(config, out, device, factory, specs, probe)
            for microbatch in _calibration_sizes(requested_microbatch):
                config["ppo"]["sequences_per_microbatch"] = microbatch
                config["performance"]["kl_sequences_per_batch"] = microbatch
                del probe.policy, probe.optimizer
                gc.collect()
                torch.cuda.empty_cache()
                configure(config.get("seed", 17), device)
                probe.policy, probe.optimizer = shared.load_parent(config, device)
                torch.cuda.reset_peak_memory_stats()
                probe.backend = None
                try:
                    result = _probe_microbatch_vram(
                        config,
                        device,
                        probe,
                        collection_peak,
                        timings,
                        envs,
                        microbatch,
                    )
                    attempts.append(result)
                    write_json(out / "memory-calibration.json", attempts)
                    if result["total_vram"] - result["peak_reserved_bytes"] < 2 * 1024**3:
                        raise torch.OutOfMemoryError("Less than 2 GiB VRAM reserve")
                    if result["available_ram"] < 4 * 1024**3:
                        raise MemoryError("Less than 4 GiB RAM reserve")
                    # Mixed-size public history uses unmodified parent weights.
                    del probe.policy, probe.optimizer, probe.episodes
                    gc.collect()
                    torch.cuda.empty_cache()
                    configure(config.get("seed", 17), device)
                    probe.policy, probe.optimizer = shared.load_parent(config, device)
                    _verify_mixed_rollout_replay(config, plan, out, device, factory, probe, envs)
                    del probe.policy, probe.optimizer, probe.episodes
                    gc.collect()
                    torch.cuda.empty_cache()
                    write_json(out / "resolved-config.json", config)
                    print(
                        f"Memory configuration fixed: {envs} envs, microbatch {microbatch}",
                        flush=True,
                    )
                    return config
                except torch.OutOfMemoryError as error:
                    del probe.backend
                    attempts.append({"envs": envs, "microbatch": microbatch, "OOM": str(error)})
                    write_json(out / "memory-calibration.json", attempts)
                    gc.collect()
                    torch.cuda.empty_cache()
            raise MemoryError("Backward does not fit with microbatch 1")
        except (torch.OutOfMemoryError, MemoryError) as error:
            attempts.append({"envs": envs, "collection_or_RAM_error": str(error)})
            write_json(out / "memory-calibration.json", attempts)
            probe.episodes = None
            gc.collect()
            torch.cuda.empty_cache()
    raise MemoryError(f"Target-scale workload does not fit {workers} envs / microbatch 1")


def initialize(config, plan, out):
    configure(config.get("seed", 17), "cpu")
    policy, optimizer = shared.load_parent(config)
    experiment = config["experiment"]
    parent_adam_counter = experiment["parent_optimizer_steps"]
    record = {
        "kind": KIND,
        "config": config,
        "evaluation_runtime_default": "auto",
        "parent_sha256": experiment["parent_sha256"],
        "manifest_hash": plan["hash"],
        "initial_model_Adam_bitwise_equal": True,
        "parent_Adam_counter": parent_adam_counter,
        "critic_preserved": True,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seed": config.get("seed", 17),
        "shuffle_seed": seed_for(
            config.get("seed", 17), 0, "shuffle", namespace=config["experiment"]["shuffle_namespace"]
        ),
    }
    record["run_id"] = digest(record)
    state = {
        **TrainingCounters().to_state_dict(protocol=KIND),
        "origin_Adam_counter": parent_adam_counter,
        "decisions": 0,
        "next_episode": experiment["start_episode"],
        "zero_accepted_streak": 0,
        "active_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "preparation_seconds": 0.0,
        "evaluated": {},
        "latest": "checkpoints/initial.pt",
        "pending_evaluation": True,
        "best": {"path": "checkpoints/initial.pt", "wins": -1, "update": 0},
        "stop_reason": None,
    }
    write_json(out / "run.json", record)
    checkpoint.save(
        out / state["latest"],
        policy,
        optimizer,
        state,
        random.Random(record["shuffle_seed"]),
        config,
        {"run_id": record["run_id"]},
    )
    write_json(out / "state.json", state)
    write_json(out / "status.json", {"status": "prepared"})


def development(config, plan, out, state, shuffle):
    started = perf_counter()
    with isolated_rng(shuffle):
        result = evaluate(config, plan, out, state["latest"], "development")
    state["evaluation_seconds"] += perf_counter() - started
    state["evaluated"][str(state["updates"])] = {
        **result,
        "decisions": state["decisions"],
    }
    wins = result["runs"]["argmax"]["cp"]["summary"]["W"]
    if wins > state["best"]["wins"]:
        state["best"] = {
            "path": state["latest"],
            "wins": wins,
            "update": state["updates"],
            "policy_hash": result["policy_hash"],
        }
    state["pending_evaluation"] = False
    write_json(out / "state.json", state)
    if state["updates"] == 0:
        write_json(
            out / "evaluation-cost.json",
            {
                "baseline_wall_seconds": result["seconds"],
                "estimated_target_evaluation_seconds": result["seconds"] * 9,
                "slices": result["slices"],
                "uncertainty": "Trajectory length and army composition change CPU planning cost",
            },
        )
    report(config, out)


def finish(config, plan, out):
    state = read_json(out / "state.json")
    if not state["stop_reason"] or state["pending_evaluation"]:
        raise ValueError("Training is incomplete")
    write_json(out / "selected.json", state["best"])
    baseline = evaluate(config, plan, out, "checkpoints/initial.pt", "confirmation")
    candidate = evaluate(config, plan, out, state["best"]["path"], "confirmation")
    same = baseline["policy_hash"] == candidate["policy_hash"]

    def outcomes(evaluation):
        directory = out / evaluation["runs"]["argmax"]["cp"]["directory"]
        results, cases, _ = collect_paired_outcomes(directory, sort=False)
        return results, cases

    before, cases = outcomes(baseline)
    after, paired = outcomes(candidate)
    if cases != paired:
        raise ValueError("Unpaired target validation")
    comparison = (
        None
        if same
        else paired_world_bootstrap(
            before, after, samples=10000, seed=20261103, cases=cases, cases_per_world=16
        )
    )
    write_json(
        out / "confirmation.json",
        {
            "baseline": baseline,
            "candidate": candidate,
            "comparison": comparison,
            "baseline_selected": same,
        },
    )
    small_before = evaluate(config, plan, out, "checkpoints/initial.pt", "small_control")
    small_after = evaluate(config, plan, out, state["best"]["path"], "small_control")
    write_json(out / "small-control.json", {"baseline": small_before, "candidate": small_after})
    directory = out / candidate["runs"]["argmax"]["cp"]["directory"]
    chosen = select_outcome_representatives(directory)
    saved = torch.load(out / state["best"]["path"], map_location="cpu", weights_only=False)
    counters = {int(v["step"]) for v in saved["optimizer"]["state"].values()}
    saved_state = saved["state"]
    expected_counter = saved_state["origin_Adam_counter"] + saved_state["optimizer_steps"]
    if counters != {expected_counter}:
        raise ValueError(f"Selected Adam counters differ: {counters} != {expected_counter}")
    write_json(
        out / "verification.json",
        {
            "selected_Adam_counter": next(iter(counters)),
            "replays": {kind: verify_replay(path) for kind, path in chosen.items()},
        },
    )


def run(phase, config, out, device):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if phase == "pause":
        (out / "pause.request").write_text("user requested pause\n", encoding="utf-8")
        print(
            "Pause requested; completes current update and preserves pending evaluation",
            flush=True,
        )
        return
    if (out / "resolved-config.json").exists():
        config = read_json(out / "resolved-config.json")
    validate_profile(config)
    with RunLock(out):
        if phase in ("train", "run", "resume"):
            assert_not_closed(out)
        try:
            with record_experiment_errors(
                out,
                phase=phase,
                on_error_report=lambda: report(config, out) if (out / "state.json").exists() else None,
                passthrough=(InvalidDataset,),
            ):
                if phase == "resume":
                    (out / "pause.request").unlink(missing_ok=True)
                if not (out / "run.json").exists():
                    if phase not in ("prepare", "run"):
                        raise ValueError("Prepare first")
                    write_json(out / "status.json", {"status": "preparing"})
                    if (out / "manifest.json").exists() and (out / "preparation.json").exists():
                        plan = read_json(out / "manifest.json")
                        if digest({k: v for k, v in plan.items() if k != "hash"}) != plan["hash"]:
                            raise ValueError("Prepared manifest differs")
                        if any(
                            case.get("placement_algorithm") != "native_visibility_regions_v2"
                            for purpose in ("development", "confirmation")
                            for case in plan[purpose]
                            if case["placement"] == "separated"
                        ):
                            raise ValueError("Prepared cases use the previous placement algorithm")
                        print(
                            "Reusing completed crop preparation; continuing memory calibration",
                            flush=True,
                        )
                    else:
                        plan = prepare(config, out)
                    config = calibrate(config, plan, out, device)
                    initialize(config, plan, out)
                else:
                    record = read_json(out / "run.json")
                    if record.get("active_config", record["config"]) != config:
                        raise ValueError("Configuration differs")
                plan = read_json(out / "manifest.json")
                if phase in ("train", "run", "resume"):
                    write_json(out / "status.json", {"status": "training"})
                    shared.train(
                        config,
                        plan,
                        out,
                        device,
                        factory_type=TargetFactory,
                        development_fn=development,
                    )
                    if read_json(out / "status.json")["status"] == "paused":
                        report(config, out)
                        return
                if phase in ("evaluate", "run", "resume"):
                    finish(config, plan, out)
                    write_json(
                        out / "status.json",
                        {
                            "status": "complete",
                            "stop_reason": read_json(out / "state.json")["stop_reason"],
                        },
                    )
                if (out / "state.json").exists():
                    report(config, out)
        except InvalidDataset as error:
            write_json(out / "invalid-dataset.json", error.details)
            write_json(
                out / "status.json",
                {"status": "invalid_dataset", "phase": phase, "reason": str(error)},
            )
            write_invalid_dataset_report(out, error)
            raise
