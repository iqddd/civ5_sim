"""One bounded local or remote CP--CP behavioral-cloning experiment on target armies."""

from __future__ import annotations

import gc
import inspect
import platform
import random
import shutil
import sys
import zipfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import psutil
import torch
from torch.nn import functional as F

from civ5rl.config import load_config as resolve_config
from civ5rl.core.model import Policy
from civ5rl.engine.imitation.core import decision_indices, label_minibatches
from civ5rl.engine.imitation.pipeline import (
    backward_minibatch_async,
    run_bc_minibatch_stream,
)
from civ5rl.engine.imitation.storage import PreparedDemonstrations, prepare_inputs
from civ5rl.engine.ppo.backend import prefixes
from civ5rl.engine.runtime.budget import ActiveBudget
from civ5rl.engine.runtime.control import (
    TrainingCounters,
    assert_not_closed,
    pause_due,
)
from civ5rl.engine.runtime.telemetry import ResourceMonitor, analyze_telemetry
from civ5rl.engine.runtime.transactions import FrozenParamsCheck, bc_optimizer_step
from civ5rl.experiments._shared.lifecycle import (
    ActiveSessionClock,
    experiment_run_guard,
)
from civ5rl.experiments._shared.outcomes import select_outcome_representatives
from civ5rl.experiments._shared.provenance_checks import verify_checkpoint_identity
from civ5rl.experiments.evaluation import verify_replay
from civ5rl.experiments.imitation_helpers import (
    CompilationClock,
    adam,
    assert_equal,
    isolated_rng,
    load_policy,
    restore_rng,
)
from civ5rl.experiments.ppo_after_imitation import play_case, policy_hash
from civ5rl.experiments.ppo_target_scale import evaluate as target_evaluate
from civ5rl.experiments.target_scale import evaluation as _target_evaluation
from civ5rl.experiments.target_scale.corpus import (
    _collect_c2,
    _entry_lookup_key,
    _format_corpus_entry,
    _load_more_data_corpus,
    _load_old_corpus,
    _restore_collection_progress,
    collect,
    collect_c2,
    entry_lookup_key,
    format_corpus_entry,
    load_more_data_corpus,
    load_old_corpus,
    restore_collection_progress,
)
from civ5rl.experiments.target_scale.evaluation import (
    _complete_eval_dir,
    baseline,
    evaluate,
)
from civ5rl.experiments.target_scale.reporting import (
    extract_environment_metadata,
    generate_target_scale_report,
    render_learning_curves_plot,
    report,
)
from civ5rl.experiments.target_scale.restoration import (
    BcRestorationResult,
    BcRestorationState,
    PreparedEpochState,
    RestoredEpochBatches,
    reconcile_state_vs_steplog,
    restore_bc_training_state,
    restore_epoch_cursor_and_batches,
    select_bc_checkpoint,
)
from civ5rl.experiments.target_scale.selection import (
    _refresh_selection,
    c2_gate_criteria,
    check_c2_gate,
    refresh_selection,
    select_best_c2_candidate,
    select_best_candidate,
)
from civ5rl.protocols.imitation_target_scale import (
    C2_KIND,
    MORE_DATA_KIND,
    SUPPORTED_KINDS,
)
from civ5rl.protocols.ppo_target_scale import (
    HOLDOUT_WORLDS,
    development_512_cases,
    holdout_126_cases,
)
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import (
    ROOT,
    configure,
    file_sha256,
    read_json,
    seed_for,
    write_json,
)
from civ5sim.engine import digest

if TYPE_CHECKING:
    from collections.abc import Sequence

    from civ5rl.engine.runtime.sequence import Span


_evaluate_c2 = _target_evaluation._evaluate_c2
_outcomes_from_dir = _target_evaluation._outcomes_from_dir
_try_reuse_saved_evaluation = _target_evaluation._try_reuse_saved_evaluation

__all__ = [
    "BcRestorationResult",
    "BcRestorationState",
    "PreparedEpochState",
    "RestoredEpochBatches",
    "_collect_c2",
    "_entry_lookup_key",
    "_format_corpus_entry",
    "_load_more_data_corpus",
    "_load_old_corpus",
    "_refresh_selection",
    "_restore_collection_progress",
    "baseline",
    "c2_gate_criteria",
    "check_c2_gate",
    "collect",
    "collect_c2",
    "entry_lookup_key",
    "evaluate",
    "extract_environment_metadata",
    "format_corpus_entry",
    "generate_target_scale_report",
    "load_config",
    "load_more_data_corpus",
    "load_old_corpus",
    "measure",
    "prepare",
    "reconcile_state_vs_steplog",
    "refresh_selection",
    "render_learning_curves_plot",
    "report",
    "restore_bc_training_state",
    "restore_collection_progress",
    "restore_epoch_cursor_and_batches",
    "run",
    "select_bc_checkpoint",
    "select_best_c2_candidate",
    "select_best_candidate",
    "train",
]


def load_config(path):
    overlay = read_json(path)
    base = resolve_config(ROOT / overlay["base_config"])
    config = {
        k: deepcopy(base[k])
        for k in (
            "model",
            "observation",
            "numerics",
            "optimizer",
            "teacher_search",
            "performance",
        )
    }
    config.update({k: v for k, v in overlay.items() if k != "performance"})
    config["performance"].update(overlay.get("performance", {}))
    config["environment"] = {"max_calls_per_turn": 2000}
    config["max_calls_per_turn"] = 2000
    if (
        config["kind"] not in SUPPORTED_KINDS
        or config["seed"] != 17
        or config["workers"] not in (4, 8, 10)
        or config["labels_per_minibatch"] != 1024
        or config["sequence_length"] != 128
    ):
        raise ValueError("Unsupported target BC profile")
    if config["optimizer"] != {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-5,
        "weight_decay": 0,
    }:
        raise ValueError("Unsupported target BC optimizer")
    if config["kind"] in (MORE_DATA_KIND, C2_KIND):
        for required in (
            "parent_policy_hash",
            "evaluation_reference",
            "evaluation_reference_sha256",
            "evaluation_reference_policy_hash",
            "old_corpus_run",
            "reference_run",
        ):
            if not config.get(required):
                raise ValueError(f"Missing required target-more-data key: {required}")
    if config["kind"] == C2_KIND:
        for required in (
            "bc10_checkpoint",
            "bc10_sha256",
            "bc10_policy_hash",
            "more_data_corpus_run",
        ):
            if not config.get(required):
                raise ValueError(f"Missing required target-c2 key: {required}")
    return config


def _native_artifacts():
    if sys.platform == "win32":
        names = ("build/combat.dll", "build/combat_ai.dll", "build/combat_rules.h")
    else:
        names = ("build/libcombat.so", "build/libcombat_ai.so", "build/combat_rules.h")
    return {str(Path(name)): file_sha256(ROOT / name) for name in names if (ROOT / name).is_file()}


def _evaluator_digest():
    import civ5rl.core.agent as agent_module
    from civ5sim.combat_ai import LIBRARY_PATH

    return digest(
        {
            "platform": platform.platform(),
            "torch": torch.__version__,
            "play_case": inspect.getsource(play_case),
            "agent": file_sha256(Path(agent_module.__file__)),
            "match": file_sha256(ROOT / "civ5sim/match.py"),
            "cp": file_sha256(ROOT / "civ5sim/cp_agent.py"),
            "model": file_sha256(ROOT / "civ5rl/core/model.py"),
            "observation": file_sha256(ROOT / "civ5rl/core/observation.py"),
            "planner": file_sha256(LIBRARY_PATH),
            "kernel": file_sha256(
                ROOT / "build" / ("combat.dll" if sys.platform == "win32" else "libcombat.so")
            ),
        }
    )


def _evaluation_key(config, plan, purpose, fingerprint, runtime=None):
    return digest(
        {
            "purpose": purpose,
            "policy": fingerprint,
            "mode": "argmax",
            "temperature": config.get("policy_temperature", 1.0),
            "cases": [c["case_hash"] for c in plan[purpose]],
            "evaluator": _evaluator_digest(),
            "search": config["teacher_search"],
            **({"runtime": runtime} if runtime is not None else {}),
        }
    )


def _verify_bc_checkpoint_inputs(config: dict[str, Any]) -> None:
    kind = config["kind"]
    if file_sha256(ROOT / config["parent"]) != config["parent_sha256"]:
        raise ValueError("Wrong training parent checkpoint SHA256")
    if kind in (MORE_DATA_KIND, C2_KIND):
        verify_checkpoint_identity(
            ROOT / config["evaluation_reference"],
            expected_sha256=config["evaluation_reference_sha256"],
            expected_policy_hash=config["evaluation_reference_policy_hash"],
            sha_error="Wrong evaluation reference checkpoint SHA256",
            policy_error="Wrong evaluation reference policy_hash",
        )
    if kind == C2_KIND:
        verify_checkpoint_identity(
            ROOT / config["bc10_checkpoint"],
            expected_sha256=config["bc10_sha256"],
            expected_policy_hash=config["bc10_policy_hash"],
            sha_error="Wrong BC10 checkpoint SHA256",
            policy_error="Wrong BC10 policy_hash",
        )


def _build_bc_evaluation_plan(config: dict[str, Any]) -> dict[str, Any]:
    kind = config["kind"]
    original = read_json(ROOT / config["target_manifest"])
    base = original["base"]
    worlds = base["worlds"]
    if len(worlds["training"]) != 57 or set(worlds["training"]) & set(worlds["validation"]):
        raise ValueError("Training/validation world leakage")
    plan = {
        "schema": 1,
        "kind": kind,
        "base": base,
        "development": original["development"],
        "confirmation": original["confirmation"],
    }
    if kind == C2_KIND:
        corpus_manifest = read_json(ROOT / "artifacts/rl/corpus/manifest.json")
        plan["development_512"] = development_512_cases(
            config,
            base,
            existing_cases=[*original["development"], *original["confirmation"]],
        )
        plan["holdout_126"] = holdout_126_cases(config, corpus_manifest, training_worlds=worlds["training"])
        if len(plan["development_512"]) != 512 or any(
            case["split"] != "validation" or case["parent_world"] not in worlds["validation"]
            for case in plan["development_512"]
        ):
            raise ValueError("Closed or incomplete development_512 cases")
        if len(plan["holdout_126"]) != 126 or any(
            case["split"] != "train" or case["parent_world"] not in HOLDOUT_WORLDS
            for case in plan["holdout_126"]
        ):
            raise ValueError("Closed or incomplete holdout_126 cases")
    for purpose, count in (("development", 128), ("confirmation", 256)):
        if len(plan[purpose]) != count or any(
            case["split"] != "validation" or case["parent_world"] not in worlds["validation"]
            for case in plan[purpose]
        ):
            raise ValueError("Closed or incomplete evaluation cases")
    plan["hash"] = digest(plan)
    return plan


def _archive_bc_sources(config: dict[str, Any], out: Path) -> None:
    paths: set[Path] = set()
    for pattern in (
        "civ5rl/**/*.py",
        "civ5sim/**/*.py",
        "native/**/*.cpp",
        "native/**/*.h",
        "native/**/*.inc",
    ):
        paths.update(ROOT.glob(pattern))
    paths.update(
        [
            ROOT / "tools/run_cp_imitation.py",
            ROOT / "configs/rl-cp-imitation-target-scale.json",
            ROOT / config["base_config"],
        ]
    )
    for extra_cfg in (
        "configs/rl-cp-imitation-target-more-data.json",
        "configs/rl-cp-imitation-target-c2.json",
    ):
        if (ROOT / extra_cfg).is_file():
            paths.add(ROOT / extra_cfg)
    with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            if path.is_file():
                archive.write(path, path.relative_to(ROOT).as_posix())


def _initialize_bc_model(config, out):
    kind = config["kind"]
    configure(17, "cpu")
    parent = torch.load(ROOT / config["parent"], map_location="cpu", weights_only=False)
    if kind == C2_KIND:
        if policy_hash(parent["model"]) != config["parent_policy_hash"]:
            raise ValueError("BC-1M reference policy_hash differs")
        policy = Policy("C2")
        for parameter in policy.value.parameters():
            parameter.requires_grad_(False)
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            ROOT / config["bc10_checkpoint"],
            out / "checkpoints/reference-bc-epoch-10.pt",
        )
        shutil.copy2(
            ROOT / config["parent"],
            out / "checkpoints/reference-bc-more-data-epoch-10.pt",
        )
        shutil.copy2(
            ROOT / config["evaluation_reference"],
            out / "checkpoints/reference-update-0037.pt",
        )
    else:
        policy = load_policy(config, ROOT / config["parent"], "cpu")
        assert_equal(policy.state_dict(), parent["model"])
    if kind == MORE_DATA_KIND:
        if policy_hash(policy.state_dict()) != config["parent_policy_hash"]:
            raise ValueError("Training parent policy_hash differs")
        (out / "checkpoints").mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            ROOT / config["evaluation_reference"],
            out / "checkpoints/reference-update-0037.pt",
        )
    optimizer = adam(policy, config, "cpu")
    assert not optimizer.state
    return policy, optimizer


def _bc_run_record(config, out, plan, policy):
    kind = config["kind"]
    run_record = {
        "evaluation_runtime_default": "auto",
        "kind": kind,
        "config": config,
        "parent": config["parent"],
        "parent_sha256": config["parent_sha256"],
        "parent_model_equal": kind != C2_KIND,
        "initialized_from_scratch": kind == C2_KIND,
        "new_Adam_empty": True,
        "seed": 17,
        "source_snapshot": file_sha256(out / "source.zip"),
        "manifest_hash": plan["hash"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "teacher_search": config["teacher_search"],
        "native": _native_artifacts(),
        "run_id": digest({"config": config, "manifest": plan["hash"]})[:16],
    }
    if kind in (MORE_DATA_KIND, C2_KIND):
        run_record.update(
            parent_policy_hash=config["parent_policy_hash"],
            evaluation_reference=config["evaluation_reference"],
            evaluation_reference_sha256=config["evaluation_reference_sha256"],
            evaluation_reference_policy_hash=config["evaluation_reference_policy_hash"],
            old_corpus_run=config["old_corpus_run"],
        )
    if kind == C2_KIND:
        run_record.update(
            bc10_checkpoint=config["bc10_checkpoint"],
            bc10_sha256=config["bc10_sha256"],
            bc10_policy_hash=config["bc10_policy_hash"],
            more_data_corpus_run=config["more_data_corpus_run"],
            initial_policy_hash=policy_hash(policy.state_dict()),
        )
    return run_record


def _save_initial_bc_boundary(config, out, policy, optimizer, run_record):
    kind = config["kind"]
    initial_state = {
        **TrainingCounters().to_state_dict(protocol=kind),
        "epoch": 0,
        "epoch_completed": True,
        "batch_cursor": 0,
        "epoch_order": [],
        "Adam_steps": 0,
        "free_labels_processed": 0,
        "active_seconds_at_save_start": 0.0,
        "history_positions": 0,
        "prefix_positions": 0,
        "epochs_history": [],
        "stop_reason": None,
    }
    shuffle_init = random.Random(
        seed_for(17, 0, "shuffle", namespace=config["experiment"]["shuffle_namespace"])
    )
    checkpoint.save(
        out / "checkpoints/initial.pt",
        policy,
        optimizer,
        initial_state,
        shuffle_init,
        config,
        {"run_id": run_record["run_id"]},
    )
    write_json(
        out / "state.json",
        {**initial_state, "active_seconds": 0.0, "latest": "checkpoints/initial.pt"},
    )
    write_json(out / "status.json", {"status": "prepared"})


def prepare(config, out):
    if (out / "run.json").exists():
        if read_json(out / "resolved-config.json") != config:
            raise ValueError("Configuration changed")
        return read_json(out / "manifest.json")
    _verify_bc_checkpoint_inputs(config)
    plan = _build_bc_evaluation_plan(config)
    write_json(out / "manifest.json", plan)
    write_json(out / "resolved-config.json", config)
    _archive_bc_sources(config, out)
    if config["kind"] in (MORE_DATA_KIND, C2_KIND):
        checkpoint.passport(out, config, plan)
    policy, optimizer = _initialize_bc_model(config, out)
    run_record = _bc_run_record(config, out, plan, policy)
    write_json(out / "run.json", run_record)
    _save_initial_bc_boundary(config, out, policy, optimizer, run_record)
    return plan


def fixed_metrics(policy, cache, micro):
    before = cache.mode
    cache.mode = "diagnostic"
    counts = Counter()
    correct = Counter()
    losses = Counter()
    unit_counts = Counter()
    unit_correct = Counter()
    try:
        with torch.no_grad():
            for spans, _ in label_minibatches(cache.chunks(), cache.masks):
                if policy.spatial_cuda_graphs:
                    torch.compiler.cudagraph_mark_step_begin()
                hidden = prefixes(policy, cache, spans)
                for start in range(0, len(spans), micro):
                    part = spans[start : start + micro]
                    batch, labels, lengths = cache.batch(part)
                    logits, _, _ = policy.forward_sequences(
                        batch,
                        lengths,
                        hidden[:, start : start + len(part)].contiguous(),
                    )
                    ce = F.cross_entropy(logits, labels["indices"], reduction="none").cpu().tolist()
                    predicted = logits.argmax(-1).cpu().tolist()
                    rows = [r for e, a, b in part for r in cache.episodes[e]["decisions"][a:b]]
                    for i, row in enumerate(rows):
                        if not row["supervised"]:
                            continue
                        target, _ = decision_indices(row)
                        action = row["actions"][target]
                        kind = action["kind"]
                        counts[kind] += 1
                        correct[kind] += predicted[i] == target
                        losses[kind] += ce[i]
                        units = {u["id"]: u["type"] for u in row["observation"]["units"]}
                        unit = units.get(action.get("unit"), "no_unit")
                        unit_counts[unit] += 1
                        unit_correct[unit] += predicted[i] == target
        n = sum(counts.values())
        return {
            "labels": n,
            "ce": sum(losses.values()) / n,
            "accuracy": sum(correct.values()) / n,
            "by_kind": {
                k: {
                    "labels": counts[k],
                    "ce": losses[k] / counts[k],
                    "accuracy": correct[k] / counts[k],
                }
                for k in counts
            },
            "by_unit_type": {
                k: {
                    "labels": unit_counts[k],
                    "accuracy": unit_correct[k] / unit_counts[k],
                }
                for k in unit_counts
            },
        }
    finally:
        cache.mode = before


def measure(config, plan, out, path, epoch, steps, policy, fixed, micro, shuffle):
    fingerprint = policy_hash(policy.state_dict())
    cache = read_json(out / "evaluations.json") if (out / "evaluations.json").exists() else {}
    previous = cache.get("development:" + fingerprint)
    if previous and "fixed_training" in previous:
        _refresh_selection(config, out)
        return previous
    with isolated_rng(shuffle):
        diagnostic = fixed_metrics(policy, fixed, micro)
        point = target_evaluate(config, plan, out, path, "development")
    point = {
        **point,
        "checkpoint": path,
        "epoch": epoch,
        "Adam_steps": steps,
        "fixed_training": diagnostic,
    }
    evaluations = read_json(out / "evaluations.json")
    evaluations["development:" + point["policy_hash"]] = point
    write_json(out / "evaluations.json", evaluations)
    _refresh_selection(config, out)
    return point


class ActiveClock(ActiveSessionClock):
    def __init__(self, offset: float = 0.0) -> None:
        super().__init__(offset, timer=lambda: perf_counter())


def _chunk_priority_order(data: dict[str, Any], chunks: list[Span]) -> list[Span]:
    entries = data["entries"]
    matches = data.get("matches", [])
    sizes = []
    for i, entry in enumerate(entries):
        if "map_size" in entry and "army_size" in entry:
            sizes.append((int(entry["map_size"]), int(entry["army_size"])))
        else:
            row = matches[i // 2]
            sizes.append((int(row["map_size"]), int(row["army_size"])))
    largest = sorted(range(len(entries)), key=lambda i: sizes[i], reverse=True)
    rank = {episode_index: pos for pos, episode_index in enumerate(largest)}
    return sorted(chunks, key=lambda span: rank[span[0]])


def _calibrate_bc_microbatch(
    config: dict[str, Any],
    out: Path,
    data: dict[str, Any],
    chunks: list[Span],
    policy: Policy,
    saved: dict[str, Any],
    cache: PreparedDemonstrations,
    monitor: ResourceMonitor,
    *,
    restart: dict[str, Any] | None,
    source: str,
    prepared: float,
    input_preparation: dict[str, Any],
) -> tuple[int, torch.optim.Optimizer]:
    """Probe real BC steps at descending microbatch sizes after resetting model and Adam."""
    warm: list[dict[str, Any]] = []
    # The largest map/army first; use real fragments, including nonzero prefixes.
    order = _chunk_priority_order(data, chunks)
    probes = list(label_minibatches(order, cache.masks))[:2]
    selected = None
    for micro in (8, 4, 2, 1):
        monitor.mark("initial_warmup", microbatch=micro)
        policy.load_state_dict(saved["model"])
        optimizer = adam(policy, config, "cuda")
        optimizer.load_state_dict(saved["optimizer"])

        def backward(
            policy: Policy,
            cache: PreparedDemonstrations,
            spans: list[Span],
            count: int,
            microbatch: int = micro,
        ) -> torch.Tensor:
            return backward_minibatch_async(policy, cache, spans, count, microbatch)

        try:
            start = perf_counter()
            try:
                swap = psutil.swap_memory()
            except RuntimeError:
                swap = None
            for spans, n in probes:
                bc_optimizer_step(
                    policy,
                    optimizer,
                    backward,
                    cache,
                    spans,
                    n,
                    max_gradient_norm=0.5,
                    asynchronous=True,
                )
                torch.cuda.synchronize()
            try:
                end_swap = psutil.swap_memory() if swap is not None else None
            except RuntimeError:
                end_swap = None
            paging = (
                max(0, end_swap.sin - swap.sin) + max(0, end_swap.sout - swap.sout)
                if swap is not None and end_swap is not None
                else 0
            )
            if paging > 512 * 1024**2 and psutil.virtual_memory().available < 1024**3:
                raise MemoryError("Pronounced paging during full BC steps")
            selected = micro
            warm.append(
                {
                    "microbatch": micro,
                    "passed": True,
                    "seconds": perf_counter() - start,
                    "free_VRAM": torch.cuda.mem_get_info()[0],
                    "available_RAM": psutil.virtual_memory().available,
                    "paging_bytes": paging,
                    "cache": cache.statistics(),
                }
            )
            break
        except (torch.cuda.OutOfMemoryError, MemoryError) as error:
            warm.append({"microbatch": micro, "passed": False, "error": str(error)})
            optimizer.zero_grad(set_to_none=True)
            del optimizer
            gc.collect()
            torch.cuda.empty_cache()
    if selected is None:
        raise MemoryError("Target BC does not fit at microbatch 1")
    if restart and (out / "memory-preparation.json").exists():
        shutil.copy2(
            out / "memory-preparation.json",
            out / "memory-preparation-before-input-fix.json",
        )
    write_json(
        out / "memory-preparation.json",
        {
            "attempts": warm,
            "selected_microbatch": selected,
            "backend": "prepared_numpy_CPU",
            "source_checkpoint": source,
            "probe_updates_discarded": True,
            "preparation_seconds": perf_counter() - prepared,
            "one_time_input_preparation_seconds": input_preparation["seconds"],
        },
    )
    # Keep the successful probe optimizer alive through the caller's reset.
    return selected, optimizer


def _compute_initial_offline_baselines(
    config: dict[str, Any],
    out: Path,
    policy: Policy,
    fixed: PreparedDemonstrations,
    holdout: PreparedDemonstrations | None,
    selected: int,
    shuffle: random.Random,
    monitor: ResourceMonitor,
    *,
    restart: dict[str, Any] | None,
    fixed_entries: list[dict[str, Any]],
    holdout_entries: list[dict[str, Any]],
    fixed_limit: int,
    holdout_limit: int,
) -> None:
    """Record fixed-policy diagnostics with isolated RNG before any accepted step."""
    kind = config["kind"]
    monitor.mark("offline_metrics")
    if not restart and not (out / "fixed-training-initial.json").exists():
        with isolated_rng(shuffle):
            initial_metrics = fixed_metrics(policy, fixed, selected)
        write_json(out / "fixed-training-initial.json", initial_metrics)
        if kind == C2_KIND and holdout is not None:
            with isolated_rng(shuffle):
                initial_holdout = fixed_metrics(policy, holdout, selected)
            write_json(out / "holdout-initial.json", initial_holdout)
            if not (out / "baseline-offline-metrics.json").exists():
                fixed_c = PreparedDemonstrations(out, fixed_entries, "cuda", fixed_limit, variant="C")
                holdout_c = PreparedDemonstrations(out, holdout_entries, "cuda", holdout_limit, variant="C")
                baseline_offline = {}
                for ref_name, ref_rel in (
                    ("bc10", "checkpoints/reference-bc-epoch-10.pt"),
                    (
                        "bc_more_data_epoch_10",
                        "checkpoints/reference-bc-more-data-epoch-10.pt",
                    ),
                ):
                    ref_policy = load_policy(config, out / ref_rel, "cuda")
                    fixed_c.spatial_cell_padding = ref_policy.spatial_cell_padding
                    holdout_c.spatial_cell_padding = ref_policy.spatial_cell_padding
                    with isolated_rng(shuffle):
                        ref_fixed = fixed_metrics(ref_policy, fixed_c, selected)
                        ref_holdout = fixed_metrics(ref_policy, holdout_c, selected)
                    baseline_offline[ref_name] = {
                        "checkpoint": ref_rel,
                        "policy_hash": policy_hash(ref_policy.state_dict()),
                        "fixed_training": ref_fixed,
                        "holdout": ref_holdout,
                    }
                    del ref_policy
                fixed_c.clear()
                holdout_c.clear()
                del fixed_c, holdout_c
                gc.collect()
                torch.cuda.empty_cache()
                write_json(out / "baseline-offline-metrics.json", baseline_offline)
            holdout.clear()
        elif (out / "evaluations.json").exists() and (out / "selection.json").exists():
            evaluations = read_json(out / "evaluations.json")
            selection = read_json(out / "selection.json")
            if kind == MORE_DATA_KIND:
                initial_key = "development:" + config["parent_policy_hash"]
                if initial_key in evaluations:
                    evaluations[initial_key]["fixed_training"] = initial_metrics
                    write_json(out / "evaluations.json", evaluations)
                    _refresh_selection(config, out)
            else:
                initial_key = "development:" + selection["baseline"]["policy_hash"]
                evaluations[initial_key]["fixed_training"] = initial_metrics
                selection["baseline"] = evaluations[initial_key]
                if selection["best"]["policy_hash"] == selection["baseline"]["policy_hash"]:
                    selection["best"] = selection["baseline"]
                write_json(out / "evaluations.json", evaluations)
                write_json(out / "selection.json", selection)


def _run_bc_epochs(
    config: dict[str, Any],
    plan: dict[str, Any],
    out: Path,
    policy: Policy,
    optimizer: torch.optim.Optimizer,
    cache: PreparedDemonstrations,
    fixed: PreparedDemonstrations,
    holdout: PreparedDemonstrations | None,
    shuffle: random.Random,
    monitor: ResourceMonitor,
    compilation: CompilationClock,
    frozen: FrozenParamsCheck,
    epoch_state: PreparedEpochState,
    *,
    checkpoint_state: dict[str, Any],
    corpus_labels: int,
    origin_steps: int,
    selected: int,
    source: str,
    resumed: bool,
) -> bool:
    """Run restored or fresh epochs; return whether training reached its final boundary."""
    kind = config["kind"]
    steps = epoch_state.steps
    labels = epoch_state.labels
    history = epoch_state.history
    previous_records = epoch_state.previous_records
    start_epoch = epoch_state.start_epoch
    resume_mid_epoch = epoch_state.resume_mid_epoch
    clock = ActiveClock(epoch_state.clock_offset)
    budget = ActiveBudget(out, elapsed=clock.seconds)
    reason = "epoch_limit"
    last_path = source
    utilization = (
        read_json(out / "GPU-utilization-review.json")
        if (out / "GPU-utilization-review.json").exists()
        else None
    )
    max_steps = config.get("max_adam_steps")

    def save_checkpoint(
        path: str,
        epoch: int,
        cursor: int,
        order: Sequence[Span],
        complete: bool,
        stop_reason: str | None = None,
    ) -> None:
        """Save the drained minibatch stream before publishing its state pointer."""
        monitor.mark("checkpoint", epoch=epoch, Adam_steps=steps)
        serial_order = [[int(s[0]), int(s[1]), int(s[2])] for s in order]
        step_counters = TrainingCounters(
            rollouts_collected=0,
            rollouts_with_accepted_steps=0,
            adam_steps_attempted=steps,
            adam_steps_accepted=steps,
        )
        state = {
            **step_counters.to_state_dict(protocol=kind),
            "epoch": epoch,
            "epoch_completed": complete,
            "batch_cursor": cursor,
            "epoch_order": serial_order,
            "Adam_steps": steps,
            "free_labels_processed": labels,
            "active_seconds_at_save_start": clock.seconds(),
            "active_seconds_limit": budget.limit(),
            "budget_revision_applied": budget.revision,
            "history_positions": cache.counters["history_positions"],
            "prefix_positions": cache.counters["prefix_positions"],
            "epochs_history": deepcopy(history),
            "stop_reason": stop_reason,
        }
        frozen.verify()
        checkpoint.save(
            out / path,
            policy,
            optimizer,
            state,
            shuffle,
            config,
            {"run_id": read_json(out / "run.json")["run_id"]},
        )
        write_json(
            out / "state.json",
            {**state, "active_seconds": clock.seconds(), "latest": path},
        )

    for epoch in range(start_epoch, config["epochs"] + 1):
        if budget.expired():
            reason = "active_time_limit"
            break
        if max_steps is not None and steps >= max_steps:
            reason = "max_adam_steps"
            break
        if resume_mid_epoch and epoch == start_epoch:
            order, cursor, epoch_labels, batches, records = restore_epoch_cursor_and_batches(
                checkpoint_state, history, previous_records, cache.masks
            )
            resume_mid_epoch = False
        else:
            order = cache.chunks()
            shuffle.shuffle(order)
            records = []
            cursor = 0
            epoch_labels = 0
            batches = iter(label_minibatches(order, cache.masks))
        completed = True
        monitor.mark("iterative_training", epoch=epoch, Adam_steps=steps)
        stop_hit = None

        def _check_stop() -> str | None:
            nonlocal stop_hit
            if budget.expired():
                stop_hit = "active_time_limit"
                return stop_hit
            return None

        def _on_step(
            record: dict[str, Any],
            completed_steps: int,
            current_epoch: int = epoch,
            current_records: list[dict[str, Any]] = records,
        ) -> None:
            nonlocal steps, labels, cursor, epoch_labels, utilization
            steps = completed_steps
            labels += record["labels"]
            cursor += 1
            epoch_labels += record["labels"]
            current_records.append(record)
            if steps == 1:
                write_json(
                    out / "first-step-check.json",
                    {"before": 0, "after": 1, "all_counters_equal": True},
                )
            if resumed and steps == origin_steps + 1:
                write_json(
                    out / "first-restarted-step-check.json",
                    {
                        "before": origin_steps,
                        "after": steps,
                        "all_counters_equal": True,
                    },
                )
            if steps % 10 == 0:
                print(
                    f"BC epoch={current_epoch}, Adam={steps}, labels={labels}, CE={record['ce']:.4f}, active={clock.seconds():.1f}s",
                    flush=True,
                )
            if clock.seconds() >= 600 and utilization is None:
                analysis = analyze_telemetry(out / "training-telemetry/resources.jsonl")
                utilization = {
                    "metrics": analysis["metrics"],
                    "last_60s": analysis["windows"][-1] if analysis["windows"] else None,
                    "measured_cache": cache.statistics(),
                    "mean_step_seconds": sum(r["seconds"] for r in current_records) / len(current_records),
                    "prefix_seconds_not_separately_profiled": True,
                    "continuation_reason": "Bounded CPU episode input, full current-weight GRU-prefix and small fixed spatial blocks constrain dispatch. Preserve the validated 1024-label objective and precision; no extra optimization campaign or artificial GPU load.",
                }
                write_json(out / "GPU-utilization-review.json", utilization)

        check_steps = {1}
        if resumed:
            check_steps.add(origin_steps + 1)

        run_bc_minibatch_stream(
            policy=policy,
            optimizer=optimizer,
            cache=cache,
            batches=batches,
            micro_size=selected,
            max_gradient_norm=0.5,
            steps_log_path=out / "steps.jsonl",
            compilation=compilation,
            epoch=epoch,
            initial_steps=steps,
            active_seconds=clock.seconds,
            max_steps=max_steps,
            should_stop=_check_stop,
            on_step_completed=_on_step,
            check_counter_steps=check_steps,
        )
        if epoch_labels < corpus_labels:
            completed = False
            if stop_hit is not None:
                reason = stop_hit
            elif max_steps is not None and steps >= max_steps:
                reason = "max_adam_steps"
        if records:
            epoch_entry = {
                "epoch": epoch,
                "completed": completed,
                "steps": len(records),
                "labels": sum(r["labels"] for r in records),
                "online_CE": sum(r["ce"] * r["labels"] for r in records) / sum(r["labels"] for r in records),
            }
            if kind == C2_KIND:
                monitor.mark("offline_metrics", epoch=epoch, Adam_steps=steps)
                with clock.pause(), isolated_rng(shuffle):
                    epoch_entry["fixed_training"] = fixed_metrics(policy, fixed, selected)
                    if holdout is not None:
                        epoch_entry["holdout"] = fixed_metrics(policy, holdout, selected)
                fixed.clear()
                if holdout is not None:
                    holdout.clear()
            history.append(epoch_entry)
            last_path = (
                f"checkpoints/epoch-{epoch:02d}.pt"
                if completed
                else f"checkpoints/partial-epoch-{epoch:02d}-step-{steps:05d}.pt"
            )
            terminal_step = (
                (not completed and reason != "paused")
                or budget.expired()
                or (max_steps is not None and steps >= max_steps)
                or (completed and epoch == config["epochs"])
            )
            if not terminal_step:
                final_reason = None
            elif max_steps is not None and steps >= max_steps:
                final_reason = "max_adam_steps"
            elif not completed or budget.expired():
                final_reason = "active_time_limit"
            else:
                final_reason = "epoch_limit"
            save_checkpoint(last_path, epoch, cursor, order, completed, stop_reason=final_reason)
        if reason == "paused" or (completed and records and pause_due(out, "epoch", epoch)):
            if (
                kind != C2_KIND
                and completed
                and records
                and epoch in config["evaluate_epochs"]
                and not config.get("skip_milestone_evaluations", False)
            ):
                monitor.mark("evaluation")
                with clock.pause():
                    measure(
                        config,
                        plan,
                        out,
                        last_path,
                        epoch,
                        steps,
                        policy,
                        fixed,
                        selected,
                        shuffle,
                    )
                fixed.clear()
            write_json(
                out / "training.json",
                {
                    "status": "paused",
                    "epochs": history,
                    "Adam_steps": steps,
                    "free_labels_processed": labels,
                    "active_seconds": clock.seconds(),
                    "latest": last_path,
                },
            )
            write_json(
                out / "status.json",
                {"status": "paused", "checkpoint": last_path},
            )
            cache.clear()
            fixed.clear()
            if holdout is not None:
                holdout.clear()
            return False
        terminal = not completed or budget.expired() or (max_steps is not None and steps >= max_steps)
        if terminal and reason == "epoch_limit":
            reason = (
                "max_adam_steps" if (max_steps is not None and steps >= max_steps) else "active_time_limit"
            )
        if (
            kind != C2_KIND
            and records
            and (epoch in config["evaluate_epochs"] or terminal)
            and not config.get("skip_milestone_evaluations", False)
        ):
            monitor.mark("evaluation")
            with clock.pause():
                measure(
                    config,
                    plan,
                    out,
                    last_path,
                    epoch,
                    steps,
                    policy,
                    fixed,
                    selected,
                    shuffle,
                )
            fixed.clear()
            monitor.mark("iterative_training", epoch=epoch)
        write_json(
            out / "training.json",
            {
                "status": "running",
                "epochs": history,
                "Adam_steps": steps,
                "free_labels_processed": labels,
                "active_seconds": clock.seconds(),
                "latest": last_path,
                "cache": cache.statistics(),
                "microbatch": selected,
            },
        )
        if terminal:
            break
    # Ensure state.json records the terminal stop_reason if we broke between epochs.
    if (out / "state.json").exists():
        final_state = read_json(out / "state.json")
        if final_state.get("stop_reason") != reason:
            final_state["stop_reason"] = reason
            final_state["active_seconds"] = clock.seconds()
            write_json(out / "state.json", final_state)
    # A budget boundary between epochs must also evaluate the last changed policy.
    if kind != C2_KIND and steps and not config.get("skip_milestone_evaluations", False):
        monitor.mark("evaluation")
        with clock.pause():
            measure(
                config,
                plan,
                out,
                last_path,
                history[-1]["epoch"],
                steps,
                policy,
                fixed,
                selected,
                shuffle,
            )
    frozen.verify()
    elapsed = clock.seconds()
    if elapsed >= 600 and (out / "training-telemetry/resources.jsonl").exists():
        analysis = analyze_telemetry(out / "training-telemetry/resources.jsonl")
        utilization = {
            "metrics": analysis["metrics"],
            "last_60s": analysis["windows"][-1] if analysis["windows"] else None,
            "windows_count": len(analysis["windows"]),
            "min_60s_core_average": min((w["core_average"] for w in analysis["windows"]), default=None),
            "max_60s_core_average": max((w["core_average"] for w in analysis["windows"]), default=None),
            "measured_cache": cache.statistics(),
            "prefix_seconds_not_separately_profiled": True,
            "continuation_reason": "Bounded CPU episode input, full current-weight GRU-prefix and small fixed spatial blocks constrain dispatch. Preserve the validated 1024-label objective and precision; no extra optimization campaign or artificial GPU load.",
        }
        write_json(out / "GPU-utilization-review.json", utilization)
    write_json(
        out / "training.json",
        {
            "status": "complete",
            "stop_reason": reason,
            "epochs": history,
            "Adam_steps": steps,
            "free_labels_processed": labels,
            "active_seconds": elapsed,
            "time_overshoot": max(0, elapsed - budget.limit()),
            "latest": last_path,
            "microbatch": selected,
            "cache": cache.statistics(),
            "value_head_unchanged": True,
            "GPU_utilization_review": utilization,
        },
    )
    cache.clear()
    fixed.clear()
    return True


def train(config: dict[str, Any], plan: dict[str, Any], out: Path, *, resume: bool = False) -> None:
    kind = config["kind"]
    restart_path = out / "restart-input-v2.json"
    restart = read_json(restart_path) if restart_path.exists() else None
    if (out / "training.json").exists():
        training_state = read_json(out / "training.json")
        if training_state.get("status") == "complete":
            return
        if kind not in (MORE_DATA_KIND, C2_KIND) and (not restart or restart.get("consumed")):
            raise ValueError("Partial training exists; optimization must not repeat automatically")
        if kind in (MORE_DATA_KIND, C2_KIND) and not resume:
            raise ValueError("Partial training exists; use explicit resume phase to continue")
    if not config.get("skip_baseline_evaluation", False):
        baseline(config, plan, out)
    data = read_json(out / "collection.json")
    if data["status"] != "complete":
        raise ValueError("Collection is incomplete")
    all_entries = list(data["entries"])
    if kind == C2_KIND and data.get("holdout_training"):
        all_entries.extend(data["holdout_training"])
    input_preparation = prepare_inputs(out, all_entries, workers=config["workers"])
    prepared_all = input_preparation["entries"]
    lookup = {_entry_lookup_key(e): e for e in prepared_all}
    prepared_entries = [lookup[_entry_lookup_key(e)] for e in data["entries"]]
    fixed_entries = [lookup[_entry_lookup_key(e)] for e in data["fixed_training"]]
    holdout_entries = (
        [lookup[_entry_lookup_key(e)] for e in data["holdout_training"]]
        if kind == C2_KIND and data.get("holdout_training")
        else []
    )
    configure(17, "cuda")
    shuffle = random.Random(seed_for(17, 0, "shuffle", namespace=config["experiment"]["shuffle_namespace"]))
    state_record = read_json(out / "state.json") if (out / "state.json").exists() else None
    resuming_more_data = (
        kind in (MORE_DATA_KIND, C2_KIND)
        and state_record is not None
        and (resume or int(state_record.get("Adam_steps", 0)) > 0 or int(state_record.get("epoch", 0)) > 0)
    )
    source = select_bc_checkpoint(out, restart_file="restart-input-v2.json", resume=resuming_more_data)
    prepared = perf_counter()
    policy = load_policy(config, out / source, "cuda")
    saved = torch.load(out / source, map_location="cpu", weights_only=False)
    initial_saved = torch.load(out / "checkpoints/initial.pt", map_location="cpu", weights_only=False)
    fixed_limit = min(config["cache_bytes"] // 16, 256 * 1024**2)
    holdout_limit = fixed_limit if holdout_entries else 0
    cache = PreparedDemonstrations(
        out,
        prepared_entries,
        "cuda",
        config["cache_bytes"] - fixed_limit - holdout_limit,
        variant=policy.variant,
    )
    fixed = PreparedDemonstrations(out, fixed_entries, "cuda", fixed_limit, variant=policy.variant)
    holdout = (
        PreparedDemonstrations(out, holdout_entries, "cuda", holdout_limit, variant=policy.variant)
        if holdout_entries
        else None
    )
    cache.spatial_cell_padding = policy.spatial_cell_padding
    fixed.spatial_cell_padding = policy.spatial_cell_padding
    if holdout is not None:
        holdout.spatial_cell_padding = policy.spatial_cell_padding
    chunks = cache.chunks()
    with (
        CompilationClock() as compilation,
        ResourceMonitor(out / "training-telemetry") as monitor,
    ):
        compilation.monitor = monitor
        if resuming_more_data and (out / "memory-preparation.json").exists():
            selected = int(read_json(out / "memory-preparation.json")["selected_microbatch"])

        else:
            selected, optimizer = _calibrate_bc_microbatch(
                config,
                out,
                data,
                chunks,
                policy,
                saved,
                cache,
                monitor,
                restart=restart,
                source=source,
                prepared=prepared,
                input_preparation=input_preparation,
            )
        # Probe updates never enter the experiment.
        policy.load_state_dict(saved["model"])
        optimizer = adam(policy, config, "cuda")
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved, shuffle)
        optimizer.zero_grad(set_to_none=True)
        assert_equal(policy.state_dict(), saved["model"])
        assert_equal(optimizer.state_dict(), saved["optimizer"])
        for name, param in policy.value.named_parameters():
            assert_equal(param, initial_saved["model"]["value." + name])
        frozen = FrozenParamsCheck(policy.value)
        _compute_initial_offline_baselines(
            config,
            out,
            policy,
            fixed,
            holdout,
            selected,
            shuffle,
            monitor,
            restart=restart,
            fixed_entries=fixed_entries,
            holdout_entries=holdout_entries,
            fixed_limit=fixed_limit,
            holdout_limit=holdout_limit,
        )
        cache.counters.clear()
        fixed.clear()
        if holdout is not None:
            holdout.clear()
        restoration, epoch_state = restore_bc_training_state(
            out,
            saved["state"],
            state_record,
            restart=restart,
            kind=kind,
            resume=resume,
            config=config,
            cache=cache,
            source=source,
            data_labels=data["labels"] if restart else None,
        )
        origin_steps = restoration.adam_steps
        steps = epoch_state.steps

        if restart:
            # Finish the planned evaluation interrupted by the requested stop.
            monitor.mark("evaluation")
            measure(
                config,
                plan,
                out,
                source,
                restart["checkpoint_boundary"]["epoch"],
                steps,
                policy,
                fixed,
                selected,
                shuffle,
            )
            fixed.clear()
            write_json(
                out / "restart-state-verification.json",
                {
                    "model_equal": True,
                    "Adam_equal": True,
                    "RNG_restored": True,
                    "source": source,
                    "Adam_counter": steps,
                    "next_Adam_counter": steps + 1,
                    "active_seconds_preserved": restart["checkpoint_boundary"]["active_seconds"],
                    "history_and_order_preserved": True,
                },
            )
            restart["consumed"] = True
            write_json(restart_path, restart)
        elif resuming_more_data and epoch_state.evaluate_needed:
            monitor.mark("evaluation")
            measure(
                config,
                plan,
                out,
                source,
                restoration.epoch,
                steps,
                policy,
                fixed,
                selected,
                shuffle,
            )
            fixed.clear()
        if not _run_bc_epochs(
            config,
            plan,
            out,
            policy,
            optimizer,
            cache,
            fixed,
            holdout,
            shuffle,
            monitor,
            compilation,
            frozen,
            epoch_state,
            checkpoint_state=saved["state"],
            corpus_labels=data["labels"],
            origin_steps=origin_steps,
            selected=selected,
            source=source,
            resumed=bool(restart or resuming_more_data),
        ):
            return
    del policy, optimizer
    gc.collect()
    torch.cuda.empty_cache()


def _check_bc_run_admission(out: Path, phase: str) -> None:
    if phase in ("train", "run", "resume"):
        assert_not_closed(out)
    if phase == "resume":
        (out / "pause.request").unlink(missing_ok=True)
    if (
        phase in ("train", "run")
        and (out / "training.json").exists()
        and read_json(out / "training.json").get("status") == "paused"
    ):
        raise ValueError(
            "Paused BC boundary needs an explicit resume procedure; automatic resume on run/train is unavailable"
        )


def _finish_bc_evaluation(out: Path) -> None:
    selected = read_json(out / "selection.json")["best"]
    directory = out / selected["runs"]["argmax"]["cp"]["directory"]
    for point in read_json(out / "evaluations.json").values():
        if not isinstance(point, dict) or "runs" not in point:
            continue
        run = point["runs"]["argmax"]["cp"]
        if not _complete_eval_dir(out / run["directory"], run["summary"]["matches"]):
            raise ValueError("Incomplete saved evaluation artifacts")
    write_json(
        out / "replay-verification.json",
        {
            outcome: verify_replay(case_directory)
            for outcome, case_directory in select_outcome_representatives(directory).items()
        },
    )
    write_json(
        out / "status.json",
        {
            "status": "complete",
            "confirmation": read_json(out / "confirmation.json")["status"],
        },
    )


def run(phase, config, out, workers, device):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if workers not in (4, 8, 10) or device != "cuda":
        raise ValueError("This experiment uses 4, 8, or 10 CPU workers and CUDA")
    if phase not in (
        "prepare",
        "collect",
        "train",
        "evaluate",
        "report",
        "run",
        "resume",
    ):
        raise ValueError("Unsupported target BC phase")
    if phase == "resume" and config["kind"] not in (MORE_DATA_KIND, C2_KIND):
        raise ValueError(
            "resume phase is supported only for rl01.cp_imitation_target_more_data and rl01.cp_imitation_target_c2"
        )
    with experiment_run_guard(
        out,
        phase=phase,
        preflight=lambda: _check_bc_run_admission(out, phase),
        on_error_report=lambda: report(config, out),
    ):
        plan = prepare(config, out)
        if (out / "status.json").exists() and read_json(out / "status.json").get("status") == "complete":
            report(config, out)
            return
        if phase in ("collect", "run", "resume") and not (
            (out / "collection.json").exists()
            and read_json(out / "collection.json").get("status") == "complete"
        ):
            write_json(out / "status.json", {"status": "collecting"})
            collect(config, plan, out)
            write_json(out / "status.json", {"status": "collection_complete"})
        if phase in ("train", "run", "resume") and not (
            (out / "training.json").exists() and read_json(out / "training.json").get("status") == "complete"
        ):
            write_json(out / "status.json", {"status": "training"})
            train(config, plan, out, resume=(phase == "resume"))
            if read_json(out / "status.json").get("status") == "paused":
                report(config, out)
                return
            write_json(out / "status.json", {"status": "training_complete"})
        if phase in ("evaluate", "run", "resume"):
            write_json(out / "status.json", {"status": "evaluating"})
            evaluate(config, plan, out)
            _finish_bc_evaluation(out)
        report(config, out)
