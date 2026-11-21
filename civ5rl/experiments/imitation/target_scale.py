"""One bounded local or remote CP--CP behavioral-cloning experiment on target armies."""

from __future__ import annotations

import gc
import platform
import random
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import psutil
import torch

from civ5rl.config import load_config as resolve_config
from civ5rl.core.model import Policy
from civ5rl.engine.evaluation.replay import verify_replay
from civ5rl.engine.imitation.core import label_minibatches
from civ5rl.engine.imitation.pipeline import (
    backward_minibatch_async,
)
from civ5rl.engine.imitation.storage import PreparedDemonstrations, prepare_inputs
from civ5rl.engine.runtime.control import (
    RunLock,
    TrainingCounters,
    assert_not_closed,
)
from civ5rl.engine.runtime.telemetry import CompilationClock, ResourceMonitor
from civ5rl.engine.runtime.transactions import FrozenParamsCheck, bc_optimizer_step
from civ5rl.experiments._shared.lifecycle import (
    experiment_run_guard,
)
from civ5rl.experiments._shared.provenance_checks import verify_checkpoint_identity
from civ5rl.experiments.imitation.corpus import (
    collect,
    entry_lookup_key,
)
from civ5rl.experiments.imitation.epochs import fixed_metrics, measure_checkpoint, run_bc_epochs
from civ5rl.experiments.imitation.helpers import (
    adam,
    assert_equal,
    isolated_rng,
    load_policy,
    restore_rng,
)
from civ5rl.experiments.imitation.restoration import (
    restore_bc_training_state,
    saved_adam_steps,
)
from civ5rl.experiments.imitation.selection import refresh_selection
from civ5rl.experiments.imitation.target_evaluation import (
    baseline,
    complete_eval_dir,
    evaluate,
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
from civ5rl.provenance.identity import policy_hash
from civ5rl.reporting.evaluation_audit import select_outcome_representatives
from civ5rl.reporting.target_scale import report
from civ5sim.engine import digest

if TYPE_CHECKING:
    from civ5rl.engine.runtime.sequence import Span


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
    if config["kind"] not in SUPPORTED_KINDS:
        raise ValueError("Unsupported target BC profile")
    for field in ("workers", "labels_per_minibatch", "sequence_length"):
        if type(config[field]) is not int or config[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
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


def _initialize_bc_model(config, out):
    kind = config["kind"]
    configure(config["seed"], "cpu")
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


def _bc_run_record(config, plan, policy):
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
        "seed": config["seed"],
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
        seed_for(config["seed"], 0, "shuffle", namespace=config["experiment"]["shuffle_namespace"])
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
    checkpoint.passport(out, config, plan)
    policy, optimizer = _initialize_bc_model(config, out)
    run_record = _bc_run_record(config, plan, policy)
    _save_initial_bc_boundary(config, out, policy, optimizer, run_record)
    write_json(out / "run.json", run_record)
    return plan


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
    source: str,
    prepared: float,
    input_preparation: dict[str, Any],
) -> tuple[int, torch.optim.Optimizer]:
    """Probe real BC steps at descending microbatch sizes after resetting model and Adam."""
    warm: list[dict[str, Any]] = []
    # The largest map/army first; use real fragments, including nonzero prefixes.
    order = _chunk_priority_order(data, chunks)
    probes = list(label_minibatches(order, cache.masks, size=config["labels_per_minibatch"]))[:2]
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
    fixed_entries: list[dict[str, Any]],
    holdout_entries: list[dict[str, Any]],
    fixed_limit: int,
    holdout_limit: int,
) -> None:
    """Record fixed-policy diagnostics with isolated RNG before any accepted step."""
    kind = config["kind"]
    monitor.mark("offline_metrics")
    if not (out / "fixed-training-initial.json").exists():
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
                    refresh_selection(config, out)
            else:
                initial_key = "development:" + selection["baseline"]["policy_hash"]
                evaluations[initial_key]["fixed_training"] = initial_metrics
                selection["baseline"] = evaluations[initial_key]
                if selection["best"]["policy_hash"] == selection["baseline"]["policy_hash"]:
                    selection["best"] = selection["baseline"]
                write_json(out / "evaluations.json", evaluations)
                write_json(out / "selection.json", selection)


def train(
    config: dict[str, Any], plan: dict[str, Any], out: Path, *, workers: int, resume: bool = False
) -> None:
    kind = config["kind"]
    if (out / "training.json").exists():
        training_state = read_json(out / "training.json")
        if training_state.get("status") == "complete":
            return
        if not resume:
            raise ValueError("Partial training exists; use explicit resume phase to continue")
    if not config.get("skip_baseline_evaluation", False):
        baseline(config, plan, out)
    data = read_json(out / "collection.json")
    if data["status"] != "complete":
        raise ValueError("Collection is incomplete")
    all_entries = list(data["entries"])
    if kind == C2_KIND and data.get("holdout_training"):
        all_entries.extend(data["holdout_training"])
    input_preparation = prepare_inputs(out, all_entries, workers=workers)
    prepared_all = input_preparation["entries"]
    lookup = {entry_lookup_key(e): e for e in prepared_all}
    prepared_entries = [lookup[entry_lookup_key(e)] for e in data["entries"]]
    fixed_entries = [lookup[entry_lookup_key(e)] for e in data["fixed_training"]]
    holdout_entries = (
        [lookup[entry_lookup_key(e)] for e in data["holdout_training"]]
        if kind == C2_KIND and data.get("holdout_training")
        else []
    )
    configure(config["seed"], "cuda")
    shuffle = random.Random(
        seed_for(config["seed"], 0, "shuffle", namespace=config["experiment"]["shuffle_namespace"])
    )
    state_record = read_json(out / "state.json") if (out / "state.json").exists() else None
    resuming = resume or bool(
        state_record and (saved_adam_steps(state_record) > 0 or int(state_record.get("epoch", 0)) > 0)
    )
    source = (
        state_record.get("latest", "checkpoints/initial.pt") if state_record else "checkpoints/initial.pt"
    )
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
    chunks = cache.chunks(config["sequence_length"])
    with (
        CompilationClock() as compilation,
        ResourceMonitor(out / "training-telemetry") as monitor,
    ):
        compilation.monitor = monitor
        if resuming and (out / "memory-preparation.json").exists():
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
            fixed_entries=fixed_entries,
            holdout_entries=holdout_entries,
            fixed_limit=fixed_limit,
            holdout_limit=holdout_limit,
        )
        cache.counters.clear()
        fixed.clear()
        if holdout is not None:
            holdout.clear()
        epoch_state = restore_bc_training_state(
            out,
            saved["state"],
            state_record,
            resuming=resuming,
            config=config,
            cache=cache,
            source=source,
        )
        origin_steps = epoch_state.steps
        steps = epoch_state.steps

        if resuming and epoch_state.evaluate_needed:
            monitor.mark("evaluation")
            measure_checkpoint(
                config,
                plan,
                out,
                source,
                epoch_state.saved_epoch,
                steps,
                policy,
                fixed,
                selected,
                shuffle,
            )
            fixed.clear()
        if not run_bc_epochs(
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
            resumed=resuming,
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
        if not complete_eval_dir(out / run["directory"], run["summary"]["matches"]):
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
    if phase == "report":
        run_path = out / "run.json"
        if not run_path.is_file():
            raise FileNotFoundError(f"Target BC run record is missing: {run_path}")
        with RunLock(out):
            run_record = read_json(run_path)
            if run_record["kind"] != config["kind"]:
                raise ValueError("Report configuration kind differs from the saved target BC run")
            return report(run_record, out)
    if phase not in (
        "prepare",
        "collect",
        "train",
        "evaluate",
        "run",
        "resume",
    ):
        raise ValueError("Unsupported target BC phase")
    if type(workers) is not int or workers <= 0 or device != "cuda":
        raise ValueError("Target BC requires a positive CPU worker count and CUDA")
    out.mkdir(parents=True, exist_ok=True)
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
            collect(config, plan, out, workers=workers)
            write_json(out / "status.json", {"status": "collection_complete"})
        if phase in ("train", "run", "resume") and not (
            (out / "training.json").exists() and read_json(out / "training.json").get("status") == "complete"
        ):
            write_json(out / "status.json", {"status": "training"})
            train(config, plan, out, workers=workers, resume=(phase == "resume"))
            if read_json(out / "status.json").get("status") == "paused":
                report(config, out)
                return
            write_json(out / "status.json", {"status": "training_complete"})
        if phase in ("evaluate", "run", "resume"):
            write_json(out / "status.json", {"status": "evaluating"})
            evaluate(config, plan, out)
            _finish_bc_evaluation(out)
        report(config, out)
