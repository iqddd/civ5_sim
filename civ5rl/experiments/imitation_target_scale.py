"""One bounded local or remote CP--CP behavioral-cloning experiment on target armies."""
import gc
import inspect
import io
import json
import platform
import random
import shutil
import sys
import traceback
import zipfile
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import psutil
import torch
from torch.nn import functional as F

from civ5rl.config import load_config as resolve_config
from civ5rl.engine.control import RunLock, assert_not_closed, pause_due
from civ5rl.engine.imitation import decision_indices, label_minibatches
from civ5rl.engine.imitation_performance import (
    backward_minibatch_async,
    run_bc_minibatch_stream,
)
from civ5rl.engine.imitation_stream import PreparedDemonstrations, prepare_inputs
from civ5rl.engine.ppo_fast import prefixes
from civ5rl.engine.resources import ResourceMonitor, analyze_telemetry
from civ5rl.engine.spans import Span
from civ5rl.engine.training import FrozenParamsCheck, bc_optimizer_step
from civ5rl.experiments.evaluation import paired_world_bootstrap, verify_replay
from civ5rl.experiments.imitation_helpers import (
    CompilationClock,
    adam,
    assert_equal,
    isolated_rng,
    load_policy,
    pool,
    restore_rng,
)
from civ5rl.core.model import Policy
from civ5rl.experiments.ppo_after_imitation import play_case, policy_hash
from civ5rl.experiments.ppo_target_scale import evaluate as target_evaluate
from civ5rl.protocols.imitation_target_scale import (
    C2_KIND,
    MORE_DATA_KIND,
    SUPPORTED_KINDS,
    match_task,
)
from civ5rl.protocols.ppo_target_scale import (
    HOLDOUT_WORLDS,
    TargetFactory,
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
    return {
        str(Path(name)): file_sha256(ROOT / name)
        for name in names
        if (ROOT / name).is_file()
    }


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


def _evaluation_key(config, plan, purpose, fingerprint):
    return digest(
        {
            "purpose": purpose,
            "policy": fingerprint,
            "mode": "argmax",
            "temperature": config.get("policy_temperature", 1.0),
            "cases": [c["case_hash"] for c in plan[purpose]],
            "evaluator": _evaluator_digest(),
            "search": config["teacher_search"],
        }
    )


def _complete_eval_dir(directory, expected_matches):
    directory = Path(directory)
    if not directory.is_dir():
        return False
    paths = list(directory.glob("*/metrics.json"))
    if len(paths) != expected_matches:
        return False
    for path in paths:
        for name in ("case.json", "result.json", "final.json", "replay.json"):
            if not (path.parent / name).is_file():
                return False
    return True


def _format_corpus_entry(entry, corpus_source, path, map_size, army_size, sha256=None):
    match_id = int(entry["match_id"])
    side = int(entry["side"])
    if corpus_source not in ("old", "new") or side not in (0, 1):
        raise ValueError("Invalid corpus_source or side")
    formatted = {
        **entry,
        "entry_id": f"{corpus_source}:match-{match_id:06d}:side-{side}",
        "corpus_source": corpus_source,
        "match_id": match_id,
        "side": side,
        "path": path,
        "map_size": int(map_size),
        "army_size": int(army_size),
    }
    if sha256 is not None:
        formatted["sha256"] = sha256
    return formatted


def _load_old_corpus(config, plan):
    old_root = ROOT / config["old_corpus_run"]
    old_collection_path = old_root / "collection.json"
    if not old_collection_path.is_file():
        raise ValueError("Missing old BC collection.json")
    old_data = read_json(old_collection_path)
    if old_data.get("status") != "complete" or old_data.get("labels") != 200483:
        raise ValueError("Unexpected old BC collection status or label count")
    old_matches = old_data.get("matches", [])
    if len(old_matches) != 748:
        raise ValueError("Unexpected old BC match count")
    entries = []
    matches = []
    for row in old_matches:
        match_id = int(row["match_id"])
        match_dir = old_root / f"data/match-{match_id:06d}"
        case_path = match_dir / "case.json"
        if not case_path.is_file():
            raise ValueError(f"Missing old BC case.json: {case_path}")
        case = read_json(case_path)
        if (
            case.get("split") != "train"
            or case.get("parent_world") not in plan["base"]["worlds"]["training"]
        ):
            raise ValueError("Old BC match world/split mismatch")
        row_entries = []
        for entry in row["entries"]:
            rel_in_old = Path(entry["path"]).as_posix()
            if Path(rel_in_old).is_absolute():
                full_path = Path(rel_in_old)
            else:
                full_path = old_root / rel_in_old
            if not full_path.is_file():
                raise ValueError(f"Missing old demonstration file: {full_path}")
            workspace_rel = full_path.resolve().relative_to(ROOT.resolve()).as_posix()
            formatted = _format_corpus_entry(
                entry,
                "old",
                workspace_rel,
                row["map_size"],
                row["army_size"],
                sha256=file_sha256(full_path),
            )
            row_entries.append(formatted)
            entries.append(formatted)
        matches.append({**row, "corpus_source": "old", "entries": row_entries})
    if sum(e["labels"] for e in entries) != 200483 or len(entries) != 1496:
        raise ValueError("Old BC entries label total or count mismatch")
    return {
        "labels": 200483,
        "positions": sum(e["positions"] for e in entries),
        "matches_count": len(matches),
        "entries": entries,
        "matches": matches,
    }


def _load_more_data_corpus(config, plan):
    more_root = ROOT / config["more_data_corpus_run"]
    more_collection_path = more_root / "collection.json"
    if not more_collection_path.is_file():
        raise ValueError("Missing more-data BC collection.json")
    more_data = read_json(more_collection_path)
    if more_data.get("status") != "complete" or more_data.get("new_labels") != 800333:
        raise ValueError("Unexpected more-data BC collection status or label count")
    new_matches = more_data.get("new_matches", [])
    if len(new_matches) != 3020:
        raise ValueError("Unexpected more-data BC match count")
    entries = []
    matches = []
    for row in new_matches:
        match_id = int(row["match_id"])
        match_dir = more_root / f"data/match-{match_id:06d}"
        case_path = match_dir / "case.json"
        if not case_path.is_file():
            raise ValueError(f"Missing more-data BC case.json: {case_path}")
        case = read_json(case_path)
        if (
            case.get("split") != "train"
            or case.get("parent_world") not in plan["base"]["worlds"]["training"]
        ):
            raise ValueError("More-data BC match world/split mismatch")
        row_entries = []
        for entry in row["entries"]:
            rel_in_more = Path(entry["path"]).as_posix()
            if Path(rel_in_more).is_absolute():
                full_path = Path(rel_in_more)
            else:
                full_path = more_root / rel_in_more
            if not full_path.is_file():
                raise ValueError(f"Missing more-data demonstration file: {full_path}")
            workspace_rel = full_path.resolve().relative_to(ROOT.resolve()).as_posix()
            sha = entry.get("sha256") or file_sha256(full_path)
            if file_sha256(full_path) != sha:
                raise ValueError(f"Demonstration SHA256 mismatch: {full_path}")
            formatted = _format_corpus_entry(
                entry,
                "new",
                workspace_rel,
                row["map_size"],
                row["army_size"],
                sha256=sha,
            )
            row_entries.append(formatted)
            entries.append(formatted)
        matches.append({**row, "corpus_source": "new", "entries": row_entries})
    if sum(e["labels"] for e in entries) != 800333 or len(entries) != 6040:
        raise ValueError("More-data BC entries label total or count mismatch")
    fixed_keys = {_entry_lookup_key(e) for e in more_data.get("fixed_training", [])}
    fixed_training = [e for e in entries if _entry_lookup_key(e) in fixed_keys]
    if sum(e["labels"] for e in fixed_training) < config["fixed_training_labels"]:
        raise ValueError("Fixed training subset from more-data corpus is incomplete")
    return {
        "labels": 800333,
        "positions": sum(e["positions"] for e in entries),
        "matches_count": len(matches),
        "entries": entries,
        "matches": matches,
        "fixed_training": fixed_training,
        "fixed_labels": sum(e["labels"] for e in fixed_training),
        "coverage": more_data.get("coverage", {}),
    }


def prepare(config, out):
    kind = config["kind"]
    if (out / "run.json").exists():
        if read_json(out / "resolved-config.json") != config:
            raise ValueError("Configuration changed")
        return read_json(out / "manifest.json")
    if file_sha256(ROOT / config["parent"]) != config["parent_sha256"]:
        raise ValueError("Wrong training parent checkpoint SHA256")
    if kind in (MORE_DATA_KIND, C2_KIND):
        if (
            file_sha256(ROOT / config["evaluation_reference"])
            != config["evaluation_reference_sha256"]
        ):
            raise ValueError("Wrong evaluation reference checkpoint SHA256")
        ref_state = torch.load(
            ROOT / config["evaluation_reference"],
            map_location="cpu",
            weights_only=False,
        )
        if (
            policy_hash(ref_state["model"])
            != config["evaluation_reference_policy_hash"]
        ):
            raise ValueError("Wrong evaluation reference policy_hash")
    if kind == C2_KIND:
        if file_sha256(ROOT / config["bc10_checkpoint"]) != config["bc10_sha256"]:
            raise ValueError("Wrong BC10 checkpoint SHA256")
        bc10_state = torch.load(
            ROOT / config["bc10_checkpoint"], map_location="cpu", weights_only=False
        )
        if policy_hash(bc10_state["model"]) != config["bc10_policy_hash"]:
            raise ValueError("Wrong BC10 policy_hash")
    original = read_json(ROOT / config["target_manifest"])
    base = original["base"]
    worlds = base["worlds"]
    if len(worlds["training"]) != 57 or set(worlds["training"]) & set(
        worlds["validation"]
    ):
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
        plan["holdout_126"] = holdout_126_cases(
            config, corpus_manifest, training_worlds=worlds["training"]
        )
        if len(plan["development_512"]) != 512 or any(
            c["split"] != "validation" or c["parent_world"] not in worlds["validation"]
            for c in plan["development_512"]
        ):
            raise ValueError("Closed or incomplete development_512 cases")
        if len(plan["holdout_126"]) != 126 or any(
            c["split"] != "train" or c["parent_world"] not in HOLDOUT_WORLDS
            for c in plan["holdout_126"]
        ):
            raise ValueError("Closed or incomplete holdout_126 cases")
    for purpose, count in (("development", 128), ("confirmation", 256)):
        if len(plan[purpose]) != count or any(
            c["split"] != "validation" or c["parent_world"] not in worlds["validation"]
            for c in plan[purpose]
        ):
            raise ValueError("Closed or incomplete evaluation cases")
    plan["hash"] = digest(plan)
    write_json(out / "manifest.json", plan)
    write_json(out / "resolved-config.json", config)
    paths = set()
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
    if kind in (MORE_DATA_KIND, C2_KIND):
        checkpoint.passport(out, config, plan)
    configure(17, "cpu")
    parent = torch.load(
        ROOT / config["parent"], map_location="cpu", weights_only=False
    )
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
    run_record = {
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
            evaluation_reference_policy_hash=config[
                "evaluation_reference_policy_hash"
            ],
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
    write_json(out / "run.json", run_record)
    initial_state = {
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
        seed_for(
            17, 0, "shuffle", namespace=config["experiment"]["shuffle_namespace"]
        )
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
    return plan


def _restore_collection_progress(config, factory, out):
    if not (out / "collection.json").exists():
        return [], 0, 0.0, 0
    saved = read_json(out / "collection.json")
    if saved.get("status") == "complete":
        return saved, None, None, None
    if config["kind"] != MORE_DATA_KIND or saved.get("status") not in (
        "collecting",
        "paused",
    ):
        raise ValueError("Incomplete collection exists; do not silently repeat it")
    rows = saved.get("new_matches", saved.get("matches", []))
    elapsed_offset = float(saved.get("new_seconds", saved.get("seconds", 0.0)))
    labels = 0
    for expected_index, row in enumerate(rows):
        if int(row.get("match_id", -1)) != expected_index:
            raise ValueError("Out-of-order match_id in saved collection wave")
        case = factory.make(expected_index)
        case["source"] = config["kind"]
        case["case_id"] = expected_index
        case["case_hash"] = digest(case)
        match_dir = out / f"data/match-{expected_index:06d}"
        if (
            not (match_dir / "case.json").is_file()
            or read_json(match_dir / "case.json") != case
            or not (match_dir / "collection-metrics.json").is_file()
        ):
            raise ValueError(f"Corrupted or mismatched saved match: {match_dir}")
        for entry in row["entries"]:
            ep_path = out / entry["path"]
            if not ep_path.is_file():
                raise ValueError(f"Missing saved demonstration: {ep_path}")
            if "sha256" in entry and file_sha256(ep_path) != entry["sha256"]:
                raise ValueError(f"Demonstration SHA256 mismatch on resume: {ep_path}")
        labels += int(row["labels"])
    if labels != int(saved.get("new_labels", saved.get("labels", 0))):
        raise ValueError("Saved collection label count disagrees with match records")
    return rows, labels, elapsed_offset, len(rows)


def _collect_c2(config, plan, out):
    if (out / "collection.json").exists():
        saved = read_json(out / "collection.json")
        if saved.get("status") == "complete":
            return saved
    old_corpus = _load_old_corpus(config, plan)
    more_corpus = _load_more_data_corpus(config, plan)
    holdout_cases = plan["holdout_126"]
    holdout_rows = []
    started = perf_counter()
    with (
        ResourceMonitor(out / "collection-telemetry") as monitor,
        pool(config["workers"]) as executor,
    ):
        monitor.mark("CP_holdout_generation")
        pending_jobs = []
        for idx, case in enumerate(holdout_cases):
            match_dir = out / f"holdout-data/match-{idx:06d}"
            metrics_path = match_dir / "collection-metrics.json"
            if (
                metrics_path.is_file()
                and (match_dir / "case.json").is_file()
                and read_json(match_dir / "case.json") == case
            ):
                row = read_json(metrics_path)
                holdout_rows.append((idx, row))
            else:
                pending_jobs.append((config, case, str(match_dir)))
        for first in range(0, len(pending_jobs), config["workers"]):
            wave = pending_jobs[first : first + config["workers"]]
            completed = list(executor.map(match_task, wave))
            for row in completed:
                holdout_rows.append((int(row["match_id"]), row))
            print(
                f"Holdout CP--CP collection: {len(holdout_rows)}/{len(holdout_cases)} matches",
                flush=True,
            )
    holdout_rows.sort(key=lambda item: item[0])
    if [i for i, _ in holdout_rows] != list(range(len(holdout_cases))):
        raise ValueError("Incomplete or out-of-order holdout match collection")
    formatted_holdout_matches = []
    holdout_entries = []
    for idx, row in holdout_rows:
        row = deepcopy(row)
        row["corpus_source"] = "holdout"
        row_entries = []
        for entry in row["entries"]:
            full_path = Path(entry["path"])
            if not full_path.is_absolute():
                full_path = out / full_path
            if not full_path.is_file():
                full_path = (
                    out
                    / f"holdout-data/match-{idx:06d}"
                    / f"demonstration-side-{entry['side']}.json.gz"
                )
            rel_path = full_path.resolve().relative_to(out.resolve()).as_posix()
            formatted = {
                **entry,
                "entry_id": f"holdout:match-{idx:06d}:side-{int(entry['side'])}",
                "corpus_source": "holdout",
                "match_id": idx,
                "side": int(entry["side"]),
                "path": rel_path,
                "map_size": int(row["map_size"]),
                "army_size": int(row["army_size"]),
                "sha256": file_sha256(full_path),
            }
            row_entries.append(formatted)
            holdout_entries.append(formatted)
        row["entries"] = row_entries
        formatted_holdout_matches.append(row)
    holdout_elapsed = perf_counter() - started
    combined_entries = [*old_corpus["entries"], *more_corpus["entries"]]
    combined_matches = [*old_corpus["matches"], *more_corpus["matches"]]
    total_labels = old_corpus["labels"] + more_corpus["labels"]
    total_positions = old_corpus["positions"] + more_corpus["positions"]
    if total_labels != 1000816 or len(combined_entries) != 7536:
        raise ValueError("Combined 1M BC corpus label or entry count mismatch")
    holdout_summary = {
        "status": "complete",
        "worlds": list(HOLDOUT_WORLDS),
        "matches_count": len(formatted_holdout_matches),
        "entries_count": len(holdout_entries),
        "labels": sum(e["labels"] for e in holdout_entries),
        "positions": sum(e["positions"] for e in holdout_entries),
        "seconds": holdout_elapsed,
        "entries": holdout_entries,
        "matches": formatted_holdout_matches,
    }
    write_json(out / "holdout-collection.json", holdout_summary)
    result = {
        "status": "complete",
        "stop_reason": "reused_combined_1m_corpus",
        "labels": total_labels,
        "old_labels": old_corpus["labels"],
        "new_labels": more_corpus["labels"],
        "old_fraction": old_corpus["labels"] / total_labels,
        "new_fraction": more_corpus["labels"] / total_labels,
        "label_overshoot": 0,
        "seconds": holdout_elapsed,
        "new_seconds": 0.0,
        "holdout_seconds": holdout_elapsed,
        "time_overshoot": 0.0,
        "positions": total_positions,
        "old_positions": old_corpus["positions"],
        "new_positions": more_corpus["positions"],
        "old_matches_count": old_corpus["matches_count"],
        "new_matches_count": more_corpus["matches_count"],
        "entries": combined_entries,
        "matches": combined_matches,
        "new_matches": more_corpus["matches"],
        "fixed_training": more_corpus["fixed_training"],
        "fixed_labels": more_corpus["fixed_labels"],
        "holdout_training": holdout_entries,
        "holdout_labels": holdout_summary["labels"],
        "holdout_positions": holdout_summary["positions"],
        "holdout_matches_count": holdout_summary["matches_count"],
        "physical_repeats": 0,
        "coverage": more_corpus["coverage"],
    }
    corpus_manifest = {
        "schema": 1,
        "kind": config["kind"],
        "old_corpus_run": config["old_corpus_run"],
        "more_data_corpus_run": config["more_data_corpus_run"],
        "old_labels": old_corpus["labels"],
        "new_labels": more_corpus["labels"],
        "total_labels": total_labels,
        "old_entries_count": len(old_corpus["entries"]),
        "new_entries_count": len(more_corpus["entries"]),
        "holdout_matches_count": holdout_summary["matches_count"],
        "holdout_labels": holdout_summary["labels"],
        "entries": [
            {
                "entry_id": e["entry_id"],
                "corpus_source": e["corpus_source"],
                "match_id": e["match_id"],
                "side": e["side"],
                "path": e["path"],
                "sha256": e["sha256"],
                "labels": e["labels"],
                "positions": e["positions"],
            }
            for e in combined_entries
        ],
    }
    write_json(out / "combined-corpus-manifest.json", corpus_manifest)
    write_json(out / "collection.json", result)
    return result


def collect(config, plan, out):
    kind = config["kind"]
    if kind == C2_KIND:
        return _collect_c2(config, plan, out)
    factory = TargetFactory(config, plan["base"])
    restored, labels, elapsed_offset, index = _restore_collection_progress(
        config, factory, out
    )
    if index is None:
        return restored
    rows = list(restored)
    old_corpus = _load_old_corpus(config, plan) if kind == MORE_DATA_KIND else None
    max_matches = config.get("max_collection_matches")
    started = perf_counter()
    with (
        ResourceMonitor(out / "collection-telemetry") as monitor,
        pool(config["workers"]) as executor,
    ):
        monitor.mark("CP_generation")
        while (
            labels < config["training_labels"]
            and elapsed_offset + (perf_counter() - started)
            < config["collection_seconds"]
            and (max_matches is None or len(rows) < max_matches)
        ):
            wave_size = config["workers"]
            if max_matches is not None:
                wave_size = min(wave_size, max_matches - len(rows))
            jobs = []
            for _ in range(wave_size):
                case = factory.make(index)
                case["source"] = kind
                case["case_id"] = index
                case["case_hash"] = digest(case)
                jobs.append((config, case, str(out / f"data/match-{index:06d}")))
                index += 1
            completed = list(executor.map(match_task, jobs))
            for row in completed:
                row["corpus_source"] = "new"
                formatted_entries = []
                for entry in row["entries"]:
                    full_path = Path(entry["path"])
                    rel_path = full_path.relative_to(out).as_posix()
                    if kind == MORE_DATA_KIND:
                        formatted_entries.append(
                            _format_corpus_entry(
                                entry,
                                "new",
                                rel_path,
                                row["map_size"],
                                row["army_size"],
                                sha256=file_sha256(full_path),
                            )
                        )
                    else:
                        entry["path"] = rel_path
                        formatted_entries.append(entry)
                row["entries"] = formatted_entries
            rows.extend(completed)
            labels += sum(r["labels"] for r in completed)
            elapsed = elapsed_offset + (perf_counter() - started)
            rate = labels / max(elapsed, 1e-9)
            remaining = max(0, config["training_labels"] - labels)
            eta = remaining / rate if rate > 0 else None
            progress = {
                "status": "collecting",
                "labels": labels,
                "new_labels": labels,
                "seconds": elapsed,
                "new_seconds": elapsed,
                "eta_seconds": eta,
                "matches": rows,
                "new_matches": rows,
            }
            write_json(out / "collection.json", progress)
            print(
                f"CP--CP collection: {len(rows)} new matches, {labels} new free labels, "
                f"{elapsed:.1f}s; {rate:.1f} labels/s"
                + (f", ETA {eta:.0f}s" if eta is not None else ""),
                flush=True,
            )
    elapsed = elapsed_offset + (perf_counter() - started)
    new_entries = [e for r in rows for e in r["entries"]]
    if not labels:
        raise RuntimeError("No free CP labels were collected")
    if max_matches is not None and len(rows) >= max_matches and labels < config["training_labels"]:
        stop_reason = "max_collection_matches"
    elif labels >= config["training_labels"]:
        stop_reason = "label_target"
    else:
        stop_reason = "collection_time_limit"
    # A fixed training diagnostic subset; never used for candidate selection.
    ids = list(range(len(rows)))
    random.Random(17).shuffle(ids)
    selected = []
    count = 0
    for i in ids:
        selected.extend(rows[i]["entries"])
        count += rows[i]["labels"]
        if count >= config["fixed_training_labels"]:
            break
    if kind == MORE_DATA_KIND:
        combined_entries = [*old_corpus["entries"], *new_entries]
        combined_matches = [*old_corpus["matches"], *rows]
        total_labels = old_corpus["labels"] + labels
        total_positions = old_corpus["positions"] + sum(
            e["positions"] for e in new_entries
        )
        result = {
            "status": "complete",
            "stop_reason": stop_reason,
            "labels": total_labels,
            "old_labels": old_corpus["labels"],
            "new_labels": labels,
            "old_fraction": old_corpus["labels"] / total_labels,
            "new_fraction": labels / total_labels,
            "label_overshoot": max(0, labels - config["training_labels"]),
            "seconds": elapsed,
            "new_seconds": elapsed,
            "time_overshoot": max(0, elapsed - config["collection_seconds"]),
            "positions": total_positions,
            "old_positions": old_corpus["positions"],
            "new_positions": sum(e["positions"] for e in new_entries),
            "old_matches_count": old_corpus["matches_count"],
            "new_matches_count": len(rows),
            "entries": combined_entries,
            "matches": combined_matches,
            "new_matches": rows,
            "fixed_training": selected,
            "fixed_labels": count,
            "physical_repeats": factory.duplicates,
            "coverage": {
                "conditions": dict(factory.conditions),
                "worlds": dict(factory.worlds),
                "crops": dict(factory.crops),
            },
        }
        corpus_manifest = {
            "schema": 1,
            "kind": kind,
            "old_corpus_run": config["old_corpus_run"],
            "old_labels": old_corpus["labels"],
            "new_labels": labels,
            "total_labels": total_labels,
            "old_entries_count": len(old_corpus["entries"]),
            "new_entries_count": len(new_entries),
            "entries": [
                {
                    "entry_id": e["entry_id"],
                    "corpus_source": e["corpus_source"],
                    "match_id": e["match_id"],
                    "side": e["side"],
                    "path": e["path"],
                    "sha256": e["sha256"],
                    "labels": e["labels"],
                    "positions": e["positions"],
                }
                for e in combined_entries
            ],
        }
        write_json(out / "combined-corpus-manifest.json", corpus_manifest)
    else:
        result = {
            "status": "complete",
            "stop_reason": stop_reason,
            "labels": labels,
            "label_overshoot": max(0, labels - config["training_labels"]),
            "seconds": elapsed,
            "time_overshoot": max(0, elapsed - config["collection_seconds"]),
            "positions": sum(e["positions"] for e in new_entries),
            "entries": new_entries,
            "matches": rows,
            "fixed_training": selected,
            "fixed_labels": count,
            "physical_repeats": factory.duplicates,
            "coverage": {
                "conditions": dict(factory.conditions),
                "worlds": dict(factory.worlds),
                "crops": dict(factory.crops),
            },
        }
    write_json(out / "collection.json", result)
    return result


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
                    ce = (
                        F.cross_entropy(logits, labels["indices"], reduction="none")
                        .cpu()
                        .tolist()
                    )
                    predicted = logits.argmax(-1).cpu().tolist()
                    rows = [
                        r
                        for e, a, b in part
                        for r in cache.episodes[e]["decisions"][a:b]
                    ]
                    for i, row in enumerate(rows):
                        if not row["supervised"]:
                            continue
                        target, _ = decision_indices(row)
                        action = row["actions"][target]
                        kind = action["kind"]
                        counts[kind] += 1
                        correct[kind] += predicted[i] == target
                        losses[kind] += ce[i]
                        units = {
                            u["id"]: u["type"] for u in row["observation"]["units"]
                        }
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


def _try_reuse_saved_evaluation(
    config, plan, out, source_run_dir, checkpoint_rel, fingerprint, purpose
):
    source_run_dir = Path(source_run_dir)
    evals_path = source_run_dir / "evaluations.json"
    if not evals_path.is_file():
        return None
    source_cfg = (
        read_json(source_run_dir / "run.json").get("config", {})
        if (source_run_dir / "run.json").is_file()
        else {}
    )
    candidate_keys = [_evaluation_key(config, plan, purpose, fingerprint)]
    if "policy_temperature" in source_cfg:
        candidate_keys.append(
            _evaluation_key(
                {**config, "policy_temperature": source_cfg["policy_temperature"]},
                plan,
                purpose,
                fingerprint,
            )
        )
    saved_evals = read_json(evals_path)
    point = None
    for key in candidate_keys:
        if key in saved_evals:
            point = saved_evals[key]
            break
    if not point:
        return None
    rel_dir = point["runs"]["argmax"]["cp"]["directory"]
    src_dir = source_run_dir / rel_dir
    expected_matches = 128 if purpose == "development" else 256
    if not _complete_eval_dir(src_dir, expected_matches):
        return None
    dst_dir = out / rel_dir
    if src_dir.resolve() != dst_dir.resolve():
        for path in sorted(src_dir.rglob("*")):
            if path.is_file():
                target = dst_dir / path.relative_to(src_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    return {
        **deepcopy(point),
        "checkpoint": checkpoint_rel,
        "reused_saved_evaluation": True,
        "reused_from": source_run_dir.relative_to(ROOT).as_posix()
        if source_run_dir.resolve().is_relative_to(ROOT.resolve())
        else str(source_run_dir),
    }


def select_best_candidate(bc_baseline, ppo_reference, candidates):
    """Select candidate by max development wins with protocol tie-breaking.

    Tie-breaking rules:
    1. Prefer existing baseline over a new BC candidate on equal wins.
    2. Between the two baselines, prefer PPO update 37 on equal wins.
    3. Among new BC checkpoints, prefer the earlier checkpoint (by Adam_steps, then epoch).
    """
    bc_wins = int(bc_baseline["runs"]["argmax"]["cp"]["summary"]["W"])
    ppo_wins = int(ppo_reference["runs"]["argmax"]["cp"]["summary"]["W"])
    if ppo_wins >= bc_wins:
        best_baseline = {**ppo_reference, "candidate_role": "evaluation_reference"}
        best_baseline_wins = ppo_wins
    else:
        best_baseline = {**bc_baseline, "candidate_role": "training_parent"}
        best_baseline_wins = bc_wins
    ordered_candidates = sorted(
        candidates,
        key=lambda c: (int(c.get("Adam_steps", 0)), int(c.get("epoch", 0))),
    )
    best_new = None
    best_new_wins = -1
    for candidate in ordered_candidates:
        wins = int(candidate["runs"]["argmax"]["cp"]["summary"]["W"])
        if wins > best_new_wins:
            best_new = {**candidate, "candidate_role": "new_bc"}
            best_new_wins = wins
    if best_new is not None and best_new_wins > best_baseline_wins:
        selected = best_new
        selected_wins = best_new_wins
        baseline_retained = False
    else:
        selected = best_baseline
        selected_wins = best_baseline_wins
        baseline_retained = True
    return {
        "training_parent": bc_baseline,
        "evaluation_reference": ppo_reference,
        "baseline": best_baseline,
        "baseline_wins": best_baseline_wins,
        "required_new_candidate_wins": best_baseline_wins + 1,
        "best_new_candidate": best_new,
        "best_new_wins": best_new_wins if best_new is not None else None,
        "best": selected,
        "best_wins": selected_wins,
        "baseline_retained": baseline_retained,
    }


def baseline(config, plan, out):
    if config["kind"] == C2_KIND or (out / "selection.json").exists():
        return
    if config["kind"] == MORE_DATA_KIND:
        ref_run = ROOT / config["reference_run"]
        old_run = ROOT / config["old_corpus_run"]
        bc_hash = config["parent_policy_hash"]
        ppo_hash = config["evaluation_reference_policy_hash"]
        bc_point = _try_reuse_saved_evaluation(
            config,
            plan,
            out,
            ref_run,
            "checkpoints/initial.pt",
            bc_hash,
            "development",
        ) or _try_reuse_saved_evaluation(
            config,
            plan,
            out,
            old_run,
            "checkpoints/initial.pt",
            bc_hash,
            "development",
        )
        if bc_point is None:
            bc_point = target_evaluate(
                config, plan, out, "checkpoints/initial.pt", "development"
            )
            bc_point["reused_saved_evaluation"] = False
        bc_point.update(
            checkpoint="checkpoints/initial.pt",
            epoch=0,
            Adam_steps=0,
            baseline_role="training_parent",
        )
        ppo_point = _try_reuse_saved_evaluation(
            config,
            plan,
            out,
            ref_run,
            "checkpoints/reference-update-0037.pt",
            ppo_hash,
            "development",
        )
        if ppo_point is None:
            ppo_point = target_evaluate(
                config,
                plan,
                out,
                "checkpoints/reference-update-0037.pt",
                "development",
            )
            ppo_point["reused_saved_evaluation"] = False
        ppo_point.update(
            checkpoint="checkpoints/reference-update-0037.pt",
            epoch=0,
            Adam_steps=0,
            baseline_role="evaluation_reference",
        )
        evaluations = (
            read_json(out / "evaluations.json")
            if (out / "evaluations.json").exists()
            else {}
        )
        evaluations["development:" + bc_point["policy_hash"]] = bc_point
        evaluations["development:" + ppo_point["policy_hash"]] = ppo_point
        write_json(out / "evaluations.json", evaluations)
        selection = select_best_candidate(bc_point, ppo_point, [])
        write_json(out / "selection.json", selection)
        print(
            f"Development baselines: BC epoch 10={bc_point['runs']['argmax']['cp']['summary']}, "
            f"PPO update 37={ppo_point['runs']['argmax']['cp']['summary']}",
            flush=True,
        )
        return
    # A cheap check of saved data and exact evaluator sources. No reconstruction.
    reused = False
    archive = ROOT / config["baseline_archive"]
    if archive.exists():
        with zipfile.ZipFile(archive) as previous:
            prefix = "artifacts/rl/ppo-target-scale/"
            old_plan = json.loads(previous.read(prefix + "manifest.json"))
            old_evals = json.loads(previous.read(prefix + "evaluations.json"))
            sources = zipfile.ZipFile(io.BytesIO(previous.read(prefix + "source.zip")))
            relevant = (
                "civ5rl/core/agent.py",
                "civ5rl/core/model.py",
                "civ5rl/core/observation.py",
                "civ5rl/experiments/ppo_after_imitation.py",
                "civ5sim/match.py",
                "civ5sim/cp_agent.py",
            )
            same_sources = all(
                sources.read(name) == (ROOT / name).read_bytes() for name in relevant
            )
            saved = torch.load(
                out / "checkpoints/initial.pt", map_location="cpu", weights_only=False
            )
            fingerprint = policy_hash(saved["model"])
            key = "development:" + fingerprint
            if (
                old_plan["development"] == plan["development"]
                and same_sources
                and key in old_evals
            ):
                point = old_evals[key]
                directory = out / point["runs"]["argmax"]["cp"]["directory"]
                old_directory = (
                    prefix + point["runs"]["argmax"]["cp"]["directory"] + "/"
                )
                for name in previous.namelist():
                    if name.startswith(old_directory) and not name.endswith("/"):
                        target = directory / name[len(old_directory) :]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(previous.read(name))
                point = {
                    **point,
                    "checkpoint": "checkpoints/initial.pt",
                    "epoch": 0,
                    "Adam_steps": 0,
                    "reused_Windows_baseline": True,
                }
                write_json(out / "evaluations.json", {key: point})
                reused = True
    if not reused:
        point = target_evaluate(
            config, plan, out, "checkpoints/initial.pt", "development"
        )
        point.update(epoch=0, Adam_steps=0, reused_Windows_baseline=False)
        evaluations = read_json(out / "evaluations.json")
        evaluations["development:" + point["policy_hash"]] = point
        write_json(out / "evaluations.json", evaluations)
    wins = point["runs"]["argmax"]["cp"]["summary"]["W"]
    write_json(
        out / "selection.json", {"baseline": point, "best": point, "best_wins": wins}
    )
    print(
        "Development baseline: " + str(point["runs"]["argmax"]["cp"]["summary"]),
        flush=True,
    )


def _refresh_selection(config, out):
    evaluations = read_json(out / "evaluations.json")
    selection = read_json(out / "selection.json")
    if config["kind"] == MORE_DATA_KIND:
        bc_hash = config["parent_policy_hash"]
        ppo_hash = config["evaluation_reference_policy_hash"]
        bc_point = evaluations["development:" + bc_hash]
        ppo_point = evaluations["development:" + ppo_hash]
        candidates = [
            v
            for k, v in evaluations.items()
            if k.startswith("development:")
            and v.get("purpose") == "development"
            and v.get("policy_hash") not in (bc_hash, ppo_hash)
            and int(v.get("Adam_steps", 0)) > 0
        ]
        updated = select_best_candidate(bc_point, ppo_point, candidates)
        write_json(out / "selection.json", updated)
        return updated
    for point in sorted(
        (
            v
            for k, v in evaluations.items()
            if k.startswith("development:") and v.get("purpose") == "development"
        ),
        key=lambda p: int(p.get("Adam_steps", 0)),
    ):
        wins = point["runs"]["argmax"]["cp"]["summary"]["W"]
        if wins > selection["best_wins"]:
            selection.update(best=point, best_wins=wins)
    write_json(out / "selection.json", selection)
    return selection


def measure(config, plan, out, path, epoch, steps, policy, fixed, micro, shuffle):
    fingerprint = policy_hash(policy.state_dict())
    cache = (
        read_json(out / "evaluations.json")
        if (out / "evaluations.json").exists()
        else {}
    )
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


class ActiveClock:
    def __init__(self, offset=0.0):
        self.started = perf_counter()
        self.excluded = 0.0
        self.offset = offset

    def seconds(self):
        return self.offset + perf_counter() - self.started - self.excluded

    @contextmanager
    def pause(self):
        start = perf_counter()
        try:
            yield
        finally:
            self.excluded += perf_counter() - start


def _entry_lookup_key(entry):
    return entry.get("entry_id", entry["path"])


def _chunk_priority_order(data, chunks):
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


def train(config, plan, out, *, resume=False):
    kind = config["kind"]
    restart_path = out / "restart-input-v2.json"
    restart = read_json(restart_path) if restart_path.exists() else None
    if (out / "training.json").exists():
        training_state = read_json(out / "training.json")
        if training_state.get("status") == "complete":
            return
        if kind not in (MORE_DATA_KIND, C2_KIND) and (
            not restart or restart.get("consumed")
        ):
            raise ValueError(
                "Partial training exists; optimization must not repeat automatically"
            )
        if kind in (MORE_DATA_KIND, C2_KIND) and not resume:
            raise ValueError(
                "Partial training exists; use explicit resume phase to continue"
            )
    if not config.get("skip_baseline_evaluation", False):
        baseline(config, plan, out)
    data = read_json(out / "collection.json")
    if data["status"] != "complete":
        raise ValueError("Collection is incomplete")
    all_entries = list(data["entries"])
    if kind == C2_KIND and data.get("holdout_training"):
        all_entries.extend(data["holdout_training"])
    input_preparation = prepare_inputs(
        out, all_entries, workers=config["workers"]
    )
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
    shuffle = random.Random(
        seed_for(
            17, 0, "shuffle", namespace=config["experiment"]["shuffle_namespace"]
        )
    )
    state_record = (
        read_json(out / "state.json") if (out / "state.json").exists() else None
    )
    resuming_more_data = (
        kind in (MORE_DATA_KIND, C2_KIND)
        and state_record is not None
        and (
            resume
            or int(state_record.get("Adam_steps", 0)) > 0
            or int(state_record.get("epoch", 0)) > 0
        )
    )
    if restart:
        source = restart["checkpoint_boundary"]["latest"]
    elif resuming_more_data:
        source = state_record["latest"]
    else:
        source = "checkpoints/initial.pt"
    prepared = perf_counter()
    policy = load_policy(config, out / source, "cuda")
    saved = torch.load(out / source, map_location="cpu", weights_only=False)
    initial_saved = torch.load(
        out / "checkpoints/initial.pt", map_location="cpu", weights_only=False
    )
    fixed_limit = min(config["cache_bytes"] // 16, 256 * 1024**2)
    holdout_limit = fixed_limit if holdout_entries else 0
    cache = PreparedDemonstrations(
        out,
        prepared_entries,
        "cuda",
        config["cache_bytes"] - fixed_limit - holdout_limit,
        variant=policy.variant,
    )
    fixed = PreparedDemonstrations(
        out, fixed_entries, "cuda", fixed_limit, variant=policy.variant
    )
    holdout = (
        PreparedDemonstrations(
            out, holdout_entries, "cuda", holdout_limit, variant=policy.variant
        )
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
            selected = int(
                read_json(out / "memory-preparation.json")["selected_microbatch"]
            )
            def backward(p, c, s, n):
                return backward_minibatch_async(p, c, s, n, selected)
        else:
            warm = []
            # The largest map/army first; use real fragments, including nonzero prefixes.
            order = _chunk_priority_order(data, chunks)
            probes = list(label_minibatches(order, cache.masks))[:2]
            selected = None
            for micro in (8, 4, 2, 1):
                monitor.mark("initial_warmup", microbatch=micro)
                policy.load_state_dict(saved["model"])
                optimizer = adam(policy, config, "cuda")
                optimizer.load_state_dict(saved["optimizer"])

                def backward(p, c, s, n, micro=micro):
                    return backward_minibatch_async(p, c, s, n, micro)

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
                        max(0, end_swap.sin - swap.sin)
                        + max(0, end_swap.sout - swap.sout)
                        if swap is not None and end_swap is not None
                        else 0
                    )
                    if (
                        paging > 512 * 1024**2
                        and psutil.virtual_memory().available < 1024**3
                    ):
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
                    warm.append(
                        {"microbatch": micro, "passed": False, "error": str(error)}
                    )
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
                    fixed_c = PreparedDemonstrations(
                        out, fixed_entries, "cuda", fixed_limit, variant="C"
                    )
                    holdout_c = PreparedDemonstrations(
                        out, holdout_entries, "cuda", holdout_limit, variant="C"
                    )
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
            elif (out / "evaluations.json").exists() and (
                out / "selection.json"
            ).exists():
                evaluations = read_json(out / "evaluations.json")
                selection = read_json(out / "selection.json")
                if kind == MORE_DATA_KIND:
                    initial_key = "development:" + config["parent_policy_hash"]
                    if initial_key in evaluations:
                        evaluations[initial_key]["fixed_training"] = initial_metrics
                        write_json(out / "evaluations.json", evaluations)
                        _refresh_selection(config, out)
                else:
                    initial_key = (
                        "development:" + selection["baseline"]["policy_hash"]
                    )
                    evaluations[initial_key]["fixed_training"] = initial_metrics
                    selection["baseline"] = evaluations[initial_key]
                    if (
                        selection["best"]["policy_hash"]
                        == selection["baseline"]["policy_hash"]
                    ):
                        selection["best"] = selection["baseline"]
                    write_json(out / "evaluations.json", evaluations)
                    write_json(out / "selection.json", selection)
        cache.counters.clear()
        fixed.clear()
        if holdout is not None:
            holdout.clear()
        origin_steps = int(saved["state"]["Adam_steps"])
        history = []
        start_epoch = 1
        resume_mid_epoch = False
        if restart:
            boundary = restart["checkpoint_boundary"]
            if (
                not saved["state"]["epoch_completed"]
                or saved["state"]["Adam_steps"] != boundary["Adam_steps"]
            ):
                raise ValueError(
                    "Input-fix restart requires its complete saved epoch boundary"
                )
            previous = [
                json.loads(line)
                for line in (out / "steps.jsonl").read_text().splitlines()
                if line.strip()
            ]
            if len(previous) != origin_steps:
                raise ValueError(
                    "Saved step log differs from checkpoint boundary"
                )
            steps_per_epoch = (data["labels"] + 1023) // 1024
            for e in range(1, boundary["epoch"] + 1):
                records = previous[
                    (e - 1) * steps_per_epoch : e * steps_per_epoch
                ]
                if sum(r["labels"] for r in records) != data["labels"]:
                    raise ValueError("Incomplete saved epoch")
                history.append(
                    {
                        "epoch": e,
                        "completed": True,
                        "steps": len(records),
                        "labels": data["labels"],
                        "online_CE": sum(r["ce"] * r["labels"] for r in records)
                        / data["labels"],
                    }
                )
            start_epoch = boundary["epoch"] + 1
            steps = boundary["Adam_steps"]
            labels = boundary["free_labels_processed"]
            for field in ("history_positions", "prefix_positions"):
                cache.counters[field] = boundary[field]
            # Finish the planned evaluation interrupted by the requested stop.
            monitor.mark("evaluation")
            measure(
                config,
                plan,
                out,
                source,
                boundary["epoch"],
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
                    "active_seconds_preserved": boundary["active_seconds"],
                    "history_and_order_preserved": True,
                },
            )
            restart["consumed"] = True
            write_json(restart_path, restart)
            clock_offset = float(boundary["active_seconds"])
        elif resuming_more_data:
            saved_state = saved["state"]
            for key in (
                "epoch",
                "epoch_completed",
                "batch_cursor",
                "Adam_steps",
                "free_labels_processed",
            ):
                if saved_state.get(key) != state_record.get(key):
                    raise ValueError(
                        f"Checkpoint state and state.json disagree on {key}"
                    )
            steps_path = out / "steps.jsonl"
            all_lines = (
                [
                    line
                    for line in steps_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                if steps_path.exists()
                else []
            )
            if len(all_lines) < origin_steps:
                raise ValueError("steps.jsonl has fewer records than saved checkpoint")
            if len(all_lines) > origin_steps:
                all_lines = all_lines[:origin_steps]
                steps_path.write_text(
                    ("\n".join(all_lines) + "\n") if all_lines else "",
                    encoding="utf-8",
                )
            previous_records = [json.loads(line) for line in all_lines]
            if [r["step"] for r in previous_records] != list(
                range(1, origin_steps + 1)
            ):
                raise ValueError("Non-sequential step numbers in steps.jsonl")
            if (
                sum(r["labels"] for r in previous_records)
                != saved_state["free_labels_processed"]
            ):
                raise ValueError("Saved label count disagrees with steps.jsonl")
            steps = origin_steps
            labels = int(saved_state["free_labels_processed"])
            for field in ("history_positions", "prefix_positions"):
                cache.counters[field] = int(saved_state.get(field, 0))
            history = list(saved_state.get("epochs_history", []))
            clock_offset = float(
                state_record.get(
                    "active_seconds",
                    saved_state.get("active_seconds_at_save_start", 0.0),
                )
            )
            saved_epoch = int(saved_state["epoch"])
            if saved_state["epoch_completed"]:
                if (
                    kind != C2_KIND
                    and saved_epoch in config["evaluate_epochs"]
                    and not config.get("skip_milestone_evaluations", False)
                ):
                    monitor.mark("evaluation")
                    measure(
                        config,
                        plan,
                        out,
                        source,
                        saved_epoch,
                        steps,
                        policy,
                        fixed,
                        selected,
                        shuffle,
                    )
                    fixed.clear()
                start_epoch = saved_epoch + 1
            else:
                start_epoch = saved_epoch
                resume_mid_epoch = True
                if history and history[-1].get("epoch") == saved_epoch and not history[
                    -1
                ].get("completed", True):
                    history.pop()
            write_json(
                out / "resume-state-verification.json",
                {
                    "model_equal": True,
                    "Adam_equal": True,
                    "RNG_restored": True,
                    "source": source,
                    "epoch": saved_epoch,
                    "epoch_completed": bool(saved_state["epoch_completed"]),
                    "batch_cursor": int(saved_state["batch_cursor"]),
                    "Adam_counter": steps,
                    "next_Adam_counter": steps + 1,
                    "active_seconds_preserved": clock_offset,
                },
            )
        else:
            steps = labels = 0
            clock_offset = 0.0
        clock = ActiveClock(clock_offset)
        reason = "epoch_limit"
        last_path = source
        utilization = (
            read_json(out / "GPU-utilization-review.json")
            if (out / "GPU-utilization-review.json").exists()
            else None
        )
        max_steps = config.get("max_adam_steps")

        def save(path, epoch, cursor, order, complete, stop_reason=None):
            monitor.mark("checkpoint", epoch=epoch, Adam_steps=steps)
            serial_order = [[int(s[0]), int(s[1]), int(s[2])] for s in order]
            state = {
                "epoch": epoch,
                "epoch_completed": complete,
                "batch_cursor": cursor,
                "epoch_order": serial_order,
                "Adam_steps": steps,
                "free_labels_processed": labels,
                "active_seconds_at_save_start": clock.seconds(),
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
            if clock.seconds() >= config["training_seconds"]:
                reason = "active_time_limit"
                break
            if max_steps is not None and steps >= max_steps:
                reason = "max_adam_steps"
                break
            if resume_mid_epoch and epoch == start_epoch:
                order = [Span(int(a), int(b), int(c)) for a, b, c in saved["state"]["epoch_order"]]
                cursor = int(saved["state"]["batch_cursor"])
                all_batches = list(label_minibatches(order, cache.masks))
                prior_steps = sum(h["steps"] for h in history)
                records = previous_records[prior_steps:]
                if len(records) != cursor:
                    raise ValueError("Mid-epoch step records disagree with batch_cursor")
                epoch_labels = sum(r["labels"] for r in records)
                batches = iter(all_batches[cursor:])
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

            def _check_stop():
                nonlocal stop_hit
                if clock.seconds() >= config["training_seconds"]:
                    stop_hit = "active_time_limit"
                    return stop_hit
                return None

            def _on_step(record, completed_steps):
                nonlocal steps, labels, cursor, epoch_labels, utilization
                steps = completed_steps
                labels += record["labels"]
                cursor += 1
                epoch_labels += record["labels"]
                records.append(record)
                if steps == 1:
                    write_json(
                        out / "first-step-check.json",
                        {"before": 0, "after": 1, "all_counters_equal": True},
                    )
                if (restart or resuming_more_data) and steps == origin_steps + 1:
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
                        f"BC epoch={epoch}, Adam={steps}, labels={labels}, CE={record['ce']:.4f}, active={clock.seconds():.1f}s",
                        flush=True,
                    )
                if clock.seconds() >= 600 and utilization is None:
                    analysis = analyze_telemetry(
                        out / "training-telemetry/resources.jsonl"
                    )
                    utilization = {
                        "metrics": analysis["metrics"],
                        "last_60s": analysis["windows"][-1]
                        if analysis["windows"]
                        else None,
                        "measured_cache": cache.statistics(),
                        "mean_step_seconds": sum(r["seconds"] for r in records)
                        / len(records),
                        "prefix_seconds_not_separately_profiled": True,
                        "continuation_reason": "Bounded CPU episode input, full current-weight GRU-prefix and small fixed spatial blocks constrain dispatch. Preserve the validated 1024-label objective and precision; no extra optimization campaign or artificial GPU load.",
                    }
                    write_json(out / "GPU-utilization-review.json", utilization)

            check_steps = {1}
            if restart or resuming_more_data:
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
                max_steps=max_steps,
                should_stop=_check_stop,
                on_step_completed=_on_step,
                check_counter_steps=check_steps,
            )
            if epoch_labels < data["labels"]:
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
                    "online_CE": sum(r["ce"] * r["labels"] for r in records)
                    / sum(r["labels"] for r in records),
                }
                if kind == C2_KIND:
                    monitor.mark("offline_metrics", epoch=epoch, Adam_steps=steps)
                    with clock.pause(), isolated_rng(shuffle):
                        epoch_entry["fixed_training"] = fixed_metrics(
                            policy, fixed, selected
                        )
                        if holdout is not None:
                            epoch_entry["holdout"] = fixed_metrics(
                                policy, holdout, selected
                            )
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
                    or clock.seconds() >= config["training_seconds"]
                    or (max_steps is not None and steps >= max_steps)
                    or (completed and epoch == config["epochs"])
                )
                if not terminal_step:
                    final_reason = None
                elif max_steps is not None and steps >= max_steps:
                    final_reason = "max_adam_steps"
                elif not completed or clock.seconds() >= config["training_seconds"]:
                    final_reason = "active_time_limit"
                else:
                    final_reason = "epoch_limit"
                save(last_path, epoch, cursor, order, completed, stop_reason=final_reason)
            if reason == "paused" or (
                completed and records and pause_due(out, "epoch", epoch)
            ):
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
                return
            terminal = (
                not completed
                or clock.seconds() >= config["training_seconds"]
                or (max_steps is not None and steps >= max_steps)
            )
            if terminal and reason == "epoch_limit":
                reason = (
                    "max_adam_steps"
                    if (max_steps is not None and steps >= max_steps)
                    else "active_time_limit"
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
        if (
            kind != C2_KIND
            and steps
            and not config.get("skip_milestone_evaluations", False)
        ):
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
                "min_60s_core_average": min(
                    (w["core_average"] for w in analysis["windows"]), default=None
                ),
                "max_60s_core_average": max(
                    (w["core_average"] for w in analysis["windows"]), default=None
                ),
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
                "time_overshoot": max(0, elapsed - config["training_seconds"]),
                "latest": last_path,
                "microbatch": selected,
                "cache": cache.statistics(),
                "value_head_unchanged": True,
                "GPU_utilization_review": utilization,
            },
        )
        cache.clear()
        fixed.clear()
    policy = optimizer = None
    gc.collect()
    torch.cuda.empty_cache()


def _outcomes_from_dir(directory):
    directory = Path(directory)
    result = {}
    cases = {}
    for path in sorted(directory.glob("*/metrics.json")):
        row = read_json(path)
        case = read_json(path.parent / "case.json")
        key = (case["parent_world"], case["case_id"], case["learner_side"])
        result[key] = int(row["winner"] == row["side"])
        cases[key] = case
    return result, cases


def _evaluate_c2(config, plan, out, training):
    out = Path(out)
    baseline_offline = (
        read_json(out / "baseline-offline-metrics.json")
        if (out / "baseline-offline-metrics.json").exists()
        else {}
    )
    epoch_by_num = {int(e["epoch"]): e for e in training.get("epochs", [])}
    evaluations = (
        read_json(out / "evaluations.json")
        if (out / "evaluations.json").exists()
        else {}
    )

    def _run_dev512(path, role, *, epoch=0, steps=0, fixed_tr=None, holdout_tr=None):
        point = target_evaluate(config, plan, out, path, "development_512")
        point = {
            **point,
            "checkpoint": path,
            "epoch": epoch,
            "Adam_steps": steps,
            "candidate_role": role,
        }
        if fixed_tr is not None:
            point["fixed_training"] = fixed_tr
        if holdout_tr is not None:
            point["holdout"] = holdout_tr
        evals = (
            read_json(out / "evaluations.json")
            if (out / "evaluations.json").exists()
            else {}
        )
        evals["development_512:" + point["policy_hash"]] = point
        write_json(out / "evaluations.json", evals)
        return point

    cp_point = _run_dev512("cp", "cp_mirror")
    bc10_off = baseline_offline.get("bc10", {})
    bc10_point = _run_dev512(
        "checkpoints/reference-bc-epoch-10.pt",
        "bc10_reference",
        fixed_tr=bc10_off.get("fixed_training"),
        holdout_tr=bc10_off.get("holdout"),
    )
    ppo37_point = _run_dev512(
        "checkpoints/reference-update-0037.pt",
        "ppo37_reference",
    )
    bc1m_off = baseline_offline.get("bc_more_data_epoch_10", {})
    bc1m_point = _run_dev512(
        "checkpoints/reference-bc-more-data-epoch-10.pt",
        "bc_more_data_baseline",
        fixed_tr=bc1m_off.get("fixed_training"),
        holdout_tr=bc1m_off.get("holdout"),
    )

    c2_candidates = []
    seen_paths = set()
    milestone_paths = []
    for e_num in config["evaluate_epochs"]:
        rel = f"checkpoints/epoch-{int(e_num):02d}.pt"
        if (out / rel).is_file():
            milestone_paths.append((int(e_num), rel))
    latest_rel = training.get("latest")
    if latest_rel and (out / latest_rel).is_file():
        last_epoch_num = (
            int(training["epochs"][-1]["epoch"]) if training.get("epochs") else 0
        )
        if all(rel != latest_rel for _, rel in milestone_paths):
            milestone_paths.append((last_epoch_num, latest_rel))

    cum_steps = 0
    steps_by_epoch = {}
    for e_row in training.get("epochs", []):
        cum_steps += int(e_row["steps"])
        steps_by_epoch[int(e_row["epoch"])] = cum_steps

    for e_num, rel in milestone_paths:
        if rel in seen_paths:
            continue
        seen_paths.add(rel)
        saved_ckpt = torch.load(out / rel, map_location="cpu", weights_only=False)
        steps = int(
            saved_ckpt.get("state", {}).get(
                "Adam_steps", steps_by_epoch.get(e_num, 0)
            )
        )
        e_info = epoch_by_num.get(e_num, {})
        point = _run_dev512(
            rel,
            "c2_candidate",
            epoch=e_num,
            steps=steps,
            fixed_tr=e_info.get("fixed_training"),
            holdout_tr=e_info.get("holdout"),
        )
        c2_candidates.append(point)

    if not c2_candidates:
        raise ValueError("No C2 milestone checkpoints found to evaluate")

    c2_candidates.sort(
        key=lambda c: (int(c.get("Adam_steps", 0)), int(c.get("epoch", 0)))
    )
    best_c2 = None
    best_c2_wins = -1
    for cand in c2_candidates:
        w = int(cand["runs"]["argmax"]["cp"]["summary"]["W"])
        if w > best_c2_wins:
            best_c2 = cand
            best_c2_wins = w

    best_c2_outcomes, cases_512 = _outcomes_from_dir(
        out / best_c2["runs"]["argmax"]["cp"]["directory"]
    )
    bc1m_outcomes, bc1m_cases = _outcomes_from_dir(
        out / bc1m_point["runs"]["argmax"]["cp"]["directory"]
    )
    ppo37_outcomes, ppo37_cases = _outcomes_from_dir(
        out / ppo37_point["runs"]["argmax"]["cp"]["directory"]
    )
    bc10_outcomes, bc10_cases = _outcomes_from_dir(
        out / bc10_point["runs"]["argmax"]["cp"]["directory"]
    )
    if (
        cases_512 != bc1m_cases
        or cases_512 != ppo37_cases
        or cases_512 != bc10_cases
    ):
        raise ValueError("Unpaired Development-512 evaluations")

    vs_bc1m = paired_world_bootstrap(
        bc1m_outcomes,
        best_c2_outcomes,
        samples=10000,
        seed=20261103,
        cases=cases_512,
        cases_per_world=32,
    )
    vs_ppo37 = paired_world_bootstrap(
        ppo37_outcomes,
        best_c2_outcomes,
        samples=10000,
        seed=20261103,
        cases=cases_512,
        cases_per_world=32,
    )
    vs_bc10 = paired_world_bootstrap(
        bc10_outcomes,
        best_c2_outcomes,
        samples=10000,
        seed=20261103,
        cases=cases_512,
        cases_per_world=32,
    )
    bc1m_wins = int(bc1m_point["runs"]["argmax"]["cp"]["summary"]["W"])
    win_rate_delta = float(vs_bc1m["delta"])
    ci_low = float(vs_bc1m["CI95"][0])
    gate_passed = bool(win_rate_delta >= 0.05 and ci_low > 0.0)

    selection = {
        "purpose": "development_512",
        "best": best_c2,
        "best_wins": best_c2_wins,
        "c2_candidates": c2_candidates,
        "bc_more_data_baseline": bc1m_point,
        "bc_more_data_wins": bc1m_wins,
        "ppo37_reference": ppo37_point,
        "ppo37_wins": int(ppo37_point["runs"]["argmax"]["cp"]["summary"]["W"]),
        "bc10_reference": bc10_point,
        "bc10_wins": int(bc10_point["runs"]["argmax"]["cp"]["summary"]["W"]),
        "cp_mirror": cp_point,
        "cp_mirror_wins": int(cp_point["runs"]["argmax"]["cp"]["summary"]["W"]),
        "comparison_vs_bc_more_data": vs_bc1m,
        "comparison_vs_ppo37": vs_ppo37,
        "comparison_vs_bc10": vs_bc10,
        "gate_criteria": {
            "min_win_rate_delta": 0.05,
            "min_ci95_low": 0.0,
            "observed_win_rate_delta": win_rate_delta,
            "observed_ci95_low": ci_low,
            "gate_passed": gate_passed,
        },
    }
    write_json(out / "selection.json", selection)

    if not gate_passed:
        write_json(
            out / "confirmation.json",
            {
                "status": "not_run_development_512_gate_not_met",
                "selected": best_c2,
                "bc_more_data_baseline": bc1m_point,
                "ppo37_reference": ppo37_point,
                "comparison_vs_bc_more_data": vs_bc1m,
                "comparison_vs_ppo37": vs_ppo37,
                "gate_criteria": selection["gate_criteria"],
            },
        )
        return

    ref_run = ROOT / config["reference_run"]
    old_run = ROOT / config["old_corpus_run"]
    bc_hash = config["parent_policy_hash"]
    ppo_hash = config["evaluation_reference_policy_hash"]

    bc1m_conf = _try_reuse_saved_evaluation(
        config,
        plan,
        out,
        ref_run,
        "checkpoints/reference-bc-more-data-epoch-10.pt",
        bc_hash,
        "confirmation",
    ) or _try_reuse_saved_evaluation(
        config,
        plan,
        out,
        old_run,
        "checkpoints/reference-bc-more-data-epoch-10.pt",
        bc_hash,
        "confirmation",
    )
    if bc1m_conf is None:
        bc1m_conf = target_evaluate(
            config,
            plan,
            out,
            "checkpoints/reference-bc-more-data-epoch-10.pt",
            "confirmation",
        )
        bc1m_conf["reused_saved_evaluation"] = False

    ppo37_conf = _try_reuse_saved_evaluation(
        config,
        plan,
        out,
        ref_run,
        "checkpoints/reference-update-0037.pt",
        ppo_hash,
        "confirmation",
    )
    if ppo37_conf is None:
        ppo37_conf = target_evaluate(
            config,
            plan,
            out,
            "checkpoints/reference-update-0037.pt",
            "confirmation",
        )
        ppo37_conf["reused_saved_evaluation"] = False

    c2_conf = target_evaluate(config, plan, out, best_c2["checkpoint"], "confirmation")
    bc1m_c_out, conf_cases = _outcomes_from_dir(
        out / bc1m_conf["runs"]["argmax"]["cp"]["directory"]
    )
    ppo37_c_out, ppo_conf_cases = _outcomes_from_dir(
        out / ppo37_conf["runs"]["argmax"]["cp"]["directory"]
    )
    c2_c_out, c2_conf_cases = _outcomes_from_dir(
        out / c2_conf["runs"]["argmax"]["cp"]["directory"]
    )
    if conf_cases != c2_conf_cases or ppo_conf_cases != c2_conf_cases:
        raise ValueError("Unpaired Additional-256 confirmation")
    conf_vs_bc1m = paired_world_bootstrap(
        bc1m_c_out,
        c2_c_out,
        samples=10000,
        seed=20261103,
        cases=conf_cases,
        cases_per_world=16,
    )
    conf_vs_ppo37 = paired_world_bootstrap(
        ppo37_c_out,
        c2_c_out,
        samples=10000,
        seed=20261103,
        cases=conf_cases,
        cases_per_world=16,
    )
    write_json(
        out / "confirmation.json",
        {
            "status": "complete",
            "baseline": bc1m_conf,
            "ppo37_reference": ppo37_conf,
            "candidate": c2_conf,
            "comparison": conf_vs_bc1m,
            "comparison_vs_ppo37": conf_vs_ppo37,
        },
    )


def evaluate(config, plan, out):
    if (out / "confirmation.json").exists():
        return
    training = read_json(out / "training.json")
    if training["status"] != "complete":
        raise ValueError("Training is incomplete")
    if config["kind"] == C2_KIND:
        _evaluate_c2(config, plan, out, training)
        return
    selection = read_json(out / "selection.json")
    best = selection["best"]
    if config["kind"] == MORE_DATA_KIND:
        if selection.get("baseline_retained", False) or (
            selection["best_wins"] <= selection["baseline_wins"]
        ):
            write_json(
                out / "confirmation.json",
                {
                    "status": "not_run_no_development_improvement",
                    "selected": best,
                    "baseline": selection["baseline"],
                    "baseline_wins": selection["baseline_wins"],
                    "required_wins": selection["required_new_candidate_wins"],
                },
            )
            return
        ref_run = ROOT / config["reference_run"]
        old_run = ROOT / config["old_corpus_run"]
        ppo_hash = config["evaluation_reference_policy_hash"]
        bc_hash = config["parent_policy_hash"]
        before = _try_reuse_saved_evaluation(
            config,
            plan,
            out,
            ref_run,
            "checkpoints/reference-update-0037.pt",
            ppo_hash,
            "confirmation",
        )
        if before is None:
            before = target_evaluate(
                config,
                plan,
                out,
                "checkpoints/reference-update-0037.pt",
                "confirmation",
            )
            before["reused_saved_evaluation"] = False
        after = target_evaluate(config, plan, out, best["checkpoint"], "confirmation")
        first, cases = _outcomes_from_dir(
            out / before["runs"]["argmax"]["cp"]["directory"]
        )
        second, paired = _outcomes_from_dir(
            out / after["runs"]["argmax"]["cp"]["directory"]
        )
        if cases != paired:
            raise ValueError("Unpaired target BC confirmation")
        interval = paired_world_bootstrap(
            first,
            second,
            samples=10000,
            seed=20261103,
            cases=cases,
            cases_per_world=16,
        )
        # Secondary comparison against already available BC epoch 10 confirmation results (no 3rd 256-match run).
        secondary = None
        for candidate_root in (ref_run, old_run):
            evals_path = candidate_root / "evaluations.json"
            if not evals_path.is_file():
                continue
            saved_evals = read_json(evals_path)
            bc_conf = next(
                (
                    v
                    for v in saved_evals.values()
                    if isinstance(v, dict)
                    and v.get("purpose") == "confirmation"
                    and v.get("policy_hash") == bc_hash
                ),
                None,
            )
            if bc_conf is not None:
                bc_dir = (
                    candidate_root / bc_conf["runs"]["argmax"]["cp"]["directory"]
                )
                if (bc_dir).is_dir() and len(list(bc_dir.glob("*/metrics.json"))) == 256:
                    bc_outcomes, bc_cases = _outcomes_from_dir(bc_dir)
                    if bc_cases == paired:
                        secondary = {
                            "reference": "bc_epoch_10",
                            "policy_hash": bc_hash,
                            "source_run": candidate_root.relative_to(ROOT).as_posix(),
                            "summary": bc_conf["runs"]["argmax"]["cp"]["summary"],
                            "comparison": paired_world_bootstrap(
                                bc_outcomes,
                                second,
                                samples=10000,
                                seed=20261103,
                                cases=cases,
                                cases_per_world=16,
                            ),
                        }
                        break
        write_json(
            out / "confirmation.json",
            {
                "status": "complete",
                "baseline": before,
                "candidate": after,
                "comparison": interval,
                "secondary_bc_epoch_10": secondary,
            },
        )
        return
    base = selection["baseline"]
    if selection["best_wins"] <= base["runs"]["argmax"]["cp"]["summary"]["W"]:
        write_json(
            out / "confirmation.json",
            {"status": "not_run_no_development_improvement", "selected": best},
        )
        return
    before = target_evaluate(config, plan, out, "checkpoints/initial.pt", "confirmation")
    after = target_evaluate(config, plan, out, best["checkpoint"], "confirmation")
    first, cases = _outcomes_from_dir(out / before["runs"]["argmax"]["cp"]["directory"])
    second, paired = _outcomes_from_dir(
        out / after["runs"]["argmax"]["cp"]["directory"]
    )
    if cases != paired:
        raise ValueError("Unpaired target BC confirmation")
    interval = paired_world_bootstrap(
        first, second, samples=10000, seed=20261103, cases=cases, cases_per_world=16
    )
    write_json(
        out / "confirmation.json",
        {
            "status": "complete",
            "baseline": before,
            "candidate": after,
            "comparison": interval,
        },
    )


def report(config, out):
    collection = (
        read_json(out / "collection.json")
        if (out / "collection.json").exists()
        else {}
    )
    training = (
        read_json(out / "training.json") if (out / "training.json").exists() else {}
    )
    selection = (
        read_json(out / "selection.json") if (out / "selection.json").exists() else {}
    )
    run_info = read_json(out / "run.json") if (out / "run.json").exists() else {}
    status_info = (
        read_json(out / "status.json") if (out / "status.json").exists() else {}
    )
    gpu_name = run_info.get("GPU", "CUDA GPU")
    if config["kind"] == C2_KIND:
        title = "# Архитектура C2: 8 hex-блоков и глобальное внимание к известным юнитам (`cp-imitation-target-c2`)"
    elif config["kind"] == MORE_DATA_KIND:
        title = "# Расширение корпуса и продолжение target BC (`cp-imitation-target-more-data`)"
    else:
        title = "# Имитация штатного CP на целевых армиях"
    lines = [
        title,
        "",
        f"Статус: {status_info}; GPU: {gpu_name}; seed 17.",
        "",
    ]
    if config["kind"] == C2_KIND:
        lines.append(
            f"Корпус: переиспользован объединённый корпус из {collection.get('labels')} свободных меток "
            f"({collection.get('positions')} позиций; старых боёв {collection.get('old_matches_count')}, "
            f"новых боёв {collection.get('new_matches_count')}); отложенный корпус Holdout-126: "
            f"{collection.get('holdout_labels')} свободных меток ({collection.get('holdout_positions')} позиций, "
            f"{collection.get('holdout_matches_count')} боёв на 7 мирах)."
        )
    elif config["kind"] == MORE_DATA_KIND:
        lines.append(
            f"Сбор: новых свободных меток {collection.get('new_labels')}, старых {collection.get('old_labels')}, "
            f"всего в объединённом корпусе {collection.get('labels')} свободных меток "
            f"({collection.get('positions')} позиций; старых боёв {collection.get('old_matches_count')}, "
            f"новых боёв {collection.get('new_matches_count')}); время нового сбора {collection.get('seconds')} с; "
            f"причина остановки: {collection.get('stop_reason')}."
        )
    else:
        lines.append(
            f"Сбор: {collection.get('labels')} свободных меток, {collection.get('positions')} позиций, "
            f"{len(collection.get('matches', []))} полных CP–CP боёв, {collection.get('seconds')} с; "
            f"причина {collection.get('stop_reason')}."
        )
    lines += [
        "",
        "Обе стороны обычного stateful CP сохранены отдельно; все исходы включены. "
        "Training-worlds: 57. Forced states и prefix не входят в знаменатель CE.",
        "",
        f"Обучение: {training.get('stop_reason')}; Adam steps {training.get('Adam_steps')}; "
        f"обработано свободных меток {training.get('free_labels_processed')}; active {training.get('active_seconds')} с; "
        f"превышение {training.get('time_overshoot')} с; microbatch {training.get('microbatch')}.",
        "",
    ]
    eval_purpose = "development_512" if config["kind"] == C2_KIND else "development"
    if config["kind"] == C2_KIND:
        lines += [
            "| Роль | Эпоха | Adam steps | Checkpoint | W/D/L (512) | Win rate | Fixed training CE / acc | Holdout-126 CE / acc |",
            "|---|---:|---:|---|---|---:|---|---|",
        ]
    else:
        lines += [
            "| Эпоха | Adam steps | Checkpoint | W/D/L | Win rate | Fixed training CE / accuracy |",
            "|---:|---:|---|---|---:|---|",
        ]
    if (out / "evaluations.json").exists():
        seen_dirs = set()
        raw_points = [
            p
            for p in read_json(out / "evaluations.json").values()
            if isinstance(p, dict) and p.get("purpose") == eval_purpose
        ]
        raw_points.sort(
            key=lambda p: (1 if p.get("candidate_role") else 0), reverse=True
        )
        points = []
        for p in raw_points:
            d = p.get("runs", {}).get("argmax", {}).get("cp", {}).get("directory")
            if d and d not in seen_dirs:
                seen_dirs.add(d)
                points.append(p)
        points.sort(
            key=lambda p: (
                int(p.get("Adam_steps", 0)),
                int(p.get("epoch", 0)),
                str(p.get("checkpoint", "")),
            )
        )
        for point in points:
            summary = point["runs"]["argmax"]["cp"]["summary"]
            metric = point.get("fixed_training") or {}
            if config["kind"] == C2_KIND:
                hold = point.get("holdout") or {}
                lines.append(
                    f"| {point.get('candidate_role', '')} | {point.get('epoch', 0)} | {point.get('Adam_steps', 0)} | {point['checkpoint']} | "
                    f"{summary['W']}/{summary['D']}/{summary['L']} | {summary['win_rate']:.2%} | "
                    f"{metric.get('ce')} / {metric.get('accuracy')} | {hold.get('ce')} / {hold.get('accuracy')} |"
                )
            else:
                lines.append(
                    f"| {point.get('epoch', 0)} | {point.get('Adam_steps', 0)} | {point['checkpoint']} | "
                    f"{summary['W']}/{summary['D']}/{summary['L']} | {summary['win_rate']:.2%} | "
                    f"{metric.get('ce')} / {metric.get('accuracy')} |"
                )
    lines += [
        "",
        "Selection: " + json.dumps(selection, ensure_ascii=False),
        "",
        "Value head неизменна: "
        + str(training.get("value_head_unchanged"))
        + ". Общий encoder менялся; калибровка critic не сохраняется.",
    ]
    if (out / "confirmation.json").exists():
        lines += [
            "",
            "Additional: "
            + json.dumps(read_json(out / "confirmation.json"), ensure_ascii=False),
        ]
    match_rows = collection.get("new_matches", collection.get("matches", []))
    if match_rows:
        commands = Counter()
        units = Counter()
        outcomes = Counter()
        strata = Counter()
        for row in match_rows:
            commands.update(row["commands"])
            units.update(row["unit_types"])
            outcomes[
                "draw" if row["winner"] is None else f"side_{row['winner']}_win"
            ] += 1
            strata[f"{row['map_size']}/{row['army_size']}/{row['placement']}"] += 1
        new_pos = collection.get("new_positions", collection.get("positions", 0))
        new_lbl = collection.get("new_labels", collection.get("labels", 0))
        lines += [
            "",
            "## Корпус",
            "",
            f"Forced positions (новый сбор): {new_pos - new_lbl}; "
            f"превышение меток {collection.get('label_overshoot')}, времени {collection.get('time_overshoot')} с; "
            f"физические повторы {collection.get('physical_repeats')}; "
            f"миров фактически {len(collection.get('coverage', {}).get('worlds', {}))}; "
            f"crops фактически {len(collection.get('coverage', {}).get('crops', {}))}.",
            "",
            "Исходы CP–CP (новый сбор): " + json.dumps(outcomes),
            "",
            "Свободные команды (новый сбор): " + json.dumps(commands),
            "",
            "Свободные метки по типу действующего юнита (новый сбор): "
            + json.dumps(units),
            "",
            "Расстановки map/army/contact (новый сбор): " + json.dumps(strata),
        ]
    if training.get("epochs"):
        if config["kind"] == C2_KIND:
            lines += [
                "",
                "## Обучение по эпохам",
                "",
                "| Эпоха | Полная | Свободные метки | Adam steps | Online CE | Fixed-training CE / acc | Holdout-126 CE / acc |",
                "|---:|---|---:|---:|---:|---|---|",
            ]
            for epoch in training["epochs"]:
                ft = epoch.get("fixed_training") or {}
                ho = epoch.get("holdout") or {}
                lines.append(
                    f"| {epoch['epoch']} | {epoch['completed']} | {epoch['labels']} | {epoch['steps']} | {epoch['online_CE']} | "
                    f"{ft.get('ce')} / {ft.get('accuracy')} | {ho.get('ce')} / {ho.get('accuracy')} |"
                )
        else:
            lines += [
                "",
                "## Обучение",
                "",
                "| Эпоха | Полная | Свободные метки | Adam steps | Online CE |",
                "|---:|---|---:|---:|---:|",
            ]
            for epoch in training["epochs"]:
                lines.append(
                    f"| {epoch['epoch']} | {epoch['completed']} | {epoch['labels']} | {epoch['steps']} | {epoch['online_CE']} |"
                )
        lines += [
            "",
            "История/prefix и кеш: " + json.dumps(training.get("cache", {})),
            "",
            "Online CE усредняется по меткам при изменявшихся весах. Fixed training CE измеряется при фиксированных весах.",
        ]
    if (out / "training-telemetry/resources.jsonl").exists():
        analysis = analyze_telemetry(out / "training-telemetry/resources.jsonl")
        write_json(
            out / "gpu-analysis.json",
            {"metrics": analysis["metrics"], "windows": analysis["windows"]},
        )
        lines += [
            "",
            "## GPU NVML",
            "",
            json.dumps(analysis["metrics"]),
            "",
            f"Непрерывных окон 60 s: {len(analysis['windows'])}. Подготовка, compilation, offline CE и боевая оценка исключены.",
            "",
            "Разбор загрузки GPU: "
            + json.dumps(training.get("GPU_utilization_review"), ensure_ascii=False),
        ]
    if selection and (out / "evaluations.json").exists():
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        points = [
            p
            for p in read_json(out / "evaluations.json").values()
            if isinstance(p, dict) and p.get("purpose") == eval_purpose
        ]
        if config["kind"] == C2_KIND:
            points = [
                p for p in points if p.get("candidate_role") == "c2_candidate"
            ]
        points.sort(key=lambda p: int(p.get("Adam_steps", 0)))
        if points:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
            axes[0].plot(
                [p.get("Adam_steps", 0) for p in points],
                [
                    100 * p["runs"]["argmax"]["cp"]["summary"]["win_rate"]
                    for p in points
                ],
                marker="o",
            )
            axes[0].set(
                xlabel="BC Adam steps",
                ylabel=f"{eval_purpose} argmax wins vs CP, %",
            )
            metrics = [p for p in points if p.get("fixed_training")]
            if metrics:
                axes[1].plot(
                    [p.get("Adam_steps", 0) for p in metrics],
                    [p["fixed_training"]["ce"] for p in metrics],
                    marker="o",
                    label="Fixed training CE",
                )
            holdout_pts = [p for p in points if p.get("holdout")]
            if holdout_pts:
                axes[1].plot(
                    [p.get("Adam_steps", 0) for p in holdout_pts],
                    [p["holdout"]["ce"] for p in holdout_pts],
                    marker="s",
                    label="Holdout-126 CE",
                )
                axes[1].legend()
            axes[1].set(xlabel="BC Adam steps", ylabel="Offline CE")
            for ax in axes:
                ax.grid(alpha=0.25)
            fig.savefig(out / "learning-curves.png", dpi=150)
            plt.close(fig)
            lines += ["", "![Кривые](learning-curves.png)"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    with RunLock(out):
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
        try:
            plan = prepare(config, out)
            if (
                (out / "status.json").exists()
                and read_json(out / "status.json").get("status") == "complete"
            ):
                report(config, out)
                return
            if phase in ("collect", "run", "resume"):
                if not (
                    (out / "collection.json").exists()
                    and read_json(out / "collection.json").get("status") == "complete"
                ):
                    write_json(out / "status.json", {"status": "collecting"})
                    collect(config, plan, out)
                    write_json(out / "status.json", {"status": "collection_complete"})
            if phase in ("train", "run", "resume"):
                if not (
                    (out / "training.json").exists()
                    and read_json(out / "training.json").get("status") == "complete"
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
                selected = read_json(out / "selection.json")["best"]
                directory = out / selected["runs"]["argmax"]["cp"]["directory"]
                for point in read_json(out / "evaluations.json").values():
                    if not isinstance(point, dict) or "runs" not in point:
                        continue
                    d = out / point["runs"]["argmax"]["cp"]["directory"]
                    paths = list(d.glob("*/metrics.json"))
                    if len(paths) != point["runs"]["argmax"]["cp"]["summary"][
                        "matches"
                    ] or any(
                        not (p.parent / name).is_file()
                        for p in paths
                        for name in (
                            "case.json",
                            "result.json",
                            "final.json",
                            "replay.json",
                        )
                    ):
                        raise ValueError("Incomplete saved evaluation artifacts")
                chosen = {}
                for path in sorted(directory.glob("*/metrics.json")):
                    r = read_json(path)
                    kind_outcome = (
                        "D"
                        if r["winner"] is None
                        else "W"
                        if r["winner"] == r["side"]
                        else "L"
                    )
                    chosen.setdefault(kind_outcome, path.parent)
                write_json(
                    out / "replay-verification.json",
                    {k: verify_replay(p) for k, p in chosen.items()},
                )
                write_json(
                    out / "status.json",
                    {
                        "status": "complete",
                        "confirmation": read_json(out / "confirmation.json")["status"],
                    },
                )
            report(config, out)
        except BaseException:
            (out / "traceback.txt").write_text(
                traceback.format_exc(), encoding="utf-8"
            )
            write_json(
                out / "status.json", {"status": "technical_error", "phase": phase}
            )
            report(config, out)
            raise
