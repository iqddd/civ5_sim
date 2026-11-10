"""Corpus preparation and collection for target scale BC experiments."""

from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

from civ5rl.engine.resources import ResourceMonitor
from civ5rl.experiments.imitation_helpers import pool
from civ5rl.protocols.imitation_target_scale import (
    C2_KIND,
    MORE_DATA_KIND,
    match_task,
)
from civ5rl.protocols.ppo_target_scale import (
    HOLDOUT_WORLDS,
    TargetFactory,
)
from civ5rl.provenance.common import (
    ROOT,
    file_sha256,
    read_json,
    write_json,
)
from civ5sim.engine import digest


def _entry_lookup_key(entry: dict[str, Any]) -> str:
    """Return unique lookup key for a corpus demonstration entry."""
    return str(entry.get("entry_id", entry["path"]))


entry_lookup_key = _entry_lookup_key


def _format_corpus_entry(
    entry: dict[str, Any],
    corpus_source: str,
    path: str,
    map_size: int,
    army_size: int,
    sha256: str | None = None,
) -> dict[str, Any]:
    """Standardize metadata for a single match side demonstration."""
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


format_corpus_entry = _format_corpus_entry


def _load_old_corpus(config: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Load and validate the initial 200,483 free label target BC demonstration corpus."""
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


load_old_corpus = _load_old_corpus


def _load_more_data_corpus(config: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """Load and validate the 800,333 free label target more-data demonstration corpus."""
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


load_more_data_corpus = _load_more_data_corpus


def _restore_collection_progress(
    config: dict[str, Any], factory: TargetFactory, out: Path
) -> tuple[list[dict[str, Any]], int | None, float | None, int | None]:
    """Resume partial collection wave for more-data target BC."""
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


restore_collection_progress = _restore_collection_progress


def _collect_c2(
    config: dict[str, Any], plan: dict[str, Any], out: Path
) -> dict[str, Any]:
    """Assemble combined 1M BC corpus and generate Holdout-126 matches for C2."""
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


collect_c2 = _collect_c2


def collect(config: dict[str, Any], plan: dict[str, Any], out: Path) -> dict[str, Any]:
    """Execute corpus collection phase for target BC experiments."""
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


__all__ = [
    "_collect_c2",
    "_entry_lookup_key",
    "_format_corpus_entry",
    "_load_more_data_corpus",
    "_load_old_corpus",
    "_restore_collection_progress",
    "collect",
    "collect_c2",
    "entry_lookup_key",
    "format_corpus_entry",
    "load_more_data_corpus",
    "load_old_corpus",
    "restore_collection_progress",
]
