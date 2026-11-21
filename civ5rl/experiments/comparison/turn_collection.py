"""Outcome-blind whole-turn collection from immutable recorded source battles."""

from __future__ import annotations

import random
import shutil
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from civ5rl.engine.comparison.storage import (
    archive_tree,
    readable_tree,
    restore_tree,
)
from civ5rl.engine.evaluation.executor import EvaluationRuntime
from civ5rl.engine.runtime import is_int_at_least
from civ5rl.engine.runtime.control import RunLock
from civ5rl.engine.runtime.telemetry import ResourceMonitor
from civ5rl.experiments._shared.provenance_checks import verify_file_hashes
from civ5rl.experiments.comparison.match import match_task
from civ5rl.experiments.comparison.turn_confirmation import prepare_confirmation, summarize_confirmation
from civ5rl.experiments.comparison.turn_executor import TurnExecutor
from civ5rl.experiments.comparison.turn_protocol import (
    CONFIRMATION_REPETITIONS,
    NAMESPACE,
    SCREENING_REPETITIONS,
    SCREENING_VARIANTS,
    branch_spec,
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

KIND = "rl01.turn_comparison_collection"


@dataclass(frozen=True, slots=True)
class _ScreeningInputs:
    source: Path
    training_worlds: Sequence[int]
    lineage: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _TurnContinuationTask:
    config: dict[str, Any]
    case: dict[str, Any]
    directory: str
    checkpoint: str
    branch: dict[str, Any]

    def worker_args(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any], str, str, dict[str, Any]]:
        """Keep the worker's five-element tuple and its pickle format."""
        return self.config, self.case, self.directory, self.checkpoint, self.branch


@dataclass(frozen=True, slots=True)
class _ContinuationBatch:
    completed: tuple[dict[str, Any], ...]
    pending: tuple[_TurnContinuationTask, ...]


def position_candidates(
    episode: int,
    case: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Require the actual first decision, never skip a forced turn opening."""
    candidates = []
    seen_turns: set[int] = set()
    side = case["learner_side"]
    for history_index, row in enumerate(history):
        turn = row["observation"]["turn"]
        if turn in seen_turns:
            continue
        seen_turns.add(turn)
        command = row["command"]
        if command == 0 or replay["steps"][command - 1]["action"]["kind"] != "end_turn":
            continue
        observation = row["observation"]
        if len(row["actions"]) <= 1 or not any(unit["side"] != side for unit in observation["units"]):
            continue
        candidates.append(
            {
                "key": [episode, turn, side],
                "episode": episode,
                "command": command,
                "world": case["parent_world"],
                "history_index": history_index,
                "observation_sha256": digest(observation),
            }
        )
    return candidates


def select_positions(
    by_world: Mapping[int, Mapping[int, Sequence[dict[str, Any]]]], maximum: int = 512
) -> list[dict[str, Any]]:
    """Round-robin shuffled worlds/battles; exhaust first picks before seconds."""
    rng = random.Random(seed_for(17, 0, "positions", namespace=NAMESPACE + "_selection"))
    worlds = sorted(by_world)
    rng.shuffle(worlds)
    episode_order: dict[int, list[int]] = {}
    for world in worlds:
        episodes = sorted(by_world[world])
        rng.shuffle(episodes)
        episode_order[world] = episodes
    battles = []
    for index in range(max((len(episodes) for episodes in episode_order.values()), default=0)):
        for world in worlds:
            if index < len(episode_order[world]):
                battles.append((world, episode_order[world][index]))
    choices = {}
    for world, episode in battles:
        rows = sorted(by_world[world][episode], key=lambda position: tuple(position["key"]))
        choices[episode] = rng.sample(rows, min(2, len(rows)))
    selected = []
    for pick in range(2):
        for _, episode in battles:
            if len(choices[episode]) > pick:
                selected.append(choices[episode][pick])
            if len(selected) >= maximum:
                return selected
    return selected


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = read_json(path)
    if cfg.get("kind") != KIND or cfg.get("seed") != 17:
        raise ValueError("Unknown whole-turn protocol")
    for key, value in (("positions", cfg.get("positions")), ("workers", cfg.get("workers", 8))):
        if not is_int_at_least(value):
            raise ValueError(f"{key} must be a positive integer")
    if "confirmation" in cfg and cfg["confirmation"]["repetitions"] != list(CONFIRMATION_REPETITIONS):
        raise ValueError("Confirmation repetitions must use the independent damage group")
    return cfg


def _prepare_screening_inputs(cfg: dict[str, Any], out: Path) -> _ScreeningInputs:
    source = ROOT / cfg["collection_input"]["run"]
    verify_file_hashes(source, cfg["collection_input"]["sha256"], label="Source changed")
    checkpoint.verify_frozen_passport(source, sources=False)
    if file_sha256(ROOT / cfg["parent"]) != cfg["parent_sha256"]:
        raise ValueError("Parent changed")
    if file_sha256(ROOT / cfg["split"]) != cfg["split_sha256"]:
        raise ValueError("World split changed")
    training_worlds = read_json(ROOT / cfg["split"])["training"]
    lineage = {
        "source": cfg["collection_input"],
        "training_worlds": training_worlds,
        "split_sha256": cfg["split_sha256"],
    }
    if (out / "passport.json").exists():
        checkpoint.verify_execution_passport(out, cfg, lineage)
    else:
        identity = checkpoint.passport(out, cfg, lineage)
        write_json(out / "run.json", {"kind": KIND, "config": cfg, "identity": identity})
    return _ScreeningInputs(source, training_worlds, lineage)


def prepare(cfg: dict[str, Any], out: Path) -> None:
    if "confirmation" in cfg:
        return prepare_confirmation(cfg, out)
    inputs = _prepare_screening_inputs(cfg, out)
    if (out / "positions.json").exists():
        if file_sha256(out / "positions.json") != read_json(out / "selection.json")["sha256"]:
            raise ValueError("Position manifest changed")
        return
    write_json(out / "status.json", {"status": "preparing", "phase": "selection"})
    source_storage = out / "sources.storage"
    if not source_storage.exists():
        shutil.copytree(inputs.source / "sources.storage", source_storage)
    started = perf_counter()
    by_world, source_battles = _screening_candidates(out, inputs)
    positions = select_positions(by_world, cfg["positions"])
    if not positions:
        raise ValueError("No eligible training positions in the source collection")
    manifest = {
        "schema": 1,
        "namespace": NAMESPACE,
        "positions": positions,
        "lineage": inputs.lineage,
        "parent_sha256": cfg["parent_sha256"],
        "variants": len(SCREENING_VARIANTS),
        "repetitions": len(SCREENING_REPETITIONS),
    }
    write_json(out / "positions.json", manifest)
    write_json(
        out / "selection.json",
        {
            "sha256": file_sha256(out / "positions.json"),
            "source_battles": source_battles,
            "eligible_battles": sum(len(episodes) for episodes in by_world.values()),
            "eligible_positions": sum(
                len(positions) for episodes in by_world.values() for positions in episodes.values()
            ),
            "selected": len(positions),
            "target_continuations": len(positions) * len(SCREENING_VARIANTS) * len(SCREENING_REPETITIONS),
            "seconds": perf_counter() - started,
            "world_counts": dict(Counter(p["world"] for p in positions)),
        },
    )
    write_json(out / "status.json", {"status": "prepared", "phase": "collection"})


def _screening_candidates(out: Path, inputs: _ScreeningInputs):
    episodes = read_json(inputs.source / "collection.json")["sources"]
    if not episodes or any(not is_int_at_least(episode) for episode in episodes):
        raise ValueError("Source episode inventory must contain positive IDs")
    if len(set(episodes)) != len(episodes):
        raise ValueError("Duplicate source episode IDs")
    source_hashes = read_json(inputs.source / "dataset.json")["source_hashes"]
    by_world: dict[int, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    source_battles = 0
    with readable_tree(out, "sources") as sources:
        for episode in sorted(episodes):
            folder = sources / str(episode)
            case = read_json(folder / "case.json")
            if case["episode_id"] != episode:
                raise ValueError(f"Source episode identity differs: {episode}")
            world = case["parent_world"]
            if world != case["map_entry"]["parent_seed"]:
                raise ValueError(f"Source world identity differs: {episode}")
            if world not in inputs.training_worlds:
                continue
            source_battles += 1
            verify_file_hashes(folder, source_hashes[str(episode)], label="Public source history changed")
            candidates = position_candidates(
                episode, case, read_json(folder / "history.json"), read_json(folder / "replay.json")
            )
            if candidates:
                by_world[world][episode] = candidates
    return by_world, source_battles


def _stage_continuations(
    cfg: dict[str, Any],
    out: Path,
    sources: Path,
    positions: Sequence[Mapping[str, Any]],
    repetitions: Sequence[int],
    *,
    confirmation: bool,
) -> _ContinuationBatch:
    """Check completed branches against their frozen identities before resuming."""
    completed = []
    pending = []
    match_config = read_json(ROOT / cfg["collection_input"]["run"] / "match-config.json")
    for position in positions:
        episode = position["episode"]
        case = read_json(sources / str(episode) / "case.json")
        variants = (0, position["selected_variant"]) if confirmation else SCREENING_VARIANTS
        for variant in variants:
            for repetition in repetitions:
                folder = out / "continuations" / f"{episode}-{position['key'][1]}" / f"{variant}-{repetition}"
                branch = branch_spec(position, variant, repetition, sources / str(episode))
                if (folder / "completed.json").exists():
                    if read_json(folder / "branch.json") != branch or not (folder / "checked.json").exists():
                        raise ValueError("Saved continuation identity/check differs")
                    completed.append(read_json(folder / "completed.json"))
                    continue
                pending.append(
                    _TurnContinuationTask(
                        match_config,
                        case,
                        str(folder),
                        str(ROOT / cfg["parent"]),
                        branch,
                    )
                )
    return _ContinuationBatch(tuple(completed), tuple(pending))


def _summarize_screening(
    out: Path,
    positions: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    elapsed: float,
) -> None:
    """Publish descriptive screening outcomes and exact sampled-turn diversity."""
    variant_counts = {}
    for variant in SCREENING_VARIANTS:
        variant_rows = [row for row in rows if row["turn_variant"] == variant]
        variant_counts[str(variant)] = {
            outcome: sum(
                ("D" if row["winner"] is None else "W" if row["winner"] == row["side"] else "L") == outcome
                for row in variant_rows
            )
            for outcome in "WDL"
        }
    indexed = {(tuple(row["position_key"]), row["turn_variant"], row["repetition"]): row for row in rows}
    paired_vs_argmax = {}
    for variant in SCREENING_VARIANTS[1:]:
        differences = [
            int(row["winner"] == row["side"]) - int(indexed[(key, 0, repetition)]["winner"] == row["side"])
            for (key, row_variant, repetition), row in indexed.items()
            if row_variant == variant
        ]
        paired_vs_argmax[str(variant)] = {
            "mean_win_delta": sum(differences) / len(differences) if differences else None,
            "comparisons": len(differences),
        }
    unique_sequence_counts = []
    for position in positions:
        episode, turn, _ = position["key"]
        for repetition in SCREENING_REPETITIONS:
            sequences = []
            for variant in SCREENING_VARIANTS:
                history = read_json(
                    out
                    / "continuations"
                    / f"{episode}-{turn}"
                    / f"{variant}-{repetition}"
                    / "turn-history.json"
                )
                sequences.append(digest([row["actions"][row["chosen"]] for row in history]))
            unique_sequence_counts.append(len(set(sequences)))
    write_json(
        out / "result.json",
        {
            "status": "complete",
            "variants": variant_counts,
            "positions": len(positions),
            "continuations": len(rows),
            "seconds": elapsed,
            "paired_vs_argmax": paired_vs_argmax,
            "unique_turn_sequences": dict(Counter(unique_sequence_counts)),
            "note": "Descriptive paired futures; no best-sample policy claim",
        },
    )


def collect(cfg: dict[str, Any], out: Path, workers: int | None = None) -> None:
    if (out / "result.json").exists() and read_json(out / "collection.json").get("complete"):
        archive_tree(out, "continuations")
        write_json(out / "status.json", {"status": "complete", "phase": "complete"})
        return
    manifest = read_json(out / "positions.json")
    runtime = EvaluationRuntime(
        "cuda", cfg.get("workers", 8) if workers is None else workers, 32, 0.001
    ).resolve()
    runtime_path = out / "collection-runtime.json"
    if runtime_path.exists() and read_json(runtime_path) != runtime.identity():
        raise ValueError("Collection runtime changed")
    write_json(runtime_path, runtime.identity())
    restore_tree(out, "continuations")
    state = (
        read_json(out / "collection.json")
        if (out / "collection.json").exists()
        else {"completed": 0, "seconds": 0.0}
    )
    confirmation = "confirmation" in cfg
    repetitions = manifest["repetitions"] if confirmation else SCREENING_REPETITIONS
    state["target"] = (
        len(manifest["tasks"])
        if confirmation
        else len(manifest["positions"]) * len(SCREENING_VARIANTS) * len(repetitions)
    )
    started = perf_counter()
    base_seconds = state["seconds"]

    with (
        ResourceMonitor(out / "telemetry") as monitor,
        readable_tree(out, "sources") as sources,
    ):
        monitor.mark("collection")
        batch = _stage_continuations(
            cfg,
            out,
            sources,
            manifest["positions"],
            repetitions,
            confirmation=confirmation,
        )
        rows = list(batch.completed)
        write_json(out / "status.json", {"status": "collecting", "phase": "collection"})
        with TurnExecutor(str(ROOT / cfg["parent"]), runtime) as executor:
            for folder in executor.run(match_task, [task.worker_args() for task in batch.pending]):
                rows.append(read_json(Path(folder) / "completed.json"))
                state.update(
                    {
                        "completed": len(rows),
                        "seconds": base_seconds + perf_counter() - started,
                    }
                )
                if len(rows) % 16 == 0:
                    write_json(out / "collection.json", state)
    state.update(
        {
            "completed": len(rows),
            "seconds": base_seconds + perf_counter() - started,
            "complete": False,
        }
    )
    write_json(out / "collection.json", state)
    if len(rows) != state["target"]:
        raise ValueError("Collection incomplete")
    if confirmation:
        summarize_confirmation(cfg, out, manifest, rows, state["seconds"])
    else:
        _summarize_screening(out, manifest["positions"], rows, state["seconds"])
    archive_tree(out, "continuations")
    state["complete"] = True
    write_json(out / "collection.json", state)
    write_json(out / "status.json", {"status": "complete", "phase": "complete"})


def run(phase: str, cfg: dict[str, Any], out: Path | str, workers: int | None = None) -> None:
    out = Path(out).resolve()
    configure(cfg["seed"], "cuda")
    with RunLock(out):
        prepare(cfg, out)
        if phase in ("collect", "resume"):
            collect(cfg, out, workers)
