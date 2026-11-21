"""Source snapshots, outcome-blind anchors, and pairs for rollout comparisons."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, TypeVar

from civ5rl.engine.runtime import is_int_at_least
from civ5rl.provenance.common import seed_for, write_json

ForkPosition: TypeAlias = tuple[int, int]
ForkActions: TypeAlias = tuple[int, int | None]
ForkLabel: TypeAlias = tuple[str, Mapping[str, Any]]

_Fork = TypeVar("_Fork", bound=Mapping[str, Any])


@dataclass(frozen=True)
class ComparisonDatasetInputs:
    anchors: list[ForkPosition]
    source_hashes: dict[str, dict[str, str]]
    fork_actions: dict[ForkPosition, ForkActions]
    learner_sides: dict[int, int]


def validate_fit_options(training: Mapping[str, Any]) -> None:
    """Validate configured capacities and KL-control settings used by both fitters."""
    for key in ("max_accepted_steps", "anchors_per_step"):
        if not is_int_at_least(training.get(key)):
            raise ValueError(f"training.{key} must be a positive integer")
    if not is_int_at_least(training.get("max_retries"), 0):
        raise ValueError("training.max_retries must be a nonnegative integer")
    for key in ("initial_lr", "kl_limit"):
        value = training.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"training.{key} must be positive and finite")


def _read_hashed_json(path: Path) -> tuple[Any, str]:
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def prepare_comparison_dataset(
    out: Path,
    state: Mapping[str, Any],
    *,
    seed: int,
    namespace: str,
    sort_sources: bool,
    load_learner_sides: bool,
) -> ComparisonDatasetInputs:
    """Read each input once and publish anchors before callers access outcomes."""
    anchor_rng = random.Random(seed_for(seed, 0, "anchors", namespace=namespace))
    free_positions: list[ForkPosition] = []
    fork_actions: dict[ForkPosition, ForkActions] = {}
    learner_sides: dict[int, int] = {}
    source_hashes: dict[str, dict[str, str]] = {}
    fork_positions = {(fork["episode"], fork["index"]) for fork in state["forks"]}
    sources = sorted(state["sources"]) if sort_sources else state["sources"]
    for episode in sources:
        folder = out / "sources" / str(episode)
        case, case_sha = _read_hashed_json(folder / "case.json")
        rows, history_sha = _read_hashed_json(folder / "history.json")
        if load_learner_sides:
            learner_sides[episode] = case["learner_side"]
        source_hashes[str(episode)] = {
            "case.json": case_sha,
            "history.json": history_sha,
        }
        free_positions.extend((episode, index) for index, row in enumerate(rows) if len(row["actions"]) > 1)
        for index, row in enumerate(rows):
            position = (episode, index)
            if position in fork_positions:
                fork_actions[position] = (row["chosen"], row["alternative"])

    sampled = anchor_rng.sample(free_positions, min(8192, len(free_positions)))
    anchors = sorted(set(sampled) | fork_positions)
    write_json(
        out / "anchors.json",
        {"positions": anchors, "selection_blind_to_continuation_outcomes": True},
    )
    return ComparisonDatasetInputs(anchors, source_hashes, fork_actions, learner_sides)


def write_labels_and_collect_pairs(
    out: Path,
    forks: Iterable[_Fork],
    fork_actions: Mapping[ForkPosition, ForkActions],
    label_fork: Callable[[_Fork], ForkLabel | None],
) -> list[dict[str, Any]]:
    """Write each protocol's label record before resolving its accepted pair."""
    pairs = []
    for fork in forks:
        labeled = label_fork(fork)
        if labeled is None:
            continue
        filename, record = labeled
        write_json(out / "labels" / filename, record)
        if record["accepted"]:
            actions = fork_actions[(fork["episode"], fork["index"])]
            pairs.append(
                {
                    "episode": fork["episode"],
                    "index": fork["index"],
                    "better": actions[record["better"]],
                    "worse": actions[1 - record["better"]],
                    "weight": record["weight"],
                }
            )
    return pairs
