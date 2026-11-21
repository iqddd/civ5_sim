"""Fixed paired-damage comparisons and training-only anchors for whole turns."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from itertools import combinations
from os import PathLike
from pathlib import Path
from typing import Any, TypedDict

from civ5rl.engine.comparison.storage import readable_members
from civ5rl.engine.runtime import is_int_at_least
from civ5rl.experiments._shared.provenance_checks import verify_file_hashes
from civ5rl.experiments.comparison.turn_protocol import (
    CONFIRMATION_REPETITIONS,
    SCREENING_REPETITIONS,
    SCREENING_VARIANTS,
)
from civ5rl.provenance.common import ROOT, read_json
from civ5rl.provenance.jsonl import read_jsonl

VARIANT_PAIRS = tuple(combinations(SCREENING_VARIANTS, 2))


class _Comparison(TypedDict):
    a: int
    b: int
    sign: int
    repeats: int


class _Position(TypedDict):
    index: int
    key: list[int]
    world: int
    comparisons: list[_Comparison]


class _Branch(TypedDict):
    position: int
    episode: int
    world: int
    prefix: int
    command: int
    key: list[int]
    variant: int
    repetition: int
    packet: str
    folder: str


@dataclass(frozen=True)
class _TurnComparisons:
    positions: list[_Position]
    branches: dict[str, _Branch]


def verify_inputs(cfg: Mapping[str, Any]) -> None:
    verify_file_hashes(ROOT, cfg["input_sha256"], label="Whole-turn input changed")


def _build_turn_comparisons(
    cfg: Mapping[str, Any],
    screening: Path,
    confirmation: Path,
    position_indices: Iterable[int] | None,
) -> _TurnComparisons:
    manifest = read_json(screening / "positions.json")
    review = {tuple(row["pos"]): row for row in read_json_lines(ROOT / cfg["review"])}
    selected = {
        tuple(row["position_key"]): row["variant"]
        for row in read_json(screening / "derived/turn-review/candidates.json")["rows"]
    }
    confirmed: defaultdict[tuple[int, ...], dict[tuple[int, int], dict[str, Any]]] = defaultdict(dict)
    for row in read_json(confirmation / "continuations-summary.json"):
        confirmed[tuple(row["position_key"])][row["turn_variant"], row["repetition"]] = row
    positions: list[_Position] = []
    branches: dict[str, _Branch] = {}
    indices = range(len(manifest["positions"])) if position_indices is None else position_indices
    for position_index in indices:
        position = manifest["positions"][position_index]
        key = tuple(position["key"])
        screened = {(row["v"], row["r"]): row for row in review[key]["rows"]}
        comparisons: list[_Comparison] = []
        for a, b in VARIANT_PAIRS:
            uses_confirmation = key in selected and (a, b) == (0, selected[key])
            repeats = CONFIRMATION_REPETITIONS if uses_confirmation else SCREENING_REPETITIONS
            for repetition in repeats:
                pair: list[int] = []
                outcomes: list[bool] = []
                for variant in (a, b):
                    identity = 1_000_000_000 + position_index * 100 + variant * 12 + repetition
                    row = (
                        confirmed[key][variant, repetition]
                        if uses_confirmation
                        else screened[variant, repetition]
                    )
                    outcomes.append(row["outcome" if uses_confirmation else "wdl"] == "W")
                    # JSON object keys are strings. Use that representation before hashing too.
                    branches[str(identity)] = {
                        "position": position_index,
                        "episode": position["episode"],
                        "world": position["world"],
                        "prefix": position["history_index"],
                        "command": position["command"],
                        "key": position["key"],
                        "variant": variant,
                        "repetition": repetition,
                        "packet": "confirmation" if uses_confirmation else "screening",
                        "folder": f"{position['episode']}-{position['key'][1]}/{variant}-{repetition}",
                    }
                    pair.append(identity)
                comparisons.append(
                    {
                        "a": pair[0],
                        "b": pair[1],
                        "sign": int(outcomes[0]) - int(outcomes[1]),
                        "repeats": len(repeats),
                    }
                )
        positions.append(
            {
                "index": position_index,
                "key": position["key"],
                "world": position["world"],
                "comparisons": comparisons,
            }
        )
    return _TurnComparisons(positions, branches)


def _select_training_world_anchors(
    cfg: Mapping[str, Any], screening: Path, anchor_limit: int | None
) -> list[list[int]]:
    all_anchors: list[list[int]] = read_json(ROOT / cfg["anchor_input"])["positions"]
    worlds = set(read_json(ROOT / cfg["split"])["training"])
    source_packet = screening / "sources.storage"
    episodes = sorted({episode for episode, _ in all_anchors})
    with readable_members(source_packet, [f"{episode}/case.json" for episode in episodes]) as source:
        allowed = {
            episode
            for episode in episodes
            if read_json(source / str(episode) / "case.json")["parent_world"] in worlds
        }
    anchors = [anchor for anchor in all_anchors if anchor[0] in allowed]
    if not anchors:
        raise ValueError("No anchors belong to the configured training worlds")
    return anchors if anchor_limit is None else anchors[:anchor_limit]


def _check_expected_counts(expected, counts):
    for key, value in expected.items():
        if key not in counts or not is_int_at_least(value, 0):
            raise ValueError(f"Invalid expected whole-turn count: {key}")
        if counts[key] != value:
            raise ValueError(f"Expected {value} {key}, found {counts[key]}")


def build_dataset(
    cfg: Mapping[str, Any],
    *,
    position_indices: Iterable[int] | None = None,
    anchor_limit: int | None = None,
) -> dict[str, Any]:
    """Include zero comparisons in denominators and use fresh futures for selected pairs."""
    verify_inputs(cfg)
    screening = ROOT / cfg["screening_run"]
    confirmation = ROOT / cfg["confirmation_run"]
    turns = _build_turn_comparisons(cfg, screening, confirmation, position_indices)
    anchors = _select_training_world_anchors(cfg, screening, anchor_limit)
    if not turns.positions or not anchors:
        raise ValueError("Whole-turn fitting requires positions and training anchors")
    if len({tuple(position["key"]) for position in turns.positions}) != len(turns.positions):
        raise ValueError("Duplicate whole-turn position keys")
    counts = {
        "positions": len(turns.positions),
        "variant_pairs": len(turns.positions) * len(VARIANT_PAIRS),
        "comparisons": sum(len(position["comparisons"]) for position in turns.positions),
        "nonzero": sum(
            comparison["sign"] != 0 for position in turns.positions for comparison in position["comparisons"]
        ),
        "positions_with_signal": sum(
            any(comparison["sign"] for comparison in position["comparisons"]) for position in turns.positions
        ),
        "anchors": len(anchors),
    }
    if position_indices is None and anchor_limit is None:
        _check_expected_counts(cfg.get("expected_counts", {}), counts)
    return {
        "schema": 1,
        "positions": turns.positions,
        "branches": turns.branches,
        "anchors": anchors,
        "counts": counts,
        "input_sha256": cfg["input_sha256"],
    }


def read_json_lines(path: str | PathLike[str]) -> list[dict[str, Any]]:
    return read_jsonl(path, streaming=True)


@contextmanager
def history_inputs(
    cfg: Mapping[str, Any], dataset: Mapping[str, Any]
) -> Iterator[tuple[Path, dict[str, Path], dict[int, _Branch]]]:
    """Read only needed public histories; extraction can be rebuilt after any interruption."""
    active = {c[k] for p in dataset["positions"] for c in p["comparisons"] if c["sign"] for k in ("a", "b")}
    branches: dict[int, _Branch] = {b: dataset["branches"][str(b)] for b in sorted(active)}
    episodes = {branch["episode"] for branch in branches.values()}
    episodes.update(e for e, _ in dataset["anchors"])
    with ExitStack() as stack:
        sources = stack.enter_context(
            readable_members(
                ROOT / cfg["screening_run"] / "sources.storage",
                [f"{e}/{name}.json" for e in episodes for name in ("case", "history")],
            )
        )
        roots: dict[str, Path] = {}
        for packet, run in [
            ("screening", cfg["screening_run"]),
            ("confirmation", cfg["confirmation_run"]),
        ]:
            names = [
                f"{b['folder']}/{name}.json"
                for b in branches.values()
                if b["packet"] == packet
                for name in ("turn-history", "completed", "branch")
            ]
            if names:
                roots[packet] = stack.enter_context(
                    readable_members(ROOT / run / "continuations.storage", names)
                )
        yield sources, roots, branches
