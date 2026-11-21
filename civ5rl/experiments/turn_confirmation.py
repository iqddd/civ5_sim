"""Independent damage futures for fixed, previously selected turn variants."""

from __future__ import annotations

import shutil
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict

import numpy as np
from numpy.typing import ArrayLike

from civ5rl.experiments._shared.outcomes import Outcome, classify_outcome
from civ5rl.experiments._shared.provenance_checks import verify_file_hashes
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import ROOT, file_sha256, read_json, write_json

INTERPRETATION = {
    "primary": "Equal-position mean paired win difference over all fixed candidates, including identical boundary states.",
    "positive": "CI95 lower bound > 0 supports average usefulness of selection on these positions and a subsequent policy-learning experiment; no training is started automatically.",
    "uncertain": "CI95 containing zero leaves the average advantage unresolved.",
    "negative": "CI95 upper bound < 0 supports an average disadvantage relative to argmax on these positions.",
    "scope": "Conditional on the selected positions; not a test of choosing variants from public information or of a learned policy in complete battles.",
    "individual": "Position signs are descriptive: eight repeats give a 12.5 percentage-point resolution.",
    "boundary": "Boundary hashes describe simulator state, excluding agent and GRU memory.",
}


class _ScreeningPosition(TypedDict):
    key: list[int]
    world: int
    episode: int
    command: int
    history_index: int
    observation_sha256: str


class _SelectedPosition(_ScreeningPosition):
    selected_variant: int


class _ScreeningCandidate(TypedDict):
    position_key: list[int]
    world: int
    variant: int
    screening_wins: list[int]


class _ConfirmationTask(TypedDict):
    position_key: list[int]
    variant: int
    repetition: int
    damage_seed: int
    sampling_seed: int


class _ConfirmationManifest(TypedDict):
    schema: int
    namespace: str
    positions: list[_SelectedPosition]
    parent_sha256: str
    tasks: list[_ConfirmationTask]
    repetitions: list[int]
    interpretation: dict[str, str]


class _PositionResult(TypedDict):
    position_key: list[int]
    world: int
    selected_variant: int
    wins: dict[str, int]
    paired_win_delta: float
    identical_boundary_states: int


class _PairedStatistics(TypedDict):
    mean_win_delta: float
    ci95: list[float]
    worlds: int
    samples: int
    seed: int
    conclusion: Literal["positive", "negative", "uncertain"]


_ContinuationKey = tuple[tuple[int, ...], int, int]
_ContinuationRow = Mapping[str, Any]
_ROLES = ("argmax", "selected")


def _verify_confirmation_inputs(cfg: dict[str, Any], source: Path) -> None:
    collection = cfg["collection_input"]
    verify_file_hashes(
        ROOT / collection["run"],
        collection["sha256"],
        label="Match configuration changed",
    )
    verify_file_hashes(source, cfg["confirmation"]["sha256"], label="Confirmation input changed")
    if file_sha256(ROOT / cfg["parent"]) != cfg["parent_sha256"]:
        raise ValueError("Parent changed")


def _screening_choice_differs(position: _ScreeningPosition, candidate: _ScreeningCandidate) -> bool:
    variant = candidate["variant"]
    wins = candidate["screening_wins"]
    expected_variant = max(range(4), key=lambda option: (wins[option], -option))
    return (
        position["world"] != candidate["world"]
        or variant not in (1, 2, 3)
        or variant != expected_variant
        or wins[variant] <= wins[0]
    )


def _confirmation_tasks(
    position: _ScreeningPosition,
    variant: int,
    repetitions: Sequence[int],
    branch_spec: Callable[[_ScreeningPosition, int, int, str], dict[str, Any]],
) -> list[_ConfirmationTask]:
    screening_seeds = {branch_spec(position, 0, repetition, "")["seed"] for repetition in range(4)}
    confirmation_seeds = [branch_spec(position, 0, repetition, "")["seed"] for repetition in repetitions]
    if len(set(confirmation_seeds)) != 8 or screening_seeds.intersection(confirmation_seeds):
        raise ValueError("Confirmation damage seeds are not new and distinct")
    tasks: list[_ConfirmationTask] = []
    for turn_variant in (0, variant):
        for repetition in repetitions:
            branch = branch_spec(position, turn_variant, repetition, "")
            tasks.append(
                {
                    "position_key": position["key"],
                    "variant": turn_variant,
                    "repetition": repetition,
                    "damage_seed": branch["seed"],
                    "sampling_seed": branch["sampling_seed"],
                }
            )
    return tasks


def _build_confirmation_manifest(cfg: dict[str, Any], source: Path) -> _ConfirmationManifest:
    # The whole-turn runner calls confirmation and owns the screening RNG scheme.
    from civ5rl.experiments.turn_comparison import NAMESPACE, branch_spec

    spec = cfg["confirmation"]
    original = read_json(source / "positions.json")
    if original["namespace"] != NAMESPACE:
        raise ValueError("Screening RNG namespace differs")
    if original["parent_sha256"] != cfg["parent_sha256"]:
        raise ValueError("Confirmation parent differs from screening parent")
    candidates: list[_ScreeningCandidate] = read_json(source / spec["candidates"])["rows"]
    positions_by_key: dict[tuple[int, ...], _ScreeningPosition] = {
        tuple(position["key"]): position for position in original["positions"]
    }
    if len(candidates) != cfg["positions"] or len(
        {tuple(candidate["position_key"]) for candidate in candidates}
    ) != len(candidates):
        raise ValueError("Candidate count or uniqueness differs")
    positions: list[_SelectedPosition] = []
    tasks: list[_ConfirmationTask] = []
    for candidate in candidates:
        position = positions_by_key[tuple(candidate["position_key"])]
        variant = candidate["variant"]
        if _screening_choice_differs(position, candidate):
            raise ValueError("Fixed candidate is inconsistent with screening selection")
        positions.append({**position, "selected_variant": variant})
        tasks.extend(_confirmation_tasks(position, variant, spec["repetitions"], branch_spec))
    return {
        "schema": 1,
        "namespace": original["namespace"],
        "positions": positions,
        "parent_sha256": cfg["parent_sha256"],
        "tasks": tasks,
        "repetitions": spec["repetitions"],
        "interpretation": INTERPRETATION,
    }


def prepare_confirmation(cfg: dict[str, Any], out: Path) -> None:
    """Freeze candidates and task identities before any confirmation outcomes."""
    spec = cfg["confirmation"]
    source = ROOT / spec["run"]
    _verify_confirmation_inputs(cfg, source)
    manifest = _build_confirmation_manifest(cfg, source)
    lineage = {"confirmation": spec, "parent_sha256": cfg["parent_sha256"]}
    if (out / "passport.json").exists():
        checkpoint.verify_execution_passport(out, cfg, lineage)
    else:
        identity = checkpoint.passport(out, cfg, lineage)
        write_json(out / "run.json", {"kind": cfg["kind"], "config": cfg, "identity": identity})
    if (out / "positions.json").exists():
        if read_json(out / "positions.json") != manifest:
            raise ValueError("Frozen confirmation manifest differs")
    else:
        write_json(out / "positions.json", manifest)
    if not (out / "sources.storage").exists():
        shutil.copytree(source / "sources.storage", out / "sources.storage")
    write_json(
        out / "selection.json",
        {
            "selected": len(manifest["positions"]),
            "target_continuations": len(manifest["tasks"]),
            "sha256": file_sha256(out / "positions.json"),
            "world_counts": dict(Counter(position["world"] for position in manifest["positions"])),
        },
    )
    write_json(
        out / "analysis-plan.json",
        {
            "interpretation": INTERPRETATION,
            "bootstrap": spec["bootstrap"],
            "repetitions": spec["repetitions"],
            "candidate_sha256": spec["sha256"][spec["candidates"]],
        },
    )
    if not (out / "collection.json").exists():
        write_json(out / "status.json", {"status": "prepared", "phase": "collection"})


def paired_statistics(
    worlds: ArrayLike,
    differences: ArrayLike,
    *,
    samples: int = 10000,
    seed: int = 20261103,
) -> _PairedStatistics:
    """Resample whole worlds while retaining equal weights for positions.

    A world drawn twice contributes every one of its positions twice. Dividing
    by the resulting position count avoids silently changing the estimand to
    an equal-world mean when cluster sizes differ.
    """
    worlds = np.asarray(worlds)
    differences = np.asarray(differences, dtype=np.float64)
    unique_worlds = np.unique(worlds)
    sums = np.array([differences[worlds == world].sum() for world in unique_worlds])
    counts = np.array([(worlds == world).sum() for world in unique_worlds])
    draws = np.random.default_rng(seed).integers(len(unique_worlds), size=(samples, len(unique_worlds)))
    values = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(values, [0.025, 0.975]).tolist()
    return {
        "mean_win_delta": float(differences.mean()),
        "ci95": [low, high],
        "worlds": len(unique_worlds),
        "samples": samples,
        "seed": seed,
        "conclusion": "positive" if low > 0 else "negative" if high < 0 else "uncertain",
    }


def _index_continuations(
    manifest: _ConfirmationManifest, rows: Sequence[_ContinuationRow]
) -> dict[_ContinuationKey, _ContinuationRow]:
    indexed = {(tuple(row["position_key"]), row["turn_variant"], row["repetition"]): row for row in rows}
    if len(indexed) != len(manifest["tasks"]) or len(rows) != len(indexed):
        raise ValueError("Confirmation is incomplete or contains duplicate results")
    return indexed


def _summarize_position(
    position: _SelectedPosition,
    repetitions: Sequence[int],
    indexed: Mapping[_ContinuationKey, _ContinuationRow],
    totals: dict[str, Counter[Outcome]],
    compact: list[dict[str, Any]],
) -> _PositionResult:
    wins: Counter[str] = Counter()
    identical_boundaries = 0
    position_key = tuple(position["key"])
    for repetition in repetitions:
        pair = [indexed[(position_key, variant, repetition)] for variant in (0, position["selected_variant"])]
        identical_boundaries += pair[0]["boundary_state_sha256"] == pair[1]["boundary_state_sha256"]
        for role, row in zip(_ROLES, pair):
            outcome = classify_outcome(row["winner"], row["side"])
            wins[role] += outcome == "W"
            totals[role][outcome] += 1
            compact.append({**row, "world": position["world"], "role": role, "outcome": outcome})
    return {
        "position_key": position["key"],
        "world": position["world"],
        "selected_variant": position["selected_variant"],
        "wins": dict(wins),
        "paired_win_delta": (wins["selected"] - wins["argmax"]) / 8,
        "identical_boundary_states": identical_boundaries,
    }


def summarize_confirmation(
    cfg: dict[str, Any],
    out: Path,
    manifest: _ConfirmationManifest,
    rows: Sequence[_ContinuationRow],
    seconds: float,
) -> dict[str, Any]:
    indexed = _index_continuations(manifest, rows)
    positions: list[_PositionResult] = []
    compact: list[dict[str, Any]] = []
    totals: dict[str, Counter[Outcome]] = {role: Counter() for role in _ROLES}
    for position in manifest["positions"]:
        positions.append(_summarize_position(position, manifest["repetitions"], indexed, totals, compact))
    stats = paired_statistics(
        [position["world"] for position in positions],
        [position["paired_win_delta"] for position in positions],
        **cfg["confirmation"]["bootstrap"],
    )
    result = {
        "status": "complete",
        "positions": len(positions),
        "continuations": len(rows),
        "seconds": seconds,
        "variants": {
            role: {"W": outcomes["W"], "D": outcomes["D"], "L": outcomes["L"]}
            for role, outcomes in totals.items()
        },
        "primary": stats,
        "position_signs": {
            "positive": sum(position["paired_win_delta"] > 0 for position in positions),
            "zero": sum(position["paired_win_delta"] == 0 for position in positions),
            "negative": sum(position["paired_win_delta"] < 0 for position in positions),
        },
        "position_resolution_percentage_points": 12.5,
        "interpretation": INTERPRETATION,
    }
    write_json(out / "continuations-summary.json", compact)
    write_json(out / "position-results.json", positions)
    write_json(out / "result.json", result)
    return result
