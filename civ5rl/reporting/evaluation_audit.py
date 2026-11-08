"""Saved-result analysis only: no match execution, policy loading or training."""

from collections import Counter
from pathlib import Path

import numpy as np

from civ5rl.protocols.common import bootstrap_ci95
from civ5rl.provenance.common import ROOT, read_json
from civ5sim.engine import digest


def paired_world_bootstrap(
    before, after, *, samples, seed, clusters=16, cases=None, cases_per_world=4
):
    """Bootstrap matched outcomes by parent world, preserving case order."""
    if before.keys() != after.keys():
        raise ValueError("Unpaired evaluation cases")
    worlds = sorted({key[0] for key in before})
    if len(worlds) != clusters:
        raise ValueError("Wrong bootstrap cluster count")
    if any(
        len([key for key in before if key[0] == world]) != cases_per_world
        for world in worlds
    ):
        raise ValueError("Wrong paired case count per parent world")
    if cases is not None:
        if cases.keys() != before.keys():
            raise ValueError("Missing paired case metadata")
        for world in worlds:
            combinations = [
                (cases[key]["placement"], key[2]) for key in before if key[0] == world
            ]
            if set(combinations) != {
                (placement, side)
                for placement in ("contact", "separated")
                for side in (0, 1)
            }:
                raise ValueError("Expected both placements and sides per parent world")
            if any(
                combinations.count((placement, side)) != cases_per_world // 4
                for placement in ("contact", "separated")
                for side in (0, 1)
            ):
                raise ValueError("Unbalanced placements/sides per parent world")
    differences = np.asarray(
        [
            np.mean(
                [after[key] - before[key] for key in sorted(before) if key[0] == world]
            )
            for world in worlds
        ]
    )
    ci = bootstrap_ci95(differences, samples=samples, seed=seed)
    return {
        "delta": float(differences.mean()),
        "CI95": [float(v) for v in ci],
        "bootstrap_clusters": clusters,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
    }


def outcome(metric):
    """Return the learner's W/D/L outcome for a completed metric."""
    side, winner = metric["side"], metric["winner"]
    if side not in (0, 1) or winner not in (None, 0, 1):
        raise ValueError("Invalid side or winner")
    return "D" if winner is None else "W" if winner == side else "L"


def load_eval_rows(directory, expected_n):
    """Load complete saved matches, checking identity and cross-file consistency."""
    directory = ROOT / Path(directory)
    rows = {}
    for match in sorted(p for p in directory.iterdir() if p.is_dir()):
        required = (
            "case.json",
            "metrics.json",
            "result.json",
            "final.json",
            "replay.json",
        )
        if any(not (match / name).is_file() for name in required):
            raise ValueError(f"Missing evaluation artifact: {match}")
        case, metric, result = (read_json(match / name) for name in required[:3])
        # Parse the remaining artifacts too; a truncated file is not complete.
        for name in required[3:]:
            read_json(match / name)
        if metric["status"] != "completed" or result["status"] != "completed":
            raise ValueError(f"Incomplete evaluation: {match}")
        side, winner = metric["side"], metric["winner"]
        outcome(metric)
        if (
            side != case["learner_side"]
            or winner != result["winner"]
            or metric["case_id"] != case["case_id"]
            or metric["parent_world"] != case["parent_world"]
            or metric["case_hash"] != case["case_hash"]
            or case["case_hash"]
            != digest({k: v for k, v in case.items() if k != "case_hash"})
        ):
            raise ValueError(f"Inconsistent evaluation identity/result: {match}")
        key = (case["parent_world"], case["case_id"], side)
        if key in rows:
            raise ValueError(f"Duplicate evaluation case: {key}")
        rows[key] = (metric, case, match)
    if len(rows) != expected_n:
        raise ValueError(f"Expected {expected_n} evaluation cases, found {len(rows)}")
    return rows


def clustered_paired_bootstrap(
    before, after, *, clusters=16, samples=10000, seed=20261103
):
    """Compare matched saved rows, resampling whole parent worlds."""
    if before.keys() != after.keys() or any(
        before[k][1] != after[k][1] for k in before
    ):
        raise ValueError("Unpaired validation cases or metadata")
    if not before or len(before) % clusters:
        raise ValueError("Unequal cases per parent world")
    win = lambda row: float(outcome(row[0]) == "W")
    report = paired_world_bootstrap(
        {k: win(v) for k, v in before.items()},
        {k: win(v) for k, v in after.items()},
        samples=samples,
        seed=seed,
        clusters=clusters,
        cases={k: v[1] for k, v in after.items()},
        cases_per_world=len(before) // clusters,
    )
    transitions = {a: {b: 0 for b in "WDL"} for a in "WDL"}
    for key in before:
        transitions[outcome(before[key][0])][outcome(after[key][0])] += 1
    report["transitions"] = transitions
    report["world_deltas"] = {
        str(world): float(
            np.mean([win(after[k]) - win(before[k]) for k in before if k[0] == world])
        )
        for world in sorted({k[0] for k in before})
    }
    report["slices"] = {}
    for name, predicate in (
        ("contact", lambda k: after[k][1]["placement"] == "contact"),
        ("separated", lambda k: after[k][1]["placement"] == "separated"),
        ("side0", lambda k: k[2] == 0),
        ("side1", lambda k: k[2] == 1),
    ):
        keys = [k for k in after if predicate(k)]
        outcomes = Counter(outcome(after[k][0]) for k in keys)
        report["slices"][name] = {
            "matches": len(keys),
            "baseline_wins": sum(win(before[k]) for k in keys),
            "candidate_wins": sum(win(after[k]) for k in keys),
            "outcomes": {value: outcomes[value] for value in "WDL"},
        }
    return report


def compare_paired_directories(
    before_path, after_path, *, samples, seed, require_artifacts=False
):
    """Read paired evaluations and return the historical tuple interface."""
    before = load_eval_rows(before_path, 64)
    after = load_eval_rows(after_path, 64)
    report = clustered_paired_bootstrap(before, after, samples=samples, seed=seed)
    slices = report.pop("slices")
    return report, slices, before, after
