"""Saved-result analysis only: no match execution, policy loading or training."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from civ5rl.protocols.common import bootstrap_ci95, learner_outcome
from civ5rl.provenance.common import ROOT, read_json
from civ5sim.engine import digest


def paired_world_bootstrap(before, after, *, samples, seed, clusters=16, cases=None, cases_per_world=4):
    """Bootstrap matched outcomes by parent world, preserving case order."""
    if before.keys() != after.keys():
        raise ValueError("Unpaired evaluation cases")
    worlds = sorted({key[0] for key in before})
    if len(worlds) != clusters:
        raise ValueError("Wrong bootstrap cluster count")
    if any(len([key for key in before if key[0] == world]) != cases_per_world for world in worlds):
        raise ValueError("Wrong paired case count per parent world")
    if cases is not None:
        if cases.keys() != before.keys():
            raise ValueError("Missing paired case metadata")
        for world in worlds:
            combinations = [(cases[key]["placement"], key[2]) for key in before if key[0] == world]
            if set(combinations) != {
                (placement, side) for placement in ("contact", "separated") for side in (0, 1)
            }:
                raise ValueError("Expected both placements and sides per parent world")
            if any(
                combinations.count((placement, side)) != cases_per_world // 4
                for placement in ("contact", "separated")
                for side in (0, 1)
            ):
                raise ValueError("Unbalanced placements/sides per parent world")
    differences = np.asarray(
        [np.mean([after[key] - before[key] for key in sorted(before) if key[0] == world]) for world in worlds]
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
    return learner_outcome(metric["winner"], metric["side"])


def _extract_plan_keys(plan):
    """Extract a set of (parent_world, case_id, learner_side) tuples from a plan."""
    if isinstance(plan, dict):
        if "cases" in plan:
            source = plan["cases"]
        elif "validation_cases" in plan:
            source = plan["validation_cases"]
        else:
            source = plan
    else:
        source = plan

    if isinstance(source, dict):
        first_key = next(iter(source.keys()), None)
        if first_key is not None and isinstance(first_key, (tuple, list)) and len(first_key) == 3:
            return {tuple(k) for k in source}
        items = list(source.values())
    elif isinstance(source, (list, tuple, set)):
        items = list(source)
    else:
        raise TypeError(f"Unsupported plan type: {type(plan)}")

    keys = set()
    for item in items:
        if isinstance(item, (tuple, list)) and len(item) == 3:
            keys.add(tuple(item))
        elif isinstance(item, dict):
            pw = item.get("parent_world")
            cid = item.get("case_id")
            if pw is None or cid is None:
                raise ValueError(f"Plan case missing parent_world or case_id: {item}")
            if "learner_side" in item:
                keys.add((pw, cid, item["learner_side"]))
            elif "side" in item:
                keys.add((pw, cid, item["side"]))
            else:
                keys.add((pw, cid, 0))
                keys.add((pw, cid, 1))
        else:
            raise ValueError(f"Unrecognized plan case item: {item}")
    return keys


def load_eval_rows(directory, expected_n=None, *, plan=None, require_artifacts=True):
    """Load complete saved matches, checking identity, plan, and cross-file consistency."""
    directory = Path(directory)
    if not directory.is_absolute():
        directory = ROOT / directory
    rows = {}
    for match in sorted(p for p in directory.iterdir() if p.is_dir()):
        if match.name.startswith((".", "__")):
            continue
        required = (
            ("case.json", "metrics.json", "result.json", "final.json", "replay.json")
            if require_artifacts
            else ("case.json", "metrics.json")
        )
        if any(not (match / name).is_file() for name in required):
            raise ValueError(f"Missing evaluation artifact: {match}")
        case = read_json(match / "case.json")
        metric = read_json(match / "metrics.json")
        result = read_json(match / "result.json") if (match / "result.json").is_file() else None
        if require_artifacts:
            for name in ("final.json", "replay.json"):
                read_json(match / name)
        elif (match / "result.json").is_file():
            for name in ("final.json", "replay.json"):
                if (match / name).is_file():
                    read_json(match / name)

        if metric.get("status") != "completed":
            raise ValueError(f"Incomplete evaluation: {match}")
        if result is not None and result.get("status") != "completed":
            raise ValueError(f"Incomplete evaluation: {match}")

        side = metric["side"]
        learner_side = case["learner_side"]
        outcome(metric)
        if side != learner_side:
            raise ValueError(f"Inconsistent learner side: metric side {side} != case side {learner_side}")

        winner = metric["winner"]
        if (
            (result is not None and winner != result["winner"])
            or metric["case_id"] != case["case_id"]
            or metric["parent_world"] != case["parent_world"]
            or metric["case_hash"] != case["case_hash"]
            or case["case_hash"] != digest({k: v for k, v in case.items() if k != "case_hash"})
        ):
            raise ValueError(f"Inconsistent evaluation identity/result: {match}")

        parent_world = case["parent_world"]
        case_id = case["case_id"]
        key = (parent_world, case_id, learner_side)
        if key in rows:
            raise ValueError(
                f"Duplicate evaluation match: parent_world={parent_world}, "
                f"case_id={case_id}, learner_side={learner_side}"
            )
        rows[key] = (metric, case, match)

    if plan is not None:
        expected_keys = _extract_plan_keys(plan)
        actual_keys = set(rows.keys())
        if actual_keys != expected_keys:
            missing = expected_keys - actual_keys
            unexpected = actual_keys - expected_keys
            raise ValueError(
                f"Evaluation results do not match evaluation plan: missing={missing}, unexpected={unexpected}"
            )

    if expected_n is not None and len(rows) != expected_n:
        raise ValueError(f"Expected {expected_n} matches, found {len(rows)}")
    return rows


def _slice_keys(rows):
    """Group rows by placement and side; each row requires a known placement."""
    if any(row[1].get("placement") not in ("contact", "separated") for row in rows.values()):
        raise ValueError("Evaluation slices require contact or separated placement")
    return {
        "contact": [key for key, row in rows.items() if row[1].get("placement") == "contact"],
        "separated": [key for key, row in rows.items() if row[1].get("placement") == "separated"],
        "side0": [key for key in rows if key[2] == 0],
        "side1": [key for key in rows if key[2] == 1],
    }


def clustered_paired_bootstrap(before, after, *, clusters=16, samples=10000, seed=20261103):
    """Compare matched saved rows, resampling whole parent worlds."""
    if before.keys() != after.keys() or any(before[k][1] != after[k][1] for k in before):
        raise ValueError("Unpaired validation cases or metadata")
    if not before or len(before) % clusters:
        raise ValueError("Unequal cases per parent world")
    slice_keys = _slice_keys(after)
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
        str(world): float(np.mean([win(after[k]) - win(before[k]) for k in before if k[0] == world]))
        for world in sorted({k[0] for k in before})
    }
    report["slices"] = {}
    for name, keys in slice_keys.items():
        outcomes = Counter(outcome(after[k][0]) for k in keys)
        report["slices"][name] = {
            "matches": len(keys),
            "baseline_wins": sum(win(before[k]) for k in keys),
            "candidate_wins": sum(win(after[k]) for k in keys),
            "outcomes": {value: outcomes[value] for value in "WDL"},
        }
    return report


def compare_paired_directories(before_path, after_path, *, samples, seed, require_artifacts=False):
    """Read paired evaluations and return the historical tuple interface."""
    before = load_eval_rows(before_path, 64, require_artifacts=require_artifacts)
    after = load_eval_rows(after_path, 64, require_artifacts=require_artifacts)
    report = clustered_paired_bootstrap(before, after, samples=samples, seed=seed)
    slices = report.pop("slices")
    return report, slices, before, after


def _summary_rows(rows_or_directory, expected_n, plan, require_artifacts):
    if isinstance(rows_or_directory, (str, Path)):
        rows = load_eval_rows(
            rows_or_directory,
            expected_n=expected_n,
            plan=plan,
            require_artifacts=require_artifacts,
        )
    else:
        rows = rows_or_directory
        if expected_n is not None and len(rows) != expected_n:
            raise ValueError(f"Expected {expected_n} matches, found {len(rows)}")
    return rows


def eval_wdl_summary(rows_or_directory, expected_n=None, *, plan=None, require_artifacts=True):
    """Compute W/D/L counts and win rate from loaded rows or an evaluation directory."""
    rows = _summary_rows(rows_or_directory, expected_n, plan, require_artifacts)
    counts = Counter(outcome(v[0]) for v in rows.values())
    n = len(rows)
    return {
        "matches": n,
        "W": counts["W"],
        "D": counts["D"],
        "L": counts["L"],
        "win_rate": (counts["W"] / n) if n > 0 else 0.0,
    }


wdl_summary = eval_wdl_summary


def eval_slices_summary(rows_or_directory, expected_n=None, *, plan=None, require_artifacts=True):
    """Compute placement and side slices from loaded rows or an evaluation directory."""
    rows = _summary_rows(rows_or_directory, expected_n, plan, require_artifacts)
    slices = {}
    for name, keys in _slice_keys(rows).items():
        outcomes = Counter(outcome(rows[k][0]) for k in keys)
        slices[name] = {
            "matches": len(keys),
            "wins": sum(float(outcome(rows[k][0]) == "W") for k in keys),
            "outcomes": {value: outcomes[value] for value in "WDL"},
        }
    return slices


def h2h_world_bootstrap(
    rows_or_directory,
    expected_matches,
    *,
    samples=10000,
    seed=20261103,
    clusters=16,
    require_artifacts=True,
):
    """Clustered world bootstrap over per-game signed outcome (+1 W, 0 D, -1 L)."""
    rows = _summary_rows(rows_or_directory, expected_matches, None, require_artifacts)
    pairs = defaultdict(set)
    by_world = defaultdict(list)
    counts = Counter()
    for (parent_world, case_id, side), (metric, _case, _path) in sorted(rows.items()):
        pairs[(int(parent_world), int(case_id))].add(int(side))
        result = outcome(metric)
        counts[result] += 1
        by_world[int(parent_world)].append(1 if result == "W" else -1 if result == "L" else 0)
    total = counts["W"] + counts["D"] + counts["L"]
    if total != expected_matches:
        raise ValueError(f"Expected {expected_matches} H2H matches, got {total}")
    if any(sides != {0, 1} for sides in pairs.values()):
        raise ValueError("H2H evaluation requires complete mirrored pairs (side 0 and side 1)")
    worlds = sorted(by_world)
    if clusters is not None and len(worlds) != clusters:
        raise ValueError(f"Wrong H2H bootstrap cluster count: expected {clusters}, got {len(worlds)}")
    if len({len(by_world[w]) for w in worlds}) != 1:
        raise ValueError("Unequal H2H cases per parent world")
    world_means = {w: float(np.mean(by_world[w])) for w in worlds}
    world_values = np.asarray([world_means[w] for w in worlds], dtype=np.float64)
    ci = bootstrap_ci95(world_values, samples=samples, seed=seed)
    delta_wins = int(counts["W"] - counts["L"])
    return {
        "matches": total,
        "W": int(counts["W"]),
        "D": int(counts["D"]),
        "L": int(counts["L"]),
        "delta_wins": delta_wins,
        "delta": delta_wins / total,
        "win_rate": counts["W"] / total,
        "draw_rate": counts["D"] / total,
        "loss_rate": counts["L"] / total,
        "CI95": [float(v) for v in ci],
        "bootstrap_clusters": len(worlds),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "world_deltas": {str(w): world_means[w] for w in worlds},
    }


def select_dual_milestone(candidates, baseline_cp_wins):
    """Select best milestone under dual non-regression + max J = delta_W_cp + delta_W_h2h.

    Admissible iff delta_w_cp >= 0, delta_w_h2h >= 0, and J > 0.
    Tie-break: higher delta_w_cp, then earlier update. Fallback: checkpoints/initial.pt.
    """
    admissible = []
    baseline_entry = None
    for row in candidates:
        delta_w_cp = int(row["cp_wins"]) - int(baseline_cp_wins)
        delta_w_h2h = int(row["h2h_wins"]) - int(row["h2h_losses"])
        j_score = delta_w_cp + delta_w_h2h
        enriched = {
            **row,
            "delta_w_cp": delta_w_cp,
            "delta_w_h2h": delta_w_h2h,
            "J": j_score,
            "admissible": delta_w_cp >= 0 and delta_w_h2h >= 0 and j_score > 0,
        }
        if row["path"] == "checkpoints/initial.pt":
            baseline_entry = enriched
        if enriched["admissible"]:
            admissible.append(enriched)
    if not admissible:
        if baseline_entry is not None:
            return {
                "path": baseline_entry["path"],
                "wins": baseline_entry["cp_wins"],
                "update": baseline_entry["update"],
                "decisions": baseline_entry["decisions"],
                "policy_hash": baseline_entry["policy_hash"],
                "delta_w_cp": 0,
                "delta_w_h2h": baseline_entry["delta_w_h2h"],
                "J": baseline_entry["delta_w_h2h"],
                "selection_rule": "dual_nonregression_max_J",
                "cp_summary": baseline_entry["cp_summary"],
                "h2h_summary": baseline_entry["h2h_summary"],
            }
        raise ValueError("Missing baseline candidate for dual selection")
    best = max(admissible, key=lambda r: (r["J"], r["delta_w_cp"], -int(r["update"])))
    return {
        "path": best["path"],
        "wins": best["cp_wins"],
        "update": best["update"],
        "decisions": best["decisions"],
        "policy_hash": best["policy_hash"],
        "delta_w_cp": best["delta_w_cp"],
        "delta_w_h2h": best["delta_w_h2h"],
        "J": best["J"],
        "selection_rule": "dual_nonregression_max_J",
        "cp_summary": best["cp_summary"],
        "h2h_summary": best["h2h_summary"],
    }


def dual_promotion_decision(cp_comparison, h2h_bootstrap):
    """Promote candidate iff both deltas >= 0 and at least one CI95 lower bound > 0."""
    cp_delta = float(cp_comparison["delta"])
    cp_low = float(cp_comparison["CI95"][0])
    h2h_delta = float(h2h_bootstrap["delta"])
    h2h_low = float(h2h_bootstrap["CI95"][0])
    both_nonnegative = cp_delta >= 0.0 and h2h_delta >= 0.0
    significant = cp_low > 0.0 or h2h_low > 0.0
    return {
        "promoted": bool(both_nonnegative and significant),
        "both_nonnegative": bool(both_nonnegative),
        "significant_ci95": bool(significant),
        "cp_delta": cp_delta,
        "cp_ci95": list(cp_comparison["CI95"]),
        "h2h_delta": h2h_delta,
        "h2h_ci95": list(h2h_bootstrap["CI95"]),
    }
