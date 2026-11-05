"""Shared paired battle analysis."""

from collections import Counter
from dataclasses import asdict
from pathlib import Path

import numpy as np

from civ5rl.protocols.common import bootstrap_ci95
from civ5rl.protocols.imitation import load_episode
from civ5rl.provenance.common import ROOT, read_json
from civ5sim import Battle
from civ5sim.engine import digest


def run_validation_cases(
    config,
    manifest,
    checkpoint,
    mode,
    directories,
    workers,
    *,
    batch_size=None,
    progress=None,
):
    """Run validation cases in their recorded world, side, opponent order."""
    from civ5rl.experiments.imitation_helpers import pool
    from civ5rl.protocols.imitation import match_task

    jobs = []
    for original in manifest["validation_cases"]:
        for side in (0, 1):
            case = {**original, "learner_side": side}
            case["case_hash"] = digest(
                {k: v for k, v in case.items() if k != "case_hash"}
            )
            for enemy, directory in directories.items():
                jobs.append(
                    (
                        config,
                        manifest,
                        case,
                        str(directory / f"case-{case['case_id']:03}-side-{side}"),
                        {"checkpoint": str(checkpoint), "mode": mode},
                        enemy,
                        "cpu",
                    )
                )
    with pool(workers) as executor:
        if batch_size is None:
            list(executor.map(match_task, jobs))
        else:
            for first in range(0, len(jobs), batch_size):
                part = jobs[first : first + batch_size]
                list(executor.map(match_task, part))
                if progress is not None:
                    progress(first + len(part))
    return len(jobs)


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


def compare_paired_directories(
    before_path, after_path, *, samples, seed, require_artifacts=False
):
    """Read and validate paired cases, then compute world bootstrap and slices."""

    def read_rows(path):
        result = {}
        for metrics_path in sorted((ROOT / Path(path)).glob("*/metrics.json")):
            metric = read_json(metrics_path)
            case = read_json(metrics_path.parent / "case.json")
            if metric["status"] != "completed":
                raise ValueError("Technical error in battle evaluation")
            if require_artifacts and any(
                not (metrics_path.parent / filename).exists()
                for filename in ("replay.json", "final.json", "result.json")
            ):
                raise ValueError("Missing evaluation artifact")
            key = (case["parent_world"], case["case_id"], metric["side"])
            if key in result:
                raise ValueError("Duplicate evaluation case")
            result[key] = (metric, case, metrics_path.parent)
        if len(result) != 64:
            raise ValueError("Expected 64 evaluation cases")
        return result

    before, after = read_rows(before_path), read_rows(after_path)
    if before.keys() != after.keys() or any(
        before[k][1] != after[k][1] for k in before
    ):
        raise ValueError("Unpaired validation cases")
    win = lambda row: float(row[0]["winner"] == row[0]["side"])
    bootstrap = paired_world_bootstrap(
        {k: win(v) for k, v in before.items()},
        {k: win(v) for k, v in after.items()},
        samples=samples,
        seed=seed,
        cases={k: v[1] for k, v in after.items()},
    )
    slices = {}
    for name, predicate in (
        ("contact", lambda k: after[k][1]["placement"] == "contact"),
        ("separated", lambda k: after[k][1]["placement"] == "separated"),
        ("side0", lambda k: k[2] == 0),
        ("side1", lambda k: k[2] == 1),
    ):
        keys = [k for k in after if predicate(k)]
        outcomes = Counter(
            "D" if after[k][0]["winner"] is None else "W" if win(after[k]) else "L"
            for k in keys
        )
        slices[name] = {
            "matches": len(keys),
            "baseline_wins": sum(win(before[k]) for k in keys),
            "candidate_wins": sum(win(after[k]) for k in keys),
            "outcomes": {outcome: outcomes[outcome] for outcome in ("W", "D", "L")},
        }
    return bootstrap, slices, before, after


def verify_replay(directory, *, demonstration=False):
    replay = read_json(directory / "replay.json")
    ep = load_episode(directory / "demonstration.json.gz") if demonstration else None
    case = read_json(directory / "case.json")
    from civ5rl.core.observation import ordered_actions

    decision = 0
    available = selected = omitted = 0
    metrics = (
        read_json(directory / "metrics.json")
        if (directory / "metrics.json").exists()
        else None
    )
    with Battle(replay["initial"]) as battle:
        for step in replay["steps"]:
            if metrics and battle.snapshot()["state"]["side"] == metrics.get("side"):
                has_attack = any(
                    a.kind in ("melee", "ranged") for a in battle.available_actions()
                )
                is_attack = step["action"]["kind"] in ("melee", "ranged")
                available += has_attack
                selected += is_attack
                omitted += has_attack and not is_attack
            if ep and battle.snapshot()["state"]["side"] == case["learner_side"]:
                row = ep["decisions"][decision]
                if row["observation"] != battle.observe(case["learner_side"]):
                    raise ValueError("Saved learner observation differs from replay")
                if row["actions"] != [
                    asdict(a) for a in ordered_actions(battle.available_actions())
                ]:
                    raise ValueError("Saved learner legal list differs from replay")
                if row["actions"][row["executed_action"]] != step["action"]:
                    raise ValueError(
                        "Replay executed the teacher command instead of the policy command"
                    )
                decision += 1
            if (
                asdict(battle.step(step["action"])) != step["result"]
                or digest(battle.snapshot()) != step["state_sha256"]
            ):
                raise ValueError("Replay result/state hash differs")
        if battle.snapshot() != read_json(directory / "final.json") or (
            ep and decision != len(ep["decisions"])
        ):
            raise ValueError("Replay final state/history differs")
    return {
        "directory": directory.resolve().relative_to(ROOT.resolve()).as_posix(),
        "commands": len(replay["steps"]),
        "learner_decisions": decision if ep else None,
        "verified": True,
        "attack_available_decisions": available,
        "attacks_chosen": selected,
        "non_attack_when_available": omitted,
    }
