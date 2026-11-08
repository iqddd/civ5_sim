"""Validation execution and replay verification; saved-result analysis is re-exported."""

from dataclasses import asdict

from civ5rl.protocols.imitation import load_episode
from civ5rl.provenance.common import ROOT, read_json
from civ5rl.reporting.evaluation_audit import (
    clustered_paired_bootstrap,
    compare_paired_directories,
    load_eval_rows,
    outcome,
    paired_world_bootstrap,
)
from civ5sim import Battle
from civ5sim.engine import digest

__all__ = [
    "clustered_paired_bootstrap",
    "compare_paired_directories",
    "load_eval_rows",
    "outcome",
    "paired_world_bootstrap",
    "run_validation_cases",
    "verify_replay",
]


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
