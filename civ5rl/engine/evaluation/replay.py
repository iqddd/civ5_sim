"""Verify saved simulator commands, public histories and outcomes."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from civ5rl.core.observation import ordered_actions
from civ5rl.provenance.common import ROOT, read_gzip_json, read_json
from civ5sim import Battle


def apply_replay_step(battle, step, *, mismatch):
    """Advance one recorded command and verify its result and state hash."""
    if asdict(battle.step(step["action"])) != step["result"] or battle.state_digest() != step["state_sha256"]:
        raise ValueError(mismatch)


def verify_replay(directory, *, demonstration=False, project_relative=True):
    """Validate a saved match and optional public demonstrations against the kernel."""
    directory = Path(directory)
    replay = read_json(directory / "replay.json")
    if replay.get("schema") != 2 or replay.get("kind") != "civ5sim.replay":
        raise ValueError("Expected a civ5sim.replay with schema 2")
    if not isinstance(replay.get("steps"), list):
        raise TypeError("Replay is missing its command list")
    ep = read_gzip_json(directory / "demonstration.json.gz") if demonstration else None
    case = read_json(directory / "case.json") if demonstration else None
    decision = 0
    available = selected = omitted = 0
    metrics = read_json(directory / "metrics.json") if (directory / "metrics.json").exists() else None
    with Battle(replay["initial"]) as battle:
        for step in replay["steps"]:
            side = battle._info().side if metrics or ep else None
            if metrics and side == metrics.get("side"):
                has_attack = any(a.kind in ("melee", "ranged") for a in battle.available_actions())
                is_attack = step["action"]["kind"] in ("melee", "ranged")
                available += has_attack
                selected += is_attack
                omitted += has_attack and not is_attack
            if ep and side == case["learner_side"]:
                row = ep["decisions"][decision]
                if row["observation"] != battle.observe(case["learner_side"]):
                    raise ValueError("Saved learner observation differs from replay")
                if row["actions"] != [asdict(a) for a in ordered_actions(battle.available_actions())]:
                    raise ValueError("Saved learner legal list differs from replay")
                if row["actions"][row["executed_action"]] != step["action"]:
                    raise ValueError("Replay executed the teacher command instead of the policy command")
                decision += 1
            apply_replay_step(battle, step, mismatch="Replay result/state hash differs")
        if battle.snapshot() != read_json(directory / "final.json") or (
            ep and decision != len(ep["decisions"])
        ):
            raise ValueError("Replay final state/history differs")
    resolved = directory.resolve()
    if project_relative:
        resolved = resolved.relative_to(ROOT.resolve())
    return {
        "directory": resolved.as_posix(),
        "commands": len(replay["steps"]),
        "learner_decisions": decision if ep else None,
        "verified": True,
        "attack_available_decisions": available,
        "attacks_chosen": selected,
        "non_attack_when_available": omitted,
    }
