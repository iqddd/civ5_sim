"""Checkpoint candidate evaluation ranking, tie-breaking, and gating for target BC."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from civ5rl.protocols.imitation_target_scale import MORE_DATA_KIND
from civ5rl.provenance.common import read_json, write_json


def select_best_candidate(
    bc_baseline: dict[str, Any],
    ppo_reference: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Select candidate by max development wins with protocol tie-breaking.

    Tie-breaking rules:
    1. Prefer existing baseline over a new BC candidate on equal wins.
    2. Between the two baselines, prefer PPO update 37 on equal wins.
    3. Among new BC checkpoints, prefer the earlier checkpoint (by Adam_steps, then epoch).
    """
    bc_wins = int(bc_baseline["runs"]["argmax"]["cp"]["summary"]["W"])
    ppo_wins = int(ppo_reference["runs"]["argmax"]["cp"]["summary"]["W"])
    if ppo_wins >= bc_wins:
        best_baseline = {**ppo_reference, "candidate_role": "evaluation_reference"}
        best_baseline_wins = ppo_wins
    else:
        best_baseline = {**bc_baseline, "candidate_role": "training_parent"}
        best_baseline_wins = bc_wins
    ordered_candidates = sorted(
        candidates,
        key=lambda c: (int(c.get("Adam_steps", 0)), int(c.get("epoch", 0))),
    )
    best_new = None
    best_new_wins = -1
    for candidate in ordered_candidates:
        wins = int(candidate["runs"]["argmax"]["cp"]["summary"]["W"])
        if wins > best_new_wins:
            best_new = {**candidate, "candidate_role": "new_bc"}
            best_new_wins = wins
    if best_new is not None and best_new_wins > best_baseline_wins:
        selected = best_new
        selected_wins = best_new_wins
        baseline_retained = False
    else:
        selected = best_baseline
        selected_wins = best_baseline_wins
        baseline_retained = True
    return {
        "training_parent": bc_baseline,
        "evaluation_reference": ppo_reference,
        "baseline": best_baseline,
        "baseline_wins": best_baseline_wins,
        "required_new_candidate_wins": best_baseline_wins + 1,
        "best_new_candidate": best_new,
        "best_new_wins": best_new_wins if best_new is not None else None,
        "best": selected,
        "best_wins": selected_wins,
        "baseline_retained": baseline_retained,
    }


def check_c2_gate(
    vs_bc1m: dict[str, Any],
    min_win_rate_delta: float = 0.05,
    min_ci95_low: float = 0.0,
) -> bool:
    """Verify whether C2 candidate passes development-512 win-rate and bootstrap CI gate."""
    win_rate_delta = float(vs_bc1m["delta"])
    ci_low = float(vs_bc1m["CI95"][0])
    return bool(win_rate_delta >= min_win_rate_delta and ci_low > min_ci95_low)


def c2_gate_criteria(
    vs_bc1m: dict[str, Any],
    min_win_rate_delta: float = 0.05,
    min_ci95_low: float = 0.0,
) -> dict[str, Any]:
    """Build the gate_criteria dictionary for C2 development-512 selection record."""
    win_rate_delta = float(vs_bc1m["delta"])
    ci_low = float(vs_bc1m["CI95"][0])
    gate_passed = bool(win_rate_delta >= min_win_rate_delta and ci_low > min_ci95_low)
    return {
        "min_win_rate_delta": min_win_rate_delta,
        "min_ci95_low": min_ci95_low,
        "observed_win_rate_delta": win_rate_delta,
        "observed_ci95_low": ci_low,
        "gate_passed": gate_passed,
    }


def select_best_c2_candidate(
    candidates: list[dict[str, Any]],
    bc1m_baseline: dict[str, Any] | None = None,
    ppo37_reference: dict[str, Any] | None = None,
    bc10_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Select best C2 candidate by max development-512 wins with tie-breaking.

    Tie-breaking rule:
    Among candidate checkpoints with equal wins, prefer earlier (Adam_steps, epoch).
    """
    if not candidates:
        raise ValueError("No C2 milestone checkpoints found to evaluate")
    ordered = sorted(
        candidates,
        key=lambda c: (int(c.get("Adam_steps", 0)), int(c.get("epoch", 0))),
    )
    best_c2 = None
    best_c2_wins = -1
    for cand in ordered:
        w = int(cand["runs"]["argmax"]["cp"]["summary"]["W"])
        if w > best_c2_wins:
            best_c2 = cand
            best_c2_wins = w
    if best_c2 is None:
        raise ValueError("No valid C2 candidates")
    if bc1m_baseline is not None or ppo37_reference is not None or bc10_reference is not None:
        bc1m_wins = (
            int(bc1m_baseline["runs"]["argmax"]["cp"]["summary"]["W"])
            if bc1m_baseline
            else None
        )
        ppo37_wins = (
            int(ppo37_reference["runs"]["argmax"]["cp"]["summary"]["W"])
            if ppo37_reference
            else None
        )
        bc10_wins = (
            int(bc10_reference["runs"]["argmax"]["cp"]["summary"]["W"])
            if bc10_reference
            else None
        )
        return {
            "purpose": "development_512",
            "best": best_c2,
            "best_wins": best_c2_wins,
            "c2_candidates": ordered,
            "bc_more_data_baseline": bc1m_baseline,
            "bc_more_data_wins": bc1m_wins,
            "ppo37_reference": ppo37_reference,
            "ppo37_wins": ppo37_wins,
            "bc10_reference": bc10_reference,
            "bc10_wins": bc10_wins,
        }
    return best_c2


def _refresh_selection(config: dict[str, Any], out: Path) -> dict[str, Any]:
    """Refresh selection.json from current evaluations.json."""
    evaluations = read_json(out / "evaluations.json")
    selection = read_json(out / "selection.json")
    if config["kind"] == MORE_DATA_KIND:
        bc_hash = config["parent_policy_hash"]
        ppo_hash = config["evaluation_reference_policy_hash"]
        bc_point = evaluations["development:" + bc_hash]
        ppo_point = evaluations["development:" + ppo_hash]
        candidates = [
            v
            for k, v in evaluations.items()
            if k.startswith("development:")
            and v.get("purpose") == "development"
            and v.get("policy_hash") not in (bc_hash, ppo_hash)
            and int(v.get("Adam_steps", 0)) > 0
        ]
        updated = select_best_candidate(bc_point, ppo_point, candidates)
        write_json(out / "selection.json", updated)
        return updated
    for point in sorted(
        (
            v
            for k, v in evaluations.items()
            if k.startswith("development:") and v.get("purpose") == "development"
        ),
        key=lambda p: int(p.get("Adam_steps", 0)),
    ):
        wins = point["runs"]["argmax"]["cp"]["summary"]["W"]
        if wins > selection["best_wins"]:
            selection.update(best=point, best_wins=wins)
    write_json(out / "selection.json", selection)
    return selection


refresh_selection = _refresh_selection

__all__ = [
    "_refresh_selection",
    "c2_gate_criteria",
    "check_c2_gate",
    "refresh_selection",
    "select_best_c2_candidate",
    "select_best_candidate",
]
