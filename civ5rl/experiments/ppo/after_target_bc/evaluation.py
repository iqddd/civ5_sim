"""Evaluate Development milestones and confirm a selected policy."""

from __future__ import annotations

from time import perf_counter

from civ5rl.engine.runtime.control import target_bc_evaluation_due
from civ5rl.experiments.evaluation.h2h import evaluate_h2h
from civ5rl.experiments.evaluation.matches import evaluate_target_scale as evaluate
from civ5rl.experiments.imitation.helpers import isolated_rng
from civ5rl.experiments.ppo.after_target_bc.initialization import u0_checkpoint
from civ5rl.protocols.ppo_after_target_bc import check_development_gain
from civ5rl.provenance.common import (
    read_json,
    write_json,
)
from civ5rl.reporting.evaluation_audit import (
    collect_paired_outcomes,
    dual_promotion_decision,
    paired_world_bootstrap,
    select_dual_milestone,
)
from civ5rl.reporting.ppo_after_target_bc import report


def development(config, plan, out, state, shuffle, clock=None):
    if clock:
        clock.pause()
    started = perf_counter()
    purpose = config.get("ppo", {}).get("evaluation_purpose", "development")
    dual = bool(config.get("self_play", {}).get("enabled"))
    try:
        with isolated_rng(shuffle):
            result = evaluate(config, plan, out, state["latest"], purpose)
            h2h = None
            if dual:
                h2h = evaluate_h2h(
                    config,
                    plan,
                    out,
                    state["latest"],
                    u0_checkpoint(config),
                    purpose,
                )
        state["evaluation_seconds"] += perf_counter() - started
        entry = {
            **result,
            "decisions": state["decisions"],
            "update": state["updates"],
        }
        if h2h is not None:
            entry["h2h"] = h2h
        state["evaluated"][result["policy_hash"]] = entry
        wins = result["runs"]["argmax"]["cp"]["summary"]["W"]
        if state["baseline_wins"] is None:
            state["baseline_wins"] = wins
            if config["experiment"].get("early_gain_min_wins") is not None:
                state["early_gain_required_wins"] = wins + config["experiment"]["early_gain_min_wins"]
        if dual:
            candidates = []
            for p_hash, item in state["evaluated"].items():
                cp_sum = item["runs"]["argmax"]["cp"]["summary"]
                h2h_sum = item["h2h"]["summary"]
                candidates.append(
                    {
                        "path": item["checkpoint"],
                        "update": int(item.get("update", 0)),
                        "decisions": int(item.get("decisions", 0)),
                        "policy_hash": p_hash,
                        "cp_wins": int(cp_sum["W"]),
                        "h2h_wins": int(h2h_sum["W"]),
                        "h2h_losses": int(h2h_sum["L"]),
                        "cp_summary": cp_sum,
                        "h2h_summary": h2h_sum,
                    }
                )
            state["best"] = select_dual_milestone(candidates, state["baseline_wins"])
        elif wins > state["best"]["wins"]:
            state["best"] = {
                "path": result["checkpoint"],
                "wins": wins,
                "update": state["updates"],
                "policy_hash": result["policy_hash"],
            }
        for point in config["experiment"]["milestones"]:
            if point <= state["decisions"]:
                state["development_points"].setdefault(str(point), result["policy_hash"])
        check_development_gain(state, config)
        write_json(out / "state.json", state)
        report(config, out)
    finally:
        if clock:
            clock.resume()


def _development_gate_rejected(comparison, min_delta, min_ci_low):
    return comparison["delta"] < float(min_delta) or comparison["CI95"][0] <= min_ci_low


def finish(config, plan, out):
    if (out / "confirmation.json").exists():
        return
    state = read_json(out / "state.json")
    if target_bc_evaluation_due(config, state):
        raise ValueError("Cannot select a candidate while Development is incomplete")
    write_json(out / "selected.json", state["best"])
    if state["best"]["path"] == "checkpoints/initial.pt":
        write_json(out / "confirmation.json", {"status": "not_run_parent_selected"})
        return

    def outcomes(point):
        directory = out / point["runs"]["argmax"]["cp"]["directory"]
        results, cases, _ = collect_paired_outcomes(directory)
        return results, cases

    dev_comparison = None
    dev_purpose = config.get("ppo", {}).get("evaluation_purpose", "development")
    min_delta = config["experiment"].get("confirmation_min_win_rate_delta")
    if min_delta is not None:
        dev_baseline = evaluate(config, plan, out, "checkpoints/initial.pt", dev_purpose)
        dev_candidate = evaluate(config, plan, out, state["best"]["path"], dev_purpose)
        dev_before, dev_cases = outcomes(dev_baseline)
        dev_after, dev_paired = outcomes(dev_candidate)
        if dev_cases != dev_paired:
            raise ValueError(f"{dev_purpose} cases are not exactly paired")
        dev_comparison = paired_world_bootstrap(
            dev_before,
            dev_after,
            samples=10000,
            seed=20261103,
            cases=dev_cases,
            cases_per_world=len(dev_before) // 16,
        )
        min_ci_low = float(config["experiment"].get("confirmation_min_ci95_low", 0.0))
        if _development_gate_rejected(dev_comparison, min_delta, min_ci_low):
            write_json(
                out / "confirmation.json",
                {
                    "status": f"not_run_{dev_purpose}_gate_not_met",
                    f"{dev_purpose}_comparison": dev_comparison,
                },
            )
            return

    baseline = evaluate(config, plan, out, "checkpoints/initial.pt", "confirmation")
    candidate = evaluate(config, plan, out, state["best"]["path"], "confirmation")
    before, cases = outcomes(baseline)
    after, paired = outcomes(candidate)
    if cases != paired:
        raise ValueError("Additional cases are not exactly paired")
    comparison = paired_world_bootstrap(
        before, after, samples=10000, seed=20261103, cases=cases, cases_per_world=16
    )
    payload = {
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
    }
    if config.get("self_play", {}).get("enabled"):
        h2h_confirmation = evaluate_h2h(
            config,
            plan,
            out,
            state["best"]["path"],
            u0_checkpoint(config),
            "confirmation",
        )
        payload["h2h_confirmation"] = h2h_confirmation
        payload["promotion"] = dual_promotion_decision(comparison, h2h_confirmation["bootstrap"])
    if dev_comparison is not None:
        payload[f"{dev_purpose}_comparison"] = dev_comparison
    write_json(out / "confirmation.json", payload)
