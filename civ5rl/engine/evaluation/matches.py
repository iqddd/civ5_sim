"""Run and save ordinary evaluation matches from prepared simulator cases."""

from __future__ import annotations

import traceback
from pathlib import Path
from time import perf_counter

from civ5rl.engine.evaluation.proxy import evaluation_agent
from civ5rl.protocols.imitation import search_config
from civ5rl.provenance.common import write_json
from civ5sim import MatchConfig
from civ5sim.cp_agent import CPAgent
from civ5sim.match import run_match


def run_saved_match(case, agents, directory, *, max_calls_per_turn, seed=0):
    """Execute a prepared case with explicit agents and save its match artifacts."""
    outcome = run_match(
        MatchConfig(
            case["scenario"]["map"],
            case["armies"],
            int(case["max_rounds"]),
            seed=seed,
            max_calls_per_turn=max_calls_per_turn,
            turn_reset_mode=case["scenario"].get("turn_reset_mode", "legacy"),
        ),
        agents,
        scenario=case["scenario"],
        agent_seeds=case["agent_seeds"],
    )
    outcome.save(directory)
    return outcome


def run_policy_match(
    config,
    case,
    checkpoint_path,
    mode,
    enemy,
    directory,
    *,
    temperature=1.0,
    collect_selection_metrics=False,
    include_combat_metrics=False,
    metric_fields=None,
):
    """Shared ordinary-match execution; protocol callers supply validated cases."""
    directory = Path(directory)
    if checkpoint_path == "cp":
        agent = CPAgent(search_config(config))
    else:
        agent = evaluation_agent(
            checkpoint_path,
            mode,
            temperature=temperature,
            collect_selection_metrics=collect_selection_metrics,
            cpu_seed=17,
        )
    if enemy == "cpv2":
        from civ5sim.cpv2_agent import CPv2Agent

        opponent = CPv2Agent()
    else:
        opponent = CPAgent(search_config(config)) if enemy == "cp" else "greedy"
    side = case["learner_side"]
    agents = [None, None]
    agents[side], agents[1 - side] = agent, opponent
    write_json(directory / "case.json", case)
    started = perf_counter()
    try:
        outcome = run_saved_match(
            case,
            agents,
            directory,
            max_calls_per_turn=config["environment"]["max_calls_per_turn"],
        )
        if outcome.result["status"] != "completed":
            raise RuntimeError(str(outcome.result["error"]))
        result = {
            "status": "completed",
            "case_id": case["case_id"],
            "side": side,
            "parent_world": case["parent_world"],
            "winner": outcome.result["winner"],
            "reason": outcome.result["reason"],
            "rounds": outcome.result["completed_rounds"],
            "seconds": perf_counter() - started,
            "remaining_hp": outcome.result["remaining_hp"],
            "case_hash": case["case_hash"],
        }
        if include_combat_metrics:
            result.update(combat_metrics(case, outcome.replay, outcome.final))
        if metric_fields is not None:
            result.update(metric_fields)
        if collect_selection_metrics:
            write_json(directory / "selection-metrics.json", agent.selection_metrics)
        # Resumed evaluations treat the presence of this file as completion.
        write_json(directory / "metrics.json", result)
        return result
    except BaseException:
        (directory / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if isinstance(agent, CPAgent):
            agent.close()
        if isinstance(opponent, CPAgent):
            opponent.close()


def combat_metrics(case, replay, final):
    """Return case strata and per-side combat totals for a completed match."""
    side = case["learner_side"]
    damage = [0, 0]
    owners = {u["id"]: u["side"] for u in replay["initial"]["units"]}
    for step in replay["steps"]:
        detail = step["result"]
        if step["action"]["kind"] in ("ranged", "melee"):
            attacker = owners[step["action"]["unit"]]
            damage[attacker] += detail["damage"]
            damage[1 - attacker] += detail["retaliation"]
    return {
        "map_size": case.get("map_size", 10),
        "army_size": len(case["armies"][side]),
        "army_bin": case.get("army_bin", -1),
        "placement": case["placement"],
        "damage_dealt": damage,
        "units_lost": [sum(u["side"] == s and u["hp"] <= 0 for u in final["units"]) for s in (0, 1)],
    }
