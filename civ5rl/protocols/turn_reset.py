"""Explicit physics selection, preserving historical cases and lineage."""

from copy import deepcopy

from civ5sim.engine import digest


def mode(config):
    value = config.get("environment", {}).get("turn_reset_mode", "legacy")
    if value not in ("legacy", "end_turn"):
        raise ValueError(f"Unknown turn reset mode: {value}")
    return value


def validate_case(case, config):
    if case["scenario"].get("turn_reset_mode", "legacy") != mode(config):
        raise ValueError("Evaluation/training case physics differs from configuration")


def evaluation_plan(parent, config):
    selected = mode(config)
    plan = deepcopy(parent)
    changed = False
    for key in ("development", "development_512", "confirmation"):
        for case in plan.get(key, []):
            scenario = case["scenario"]
            if scenario.get("turn_reset_mode", "legacy") != selected:
                scenario["turn_reset_mode"] = selected
                case["scenario_hash"] = digest(scenario)
                if "case_hash" in case:
                    case["case_hash"] = digest({k: v for k, v in case.items() if k != "case_hash"})
                changed = True
            validate_case(case, config)
    if changed:
        plan["physics_parent_manifest_hash"] = parent["hash"]
        plan["turn_reset_mode"] = selected
        plan["hash"] = digest({k: v for k, v in plan.items() if k != "hash"})
    return plan
