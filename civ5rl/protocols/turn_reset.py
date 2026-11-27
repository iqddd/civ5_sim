"""Explicit physics and contact-placement selection, preserving historical cases and lineage."""

import random
from collections import Counter
from copy import deepcopy

from civ5map.maps import components
from civ5rl.protocols.common import case_content_hash, own_state_physical_hash
from civ5rl.protocols.contact_placement import (
    CONTACT_TWO_VISIBLE,
    CONTACT_V1,
    contact_mode,
    directed_visibility_tables,
    passable_graph,
    sample_two_visible_compact_positions,
    visible_enemy_counts,
)
from civ5sim.engine import digest


def mode(config):
    value = config.get("environment", {}).get("turn_reset_mode", "legacy")
    if value not in ("legacy", "end_turn"):
        raise ValueError(f"Unknown turn reset mode: {value}")
    return value


def validate_case(case, config):
    if case["scenario"].get("turn_reset_mode", "legacy") != mode(config):
        raise ValueError("Evaluation/training case physics differs from configuration")
    if case.get("placement") == "contact":
        expected_contact = contact_mode(config)
        if case.get("placement_algorithm", CONTACT_V1) != expected_contact:
            raise ValueError("Evaluation/training contact placement differs from configuration")


def _upgrade_contact_cases(plan, selected_turn_mode, selected_contact_mode):
    purposes = [key for key in ("development", "confirmation", "development_512") if key in plan]
    used = {
        own_state_physical_hash(case["scenario"])
        for key in purposes
        for case in plan[key]
        if case.get("placement") != "contact" and "scenario" in case
    }
    map_cache = {}
    changed = False

    for key in purposes:
        grouped = {}
        for index, case in enumerate(plan[key]):
            if case.get("placement") != "contact":
                continue
            if case.get("placement_algorithm", CONTACT_V1) == selected_contact_mode:
                used.add(own_state_physical_hash(case["scenario"]))
                continue
            layout_key = case.get("layout_id", case.get("case_id", index))
            grouped.setdefault(layout_key, []).append(case)

        for layout_cases in grouped.values():
            exemplar = layout_cases[0]
            m = exemplar["scenario"]["map"]
            map_id = exemplar.get("map_entry", {}).get("crop_hash") or digest(m)
            if map_id not in map_cache:
                group = sorted(components(m)[0])
                neighbors = passable_graph(m, group)
                sees, seen_by = directed_visibility_tables(
                    m, group, exemplar["streams"]["battle"]
                )
                map_cache[map_id] = (group, neighbors, sees, seen_by)
            group, neighbors, sees, seen_by = map_cache[map_id]
            roster = exemplar["armies"][0]
            rng = random.Random(exemplar["streams"]["placement"])
            reasons = Counter()
            placed = None
            for attempt in range(10000):
                sampled = sample_two_visible_compact_positions(
                    group,
                    len(roster),
                    rng,
                    sees,
                    seen_by,
                    neighbors,
                    reasons=reasons,
                )
                if sampled is None:
                    continue
                positions, meta = sampled
                scenario = {
                    "schema": 1,
                    "kind": "civ5sim.scenario",
                    "map": deepcopy(m),
                    "fog": True,
                    "random_damage": True,
                    "seed": exemplar["streams"]["battle"],
                    "active_side": 0,
                    "units": [
                        {"type": kind, "side": side, "tile": tile, "hp": 100}
                        for side in (0, 1)
                        for kind, tile in zip(roster, positions[side])
                    ],
                }
                if selected_turn_mode != "legacy":
                    scenario["turn_reset_mode"] = selected_turn_mode
                if visible_enemy_counts(scenario) != [2, 2]:
                    reasons["visibility"] += 1
                    continue
                phys_hash = own_state_physical_hash(scenario)
                if phys_hash in used:
                    reasons["duplicate_layout"] += 1
                    continue
                used.add(phys_hash)
                placed = (scenario, phys_hash, attempt + 1, meta)
                break
            if placed is None:
                raise RuntimeError(
                    f"Failed to upgrade contact layout {exemplar.get('case_id')} in {key}: {dict(reasons)}"
                )
            scenario, phys_hash, attempts, meta = placed
            scen_hash = digest(scenario)
            for case in layout_cases:
                case["scenario"] = deepcopy(scenario)
                case["scenario_hash"] = scen_hash
                if "physical_hash" in case:
                    case["physical_hash"] = phys_hash
                case["placement_algorithm"] = selected_contact_mode
                case["placement_method"] = "two_visible_compact"
                case["placement_attempts"] = attempts
                case["placement_radius"] = meta["max_hidden_distance"]
                case["rejections"] = dict(reasons)
                if "case_hash" in case:
                    case["case_hash"] = case_content_hash(case)
            changed = True
    return changed


def evaluation_plan(parent, config):
    selected = mode(config)
    selected_contact = contact_mode(config)
    plan = deepcopy(parent)
    physics_changed = False
    for key in ("development", "development_512", "confirmation"):
        for case in plan.get(key, []):
            scenario = case["scenario"]
            if scenario.get("turn_reset_mode", "legacy") != selected:
                scenario["turn_reset_mode"] = selected
                case["scenario_hash"] = digest(scenario)
                if "case_hash" in case:
                    case["case_hash"] = case_content_hash(case)
                physics_changed = True
    contact_changed = False
    if selected_contact == CONTACT_TWO_VISIBLE:
        contact_changed = _upgrade_contact_cases(plan, selected, selected_contact)
    for key in ("development", "development_512", "confirmation"):
        for case in plan.get(key, []):
            validate_case(case, config)
    if physics_changed:
        plan["physics_parent_manifest_hash"] = parent["hash"]
        plan["turn_reset_mode"] = selected
    if contact_changed:
        plan["contact_placement_parent_manifest_hash"] = parent["hash"]
        plan["contact_placement"] = selected_contact
    if physics_changed or contact_changed:
        plan["hash"] = digest({k: v for k, v in plan.items() if k != "hash"})
    return plan
