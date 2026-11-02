"""Fresh ABI6 tactical requests and execution checks; no legacy plan oracle."""

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from civ5sim import Action, Battle
from civ5sim import combat_ai as ai
from civ5sim.engine import UNIT_TYPES, digest
from civ5sim.evaluation import synthetic_map, write_json


def load_request(value):
    value = json.loads(json.dumps(value))
    state = value.pop("state")
    if any("damage_taken_last_turn" not in unit for unit in state["units"]):
        raise ValueError("Legacy tactical request has no damage history; regenerate it")
    state["tiles"] = tuple(ai.PlanningTile(**tile) for tile in state["tiles"])
    state["units"] = tuple(ai.PlanningUnit(**unit) for unit in state["units"])
    state["visibility"] = tuple(state["visibility"])
    value["state"] = ai.PlanningState(**state)
    contacts = value.pop("contacts")
    contacts["known_enemies"] = tuple(ai.Contact(**contact) for contact in contacts["known_enemies"])
    value["contacts"] = ai.ContactMemory(**contacts)
    value["selected_units"] = tuple(value["selected_units"])
    if value["aggression"] is not None:
        value["aggression"] = ai.Aggression(value["aggression"])
    return ai.SearchRequest(**value)


def fresh_cases():
    cases = {}

    def save(name, battle, aggression=ai.Aggression.MEDIUM):
        side = battle.observe()["active_side"]
        state = ai.PlanningState.from_battle(battle, side)
        selected = tuple(unit.id for unit in state.units if unit.side == side and unit.hp > 0)
        target = next(unit.tile for unit in state.units if unit.side != side and unit.hp > 0)
        request = ai.SearchRequest(state, side, state.turn, target, selected,
                                   ai.ContactMemory().updated(battle.observe(side)), aggression)
        config = ai.SearchConfig(position_capacity=800, max_completed=80)
        cases[name] = {"request": asdict(request), "snapshot": battle.snapshot(), "config": asdict(config)}

    for fog in (False, True):
        for kind in UNIT_TYPES:
            scenario = {"schema": 1, "kind": "civ5sim.scenario", "map": synthetic_map(),
                        "fog": fog, "random_damage": False, "seed": 42,
                        "units": [{"type": kind, "side": 0, "tile": [6, 7]},
                                  {"type": "warrior", "side": 1, "tile": [8, 7]}]}
            with Battle(scenario) as battle:
                save(f"{kind}-fog{int(fog)}", battle)
    for attacks in (1, 3):
        scenario = {"schema": 1, "kind": "civ5sim.scenario", "map": synthetic_map(),
                    "fog": False, "random_damage": False, "seed": 42,
                    "units": [{"type": "archer", "side": 0, "tile": [6 + i, 7]} for i in range(3)]
                             + [{"type": "warrior", "side": 1, "tile": [7, 8]}]}
        with Battle(scenario) as battle:
            for attacker in range(attacks):
                assert battle.step(Action("ranged", attacker, 7 + 8 * 16)).ok
            assert battle.step(Action("end_turn")).ok
            save(f"history-{attacks}-attacks", battle, ai.Aggression.BRAVEHEART)
    return cases


def replay_plan(case, result):
    commands = []
    initial_hp = {unit["id"]: unit["hp"] for unit in case["snapshot"]["units"]}
    side = case["request"]["side"]
    with Battle(case["snapshot"]) as battle:
        for assignment in result.assignments:
            if assignment.kind in (ai.AssignmentKind.INITIAL, ai.AssignmentKind.RESTART,
                                    ai.AssignmentKind.SWAP_REVERSE, ai.AssignmentKind.BLOCKED,
                                    ai.AssignmentKind.FINISH_TEMP):
                continue
            unit = battle.snapshot()["units"][assignment.unit]
            if assignment.kind == ai.AssignmentKind.FINISH:
                if unit["hp"] <= 0 or unit["moves"] <= 0 or unit["waiting"]:
                    continue
                action = Action("skip", assignment.unit)
            else:
                if unit["tile"] != assignment.from_tile:
                    raise ValueError("Plan start mismatch")
                kind = {ai.AssignmentKind.MOVE: "move", ai.AssignmentKind.MELEE: "melee",
                        ai.AssignmentKind.RANGED: "ranged", ai.AssignmentKind.SWAP: "swap"}[assignment.kind]
                action = Action(kind, assignment.unit, assignment.command_target)
            if action not in battle.available_actions():
                raise ValueError(f"Illegal plan action: {action}")
            step = battle.step(action)
            if not step.ok or step.damage != assignment.damage or step.retaliation != assignment.self_damage:
                raise ValueError("Plan transition mismatch")
            now = battle.snapshot()["units"][assignment.unit]
            if assignment.kind in (ai.AssignmentKind.MOVE, ai.AssignmentKind.SWAP) and now["tile"] != assignment.to_tile:
                raise ValueError("Plan destination mismatch")
            if now["hp"] > 0 and now["moves"] != assignment.remaining_moves:
                raise ValueError("Plan movement mismatch")
            if assignment.killed_unit >= 0 and battle.snapshot()["units"][assignment.killed_unit]["hp"] != 0:
                raise ValueError("Plan kill mismatch")
            commands.append(asdict(action))
        snapshot = battle.snapshot()
        return {"checked": True, "commands": commands, "final_sha256": digest(snapshot),
                "damage": sum(initial_hp[u["id"]] - u["hp"] for u in snapshot["units"] if u["side"] != side),
                "own_damage": sum(initial_hp[u["id"]] - u["hp"] for u in snapshot["units"] if u["side"] == side)}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "build/tactical-current")
    args = parser.parse_args()
    cases = fresh_cases()
    write_json(args.out / "inputs.json", {"schema": 2, "ai_abi": 6, "cases": cases})
    rows = []
    for name, case in cases.items():
        config = dict(case["config"])
        config["aggression"] = ai.Aggression(config["aggression"])
        with ai.Planner(ai.SearchConfig(**config)) as planner:
            start = time.perf_counter()
            result = planner.search(load_request(case["request"]))
            elapsed = time.perf_counter() - start
            version = planner.version
        rows.append({"case": name, "seconds": elapsed, "result": asdict(result),
                     "validation": replay_plan(case, result)})
    write_json(args.out / "results.json", {"version": version, "abi": 6, "rows": rows})
    print(f"{len(rows)} fresh requests, all plans execution-checked: {args.out}")


if __name__ == "__main__":
    main()
