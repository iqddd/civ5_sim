"""Archive CP-07 coordinator acceptance without restoring legacy replay oracles."""

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from civ5sim import Battle
from civ5sim.combat_ai import (
    ContactMemory,
    Planner,
    PlanningState,
    SearchRequest,
    SupportMode,
)
from civ5sim.evaluation import write_json
from tools.benchmark_combat_ai import aggregate


def compressed(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))


def verify_waits(after, records):
    """Recheck explicit coordinator waits, rather than every tactical FINISH.

    Ordinary CP may choose a FINISH over an attack. This obligation concerns
    the dispatcher's no-useful-task fallback, which has no accepted CP plan.
    Observe only the acting side before/after its own commands, as real agents
    do, without granting contacts glimpsed during the opposing turn.
    """
    checked = []
    with Planner() as planner:
        for record in records:
            name = record["case"]["id"]
            replay = json.loads((after / "matches" / name / "0/replay.json").read_text(encoding="utf-8"))
            events = {}
            for event in replay.get("ai_events", []):
                if event["event"] == "waiting":
                    if event["reason"] in ("call_search_limit", "search_limit", "support_limit"):
                        raise AssertionError(f"Computational wait in {name}")
                    events.setdefault(event["command"], []).append(event)
            if not events:
                continue
            memories = [ContactMemory(), ContactMemory()]
            with Battle(replay["initial"]) as battle:
                for index, step in enumerate(replay["steps"]):
                    side = battle.observe()["active_side"]
                    observation = battle.observe(side)
                    memories[side] = memories[side].updated(observation)
                    for event in events.get(index, []):
                        uid = event["unit"]
                        own = next(u for u in observation["units"] if u["id"] == uid)
                        request = SearchRequest(PlanningState.from_battle(battle, side), side,
                            observation["turn"], own["tile"], (uid,), memories[side])
                        result = planner.support(request, SupportMode.RECOVERY)
                        if result.diagnostics.recovery_kind != "wait":
                            raise AssertionError(f"Unjustified wait in {name}/{index}: {result.diagnostics}")
                        checked.append({"case": name, "command": index, "unit": uid,
                                        "reason": event["reason"], "recovery_kind": "wait"})
                    if not battle.step(step["action"]).ok:
                        raise AssertionError(f"Wait audit replay failed: {name}/{index}")
                    memories[side] = memories[side].updated(battle.observe(side))
    return checked


def engagement_audit(after, records):
    """Observed entry into combat; these measurements do not assign blame.

    'Incoming group' is the coordinator's request, before CP admission/pruning.
    HP lost before the first own attack may result from both prior movement and
    tactical choices, and must not be labelled coordinator-only damage.
    """
    rows = []
    for record in records:
        name = record["case"]["id"]
        sides = [side for side, agent in enumerate(record["case"]["agents"]) if agent == "cp"]
        replay = json.loads((after / "matches" / name / "0/replay.json").read_text(encoding="utf-8"))
        data = {side: {"case": name, "side": side, "contact": None, "first_attack": None,
                       "first_search": next((event for event in replay.get("ai_events", [])
                                             if event["event"] == "search" and event["side"] == side), None)}
                for side in sides}
        initial_hp = {side: sum(unit["hp"] for unit in replay["initial"]["units"] if unit["side"] == side)
                      for side in sides}
        def observe(battle, side, command, data=data):
            observation = battle.observe(side)
            own = [u for u in observation["units"] if u["side"] == side]
            enemies = [u for u in observation["units"] if u["side"] != side]
            if data[side]["contact"] is None and enemies:
                data[side]["contact"] = {"command": command, "turn": observation["turn"],
                                         "own_units": own, "visible_enemies": enemies,
                                         "hp": sum(u["hp"] for u in own)}
            return own
        with Battle(replay["initial"]) as battle:
            for index, step in enumerate(replay["steps"]):
                side = battle.observe()["active_side"]
                if side in sides:
                    own = observe(battle, side, index)
                    if data[side]["first_attack"] is None and step["action"]["kind"] in ("melee", "ranged"):
                        hp = sum(unit["hp"] for unit in own)
                        data[side]["first_attack"] = {"command": index, "action": step["action"],
                                                      "hp": hp, "lost_hp_before_attack": initial_hp[side] - hp}
                if not battle.step(step["action"]).ok:
                    raise AssertionError(f"Engagement audit replay failed: {name}/{index}")
                if side in sides:
                    observe(battle, side, index + 1)
        rows.extend(data.values())
    return rows


def summarize(before, after, out):
    final = json.loads((after / "summary.json").read_text(encoding="utf-8"))
    baseline = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(before.glob("*/summary.json"))]
    if len(baseline) != 16 or any(len(value["records"]) != 1 or value["failures"] for value in baseline):
        raise ValueError("The 16-case baseline did not finish cleanly")
    before_records = [value["records"][0] for value in baseline]
    names = {record["case"]["id"] for record in before_records}
    after_records = [record for record in final["records"] if record["case"]["id"] in names]
    if len(after_records) != 16:
        raise ValueError("Final series does not cover the same 16 baseline cases")
    waits = verify_waits(after, final["records"])
    engagements = engagement_audit(after, final["records"])
    out.mkdir(parents=True, exist_ok=True)
    compressed(out / "before-records.json.gz", before_records)
    compressed(out / "after-records.json.gz", final["records"])
    write_json(out / "waiting-audit.json", {"checked": len(waits), "rows": waits})
    compressed(out / "engagement-audit.json.gz", engagements)
    comparison = {"before": aggregate(before_records), "after_same_cases": aggregate(after_records)}
    write_json(out / "comparison.json", comparison)
    write_json(out / "acceptance.json", {key: value for key, value in final.items() if key != "records"})
    for name in ("report.md", "passport.json", "tactical.json"):
        (out / name).write_bytes((after / name).read_bytes())
    passports = {directory.name: json.loads((directory / "passport.json").read_text(encoding="utf-8"))
                 for directory in sorted(before.iterdir()) if (directory / "passport.json").exists()}
    compressed(out / "before-passports.json.gz", passports)
    selected = []
    for opponent, outcome in (("greedy", "win"), ("greedy", "loss"), ("random", "win")):
        for record in final["records"]:
            case = record["case"]
            if opponent not in case["agents"]:
                continue
            side = case["agents"].index("cp")
            winner = record["result"]["winner"]
            if (outcome == "win" and winner == side) or (outcome == "loss" and winner == 1-side):
                selected.append(case["id"])
                break
    selected.extend(record["case"]["id"] for record in final["records"] if record["case"]["field"] == "open-26")
    replay_manifest = {}
    for name in dict.fromkeys(selected):
        path = after / "matches" / name / "0/replay.json"
        replay = json.loads(path.read_text(encoding="utf-8"))
        compressed(out / "replays" / (name + ".json.gz"), replay)
        replay_manifest[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                 "source": str(path.relative_to(ROOT)), "steps": len(replay["steps"])}
    write_json(out / "replays.json", replay_manifest)
    lines = ["# Coordinator before/after comparison", "",
             "Same 16 cases; two deterministic runs each. Baseline commit 66b5b31.",
             "Native search settings, seeds, side placements and 600-second match timeout unchanged.",
             "These are measured outcomes, not an optimality claim.", "",
             "| Metric | before | after |", "|---|---:|---:|"]
    for name in ("decision_ms", "turn_ms", "search_ms", "support_ms"):
        for quantile in ("median", "p95", "max"):
            values = [comparison[label]["timings"].get("cp/" + name, {}).get(quantile)
                      for label in ("before", "after_same_cases")]
            lines.append(f"| {name}/{quantile} | {values[0]} | {values[1]} |")
    for key in ("empty_search_fraction", "replanning_search_fraction", "recovery_kinds", "waiting_reasons"):
        lines.append(f"| {key} | {comparison['before'].get(key)} | {comparison['after_same_cases'].get(key)} |")
    (out / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Archived {len(final['records'])} cases, {len(replay_manifest)} representative replays; acceptance={final['acceptance']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--before", type=Path, default=ROOT / "build/cp07-coordinator-before")
    parser.add_argument("--after", type=Path, default=ROOT / "build/cp07-coordinator-final")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/combat-ai/coordinator-acceptance-20261102")
    args = parser.parse_args()
    summarize(args.before.resolve(), args.after.resolve(), args.out.resolve())
