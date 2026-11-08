"""Explain any recorded command or standalone planning query without editing its source."""

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from civ5sim import Action, Battle
from civ5sim.decision_diagnostics import load_package, query_with_trace, repeat_query
from civ5sim.engine import digest


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def verified_states(replay, commands):
    if replay.get("schema") != 2 or replay.get("kind") != "civ5sim.replay":
        raise ValueError("Expected civ5sim.replay schema 2")
    if not commands or any(n < 1 or n > len(replay["steps"]) for n in commands):
        raise ValueError("Command numbers are 1-based and must exist in the replay")
    states = {}
    with Battle(replay["initial"]) as battle:
        for number, step in enumerate(replay["steps"][:max(commands)], 1):
            if number in commands:
                side = battle.snapshot()["state"]["side"]
                states[number] = {"before": battle.snapshot(), "observation": battle.observe(side)}
            result = battle.step(step["action"])
            if asdict(result) != step["result"] or digest(battle.snapshot()) != step["state_sha256"]:
                raise ValueError(f"Replay diverged at command {number}; use its matching physical kernel")
            if number in commands:
                states[number]["after"] = battle.snapshot()
    return states


def alternative(replay, number, actions, *, reply=None, match_result=None):
    """Execute explicit commands on a disposable battle; never replace the replay."""
    if not isinstance(actions, list) or not actions:
        raise ValueError("Alternative must be a nonempty JSON array of actions")
    greedy = None
    greedy_side = None
    if reply:
        from civ5sim.greedy_agent import GreedyAgent
        if match_result is None:
            raise ValueError("Greedy response requires result.json with agent seeds and configuration")
        sides = [i for i, name in enumerate(match_result["agents"])
                 if name in ("greedy", "civ5sim.greedy_agent:GreedyAgent")]
        if len(sides) != 1:
            raise ValueError("Exactly one recorded greedy agent is required")
        greedy_side = sides[0]
        config = match_result["config"]
        greedy = GreedyAgent()
        greedy.reset({"map": config["map"], "side": greedy_side, "armies": config["armies"],
                      "max_rounds": config["max_rounds"], "fog": config["fog"],
                      "random_damage": config["random_damage"],
                      "seed": match_result["seeds"][f"agent{greedy_side}"]})
    with Battle(replay["initial"], log=True) as battle:
        for index, step in enumerate(replay["steps"][:number-1], 1):
            if greedy is not None and battle.snapshot()["state"]["side"] == greedy_side:
                chosen = greedy.act(battle.observe(greedy_side), tuple(battle.available_actions()))
                if asdict(chosen) != step["action"]:
                    raise ValueError(f"Cannot restore historical greedy state: command {index} differs")
            result = battle.step(step["action"])
            if asdict(result) != step["result"] or digest(battle.snapshot()) != step["state_sha256"]:
                raise ValueError(f"Replay diverged at command {index}")
        before = battle.snapshot()
        side = before["state"]["side"]
        for index, raw_action in enumerate(actions, 1):
            action = asdict(Action(**raw_action))
            if battle.snapshot()["state"]["side"] != side:
                raise ValueError("Alternative may only issue commands for the original active side")
            if action not in [asdict(a) for a in battle.available_actions()]:
                raise ValueError(f"Illegal alternative command {index}: {action}")
            result = battle.step(action)
            if not result.ok:
                raise ValueError(f"Alternative command {index} failed: {result.error}")
        if greedy is not None and not any(battle.army_status()["destroyed"]):
            if side == greedy_side:
                raise ValueError("Greedy must be the opponent of the alternative's active side")
            if battle.snapshot()["state"]["side"] == side:
                battle.step({"kind": "end_turn", "unit": -1, "target": -1})
            limit = match_result["config"]["max_calls_per_turn"]
            for _ in range(limit):
                if any(battle.army_status()["destroyed"]) or battle.snapshot()["state"]["side"] != greedy_side:
                    break
                action = greedy.act(battle.observe(greedy_side), tuple(battle.available_actions()))
                if action not in battle.available_actions():
                    raise ValueError("Greedy returned an illegal response")
                battle.step(action)
            else:
                if battle.snapshot()["state"]["side"] == greedy_side:
                    raise ValueError("Greedy response exceeded turn call limit")
        return {"kind": "physical_counterfactual", "command": number, "before": before,
                "after": battle.snapshot(), "steps": battle.replay()["steps"][number-1:],
                "reply": reply, "claim": "One continuation, not an optimality or long-term safety proof"}


def explain(source, commands, *, repeat=False, package_path=None):
    source = Path(source)
    replay = read(source)
    path = Path(package_path) if package_path else source.with_name("diagnostics.json.gz")
    if package_path is not None and not path.exists():
        raise ValueError("Specified diagnostic package does not exist")
    package = load_package(path, replay) if path.exists() else None
    states = verified_states(replay, commands)
    reports = []
    queries = {}
    for number in commands:
        calls = [c for c in package["calls"] if c["command"] == number] if package else []
        ids = set()
        for call in calls:
            ids.update(call["queries"])
            if call.get("origin"):
                ids.add(call["origin"]["query"])
        for query_id in sorted(ids):
            if str(query_id) in queries:
                continue
            query = query_with_trace(package, package["queries"][query_id])
            queries[str(query_id)] = {"recorded": query, "state": package["states"][query["request"]["state"]],
                                      "origin_call": package["calls"][query["call"]]}
            if repeat:
                queries[str(query_id)]["experiment"] = repeat_query(package, query)
        reports.append({"command": number, "step": replay["steps"][number-1], **states[number],
                        "calls": calls, "queries": sorted(ids),
                        "legacy_events": [e for e in replay.get("ai_events", [])
                                          if number-1 <= e["command"] <= number],
                        "missing": [] if calls else ["No recorded call/request provenance; historical decision cannot be reconstructed"]})
    return {"kind": "combat-decision-report", "replay_sha256": digest(replay),
            "recording": None if package is None else {k: package[k] for k in ("mode", "complete", "errors", "provenance")},
            "commands": reports, "queries": queries}


def tile(value, width):
    return f"({value % width},{value // width}) [ID {value}]" if value >= 0 else "—"


def markdown(report):
    lines = ["# Диагностика решений", "", ("Записанное решение и повтор текущим кодом разделены. "
             "Отказы описывают исследованные варианты, а не доказывают отсутствие всех возможных планов."), ""]
    recording = report.get("recording")
    if recording is not None and not recording["complete"]:
        lines += [f"**Диагностический пакет неполон:** `{recording['errors']}`", ""]
    for command in report.get("commands", []):
        lines += [f"## Команда {command['command']}", "", f"Действие: `{command['step']['action']}`.", ""]
        width = command["before"]["map"]["width"]
        lines += ["Известные юниты перед действием:", "",
                  "| Юнит | Сторона | Клетка | HP | ОП |", "|---|---|---|---|---|"]
        for unit in command["observation"]["units"]:
            lines.append(
                f"| {unit['type']} №{unit['id']} | {unit['side']} | "
                f"{tile(unit['tile'], width)} | {unit['hp']} | {unit.get('moves', 'неизвестно')} |"
            )
        lines.append("")
        for missing in command["missing"]:
            lines += [f"Недостающие данные: {missing}", ""]
        for call in command["calls"]:
            lines += [f"Источник: `{call['agent']}`, вызов {call['id']}; назначение: `{call.get('origin', 'правило агента без записанного назначения')}`.", ""]
            coordinator = call.get("coordinator_before")
            if coordinator is not None:
                lines += [
                    (f"Состояние координатора: фаза {coordinator['phase']}; "
                     f"задача `{coordinator['task']}`; в очереди {len(coordinator['queued_assignments'])}; "
                     f"поисков {coordinator['searches']} из {coordinator['search_limit']}; "
                     f"участники RECOVERY: {coordinator.get('recovery_units', [])}."),
                    "",
                ]
            for event in call["events"]:
                if event["event"] in ("task_assigned", "task_restarted", "task_finished", "plan_cancelled", "waiting", "target_comparison", "group_recovery"):
                    lines += [f"- `{event['event']}`: `{json.dumps(event, ensure_ascii=False)}`"]
            lines.append("")
    for query_id, item in report["queries"].items():
        query, state = item["recorded"], item["state"]
        width = state["width"]
        request = query["request"]
        from civ5sim.combat_ai import Aggression, SupportMode
        mode = SupportMode(query["mode"]).name if isinstance(query["mode"], int) else query["mode"]
        aggression = request["aggression"] if request["aggression"] is not None else query["config"]["aggression"]
        lines += [f"## Запрос {query_id}: {mode}", "",
                  (f"Цель: {tile(request['target'], width)}. Состав: {request['selected_units']}. "
                  f"Агрессия: {Aggression(aggression).name}."), ""]
        if "origin_call" in item:
            origin = item["origin_call"]
            lines += [f"Запрос сделан перед командой {origin['command']}, вызов {origin['id']}.", ""]
        result = query["result"]
        if isinstance(result, dict):
            lines += [f"Итог: `{result['reason']}`; счётчики: `{result['stats']}`; отказы: `{result['diagnostics']}`.", "",
                      "| № | Назначение | Юнит | Откуда | Куда | ОП | Балл | Урон / ответный |", "|---|---|---|---|---|---|---|---|"]
            from civ5sim.combat_ai import AssignmentKind
            for index, a in enumerate(result["assignments"]):
                kind = AssignmentKind(a["kind"]).name
                unit = state["units"][a["unit"]]
                names = ("воин", "лучник", "колесничный лучник", "копейщик", "всадник")
                lines.append(f"| {index} | {kind} | {names[unit['type']]} №{a['unit']} | "
                             f"{tile(a['from_tile'],width)} | {tile(a['to_tile'],width)} | "
                             f"{a['remaining_moves']} | {a['score']} | {a['damage']} / {a['self_damage']} |")
                if kind == "RESTART":
                    lines.append("| | **Граница исполнимой части: последующие действия требуют перепланирования** | | | | | | |")
            lines.append("")
        trace = query.get("trace")
        if "experiment" in item:
            experiment = item["experiment"]
            lines += [(f"**Повтор текущим кодом**: совпадение полного результата — {experiment['matches_recorded_result']}; совпадение среды — {experiment['matches_recorded_environment']}. "
                      "Это эксперимент; версия и хеши сохранены в JSON."), ""]
            trace = experiment["trace"]
        if trace:
            events = trace.get("events", [])
            lines += [f"Трасса: {len(events)} событий; усечена: {trace.get('truncated', False)}.", "",
                      f"Сводка: `{dict(Counter(e['event'] for e in events))}`.", ""]
            # Select winner ancestry, not unrelated end scores from all explored branches.
            results = [e for e in events if e["event"] == "search_result"]
            parents = {e["position"]: e["parent"] for e in events if e["event"] == "child"}
            ancestry = set()
            position = results[-1]["best_position"] if results else -1
            while position >= 0 and position not in ancestry:
                ancestry.add(position)
                position = parents.get(position, -1)
            selected_events = [e for e in events if e["event"].startswith(("recovery_", "support_", "recruit_")) or
                               (e["position"] in ancestry and e["event"] in
                                ("turn_end", "final_evaluation", "end_skip_blocked", "end_skip_pending"))]
            lines += ["Подробности выбранной ветви / RECOVERY (полная трасса — в JSON):", ""]
            for event in selected_events[-80:]:
                formatted = dict(event)
                if "tile" in formatted:
                    formatted["tile"] = tile(formatted["tile"], width)
                lines.append(f"- `{json.dumps(formatted, ensure_ascii=False)}`")
            lines.append("")
        else:
            lines += ["Подробная нативная трасса не записана. Используйте `--repeat` для отдельного эксперимента текущим кодом.", ""]
    if "alternative" in report:
        experiment = report["alternative"]
        lines += ["## Физический эксперимент с альтернативой", "",
                  "Одна последовательность и ответ противника не доказывают оптимальность или дальнейшую безопасность.", "",
                  "| Юнит | HP до | HP после |", "|---|---|---|"]
        for uid, (before, after) in enumerate(zip(experiment["before"]["units"], experiment["after"]["units"])):
            if before["hp"] != after["hp"]:
                lines.append(f"| {before['type']} №{uid} | {before['hp']} | {after['hp']} |")
        lines += ["", f"Исполнено команд: {len(experiment['steps'])}; ответ: {experiment['reply'] or 'не запускался'}.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Replay JSON, or diagnostics.json.gz with --query")
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--commands", nargs="+", type=int)
    selection.add_argument("--query", type=int)
    parser.add_argument("--package", type=Path)
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--alternative", type=Path, help="JSON array of simulator commands")
    parser.add_argument("--reply", choices=("greedy",))
    parser.add_argument("--out", "--output", dest="output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output must be a new directory")
    if args.query is not None:
        if args.alternative or args.reply:
            parser.error("Physical alternatives require a replay and one command")
        package = load_package(args.source)
        if not 0 <= args.query < len(package["queries"]):
            parser.error("Query ID is out of range")
        query = query_with_trace(package, package["queries"][args.query])
        item = {"recorded": query, "state": package["states"][query["request"]["state"]]}
        if args.repeat:
            item["experiment"] = repeat_query(package, query)
        report = {"kind": "standalone-query-report", "queries": {str(args.query): item}}
    else:
        report = explain(args.source, args.commands, repeat=args.repeat, package_path=args.package)
        if args.reply and not args.alternative:
            parser.error("--reply requires --alternative")
        if args.alternative:
            if len(args.commands) != 1:
                parser.error("Alternative requires exactly one command")
            result_path = args.source.with_name("result.json")
            report["alternative"] = alternative(read(args.source), args.commands[0], read(args.alternative),
                                                reply=args.reply, match_result=read(result_path) if result_path.exists() else None)
    args.output.mkdir(parents=True)
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "report.md").write_text(markdown(report), encoding="utf-8")
    print(args.output / "report.md")


if __name__ == "__main__":
    main()
