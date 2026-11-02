"""Fixed CP-07 corpus, artifact checks, and measurement summaries."""

import ctypes
import json
import math
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from .combat_ai import Aggression, Planner, SearchConfig
from .engine import Battle, digest
from .match import MatchConfig

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reusable_check(directory, signature, case_id):
    """A passport alone cannot authenticate old or partially overwritten cases."""
    directory = Path(directory)
    try:
        checked = json.loads((directory / "checked.json").read_text(encoding="utf-8"))
        if checked.get("run_signature") != signature or checked.get("case") != case_id:
            return False
        artifacts = {name: json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
                     for name in ("result", "replay", "final")}
        json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        return checked.get("replay_verified") is True and digest(artifacts) == checked["deterministic_sha256"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def write_report(path, summary, full, failures, suite, *, repeats=2):
    search = suite["search"]
    lines = ["# CP-07: качество и стоимость Combat AI v1", "",
             f"Завершённых случаев: {summary['matches']}; исполнений каждого: {repeats}, с проверкой реплея.",
             f"Полная приёмка: {'да' if full and not failures else 'нет'}. Технических сбоев: {len(failures)}.", "",
             (f"Профиль CP: {search['position_capacity']}/{search['max_completed']}/"
              f"{search['max_branches']}/{search['choices_per_unit']}, {search['aggression']}, "
              f"риск {search['risk_budget_base']}+{search['risk_budget_per_kill']} за убийство, "
              f"privileged_cp_queries={str(search['privileged_cp_queries']).lower()}."),
             "Greedy и random получают обычное наблюдение; CP использует согласованные привилегированные запросы v1.", "",
             "## Исходы и оставшиеся силы", "",
             "| Соперник | Случаи CP | Победы | Ничьи | Поражения | Осталось юнитов, медиана | HP, медиана |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for key, value in summary["outcomes"].items():
        if key.startswith("opponent/"):
            counts = value["outcomes"]
            lines.append(f"| {key.split('/')[1]} | {value['count']} | {counts.get('win', 0)} | "
                         f"{counts.get('draw', 0)} | {counts.get('loss', 0)} | "
                         f"{value['remaining_units']['median']} | {value['remaining_hp']['median']} |")
    lines += ["", "В CP–CP каждый матч даёт две оценки сторон; это не два независимых матча.", "",
              "## Время и память", "",
              "Время — миллисекунды первого исполнения. Повтор, если запрошен, проверяет детерминизм.",
              "Решение включает подготовку, вызов агента и callback; поиск уже входит в вызов агента.",
              "Ход суммирует решения стороны; последний незавершённый ход также учитывается.", "",
              "| Измерение | N | Медиана | p95 | Максимум |", "|---|---:|---:|---:|---:|"]
    for key, value in summary["timings"].items():
        if key.startswith("cp/") and key.split('/')[1] in ("decision_ms", "turn_ms", "search_ms", "support_ms"):
            lines.append(f"| {key} | {value['count']} | {value['median']:.3f} | "
                         f"{value['p95']:.3f} | {value['max']:.3f} |")
    memory = summary["peak_process_rss_bytes"]
    if memory["count"]:
        lines += ["", (f"Пиковая RSS процесса: медиана {memory['median']/2**20:.2f} МиБ, "
                  f"максимум {memory['max']/2**20:.2f} МиБ.")]
    lines += ["Память включает Python, ядро, AI и накопленный реплей; это пик к концу матча до проверки реплея.",
              "Начальный уровень и прирост каждого процесса приведены в metrics.json.", "",
              "## Поиск и ограничения", "",
              (f"Пустых локальных планов: {summary['empty_search_fraction']:.2%}; "
              f"поисков после отмены плана: {summary['replanning_search_fraction']:.2%}."),
              f"Счётчики CP: `{json.dumps(summary['cp_diagnostics'], ensure_ascii=False)}`.",
              f"Причины отмены: `{json.dumps(summary['cancellations'], ensure_ascii=False)}`.", "",
              f"Тактическое продолжение: `{json.dumps(summary.get('recovery_kinds', {}), ensure_ascii=False)}`.",
              f"Причины ожидания: `{json.dumps(summary.get('waiting_reasons', {}), ensure_ascii=False)}`.", "",
              (f"Набор ограничен {len(suite['fields'])} конфигурациями и {len(suite['seeds'])} seed; "
               "он не доказывает превосходство над оригинальным DLL AI."),
              ("Поиск оценивает гипотетический урон при броске 600 и допускает привилегированные запросы; "
              "это влияет на сравнение с обычными агентами."),
              "Победы над random сами по себе не доказывают качество переноса. Пороги силы и скорости не заявлены.",
              "Полные разрезы по картам, режимам, сторонам, исходам и оставшимся силам находятся в summary.json.", ""]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def synthetic_map(shape="open"):
    tiles = [{"terrain": "grass", "elevation": "flat", "feature": None,
              "rivers": [-1, -1, -1]} for _ in range(256)]
    if shape == "passage":
        for y in range(16):
            if y not in (7, 8):
                tiles[y * 16 + 7]["elevation"] = "mountain"
    return {"schema": 1, "width": 16, "height": 16, "row_parity": 0,
            "wrap_x": False, "metadata": {}, "tiles": tiles,
            "spawn_candidates": [[[x, y] for y in range(16) for x in xs]
                                 for xs in ((0, 1), (14, 15))]}


def load_suite(path):
    path = Path(path).resolve()
    suite = json.loads(path.read_text(encoding="utf-8"))
    if suite["schema"] != 1:
        raise ValueError("Expected evaluation schema 1")
    for field in [*suite["fields"], suite["stress"]]:
        source = field["map"]
        field["map"] = (synthetic_map(source) if source in ("open", "passage") else
                        json.loads((path.parent / source).read_text(encoding="utf-8")))
    return suite


def search_config(suite):
    value = dict(suite["search"])
    value["aggression"] = Aggression[value["aggression"]]
    return SearchConfig(**value)


def cases(suite):
    result = []

    def add(field, seed, fog, random_damage, agents):
        case_id = (f"{field['id']}-s{seed}-f{int(fog)}-r{int(random_damage)}-"
                   + "-".join(agents))
        cycle = suite["unit_cycle"]
        army = [cycle[i % len(cycle)] for i in range(field["units"])]
        result.append({"id": case_id, "field": field["id"], "agents": agents,
                       "config": asdict(MatchConfig(
                           field["map"], [army, army.copy()], suite["max_rounds"],
                           seed, fog, random_damage, suite["max_calls_per_turn"]))})

    for field in suite["fields"]:
        for seed in suite["seeds"]:
            for fog in (False, True):
                for damage in (False, True):
                    for opponent in ("random", "greedy"):
                        for side in (0, 1):
                            add(field, seed, fog, damage,
                                ["cp", opponent] if side == 0 else [opponent, "cp"])
        for fog in (False, True):
            for damage in (False, True):
                add(field, suite["seeds"][0], fog, damage, ["cp", "cp"])
    for fog in (False, True):
        for damage in (False, True):
            add(suite["stress"], suite["seeds"][0], fog, damage, ["cp", "cp"])
    if len({case["id"] for case in result}) != len(result):
        raise ValueError("Duplicate evaluation case ID")
    return result


def distribution(values):
    values = sorted(values)
    if not values:
        return {"count": 0, "median": None, "p95": None, "max": None}
    n = len(values)
    return {"count": n, "median": (values[(n-1)//2] + values[n//2]) / 2,
            "p95": values[math.ceil(n * .95)-1], "max": values[-1]}


def timing_summary(samples):
    groups = defaultdict(list)
    turns = defaultdict(float)
    decisions = defaultdict(list)
    pending = defaultdict(float)
    for sample in samples:
        side, turn = sample["side"], sample["turn"]
        kind, ms = sample["kind"], sample["ns"] / 1e6
        groups[f"side{side}/{kind}_ms"].append(ms)
        if kind.startswith("prepare_"):
            pending[side] += ms
        elif kind == "act":
            decisions[side].append(pending[side] + ms)
            pending[side] = 0
        elif kind == "callback":
            decisions[side][-1] += ms
        if kind.startswith("prepare_") or kind in ("act", "callback"):
            turns[side, turn] += ms
    for side, values in decisions.items():
        groups[f"side{side}/decision_ms"] = values
    for (side, _turn), ms in turns.items():
        groups[f"side{side}/turn_ms"].append(ms)
    return {kind: distribution(values) for kind, values in sorted(groups.items())}


def diagnostics(replay, config):
    result = []
    for side in (0, 1):
        events = [e for e in replay.get("ai_events", []) if e["side"] == side]
        searches = [e for e in events if e["event"] == "search"]
        support = [e for e in events if e["event"] == "support"]
        recovery = [e for e in events if e["event"] == "recovery"]
        for event in searches:
            stats = event["stats"]
            if not (0 <= stats["completed"] <= config.max_completed and
                    0 <= stats["peak_positions"] <= config.position_capacity and
                    stats["generated"] <= config.position_capacity):
                raise AssertionError(f"Search budget exceeded: {event}")
        for event in searches + support + recovery:
            if not 1 <= len(event["selected_units"]) <= 13:
                raise AssertionError("Invalid local group size")
            if event["reason"] in ("invalid_request", "not_implemented", "turn_changed"):
                raise AssertionError(f"Unexpected planning failure: {event}")
        for turn in {e["turn"] for e in events}:
            turn_events = [e for e in events if e["turn"] == turn]
            # Initial army size is a safe upper bound on available units each turn.
            n = sum(u["side"] == side for u in replay["initial"]["units"])
            if (sum(e["event"] == "search" for e in turn_events) > 2*n + (n+12)//13 or
                sum(e["event"] == "support" for e in turn_events) > 3*n + (n+12)//13):
                raise AssertionError("Coordinator request budget exceeded")
            if sum(e["event"] == "recovery" for e in turn_events) > n * (5*n + 1):
                raise AssertionError("Coordinator recovery budget exceeded")
            if any(e["event"] == "waiting" and e.get("reason") in
                   ("call_search_limit", "search_limit", "support_limit") for e in turn_events):
                raise AssertionError("Computational limit became a wait order")
        result.append({
            "searches": len(searches), "support_calls": len(support),
            "empty_searches": sum(e["assignments"] == 0 for e in searches),
            "empty_support": sum(e["assignments"] == 0 for e in support),
            "replanning_searches": sum(e["replanning"] for e in searches),
            "recovery_calls": len(recovery),
            "recovery_kinds": dict(Counter(e.get("diagnostics", {}).get("recovery_kind", "none") for e in recovery)),
            "group_reductions": sum(e["event"] == "group_reduced" for e in events),
            "waiting_reasons": dict(Counter(e["reason"] for e in events if e["event"] == "waiting")),
            "end_risk_rejections": sum(e.get("diagnostics", {}).get("risk_rejections", 0) for e in searches),
            "cancellations": dict(Counter(e["reason"] for e in events
                                          if e["event"] == "plan_cancelled")),
            "search_reasons": dict(Counter(e["reason"] for e in searches)),
            "support_reasons": dict(Counter(e["reason"] for e in support)),
            "positions": {key: sum(e["stats"][key] for e in searches)
                          for key in ("checked", "generated", "completed", "pruned")},
            "peak_positions": max((e["stats"]["peak_positions"] for e in searches), default=0),
        })
    return result


def verify_replay(outcome):
    with Battle(outcome.replay["initial"]) as battle:
        for index, entry in enumerate(outcome.replay["steps"]):
            action = entry["action"]
            from .engine import action_value
            if action_value(action) not in battle.available_actions():
                raise AssertionError(f"Illegal replay action {index}")
            if (asdict(battle.step(action)) != entry["result"] or
                    digest(battle.snapshot()) != entry["state_sha256"]):
                raise AssertionError(f"Replay diverged at {index}")
        if battle.snapshot() != outcome.final:
            raise AssertionError("Replay final state differs")


def process_memory():
    """OS working-set peak, including Python, native code and replay storage."""
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [
                (name, ctypes.c_size_t) for name in (
                    "peak", "working", "peak_paged", "paged", "peak_nonpaged",
                    "nonpaged", "pagefile", "peak_pagefile")]
        value = Counters()
        value.cb = ctypes.sizeof(value)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters),
                                             ctypes.c_ulong]
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(value), value.cb):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"peak_rss_bytes": value.peak, "current_rss_bytes": value.working}
    import resource
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"peak_rss_bytes": int(peak * (1 if sys.platform == "darwin" else 1024))}


def passport(config_path, suite):
    paths = sorted([*ROOT.glob("civ5sim/*.py"), *ROOT.glob("civ5map/*.py"), *ROOT.glob("native/**/*.cpp"),
                    *ROOT.glob("native/**/*.h"), ROOT / "tools/benchmark_combat_ai.py",
                    ROOT / "tools/build_combat_ai.py", ROOT / "tools/build_combat.py",
                    ROOT / "build/combat_rules.json", ROOT / "docs/combat-ai-sources.json"])
    sources = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}
    with Planner(search_config(suite)) as planner:
        version = planner.version
    cp = json.loads((ROOT / "docs/combat-ai-sources.json").read_text(encoding="utf-8"))
    return {"schema": 1, "kind": "civ5sim.combat_ai_passport", "port": version, "abi": 7,
            "cp": cp, "search": suite["search"], "seeds": suite["seeds"],
            "configuration_sha256": sha256(Path(config_path).read_bytes()).hexdigest(),
            "resolved_suite_sha256": digest(suite), "source_sha256": sources,
            "source_set_sha256": digest(sources),
            "map_sha256": {f["id"]: digest(f["map"]) for f in [*suite["fields"], suite["stress"]]},
            "python": sys.version, "platform": platform.platform(),
            "processor": platform.processor(), "cpu_count": os.cpu_count(),
            "build": {p.name: p.read_text(encoding="utf-8") for p in
                      (ROOT / "build/compile_combat_ai.cmd", ROOT / "build/compile_combat.cmd")
                      if p.exists()},
            "binary_sha256": {p.name: sha256(p.read_bytes()).hexdigest() for p in
                              sorted((ROOT / "build").glob("*combat*"))
                              if p.suffix in (".dll", ".so", ".dylib", ".exe") or
                              p.name in ("combat_ai_model_test", "combat_ai_planner_test")},
            "git_head_before_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "privileged_cp_queries": suite["search"]["privileged_cp_queries"]}
