"""Archive and compare the bounded coordinator target experiment against 0.9.3."""

import argparse
import gzip
import json
import shutil
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from civ5sim.evaluation import load_suite, write_json
from tools.benchmark_combat_ai import run_bounded
from tools.archive.publish_greedy_replays import compressed, read


def prepare(baseline, archive):
    if archive.exists():
        raise ValueError("Refusing to overwrite an existing experiment archive")
    summary = read(baseline / "summary.json")
    records = summary["records"]
    if not summary["scope_complete"] or len(records) != 64 or summary["failures"]:
        raise ValueError("Expected the complete verified 64-case baseline")
    archive.mkdir(parents=True)
    compressed(archive / "before-records.json.gz", records)
    compressed(archive / "before-matches.json.gz", {
        r["case"]["id"]: {name: read(baseline / "matches" / r["case"]["id"] / "0" / name)
                          for name in ("replay.json", "result.json", "checked.json")}
        for r in records})
    rows = []
    for r in records:
        case, result = r["case"], r["result"]
        side = case["agents"].index("cp")
        rows.append({"case": case["id"], "cp_side": side,
                     "outcome": outcome(r), "seed": case["config"]["seed"],
                     "fog": case["config"]["fog"], "random_damage": case["config"]["random_damage"],
                     "remaining_hp": result["remaining_hp"][side],
                     "remaining_units": result["alive"][side], "result": result})
    write_json(archive / "before-results.json", rows)
    write_json(archive / "before-summary.json", {k: v for k, v in summary.items() if k != "records"})
    shutil.copyfile(baseline / "passport.json", archive / "before-passport.json")
    write_json(archive / "change.json", {"description":
        "0.9.4: координатор сравнивает ближайшую цель и одну цель с более многочисленной/компактной "
        "группой по физическому результату до первого нового контакта. Тактические эвристики CP не менялись."})


def outcome(record):
    winner = record["result"]["winner"]
    return "draw" if winner is None else "win" if winner == record["case"]["agents"].index("cp") else "loss"


def cp_seconds(record):
    side = record["case"]["agents"].index("cp")
    return sum(s["ns"] for s in record["metrics"]["samples"] if s.get("side") == side
               and (s["kind"].startswith("prepare_") or s["kind"] in ("act", "callback"))) / 1e9


def control_timings(names, baseline_source, archive, before, after):
    """Sequential paired runs only for apparent >2x cases; no publication."""
    controls = read(archive / "timing-controls.json") if (archive / "timing-controls.json").exists() else {}
    for name in names:
        if name in controls:
            continue
        samples = []
        # Three alternating pairs reduce scheduling noise; each run is verified.
        for repeat in range(3):
            pair = {}
            for label, source, reference in (("baseline", baseline_source, before[name]),
                                             ("experiment", ROOT, after[name])):
                directory = ROOT / "build/target-timing-controls" / name / f"{label}-{repeat}"
                directory.mkdir(parents=True, exist_ok=True)
                config = source / "scenarios/combat-ai-v1.json"
                timeout = load_suite(config)["timeout_seconds"]
                command = [sys.executable, str(source / "tools/benchmark_combat_ai.py"),
                           "--config", str(config), "--out", str(directory), "--worker", name,
                           "--signature", "timing-control"]
                completed = run_bounded(command, timeout)
                if completed.returncode:
                    raise RuntimeError(completed.stdout + completed.stderr)
                if read(directory / "checked.json")["deterministic_sha256"] != reference["checked"]["deterministic_sha256"]:
                    raise AssertionError(f"Timing control changed deterministic behavior: {name}/{label}")
                pair[label] = read(directory / "metrics.json")["match_seconds"]
            samples.append(pair)
            print(f"Timing {name} pair {repeat+1}: {pair}", flush=True)
        controls[name] = {"samples": samples,
                          "ratio": median(p["experiment"] for p in samples) / median(p["baseline"] for p in samples)}
        write_json(archive / "timing-controls.json", controls)
    return controls


def compare(batch, archive, baseline_source, *, controls=False):
    with gzip.open(archive / "before-records.json.gz", "rt", encoding="utf-8") as stream:
        before = {r["case"]["id"]: r for r in json.load(stream)}
    summary = read(batch / "summary.json")
    if not summary["scope_complete"] or summary["failures"] or summary["repeats"] != 1:
        raise ValueError("Experiment has not completed cleanly")
    after = {r["case"]["id"]: r for r in summary["records"]}
    if before.keys() != after.keys():
        raise ValueError("Case sets differ")
    suspicious = [name for name in before if after[name]["metrics"]["match_seconds"] >
                  2 * before[name]["metrics"]["match_seconds"]]
    timing = control_timings(suspicious, baseline_source, archive, before, after) if controls else {}
    comparisons = []
    choices = probes = switches = 0
    for name, old in before.items():
        new = after[name]
        side = new["case"]["agents"].index("cp")
        if old["result"]["initial_state_sha256"] != new["result"]["initial_state_sha256"]:
            raise ValueError(f"Initial position or RNG changed: {name}")
        old_time, new_time = old["metrics"]["match_seconds"], new["metrics"]["match_seconds"]
        counters = lambda r, side=side: r["checked"]["diagnostics"][side]
        row = {"case": name, "before": outcome(old), "after": outcome(new),
               "before_hp": old["result"]["remaining_hp"], "after_hp": new["result"]["remaining_hp"],
               "before_units": old["result"]["alive"], "after_units": new["result"]["alive"],
               "before_commands": old["result"]["commands"], "after_commands": new["result"]["commands"],
               "before_match_seconds": old_time, "after_match_seconds": new_time,
               "raw_time_ratio": new_time / old_time,
               "verified_time_ratio": timing.get(name, {}).get("ratio", new_time / old_time),
               "timing_controlled": name in timing,
               "before_cp_seconds": cp_seconds(old), "after_cp_seconds": cp_seconds(new),
               "before_searches": counters(old)["searches"], "after_searches": counters(new)["searches"],
               "before_generated": counters(old)["positions"]["generated"],
               "after_generated": counters(new)["positions"]["generated"]}
        comparisons.append(row)
        replay = read(batch / "matches" / name / "0/replay.json")
        events = [e for e in replay["ai_events"] if e["side"] == side]
        choices += sum(e["event"] == "target_comparison" and len(e["candidates"]) == 2 for e in events)
        switches += sum(e["event"] == "target_comparison" and e["chosen_index"] == 1 for e in events)
        probes += sum(e["event"] == "search" and e.get("probe", False) for e in events)
    points = lambda records: sum({"win": 1, "draw": .5, "loss": 0}[outcome(r)] for r in records.values())
    exceeded = [r["case"] for r in comparisons if r["verified_time_ratio"] > 2]
    verification = {"baseline_commit": "6b7746d", "cases": 64, "repeats": 1,
                    "before_points": points(before), "after_points": points(after),
                    "full_match_time_ratio": sum(r["after_match_seconds"] for r in comparisons) /
                                             sum(r["before_match_seconds"] for r in comparisons),
                    "over_2x_cases": exceeded, "timing_controls_complete": controls or not suspicious,
                    "target_comparisons": choices, "alternative_wins": switches, "additional_searches": probes,
                    "cp_time_ratio": sum(r["after_cp_seconds"] for r in comparisons) /
                                     sum(r["before_cp_seconds"] for r in comparisons),
                    "max_match_time_ratio": max(r["verified_time_ratio"] for r in comparisons),
                    "controlled_cases": len(timing),
                    "success": points(after) > points(before) and not exceeded,
                    "full_cp07_acceptance": False}
    write_json(archive / "comparison.json", {"summary": verification, "cases": comparisons})
    lines = ["# Выбор целей координатора: сравнение с 0.9.3", "",
             (f"Очки CP: {points(before)} → {points(after)}. Дополнительных поисков: {probes}; "
              f"сравнений двух целей: {choices}; выбрана альтернатива: {switches}."),
             (f"Суммарное время матчей: ×{verification['full_match_time_ratio']:.3f}; "
              f"случаев выше ×2: {len(exceeded)}; контрольных случаев: {len(timing)}."),
             ("Сравнение не доказывает оптимальность планов. Изменён только координатор; "
              "64 случая выполнены один раз, контрольные повторы служат только проверке времени."), "",
             "| Матч | До | После | HP A/B до → после | Поиски до → после | Время, × | Контроль |",
             "|---|---|---|---|---:|---:|---|"]
    for r in comparisons:
        lines.append(f"| {r['case']} | {r['before']} | {r['after']} | {r['before_hp']} → {r['after_hp']} | "
                     f"{r['before_searches']} → {r['after_searches']} | {r['verified_time_ratio']:.3f} | "
                     f"{'да' if r['timing_controlled'] else 'нет'} |")
    lines += ["", ("Исходы, составы и команды каждого изменившегося матча доступны в сохранённых "
              "before/after-matches.json.gz и на веб-панели. Полные показатели времени CP, "
              "числа позиций, команд и оставшихся сил находятся в comparison.json.")]
    (archive / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "build/cp07-greedy-defender-wounds")
    parser.add_argument("--baseline-source", type=Path, default=ROOT / "build/target-selection-baseline")
    parser.add_argument("--batch", type=Path, default=ROOT / "build/cp07-greedy-target-selection")
    parser.add_argument("--archive", type=Path, default=ROOT / "artifacts/combat-ai/greedy-target-selection-20261102")
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--control-timings", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.baseline, args.archive)
    else:
        compare(args.batch, args.archive, args.baseline_source, controls=args.control_timings)
