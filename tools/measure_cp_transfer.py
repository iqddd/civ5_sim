"""Reproducible before/after and factorial ablations of the CP tactical transfer.

Uses the archived 49 inputs. Baseline must be preserved before editing native
sources. All builds are isolated MSVC /O2; instrumented time is never a speed
measurement. Run prepare, build, run, report, then archive.
"""

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from civ5sim import combat_ai as ai
from civ5sim.evaluation import process_memory, write_json
from tools.benchmark_combat_ai import run_bounded
from tools.experiment_combat_ai import load_request, replay_plan

VARIANTS = ("baseline", "common", "area", "combos", "combined")
COUNTERS = ("copies", "applies", "views", "forecasts", "danger_queries",
            "preferred_calls", "pair_checks", "reach_queries", "formed_pairs")
SOURCES = ("combat_ai/combat_ai.cpp", "combat_ai/model.cpp", "combat_ai/planner.cpp",
           "combat_ai/tactical.cpp", "combat_ai/support.cpp", "combat_state.cpp", "geometry.cpp")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one instrumentation site: {old}")
    return text.replace(old, new)


def prepare(out, preserved):
    if (out / "manifest.json").exists():
        raise ValueError("Use a fresh --out")
    baseline = out / "baseline"
    if preserved != baseline:
        shutil.copytree(preserved, baseline)
    before = json.loads((baseline / "manifest.json").read_text())
    for relative, expected in before.items():
        normalized = relative.replace("\\", "/")
        required = normalized.startswith("native/") or normalized in ("build/combat_rules.h", "build/cp_geometry.inc")
        if required and sha(baseline / relative) != expected:
            raise ValueError(f"Changed baseline: {relative}")
    corpus = json.loads((ROOT / "artifacts/combat-ai/experiments-v1/inputs.json").read_text())
    frozen = out / "frozen"
    shutil.copytree(ROOT / "native", frozen / "native")
    for name in ("combat_rules.h", "cp_geometry.inc"):
        shutil.copyfile(ROOT / "build" / name, frozen / name)
    for name, case in corpus.items():
        write_json(out / "inputs" / f"{name}.json", case)
    write_json(out / "manifest.json", {
        "cp_commit": "029e5988b7726299d6446ddddaaf3782a9c46d61",
        "inputs": {n: sha(out / "inputs" / f"{n}.json") for n in corpus},
        "sources": {str(p.relative_to(frozen)): sha(p) for p in frozen.rglob("*") if p.is_file()},
        "baseline_sources": before,
        "variants": {"baseline": "preserved production source before transfer",
                     "common": "new shared CP evaluation; tactical area and all combo construction disabled",
                     "area": "common + CP tactical area/destination filtering",
                     "combos": "common + CP blocker combinations and depth-first finish pairing",
                     "combined": "production: shared evaluation + both mechanisms"},
        "timing": [n for n in corpus if not n.startswith("differential")],
        "differential": [n for n in corpus if n.startswith("differential")],
    })
    print(f"Frozen {len(corpus)} inputs and production sources", flush=True)


def instrument(directory, baseline):
    native = directory / "native/combat_ai"
    for name in ("planner.cpp", "model.cpp", "tactical.cpp", "combat_ai.cpp"):
        p = native / name
        text = '#include "measure.h"\n' + p.read_text()
        if name == "planner.cpp":
            text = replace(text, "    auto child = std::make_unique<TacticalPosition>();",
                           "    COUNT(0); auto child = std::make_unique<TacticalPosition>();")
            marker = "    auto old = view(*child.branch);"
            if marker not in text:
                marker = "  auto old = TacticalView(*child.branch,side);"
            text = replace(text, marker, "    COUNT(1);\n" + marker)
            if baseline:
                text = replace(text, "          auto combo = copy(*first);",
                               "          COUNT(6); auto combo = copy(*first);")
        if name == "model.cpp":
            marker = "    const auto& b = branch(base); unit(b, id);"
            # Baseline has only reachable(), production also has move_intents.
            if baseline:
                text = replace(text, marker, "    COUNT(7);\n" + marker)
                marker = "  int32_t incoming(const Branch& base, int32_t defender, int32_t target) const {"
                text = replace(text, marker, marker + "\n    COUNT(4);")
            else:
                marker = "  std::vector<ReachableTile> move_intents(const Branch& base, int32_t id) const override {"
                text = replace(text, marker, marker + "\n    COUNT(7);")
        if name == "tactical.cpp":
            marker = "TacticalView::TacticalView(const Branch& branch, int32_t side) {"
            text = replace(text, marker, marker + "\n  COUNT(2);")
        if name == "combat_ai.cpp":
            text += '\nextern "C" CAI_API long long transfer_counter(int n) { return counts[n]; }\n'
        p.write_text(text, encoding="utf-8")
    if not baseline:
        p = native / "cp_tactical.cpp"
        text = '#include "measure.h"\n' + p.read_text()
        # Preserve instrumentation of archived sources as well as the current
        # explicit names/formatting. Fail if neither form matches exactly once.
        for alternatives, index in (
            (("  auto value=adapter.forecast(root,attacker,from,defender,to,extra,true,unknown_origin);",
              "  auto value = adapter.forecast(root, attacker, from, defender, to, extra, true, unknown_origin);"), 3),
            (("  std::vector<int> key{unit,tile,self_damage};",
              "  std::vector<int> key{unit, tile, self_damage};"), 4),
            (("  std::vector<CpMove> moves, shots;",), 5),
            (("    for (auto freeing : overall) if (freeing.unit==blocker && freeing.kind==CP_MOVE) {",
              "    for (auto freeing : overall)\n"), 6),
            (("combo.moves.push_back(second);", "combo.moves.push_back(*second);"), 8),
        ):
            matches = [candidate for candidate in alternatives if candidate in text]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous or missing CP instrumentation: {alternatives}")
            marker = matches[0]
            text = replace(text, marker, f"COUNT({index}); " + marker)
        p.write_text(text, encoding="utf-8")
    (directory / "measure.h").write_text(
        "#pragma once\ninline thread_local long long counts[9]={};\n"
        "#ifdef TRANSFER_METRICS\n#define COUNT(n) (++counts[n])\n"
        "#else\n#define COUNT(n) ((void)0)\n#endif\n", encoding="utf-8")


def build(out, variants):
    if os.name != "nt":
        raise RuntimeError("Recorded toolchain: MSVC /O2")
    vswhere = Path("C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe")
    vs = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
                                  "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property",
                                  "installationPath"], text=True).strip()
    for variant in variants:
        directory = out / "variants" / variant
        if directory.exists():
            raise ValueError(f"Already exists: {directory}")
        baseline = variant == "baseline"
        if baseline:
            shutil.copytree(out / "baseline/native", directory / "native")
            for name in ("combat_rules.h", "cp_geometry.inc"):
                shutil.copyfile(out / "baseline/build" / name, directory / name)
        else:
            shutil.copytree(out / "frozen", directory)
            if variant in ("common", "area"):
                p = directory / "native/combat_ai/cp_tactical.cpp"
                s = p.read_text()
                # Older frozen sources guarded blocker combos only. Current
                # sources use the macro for both mechanisms without patching.
                old = "if (depth_first && k+1<overall.size()"
                guarded = "if (CAI_CP_COMBOS && depth_first && k+1<overall.size()"
                if old in s:
                    s = replace(s, old, guarded)
                # Current C++17 source guards the whole pairing block.
                elif guarded not in s and "if constexpr (CAI_CP_COMBOS != 0)" not in s:
                    raise ValueError("Unrecognized CP finish-pairing condition")
                p.write_text(s, encoding="utf-8")
        instrument(directory, baseline)
        sources = [directory / "native" / p for p in SOURCES]
        if not baseline:
            sources.append(directory / "native/combat_ai/cp_tactical.cpp")
        flags = []
        if not baseline:
            flags = [f'/DCAI_CP_AREA={int(variant in ("area", "combined"))}',
                     f'/DCAI_CP_COMBOS={int(variant in ("combos", "combined"))}']
        lines = [f'@cd /d "{directory}"', f'@call "{Path(vs)/"VC/Auxiliary/Build/vcvars64.bat"}" >nul']
        for metrics in (False, True):
            output = "metrics.dll" if metrics else "search.dll"
            options = [*flags, *(["/DTRANSFER_METRICS"] if metrics else [])]
            lines += ['@cl /nologo /std:c++17 /EHsc /O2 /LD /utf-8 ' + " ".join(options)
                      + f' /I"{directory}" /I"{directory/"native/combat_ai"}" '
                      + " ".join(f'"{p}"' for p in sources) + f' /link /OUT:{output}',
                      '@if errorlevel 1 exit /b %errorlevel%']
        (directory / "compile.cmd").write_text("\n".join(lines) + "\n", encoding="utf-8")
        result = run_bounded(["cmd", "/c", str(directory / "compile.cmd")], 120)
        (directory / "compile.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        write_json(directory / "build.json", {
            "search_dll": sha(directory / "search.dll"), "metrics_dll": sha(directory / "metrics.dll"),
            "flags": flags, "sources": {str(p.relative_to(directory)): sha(p) for p in directory.rglob("*.cpp")},
        })
        print(f"Built {variant}", flush=True)


def worker(out, variant, name, repeat, metrics):
    dll = out / "variants" / variant / ("metrics.dll" if metrics else "search.dll")
    ai.LIBRARY_PATH = dll
    ai._library.cache_clear()
    case = json.loads((out / "inputs" / f"{name}.json").read_text())
    config = dict(case["config"])
    config["aggression"] = ai.Aggression(config["aggression"])
    with ai.Planner(ai.SearchConfig(**config)) as planner:
        planner._lib.transfer_counter.restype = ctypes.c_longlong
        before = [planner._lib.transfer_counter(i) for i in range(len(COUNTERS))]
        memory_before = process_memory()
        start, cpu_start = time.perf_counter(), time.process_time()
        result = planner.search(load_request(case["request"]))
        elapsed, cpu = time.perf_counter() - start, time.process_time() - cpu_start
        memory_after = process_memory()
        counts = {key: planner._lib.transfer_counter(i) - before[i] for i, key in enumerate(COUNTERS)}
    write_json(out / "records" / f'{variant}-{name}-{repeat}-{"metrics" if metrics else "clean"}.json', {
        "variant": variant, "case": name, "repeat": repeat, "metrics": metrics,
        "seconds": elapsed, "cpu_seconds": cpu, "result": asdict(result),
        "counters": counts if metrics else None, "validation": replay_plan(case, result),
        "input_sha256": sha(out / "inputs" / f"{name}.json"), "library_sha256": sha(dll),
        "memory_before": memory_before, "memory_after": memory_after,
    })


def run(out, variants, suite, repeats, names):
    manifest = json.loads((out / "manifest.json").read_text())
    selected = names or manifest["differential" if suite == "differential" else "timing"]
    for name in selected:
        if sha(out / "inputs" / f"{name}.json") != manifest["inputs"][name]:
            raise ValueError(f"Input changed: {name}")
    for repeat in range(1 if suite in ("differential", "metrics") else repeats):
        for name in selected:
            order = variants[repeat % len(variants):] + variants[:repeat % len(variants)]
            for variant in order:
                metrics = suite == "metrics"
                record = out / "records" / f'{variant}-{name}-{repeat}-{"metrics" if metrics else "clean"}.json'
                if record.exists():
                    continue
                command = [sys.executable, str(Path(__file__).resolve()), "worker", "--out", str(out),
                           "--variants", variant, "--names", name, "--repeat", str(repeat)]
                if metrics:
                    command.append("--metrics")
                completed = run_bounded(command, 120)
                if completed.returncode:
                    raise RuntimeError(f"{variant}/{name}: {completed.stdout}{completed.stderr}")
                value = json.loads(record.read_text())
                print(f'{variant} {name} {repeat}: {value["seconds"]:.4f}s {value["result"]["reason"]}', flush=True)


def report(out):
    records = [json.loads(p.read_text()) for p in sorted((out / "records").glob("*.json"))]
    groups = {}
    for r in records:
        groups.setdefault((r["case"], r["variant"]), []).append(r)
        if sha(out / "inputs" / f'{r["case"]}.json') != r["input_sha256"]:
            raise ValueError("Input hash mismatch")
        dll = out / "variants" / r["variant"] / ("metrics.dll" if r["metrics"] else "search.dll")
        if sha(dll) != r["library_sha256"]:
            raise ValueError("Library hash mismatch")
    rows = []
    for (name, variant), values in sorted(groups.items()):
        clean = [r for r in values if not r["metrics"]]
        metrics = [r for r in values if r["metrics"]]
        if not clean:
            raise ValueError(f"No clean result: {variant}/{name}")
        result = clean[0]["result"]
        if any(r["result"] != result for r in values):
            raise ValueError(f"Non-determinism or metrics mismatch: {variant}/{name}")
        times = [r["seconds"] for r in clean]
        rows.append({"case": name, "variant": variant, "repeats": len(clean),
                     "median_seconds": statistics.median(times), "range_seconds": [min(times), max(times)],
                     "median_cpu_seconds": statistics.median(r["cpu_seconds"] for r in clean),
                     "result": result, "validation": clean[0]["validation"],
                     "counters": metrics[0]["counters"] if metrics else None})
    write_json(out / "summary.json", {"searches": len(records), "rows": rows})
    lines = ["# Перенос области и комбинаций CP Only", "",
             "MSVC /O2, чистые сборки; время включает упаковку Python→ABI и весь search.",
             "Инструментированные результаты совпадают с чистыми; их время исключено.", "",
             "common — новые общие оценки CP без обоих механизмов; area/combos — независимые включения; combined — рабочий вариант.",
             "Изменения планов и no_plan не являются доказательством ухудшения качества игры.", "",
             "| Вход | Вариант | Повторы | Медиана, с | Диапазон, с | generated | completed | Ответ |",
             "|---|---|---:|---:|---|---:|---:|---|"]
    for row in rows:
        if row["case"].startswith("differential"):
            continue
        stats = row["result"]["stats"]
        lines.append(f'| {row["case"]} | {row["variant"]} | {row["repeats"]} | '
                     f'{row["median_seconds"]:.6f} | {row["range_seconds"]} | '
                     f'{stats["generated"]} | {stats["completed"]} | {row["result"]["reason"]} |')
    lines += ["", f"Всего поисков: {len(records)}. Все непустые планы на входах со snapshot исполнены штатным ядром.",
              "Большой вход distant26 имеет только PlanningState; проверка исполнения к нему неприменима.",
              "copies/applies — одинаковые места копирования/исполнения в обеих версиях. pair_checks имеют разный смысл:",
              "baseline считает проверку второй части с временной копией; новые варианты — просмотр предпочтительного выхода блокировщика.",
              "formed_pairs отдельно считает сформированные пары; views включает построение TacticalView."]
    by_key = {(r["case"], r["variant"]): r for r in rows}
    if all(("distant26", v) in by_key for v in VARIANTS):
        lines += ["", "## Что изменилось", "",
                  "На distant26 все ответы пустые. У baseline исчерпан бюджет 6000; область CP отсекает все движения,",
                  "после чего поиск перебирает BLOCKED. Поэтому здесь ускорение combined поверх area даёт",
                  "объединение завершений в depth-first, а не освобождение занятых клеток.", "",
                  "| Вариант | copies | applies | views | generated |",
                  "|---|---:|---:|---:|---:|"]
        for variant in VARIANTS:
            r = by_key["distant26", variant]
            counts = r["counters"] or {}
            lines.append(f'| {variant} | {counts.get("copies")} | {counts.get("applies")} | '
                         f'{counts.get("views")} | {r["result"]["stats"]["generated"]} |')
        lines += ["", "## Различия ответов", "",
                  "| Вариант | Непустых ответов из 49 |",
                  "|---|---:|"]
        for variant in VARIANTS:
            nonempty = sum(bool(r["result"]["assignments"]) for r in rows if r["variant"] == variant)
            lines.append(f"| {variant} | {nonempty} |")
        lines += ["", "Четыре новых no_plan (differential04/10/26/30) возникают уже в common:",
                  "поиск завершается ниже бюджета. Это изменение общих оценок/отбора CP, не эффект включения",
                  "области или комбинаций. В differential33 при 180 нет завершённого плана во всех новых вариантах.",
                  "Это наблюдения о поиске, не математический эталон качества.", "",
                  "| Этюд | Вариант | Завершённых планов | Урон | Ответный урон |",
                  "|---|---|---:|---:|---:|"]
        for case in ("differential14", "differential33", "differential37"):
            for variant in ("baseline", "combined"):
                r = by_key.get((case, variant))
                if r:
                    lines.append(f'| {case} | {variant} | {r["result"]["stats"]["completed"]} | '
                                 f'{r["validation"].get("damage")} | {r["validation"].get("own_damage")} |')
        probe = out / "budget-probe.json"
        if probe.exists():
            lines += ["", "Отдельная проверка бюджета differential33 (не включена во времена выше):", "",
                      "| Вариант | Пул | generated | completed |",
                      "|---|---:|---:|---:|"]
            for row in json.loads(probe.read_text()):
                stats = row["result"]["stats"]
                lines.append(f'| {row["variant"]} | {row["config"]["position_capacity"]} | '
                             f'{stats["generated"]} | {stats["completed"]} |')
            lines += ["", "Увеличение пула до 360 уже даёт 12 завершённых планов; все непустые ответы этой проверки",
                      "также исполнены ядром. Основной сохранённый вход с бюджетом 180 не изменён."]
        lines += ["", "Версия ABI5 явно использует нулевую отсутствующую историю урона за прошлый ход и offense=0.",
                  "Сила игры и полная приёмка CP-07 этой локальной серией не устанавливаются."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Verified {len(records)} searches; {len(rows)} case/variant groups", flush=True)


def budget_probe(out):
    """Separate changed-budget diagnostics; never include these in timing rows."""
    name = "differential33"
    case = json.loads((out / "inputs" / f"{name}.json").read_text())
    rows = []
    for variant in ("common", "combined"):
        dll = out / "variants" / variant / "search.dll"
        ai.LIBRARY_PATH = dll
        ai._library.cache_clear()
        for capacity in (180, 360, 720, 6000):
            config = {**case["config"], "position_capacity": capacity}
            config["aggression"] = ai.Aggression(config["aggression"])
            with ai.Planner(ai.SearchConfig(**config)) as planner:
                result = planner.search(load_request(case["request"]))
            rows.append({"case": name, "variant": variant, "config": config,
                         "input_sha256": sha(out / "inputs" / f"{name}.json"),
                         "library_sha256": sha(dll), "result": asdict(result),
                         "validation": replay_plan(case, result)})
            print(variant, capacity, result.reason, result.stats.completed, flush=True)
    write_json(out / "budget-probe.json", rows)


def archive(out):
    target = ROOT / "artifacts/combat-ai/cp-transfer-20261102"
    target.mkdir(parents=True, exist_ok=True)
    for name in ("manifest.json", "summary.json", "report.md"):
        shutil.copyfile(out / name, target / name)
    for name in ("budget-probe.json", "review.md"):
        if (out / name).exists():
            shutil.copyfile(out / name, target / name)
    write_json(target / "results.json", [json.loads(p.read_text()) for p in sorted((out / "records").glob("*.json"))])
    write_json(target / "builds.json", {v: json.loads((out / "variants" / v / "build.json").read_text()) for v in VARIANTS})
    for name in ("baseline", "frozen"):
        source = out / name
        shutil.copytree(source / "native", target / name / "native", dirs_exist_ok=True)
        generated = source / "build" if name == "baseline" else source
        destination = target / name / "build" if name == "baseline" else target / name
        destination.mkdir(parents=True, exist_ok=True)
        for filename in ("combat_rules.h", "cp_geometry.inc"):
            shutil.copyfile(generated / filename, destination / filename)
        if name == "baseline":
            shutil.copyfile(source / "manifest.json", target / name / "manifest.json")
            if (source / "tests").exists():
                shutil.copytree(source / "tests", target / name / "tests", dirs_exist_ok=True)
    print(target)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "build", "run", "worker", "report", "budget-probe", "archive"))
    parser.add_argument("--out", type=Path, default=ROOT / "build/cp-transfer-20261102-reviewed")
    parser.add_argument("--baseline", type=Path, default=ROOT / "build/cp-transfer-20261102/baseline")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--suite", choices=("timing", "differential", "metrics"), default="timing")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--metrics", action="store_true")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage == "prepare":
        prepare(out, args.baseline.resolve())
    elif args.stage == "build":
        build(out, args.variants)
    elif args.stage == "run":
        run(out, args.variants, args.suite, args.repeats, args.names)
    elif args.stage == "worker":
        worker(out, args.variants[0], args.names[0], args.repeat, args.metrics)
    elif args.stage == "report":
        report(out)
    elif args.stage == "budget-probe":
        budget_probe(out)
    elif args.stage == "archive":
        archive(out)


if __name__ == "__main__":
    main()
