"""Isolated /O2 timing of the archived ABI5 port and successive ABI6 storage changes.

Historical requests are timing workloads, never plan-quality oracles. Conversion
of missing damage history to zero is explicit and confined to this tool.
"""

import argparse
import ctypes
import gzip
import hashlib
import importlib.util
import json
import platform
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from civ5sim import Battle
from civ5sim import combat_ai as ai
from civ5sim.engine import UNIT_TYPES, digest
from civ5sim.evaluation import process_memory, synthetic_map, write_json
from tools.benchmark_combat_ai import run_bounded
from tools.tactical_corpus import fresh_cases, replay_plan

DEFAULT_OUT = ROOT / "artifacts/combat-ai/storage-transfer-20261102"
ARCHIVE = ROOT / "artifacts/combat-ai/cp-transfer-20261102/baseline"


def version_directory(out, label):
    return ROOT / "build" / (out.name + "-versions") / label


def records_directory(out):
    return ROOT / "build" / (out.name + "-records")


def load_cases(out):
    with gzip.open(out / "inputs.json.gz", "rt", encoding="utf-8") as stream:
        return json.load(stream)


def corpus():
    historical = json.loads((ROOT / "artifacts/combat-ai/experiments-v1/inputs.json").read_text())
    cases = {}
    for name, case in historical.items():
        if name.startswith("differential"):
            continue
        case.pop("snapshot", None)  # Old replay schemas are not reintroduced.
        for unit in case["request"]["state"]["units"]:
            unit["damage_taken_last_turn"] = 0
        case["historical"] = True
        cases["old-" + name] = case
    cases.update(fresh_cases())
    for count in (1, 4, 8, 13):
        for terrain in ("open", "passage", "distant"):
            for fog in (False, True):
                map_value = synthetic_map()
                if terrain == "passage":
                    for y in range(16):
                        if y not in (6, 7):
                            map_value["tiles"][y * 16 + 8]["elevation"] = "mountain"
                own = [{"type": UNIT_TYPES[i % len(UNIT_TYPES)], "side": 0,
                        "tile": [2 + i // 5, 4 + i % 5]} for i in range(count)]
                enemy_x = 13 if terrain == "distant" else 7
                enemies = [{"type": UNIT_TYPES[i % len(UNIT_TYPES)], "side": 1,
                            "tile": [enemy_x, 4 + i]} for i in range(4)]
                scenario = {"schema": 1, "kind": "civ5sim.scenario", "map": map_value,
                            "fog": fog, "random_damage": False, "seed": 42,
                            "units": own + enemies}
                with Battle(scenario) as battle:
                    state = ai.PlanningState.from_battle(battle, 0)
                    request = ai.SearchRequest(state, 0, state.turn,
                                               enemies[0]["tile"][0] + 16 * enemies[0]["tile"][1],
                                               tuple(range(count)),
                                               ai.ContactMemory().updated(battle.observe(0)))
                    cases[f"group{count}-{terrain}-fog{int(fog)}"] = {
                        "request": asdict(request), "snapshot": battle.snapshot(),
                        "config": asdict(ai.SearchConfig(position_capacity=6000, max_completed=2560))}
    return cases


def freeze(out, label):
    directory = version_directory(out, label)
    if directory.exists():
        raise ValueError(f"Version already frozen: {label}")
    source = ARCHIVE if label == "historical" else ROOT
    shutil.copytree(source / "native", directory / "native", dirs_exist_ok=True)
    for name in ("combat_rules.h", "cp_geometry.inc"):
        shutil.copyfile(source / "build" / name, directory / name)
    write_json(directory / "manifest.json", {
        "git": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "source": "archived pre-transfer ABI5 port" if label == "historical" else "working native sources",
        "files": {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in directory.rglob("*") if p.is_file()},
    })
    vswhere = Path("C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe")
    vs = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
                                  "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property",
                                  "installationPath"], text=True).strip()
    names = ["combat_ai/combat_ai.cpp", "combat_ai/model.cpp", "combat_ai/planner.cpp",
             "combat_ai/tactical.cpp", "combat_ai/support.cpp", "combat_state.cpp", "geometry.cpp"]
    if label != "historical":
        names.append("combat_ai/cp_tactical.cpp")
    command = (f'@cd /d "{directory}"\n@call "{vs}/VC/Auxiliary/Build/vcvars64.bat" >nul\n'
               '@cl /nologo /std:c++17 /EHsc /O2 /LD /utf-8 /W4 '
               f'/I"{directory}" /I"{directory / "native/combat_ai"}" '
               + " ".join(f'"{directory / "native" / name}"' for name in names)
               + ' /link /OUT:search.dll\n@exit /b %errorlevel%\n')
    (directory / "compile.cmd").write_text(command, encoding="utf-8")
    result = run_bounded(["cmd", "/c", str(directory / "compile.cmd")], 120)
    (directory / "compile.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    shutil.copyfile(directory / "manifest.json", out / f"{label}-manifest.json")
    if label == "historical":
        (directory / "legacy_binding.py").write_bytes(subprocess.check_output(
            ["git", "show", "376d01b:civ5sim/combat_ai.py"], cwd=ROOT))
    print(f"Frozen and built {label}", flush=True)


def binding(out, label):
    label = label.removesuffix("-cold")
    if label == "historical":
        spec = importlib.util.spec_from_file_location("civ5sim._storage_legacy", version_directory(out, label) / "legacy_binding.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    else:
        module = ai
    module.LIBRARY_PATH = version_directory(out, label) / "search.dll"
    module._library.cache_clear()
    return module


def metrics_build(out, label):
    source = version_directory(out, label)
    directory = version_directory(out, label + "-metrics")
    shutil.copytree(source / "native", directory / "native", dirs_exist_ok=True)
    for name in ("combat_rules.h", "cp_geometry.inc"):
        shutil.copyfile(source / name, directory / name)
    header = ROOT / "native/combat_ai/storage_metrics.h"
    shutil.copyfile(header, directory / "native/combat_ai/storage_metrics.h")
    if label == "before":
        sites = {
            "model.cpp": [("    auto copy = std::make_unique<OwnedBranch>(*this);", "BranchAllocation"),
                          ("  std::vector<ReachableTile> move_intents(const Branch& base, int32_t id) const override {", "ReachMiss")],
            "cp_tactical.cpp": [("CpBoard CpTactics::board(const TacticalPosition &position) const {", "BoardRebuild")],
            "planner.cpp": [("    auto child = std::make_unique<TacticalPosition>();", "PositionCopy")],
        }
        for name, markers in sites.items():
            path = directory / "native/combat_ai" / name
            text = '#include "storage_metrics.h"\n' + path.read_text()
            for marker, counter in markers:
                if text.count(marker) != 1:
                    raise ValueError(f"Instrumentation site changed: {marker}")
                text = text.replace(marker, marker + f"\n    CAI_STORAGE_COUNT({counter});")
            path.write_text(text, encoding="utf-8")
    exports = directory / "metrics_exports.cpp"
    exports.write_text('#include "combat_ai.h"\n#include "storage_metrics.h"\n'
                       'extern "C" CAI_API long long cai_storage_metric(int index) {\n'
                       'return index >= 0 && index < combat_ai::StorageMetricCount ? '
                       'combat_ai::storage_metrics[index] : -1; }\n', encoding="utf-8")
    command = (source / "compile.cmd").read_text()
    # Earlier versions may have been frozen under the artifact directory.
    original = command.splitlines()[0].split('"')[1]
    command = command.replace(original, str(directory))
    command = command.replace(" /link /OUT:search.dll", f' /DCAI_STORAGE_METRICS "{exports}" /link /OUT:search.dll')
    (directory / "compile.cmd").write_text(command, encoding="utf-8")
    result = run_bounded(["cmd", "/c", str(directory / "compile.cmd")], 120)
    (directory / "compile.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    print(f"Built separate metrics DLL: {label}", flush=True)


def request(module, value):
    value = json.loads(json.dumps(value))
    state = value.pop("state")
    if module is not ai:
        for unit in state["units"]:
            unit.pop("damage_taken_last_turn")
    state["tiles"] = tuple(module.PlanningTile(**tile) for tile in state["tiles"])
    state["units"] = tuple(module.PlanningUnit(**unit) for unit in state["units"])
    state["visibility"] = tuple(state["visibility"])
    value["state"] = module.PlanningState(**state)
    contacts = value.pop("contacts")
    contacts["known_enemies"] = tuple(module.Contact(**c) for c in contacts["known_enemies"])
    value["contacts"] = module.ContactMemory(**contacts)
    value["selected_units"] = tuple(value["selected_units"])
    if value["aggression"] is not None:
        value["aggression"] = module.Aggression(value["aggression"])
    return module.SearchRequest(**value)


def worker(out, labels, name, repeat):
    case = load_cases(out)[name]
    rows = []
    for label in labels:
        if label == "historical" and not case.get("historical"):
            continue
        module = binding(out, label)
        config = dict(case["config"])
        config["aggression"] = module.Aggression(config["aggression"])
        input_request = request(module, case["request"])
        before = digest(asdict(input_request))
        with module.Planner(module.SearchConfig(**config)) as planner:
            first_start = time.perf_counter()
            warm = planner.search(input_request)
            first_seconds = time.perf_counter() - first_start
            counter_names = ["reach_hit", "reach_miss", "range_hit", "range_miss", "board_rebuild",
                             "board_update", "branch_allocation", "branch_reuse", "position_copy"]
            metrics = label.endswith("-metrics")
            if metrics:
                planner._lib.cai_storage_metric.restype = ctypes.c_longlong
                counts_before = [planner._lib.cai_storage_metric(i) for i in range(len(counter_names))]
            start, cpu_start = time.perf_counter(), time.process_time()
            result = planner.search(input_request)
            elapsed, cpu = time.perf_counter() - start, time.process_time() - cpu_start
            version = planner.version
            counts = ({key: planner._lib.cai_storage_metric(i) - counts_before[i]
                       for i, key in enumerate(counter_names)} if metrics else None)
        if asdict(warm) != asdict(result) or before != digest(asdict(input_request)):
            raise ValueError("Non-deterministic search or mutated request")
        validation = replay_plan(case, result) if "snapshot" in case else {"checked": False, "reason": "historical physical request only"}
        rows.append({"label": label, "case": name, "repeat": repeat, "seconds": elapsed,
                     "cpu_seconds": cpu, "result": asdict(result), "validation": validation,
                     "memory": process_memory(), "version": version,
                     "first_call_seconds": first_seconds})
        if counts is not None:
            rows[-1]["counters"] = counts
    write_json(records_directory(out) / f"{name}-{repeat}-{'-'.join(labels)}.json", rows)


def run(out, labels, names, repeats):
    cases = load_cases(out)
    for repeat in range(repeats):
        order = labels[repeat % len(labels):] + labels[:repeat % len(labels)]
        for name in names or cases:
            target = records_directory(out) / f"{name}-{repeat}-{'-'.join(order)}.json"
            if target.exists():
                continue
            try:
                result = run_bounded([sys.executable, str(Path(__file__).resolve()), "worker",
                                      "--out", str(out), "--labels", *order, "--names", name,
                                      "--repeat", str(repeat)], 120 * len(order) + 10)
            except subprocess.TimeoutExpired:
                write_json(target, [{"case": name, "label": label, "repeat": repeat,
                                     "timeout": True, "worker_seconds_limit": 120 * len(order) + 10}
                                    for label in order])
                continue
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
        print(f"Measured batch {repeat + 1}/{repeats}: {','.join(labels)}", flush=True)


def report(out):
    grouped = {}
    raw = []
    timeouts = []
    for path in sorted(records_directory(out).glob("*.json")):
        for row in json.loads(path.read_text()):
            raw.append(row)
            if row.get("timeout"):
                timeouts.append(row)
                continue
            grouped.setdefault((row["case"], row["label"]), []).append(row)
    summary = []
    for (name, label), rows in sorted(grouped.items()):
        if any(row["result"] != rows[0]["result"] for row in rows):
            raise ValueError(f"Non-deterministic {name}/{label}")
        times = [row["seconds"] for row in rows]
        summary.append({"case": name, "label": label, "repeats": len(rows),
                        "median": statistics.median(times), "min": min(times), "max": max(times),
                        "cpu_median": statistics.median(row["cpu_seconds"] for row in rows),
                        "process_peak_rss_median": statistics.median(row["memory"]["peak_rss_bytes"] for row in rows),
                        "first_call_median": (statistics.median(row["first_call_seconds"] for row in rows)
                                              if all("first_call_seconds" in row for row in rows) else None),
                        "result_sha256": digest(rows[0]["result"]),
                        "reason": rows[0]["result"]["reason"], "stats": rows[0]["result"]["stats"]})
    with gzip.open(out / "summary.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(summary, stream, separators=(",", ":"))
    write_json(out / "timeouts.json", timeouts)
    with gzip.open(out / "raw-results.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(raw, stream, separators=(",", ":"))
    labels = [label for label in ("historical", "before", "caches", "boards", "compact", "pool", "final")
              if any(row["label"] == label for row in summary)]
    by_case = {}
    for row in summary:
        by_case.setdefault(row["case"], {})[row["label"]] = row
    changes = {label: [name for name, versions in by_case.items()
                       if label in versions and versions[label]["result_sha256"] != versions["before"]["result_sha256"]]
               for label in labels if label not in ("historical", "before")}
    diagnostic_changes = {label: [name for name, versions in by_case.items()
                                 if label in versions and versions[label]["result_sha256"] !=
                                 versions[label.removesuffix("-metrics").removesuffix("-cold")]["result_sha256"]]
                          for label in ("before-metrics", "pool-metrics", "final-metrics", "before-cold", "pool-cold", "final-cold")
                          if any(label in versions for versions in by_case.values())}
    if any(diagnostic_changes.values()):
        raise ValueError(f"Diagnostic build changed search: {diagnostic_changes}")
    write_json(out / "equivalence.json", {"changed_full_results": changes,
                                          "diagnostic_changes": diagnostic_changes,
                                          "interpretation": "storage compatibility, not plan optimality"})
    last_label = "final" if "final" in labels else "pool"
    lines = ["# CP tactical storage transfer measurements", "",
             "MSVC 14.44 x64, C++17 /O2; one warmup and five measured calls per case/version.",
             "Serial workers; request construction, compilation, DLL loading and replay excluded.",
             "Times include Python ABI marshaling, native adapter/geometry setup and search.",
             "Historical ABI5 inputs explicitly use zero damage history in ABI6; current offense=5.",
             "Historical comparison includes heuristic changes; it is not a storage-only speedup.",
             "Metrics DLLs are separate and excluded from timing comparisons.", "",
             "process_peak_rss_median is the whole worker's high-water RSS, after replay and preceding labels; it is not per-version memory consumption.", "",
             "Full-result differences from before: " + json.dumps(changes), "",
             "| Case | " + " | ".join(labels) + f" | before/{last_label} |",
             "|---|" + "---:|" * (len(labels) + 1)]
    for name, versions in sorted(by_case.items()):
        cells = [f"{versions[label]['median']:.6f}" if label in versions else "—" for label in labels]
        speed = (f"{versions['before']['median'] / versions[last_label]['median']:.2f}×"
                 if last_label in versions else "—")
        lines.append("| " + name + " | " + " | ".join(cells) + " | " + speed + " |")
    cold_label = ("final" if any("final" in v and v["final"]["first_call_median"] is not None
                                 for v in by_case.values()) else "pool-cold")
    cold = [(name, versions) for name, versions in sorted(by_case.items())
            if "before-cold" in versions and cold_label in versions]
    if cold:
        lines += ["", "## First call in a new Planner (five repeats, distinct from warm-call latency)", "",
                  "Includes native pool allocation, excludes DLL loading and Python request construction.",
                  f"| Case | before | {cold_label} | before/{cold_label} |", "|---|---:|---:|---:|"]
        for name, versions in cold:
            before = versions["before-cold"]["first_call_median"]
            after = versions[cold_label]["first_call_median"]
            lines.append(f"| {name} | {before:.6f} | {after:.6f} | {before / after:.2f}× |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(out / "environment.json", {"platform": platform.uname()._asdict(),
                                           "python": sys.version, "source_commit_before": "b6ace93",
                                           "cp_commit": "029e5988b7726299d6446ddddaaf3782a9c46d61"})
    write_json(out / "inputs-manifest.json", {
        "sha256_gzip": hashlib.sha256((out / "inputs.json.gz").read_bytes()).hexdigest(),
        "case_sha256": {name: digest(case) for name, case in load_cases(out).items()},
    })
    snapshots = {}
    for label in labels:
        directory = version_directory(out, label)
        source_files = [*sorted((directory / "native").rglob("*.h")),
                        *sorted((directory / "native").rglob("*.cpp")),
                        directory / "combat_rules.h", directory / "cp_geometry.inc"]
        snapshots[label] = {str(p.relative_to(directory)): p.read_bytes().decode("utf-8")
                            for p in source_files}
        write_json(out / f"{label}-manifest.json", {
            "metadata": json.loads((directory / "manifest.json").read_text()),
            "sha256": {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [*source_files, directory / "search.dll"]},
            "command": (directory / "compile.cmd").read_text(),
        })
    with gzip.open(out / "source-snapshots.json.gz", "wt", encoding="utf-8") as stream:
        json.dump(snapshots, stream, separators=(",", ":"))
    print(f"Summarized {len(summary)} case/version pairs")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("command", choices=("prepare", "freeze", "metrics-build", "run", "worker", "report"))
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--labels", nargs="+", default=["before"])
    parser.add_argument("--names", nargs="*")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=0)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.command == "prepare":
        if (out / "inputs.json.gz").exists():
            raise ValueError("Inputs already frozen")
        out.mkdir(parents=True, exist_ok=True)
        with gzip.open(out / "inputs.json.gz", "wt", encoding="utf-8") as stream:
            json.dump(corpus(), stream, separators=(",", ":"))
    elif args.command == "freeze":
        for label in args.labels:
            freeze(out, label)
    elif args.command == "metrics-build":
        for label in args.labels:
            metrics_build(out, label)
    elif args.command == "worker":
        worker(out, args.labels, args.names[0], args.repeat)
    elif args.command == "run":
        run(out, args.labels, args.names, args.repeats)
    else:
        report(out)


if __name__ == "__main__":
    main()
