"""Isolated CP-07 experiments: frozen inputs, standalone changes and combinations.

All generated C++/DLLs/results live under --out; production code is read only.
Run prepare, build, run, report in order. Worker processes have tree timeouts.
"""

import argparse
import ctypes
import hashlib
import json
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from civ5sim import Action, Battle
from civ5sim import combat_ai as ai
from civ5sim.engine import digest
from civ5sim.evaluation import process_memory, write_json
from tools.benchmark_combat_ai import run_bounded

EXACT = {
    "base", "local", "distance", "view", "clone", "reach", "small", "hash", "coverage",
    "local-distance", "local-distance-view", "local-distance-view-reach",
    "distance-view",
    "local-distance-view-reach-clone-small-hash",
}
ALGORITHM = {"area", "area-strict", "combos", "combos-cp", "bundle", "area-strict-bundle"}
COUNTERS = ("copy", "apply", "candidates", "combo", "unsafe", "no_improvement", "complete", "view")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protected():
    paths = list((ROOT / "native").rglob("*")) + list((ROOT / "civ5sim").glob("*.py"))
    paths += [ROOT / "build" / p for p in ("combat_ai.dll", "combat.dll", "combat_rules.h", "cp_geometry.inc")]
    return {str(p): sha(p) for p in paths if p.is_file()}


def request(battle, side, target, selected, memory, aggression=ai.Aggression.LOW):
    return asdict(ai.SearchRequest(ai.PlanningState.from_battle(battle, side), side,
                                  battle.observe(side)["turn"], target, tuple(selected), memory, aggression))


def from_replay(replay, command, side, target, selected, aggression):
    memories = [ai.ContactMemory(), ai.ContactMemory()]
    with Battle(replay["initial"]) as b:
        for s in (0, 1):
            memories[s] = memories[s].updated(b.observe(s))
        for i, step in enumerate(replay["steps"][:command], 1):
            result = b.step(step["action"])
            if asdict(result) != step["result"] or digest(b.snapshot()) != step["state_sha256"]:
                raise ValueError(f"Replay mismatch {i}")
            for s in (0, 1):
                memories[s] = memories[s].updated(b.observe(s))
        snapshot = b.snapshot()
        snapshot["state"]["random_damage"] = False
        return {"request": request(b, side, target, selected, memories[side], aggression),
                "snapshot": snapshot, "config": asdict(ai.SearchConfig())}


def load_request(value):
    value = json.loads(json.dumps(value))
    s = value.pop("state")
    s["tiles"] = tuple(ai.PlanningTile(**x) for x in s["tiles"])
    s["units"] = tuple(ai.PlanningUnit(**x) for x in s["units"])
    s["visibility"] = tuple(s["visibility"])
    value["state"] = ai.PlanningState(**s)
    c = value.pop("contacts")
    c["known_enemies"] = tuple(ai.Contact(**x) for x in c["known_enemies"])
    value["contacts"] = ai.ContactMemory(**c)
    value["selected_units"] = tuple(value["selected_units"])
    value["aggression"] = ai.Aggression(value["aggression"])
    return ai.SearchRequest(**value)


def freeze_corpus(out, before, corpus):
    for name, value in corpus.items():
        write_json(out / "inputs" / f"{name}.json", value)
    manifest = {"schema": 1, "protected": before,
                "inputs": {name: sha(out / "inputs" / f"{name}.json") for name in corpus},
                "timing": [n for n in corpus if not n.startswith("differential")],
                "differential": [n for n in corpus if n.startswith("differential")],
                "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
    assert before == protected()
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"timing": manifest["timing"], "differential": len(manifest["differential"])}))


def prepare(out, archived=None, source_root=None):
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise ValueError("Use the existing frozen corpus or a fresh --out")
    before = protected()
    source_root = source_root or ROOT / "artifacts/combat-ai/cp-transfer-20261102/baseline"
    if not (source_root / "native").exists():
        raise ValueError("Historical variants need --source-root with the pre-transfer native sources")
    shutil.copytree(source_root / "native", out / "frozen/native", dirs_exist_ok=True)
    if (source_root / "tests").exists():
        shutil.copytree(source_root / "tests", out / "frozen/tests", dirs_exist_ok=True)
    for name in ("combat_rules.h", "cp_geometry.inc"):
        shutil.copyfile(source_root / "build" / name, out / "frozen" / name)
    if archived is not None and archived.exists():
        freeze_corpus(out, before, json.loads(archived.read_text(encoding="utf-8")))
        return
    corpus = {}
    replay = json.loads((ROOT / "tests/fixtures/cp07_contact.json").read_text(encoding="utf-8"))
    for name, command, target, selected in (
        ("contact036", 35, 6, (7,)), ("unsafe051", 50, 23, (7, 9)),
        ("solo052", 51, 54, (9,)), ("cover065", 64, 86, (9, 5, 8)),
    ):
        corpus[name] = from_replay(replay, command, 1, target, selected, ai.Aggression.LOW)
    old = ROOT / "build/cp07-search-audit-20261102/big.json"
    corpus["distant26"] = {"request": json.loads(old.read_text()), "snapshot": None,
                            "config": asdict(ai.SearchConfig())}
    # Saved real requests on terrain, fog and 26-unit armies. No new policy run.
    for prefix in ("passage-5-s17-f0-r0-cp-cp", "crop01-10-s17-f1-r0-cp-cp",
                   "crop04-14-s17-f0-r0-cp-cp", "open-26-s17-f0-r0-cp-cp"):
        path = ROOT / "build/combat-ai-v1-final/matches" / prefix / "0/replay.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text())
        options = [e for e in r["ai_events"] if e["event"] == "search" and
                   1 < len(e["selected_units"]) <= 6 and 30 <= e["stats"]["generated"] <= 1500]
        if not options:
            continue
        e = options[0]
        corpus[prefix.split("-s17")[0]] = from_replay(
            r, e["command"], e["side"], e["target"], e["selected_units"], ai.Aggression.LOW)
    # Differential corpus: wrapped seam, both row parities, rough terrain,
    # partially unavailable units, wounds, varying aggression and visibility.
    rng = random.Random(20261102)
    for n in range(40):
        w, h = 8, 6
        tiles = [{"terrain": "grass", "elevation": "flat", "feature": None,
                  "rivers": [-1, -1, -1]} for _ in range(w*h)]
        positions = rng.sample(range(w*h), 6)
        for tile in rng.sample(range(w*h), 12):
            tiles[tile]["elevation"] = "hills"
        m = {"schema": 1, "width": w, "height": h, "row_parity": n % 2,
             "wrap_x": bool(n % 3 == 0), "metadata": {}, "tiles": tiles}
        units = []
        for i, tile in enumerate(positions):
            kind = ("warrior", "archer", "chariot_archer", "spearman", "horseman")[(n+i) % 5]
            extra = {"waiting": True} if i == 2 and n % 4 == 0 else {}
            units.append({"type": kind, "side": i//3, "tile": tile,
                          "hp": rng.choice((25, 60, 100)), **extra})
        scenario = {"schema": 1, "kind": "civ5sim.scenario", "map": m, "units": units,
                    "fog": bool(n % 2), "random_damage": False, "seed": n}
        with Battle(scenario) as b:
            memory = ai.ContactMemory().updated(b.observe(0))
            visible = [u for u in b.observe(0)["units"] if u["side"] == 1]
            target = visible[0]["tile"] if visible else positions[3]
            selected = (0, 1) if n % 5 == 0 else (0, 1, 2)
            corpus[f"differential{n:02}"] = {
                "request": request(b, 0, target, selected, memory, ai.Aggression(n % 4 + 1)),
                "snapshot": b.snapshot(),
                "config": asdict(ai.SearchConfig(position_capacity=180, max_completed=12)),
            }
    freeze_corpus(out, before, corpus)


def change(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Expected one occurrence: {old[:100]!r}; got {text.count(old)}")
    return text.replace(old, new)


def transform(directory, variant):
    """Every variant is generated afresh from the same frozen production source."""
    flags = set(variant.split("-"))
    if variant == "combos-cp":
        from tools.combat_combos_variant import transform as blocker_combos
        blocker_combos(directory)
    native = directory / "native"
    paths = {"planner": native/"combat_ai/planner.cpp", "model": native/"combat_ai/model.cpp",
             "header": native/"combat_ai/model.h", "workspace": native/"combat_ai/planner.h"}
    texts = {k: p.read_text(encoding="utf-8") for k, p in paths.items()}
    p, m, h, w = (texts[k] for k in ("planner", "model", "header", "workspace"))
    if "local" in flags:
        p = change(p, "      for (int ours : selected)\n        if (request.state->units[ours].moves > 0 && !request.state->units[ours].waiting &&\n            adapter->distance(v.units[ours].tile, v.units[id].tile) <= 4) local = true;",
                   "      for (int ours : selected) {\n        if (local) break;\n        if (request.state->units[ours].moves > 0 && !request.state->units[ours].waiting &&\n            adapter->distance(v.units[ours].tile, v.units[id].tile) <= 4) local = true;\n      }")
    if "distance" in flags:
        m = change(m, "class Adapter final : public RulesAdapter {",
                   "class Adapter final : public RulesAdapter {\n  std::vector<uint16_t> distances;")
        needle = "    map_tiles = std::make_shared<const std::vector<CaiTile>>(s.tiles, s.tiles + s.tile_count);"
        m = change(m, needle, needle + "\n    if(s.tile_count<=256) {\n      distances.resize(size_t(s.tile_count)*s.tile_count);\n      for(int a=0;a<s.tile_count;++a) for(int b=0;b<s.tile_count;++b)\n        distances[size_t(a)*s.tile_count+b]=uint16_t(geo->distance(a,b));\n    }")
        m = change(m, "return geo->distance(a,b); }", "return distances.empty() ? geo->distance(a,b) : distances[size_t(a)*geo->tiles.size()+b]; }")
    if "view" in flags:
        h = change(h, "class Branch {", "struct TacticalView;\nclass Branch {")
        h = change(h, "  virtual ~Branch() = default;", "  mutable std::shared_ptr<const TacticalView> experiment_view;\n  virtual ~Branch() = default;")
        p = change(p, "    TacticalView result(branch, request.side);\n    result.enemies = local_enemies(result);\n    return result;",
                   "    if(!branch.experiment_view) {\n      auto result=std::make_shared<TacticalView>(branch,request.side);\n      result->enemies=local_enemies(*result);\n      branch.experiment_view=result;\n    }\n    return *branch.experiment_view;")
        m = change(m, "    auto& b = const_cast<OwnedBranch&>(branch(base));",
                   "    auto& b = const_cast<OwnedBranch&>(branch(base));\n    b.experiment_view.reset();")
    if "clone" in flags:
        m = change(m, "    auto copy = std::make_unique<OwnedBranch>(*this);\n    copy->danger_cache.clear();",
                   "    auto copy = std::make_unique<OwnedBranch>(rules.g,tiles,observer);\n    copy->rules=rules; copy->memory=memory; copy->last_visible=last_visible;" +
                   ("\n    copy->experiment_view=experiment_view;" if "view" in flags else ""))
    if "reach" in flags:
        m = change(m, "  State rules;", "  mutable std::map<int32_t,std::vector<ReachableTile>> experiment_reach;\n  State rules;")
        m = change(m, "    copy->danger_cache.clear();", "    copy->danger_cache.clear();\n    copy->experiment_reach.clear();") if "clone" not in flags else m
        m = change(m, "    danger_cache.clear();", "    danger_cache.clear();\n    experiment_reach.clear();")
        m = change(m, "    std::vector<int> remaining, previous;\n    b.rules.paths(id, remaining, previous);",
                   "    auto hit=b.experiment_reach.find(id);\n    if(hit!=b.experiment_reach.end()) return hit->second;\n    std::vector<int> remaining, previous;\n    b.rules.paths(id, remaining, previous);")
        m = change(m, "    return out;\n  }\n  std::vector<Command> actions", "    b.experiment_reach.emplace(id,out);\n    return out;\n  }\n  std::vector<Command> actions")
    if "small" in flags:
        p = change(p, "struct Candidate {\n  std::vector<Command> commands;", """struct Commands {
  Command items[2]{}; size_t count=0;
  Commands()=default;
  Commands(std::initializer_list<Command> values) : count(values.size()) {
    if(count>2) throw std::logic_error("too many commands");
    std::copy(values.begin(),values.end(),items);
  }
  bool empty() const {return count==0;} size_t size() const {return count;}
  const Command& front() const {return items[0];}
  const Command* begin() const {return items;} const Command* end() const {return items+count;}
};
struct Candidate {
  Commands commands;""")
        p = p.replace('#include <tuple>', '#include <tuple>\n#include <stdexcept>')
    if "hash" in flags:
        w = change(w, '#include <map>', '#include <map>\n#include <unordered_map>')
        w = change(w, "struct SearchWorkspace {", """struct ExperimentHash {
  size_t operator()(const std::vector<int32_t>& key) const {
    size_t h=1469598103934665603ull;
    for(int32_t x:key) {h^=uint32_t(x);h*=1099511628211ull;} return h;
  }
};
struct SearchWorkspace {""")
        w = change(w, "std::map<std::vector<int32_t>, std::pair<int32_t, int32_t>> equivalent;",
                   "std::unordered_map<std::vector<int32_t>, std::pair<int32_t,int32_t>,ExperimentHash> equivalent;")
    if "coverage" in flags:
        # CP-inspired attacker index, but retain the current exact danger
        # calculation. Ignore ALL other units only while forming a conservative
        # terrain-only superset. Recheck occupancy/ZOC/damage in the real branch.
        m = change(m, "class Adapter final : public RulesAdapter {",
                   "class Adapter final : public RulesAdapter {\n  mutable std::map<int,std::vector<uint8_t>> experiment_coverage;")
        needle = "      State response = projected;"
        m = change(m, needle, """      auto coverage=experiment_coverage.find(id);
      if(coverage==experiment_coverage.end()) {
        State free=projected;
        for(int j=0;j<int(free.units.size());++j) if(j!=id) free.units[j].hp=0;
        auto& attacker=free.units[id];
        attacker.moves=combat_rules::moves[attacker.type]; attacker.waiting=attacker.attacked=0;
        free.rebuild();
        std::vector<int> left,previous; free.paths(id,left,previous);
        std::vector<uint8_t> cells(geo->tiles.size(),0);
        for(int from=0;from<int(left.size());++from) if(left[from]>0) {
          if(shot) {for(int t:geo->targets[from]) cells[t]=1;}
          else {for(int t:geo->neighbors[from]) if(t>=0) cells[t]=1;}
        }
        coverage=experiment_coverage.emplace(id,std::move(cells)).first;
      }
      if(!coverage->second[target]) continue;
      State response = projected;""")
    if "area" in flags:
        p = change(p, "if (c.kind == MOVE && (parent.moved[i] || !in_area(old, c.target))) continue;",
                   "if (c.kind == MOVE && (parent.moved[i] || !in_area(old, c.target) || adapter->distance(request.target,c.target)>6)) continue;")
        if "strict" in flags:
            p = change(p, "      if (c.kind == SWAP) {", "      if (c.kind == SWAP) {\n        if(adapter->distance(request.target,c.target)>6 || adapter->distance(request.target,old.units[c.unit].tile)>6) continue;")
            p = change(p, "        int freed = old.units[selected[i]].tile;", "        int freed = old.units[selected[i]].tile;\n        if(adapter->distance(request.target,freed)>6) continue;")
    if variant == "combos":
        # Test a restrictive interaction criterion, not a claim of exact CP port:
        # only form blocker/follower pairs when follower had a preferred move
        # blocked by that friendly in the parent (adjacent, combat-facing).
        p = change(p, "          if (j == i || parent.done[j] || parent.moved[j]) continue;",
                   "          if (j == i || parent.done[j] || parent.moved[j]) continue;\n          if(adapter->distance(old.units[selected[j]].tile,freed)>1 ||\n             enemy_distance(old,*adapter,freed)>2) continue;")
    if "bundle" in flags:
        p = change(p, "  int32_t score, selected_order, kind, tile;", "  int32_t score, selected_order, kind, tile;\n  int finish_partner=-1;")
        needle = "    return result;\n  }\n  std::vector<int32_t> key"
        p = change(p, needle, """    if(depth_first) {
      for(size_t i=0;i+1<result.size();++i) {
        auto& first=result[i]; const auto& second=result[i+1];
        if(first.commands.empty() && second.commands.empty() && first.selected_order!=second.selected_order) {
          first.finish_partner=second.selected_order; first.score+=second.score;
          result.erase(result.begin()+i+1);
        }
      }
    }
    return result;
  }
  std::vector<int32_t> key""")
        needle = "            current_view.units[id].tile, current_view.units[id].moves, roles[i], 0, 0, 0, -1, -1});"
        p = change(p, needle, needle + """
          if(candidate.finish_partner>=0) {
            int j=candidate.finish_partner,other=selected[j]; child->done[j]=1;
            child->assignments.push_back({CAI_FINISH_TEMP,other,current_view.units[other].tile,
              current_view.units[other].tile,current_view.units[other].moves,roles[j],0,0,0,-1,-1});
          }
""")
    # Counters have zero cost in clean timed builds. A second metrics DLL
    # records the same path without wall-time comparisons against clean DLLs.
    p = p.replace('#include <tuple>', '#include <tuple>\n#include "experiment_metrics.h"')
    for old, new in (
        ("TacticalView view(const Branch& branch) const {", "TacticalView view(const Branch& branch) const { EXP(7);"),
        ("std::unique_ptr<TacticalPosition> copy(const TacticalPosition& parent) {", "std::unique_ptr<TacticalPosition> copy(const TacticalPosition& parent) { EXP(0);"),
        ("bool apply(TacticalPosition& child, Command command, int i) {", "bool apply(TacticalPosition& child, Command command, int i) { EXP(1);"),
        ("std::vector<Candidate> candidates(const TacticalPosition& parent) {", "std::vector<Candidate> candidates(const TacticalPosition& parent) { EXP(2);"),
        ("    int losses = 0, positive = 0", "    EXP(6); int losses = 0, positive = 0"),
        ("if (losses > config.risk_budget_base + config.risk_budget_per_kill * int(p.killed.size())) return false;",
         "if (losses > config.risk_budget_base + config.risk_budget_per_kill * int(p.killed.size())) {EXP(4);return false;}"),
        ("    return improved;", "    if(!improved) EXP(5);\n    return improved;"),
    ):
        p = change(p, old, new)
    if variant == "combos-cp":
        p = change(p, "        result.push_back({{freeing.commands.front(), primary.commands.front()},",
                   "        EXP(3); result.push_back({{freeing.commands.front(), primary.commands.front()},")
    else:
        p = change(p, "          auto combo = copy(*first);", "          EXP(3); auto combo = copy(*first);")
    for key, value in zip(("planner", "model", "header", "workspace"), (p, m, h, w)):
        paths[key].write_text(value, encoding="utf-8")
    (native / "combat_ai/experiment_metrics.h").write_text("""#pragma once
#ifdef CP_EXPERIMENT_METRICS
inline thread_local long long experiment_counts[8]={};
#define EXP(n) (++experiment_counts[n])
extern "C" __declspec(dllexport) long long experiment_counter(int n) {return experiment_counts[n];}
#else
#define EXP(n) ((void)0)
#endif
""")


def build(out, variants):
    if os.name != "nt":
        raise RuntimeError("This experiment recipe records the MSVC /O2 setup used by this workspace")
    vswhere = Path("C:/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe")
    vs = subprocess.check_output([str(vswhere), "-latest", "-products", "*", "-requires",
                                  "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property",
                                  "installationPath"], text=True).strip()
    for variant in variants:
        if variant not in EXACT | ALGORITHM:
            raise ValueError(variant)
        directory = out / "variants" / variant
        if (directory / "build.json").exists():
            raise ValueError(f"Variant already built: {directory}")
        shutil.copytree(out / "frozen", directory, dirs_exist_ok=True)
        transform(directory, variant)
        sources = [directory/"native"/p for p in (
            "combat_ai/combat_ai.cpp", "combat_ai/model.cpp", "combat_ai/planner.cpp",
            "combat_ai/tactical.cpp", "combat_ai/support.cpp", "combat_state.cpp", "geometry.cpp")]
        lines = [f'@cd /d "{directory}"', f'@call "{Path(vs)/"VC/Auxiliary/Build/vcvars64.bat"}" >nul']
        for metrics in (False, True):
            output = "metrics.dll" if metrics else "search.dll"
            command = '@cl /nologo /std:c++17 /EHsc /O2 /LD /utf-8 '
            if metrics:
                command += '/DCP_EXPERIMENT_METRICS '
            command += f'/I"{directory}" /I"{directory/"native/combat_ai"}" '
            command += ' '.join(f'"{p}"' for p in sources) + f' /link /OUT:{output}'
            lines += [command, '@if errorlevel 1 exit /b %errorlevel%']
        (directory / "compile.cmd").write_text('\n'.join(lines)+'\n', encoding="utf-8")
        started = time.perf_counter()
        result = run_bounded(["cmd", "/c", str(directory / "compile.cmd")], 120)
        (directory / "compile.log").write_text(result.stdout+result.stderr, encoding="utf-8")
        if result.returncode:
            raise RuntimeError(result.stdout+result.stderr)
        write_json(directory / "build.json", {"seconds": time.perf_counter()-started,
                   "source_hashes": {str(p.relative_to(directory)): sha(p) for p in directory.rglob("*.cpp")},
                   "dll": sha(directory/"search.dll"), "metrics_dll": sha(directory/"metrics.dll")})
        print(f"built {variant}", flush=True)


def replay_plan(case, result):
    if case["snapshot"] is None:
        return {"checked": False, "reason": "raw planning state without simulator snapshot"}
    commands = []
    initial_hp = {u["id"]: u["hp"] for u in case["snapshot"]["units"]}
    side = case["request"]["side"]
    with Battle(case["snapshot"]) as b:
        for a in result.assignments:
            if a.kind in (ai.AssignmentKind.INITIAL, ai.AssignmentKind.RESTART,
                          ai.AssignmentKind.SWAP_REVERSE, ai.AssignmentKind.BLOCKED,
                          ai.AssignmentKind.FINISH_TEMP):
                continue
            state = b.snapshot()
            u = state["units"][a.unit]
            if a.kind == ai.AssignmentKind.FINISH:
                if u["hp"] <= 0 or u["moves"] <= 0 or u["waiting"]:
                    continue
                action = Action("skip", a.unit)
            else:
                if u["tile"] != a.from_tile:
                    raise ValueError(f"Plan start mismatch: {a}")
                action = Action({ai.AssignmentKind.MOVE: "move", ai.AssignmentKind.MELEE: "melee",
                                 ai.AssignmentKind.RANGED: "ranged", ai.AssignmentKind.SWAP: "swap"}[a.kind],
                                a.unit, a.command_target)
            if action not in b.available_actions():
                raise ValueError(f"Illegal plan action: {action}")
            step = b.step(action)
            if not step.ok or step.damage != a.damage or step.retaliation != a.self_damage:
                raise ValueError(f"Plan transition mismatch: {a}, {step}")
            now = b.snapshot()["units"][a.unit]
            if a.kind in (ai.AssignmentKind.MOVE, ai.AssignmentKind.SWAP) and now["tile"] != a.to_tile:
                raise ValueError(f"Plan destination mismatch: {a}, {now}")
            if now["hp"] > 0 and now["moves"] != a.remaining_moves:
                raise ValueError(f"Plan movement mismatch: {a}, {now}")
            commands.append(asdict(action))
        units = b.snapshot()["units"]
        dealt = sum(initial_hp[u["id"]]-u["hp"] for u in units if u["side"] != side)
        lost = sum(initial_hp[u["id"]]-u["hp"] for u in units if u["side"] == side)
        return {"checked": True, "commands": commands, "damage": dealt, "own_damage": lost,
                "final_sha256": digest(b.snapshot()),
                "moved": [u["id"] for u in units if u["side"] == side and
                          u["tile"] != case["snapshot"]["units"][u["id"]]["tile"]]}


def native_checks(out, variants):
    standard_groups = ("attacks", "swaps_and_freeing", "risk", "cover_and_retreat", "budgets_and_determinism",
                       "contact", "random_replay", "scale_and_last_tile", "support_healing", "support_approach")
    for variant in variants:
        directory = out/"variants"/variant
        tests = directory/"tests/native"
        tests.mkdir(parents=True, exist_ok=True)
        source = out / "frozen/tests/native/combat_ai_planner_test.cpp"
        if not source.exists():
            source = ROOT / "artifacts/combat-ai/cp-transfer-20261102/baseline/tests/native/combat_ai_planner_test.cpp"
        shutil.copyfile(source, tests/"combat_ai_planner_test.cpp")
        groups = standard_groups
        if variant == "combos-cp":
            path = tests / "combat_ai_planner_test.cpp"
            text = path.read_text(encoding="utf-8")
            extra = (ROOT / "tools/experiments/combat_combos_tests.inc").read_text(encoding="utf-8")
            text = change(text, "int main(int argc,char** argv) {", extra + "\nint main(int argc,char** argv) {")
            text = change(text, '    if(name=="attacks") attacks();',
                          '    if(name=="combo_mechanism") combo_mechanism();\n    else if(name=="attacks") attacks();')
            path.write_text(text, encoding="utf-8")
            groups += ("combo_mechanism",)
        script = (directory/"compile.cmd").read_text()
        command = next(line for line in script.splitlines() if line.startswith("@cl"))
        command = command.replace('/LD ', '').replace('/OUT:search.dll', '/OUT:planner_test.exe')
        command = command.replace(' /link ', f' "{tests/"combat_ai_planner_test.cpp"}" /link ')
        lines = script.splitlines()[:2]+[command]
        (directory/"native-check.cmd").write_text('\n'.join(lines)+'\n')
        compiled = run_bounded(["cmd", "/c", str(directory/"native-check.cmd")], 120)
        if compiled.returncode:
            raise RuntimeError(compiled.stdout+compiled.stderr)
        records = []
        for group in groups:
            result = run_bounded([str(directory/"planner_test.exe"), group], 60)
            records.append({"group": group, "returncode": result.returncode,
                            "stdout": result.stdout, "stderr": result.stderr})
            print(variant, group, result.returncode, flush=True)
        write_json(directory/"native-checks.json", records)
    verify_frozen(out)


def worker(out, variant, names, repeat, metrics):
    dll = out / "variants" / variant / ("metrics.dll" if metrics else "search.dll")
    ai.LIBRARY_PATH = dll
    ai._library.cache_clear()
    records = []
    for name in names:
        case = json.loads((out / "inputs" / f"{name}.json").read_text())
        req = load_request(case["request"])
        config = dict(case["config"])
        config["aggression"] = ai.Aggression(config["aggression"])
        memory_before = process_memory()
        with ai.Planner(ai.SearchConfig(**config)) as planner:
            if metrics:
                planner._lib.experiment_counter.restype = ctypes.c_longlong
                before = [planner._lib.experiment_counter(i) for i in range(len(COUNTERS))]
            start = time.perf_counter()
            cpu_start = time.process_time()
            result = planner.search(req)
            elapsed = time.perf_counter()-start
            cpu_elapsed = time.process_time()-cpu_start
            memory_after = process_memory()
            counts = {key: planner._lib.experiment_counter(i)-before[i] for i, key in enumerate(COUNTERS)} if metrics else None
        records.append({"variant": variant, "case": name, "repeat": repeat, "metrics": metrics,
                        "seconds": elapsed, "result": asdict(result), "counters": counts,
                        "cpu_seconds": cpu_elapsed,
                        "input_sha256": sha(out / "inputs" / f"{name}.json"), "library_sha256": sha(dll),
                        "memory_before": memory_before, "memory_after": memory_after,
                        "validation": replay_plan(case, result)})
    write_json(out / "records" / f'{variant}-{repeat}-{names[0]}-{"metrics" if metrics else "clean"}.json', records)
    print(json.dumps({"variant": variant, "cases": len(names), "seconds": sum(r["seconds"] for r in records)}))


def verify_frozen(out):
    manifest = json.loads((out/"manifest.json").read_text())
    current = protected()
    if current != manifest["protected"]:
        raise ValueError("Protected working sources/binaries changed since corpus freeze")
    for name, value in manifest["inputs"].items():
        if sha(out/"inputs"/f"{name}.json") != value:
            raise ValueError(f"Input changed: {name}")
    return manifest


def run(out, variants, suite, repeats, requested=None):
    manifest = verify_frozen(out)
    names = manifest["timing"] if suite != "differential" else manifest["differential"]
    if requested:
        if not set(requested) <= set(names):
            raise ValueError("Requested cases do not belong to the selected suite")
        names = list(requested)
    metrics = suite == "metrics"
    log = []
    for variant in variants:
        directory = out/"variants"/variant
        built = json.loads((directory/"build.json").read_text())
        if sha(directory/"search.dll") != built["dll"] or sha(directory/"metrics.dll") != built["metrics_dll"]:
            raise ValueError(f"Variant DLL changed: {variant}")
    for repeat in range(repeats):
        # Rotate the order of variants; no parallel timed searches.
        order = variants[repeat % len(variants):] + variants[:repeat % len(variants)]
        groups = [[name] for name in names] if suite != "differential" else [names]
        for group in groups:
            for variant in order:
                record = out/"records"/f'{variant}-{repeat}-{group[0]}-{"metrics" if metrics else "clean"}.json'
                if record.exists():
                    saved = json.loads(record.read_text())
                    dll = out/"variants"/variant/("metrics.dll" if metrics else "search.dll")
                    if [r["case"] for r in saved] != group or any(
                        r["variant"] != variant or r["repeat"] != repeat or r["metrics"] != metrics or
                        ("library_sha256" in r and r["library_sha256"] != sha(dll)) or
                        ("input_sha256" in r and r["input_sha256"] != manifest["inputs"][r["case"]])
                        for r in saved
                    ):
                        raise ValueError(f"Stale or mismatched saved record: {record}")
                    continue
                cmd = [sys.executable, str(Path(__file__).resolve()), "worker", "--out", str(out),
                       "--variants", variant, "--names", *group, "--repeat", str(repeat)]
                if metrics:
                    cmd += ["--metrics"]
                result = run_bounded(cmd, 120)
                log.append({"command": cmd, "returncode": result.returncode,
                            "stdout": result.stdout, "stderr": result.stderr})
                write_json(out/"logs"/f"{suite}-{repeat}-{group[0]}-{variant}.json", log[-1])
                print(result.stdout.strip() or result.stderr.strip(), flush=True)
                if result.returncode:
                    raise RuntimeError(result.stderr)
    verify_frozen(out)


def report(out):
    manifest = verify_frozen(out)
    records = [r for p in sorted((out/"records").glob("*.json")) for r in json.loads(p.read_text()) if r["repeat"] >= 0]
    baseline = {r["case"]: r for r in records if r["variant"] == "base" and not r["metrics"]}
    summaries = {}
    for variant in sorted({r["variant"] for r in records}):
        rows, differences, lost_plans, changed_finals = {}, [], [], []
        for name in manifest["inputs"]:
            rr = [r for r in records if r["variant"] == variant and r["case"] == name and not r["metrics"]]
            if not rr:
                continue
            deterministic = all(r["result"] == rr[0]["result"] for r in rr)
            if not deterministic:
                raise ValueError(f"Non-deterministic result: {variant}/{name}")
            equal = name in baseline and rr[0]["result"] == baseline[name]["result"]
            if not equal:
                differences.append(name)
            if name in baseline:
                if baseline[name]["result"]["assignments"] and not rr[0]["result"]["assignments"]:
                    lost_plans.append(name)
                if (rr[0]["validation"].get("checked") and baseline[name]["validation"].get("checked") and
                    rr[0]["validation"]["final_sha256"] != baseline[name]["validation"]["final_sha256"]):
                    changed_finals.append(name)
            cpu = [r["cpu_seconds"] for r in rr if "cpu_seconds" in r]
            cpu_complete = all("cpu_seconds" in r for r in rr)
            rows[name] = {"count": len(rr), "median": statistics.median(r["seconds"] for r in rr),
                          "min": min(r["seconds"] for r in rr), "max": max(r["seconds"] for r in rr),
                          "cpu_median": statistics.median(cpu) if cpu else None,
                          "cpu_count": len(cpu), "cpu_complete": cpu_complete,
                          "equal_baseline": equal, "result": rr[0]["result"], "validation": rr[0]["validation"]}
        summaries[variant] = {"expected_exact": variant in EXACT, "differences": differences,
                              "lost_plans": lost_plans, "changed_finals": changed_finals, "cases": rows}
    metric_checks = []
    for r in records:
        if r["metrics"]:
            clean = next((c for c in records if not c["metrics"] and c["variant"] == r["variant"] and
                          c["case"] == r["case"]), None)
            if clean is None or clean["result"] != r["result"]:
                raise ValueError(f"Metrics build changed behavior: {r['variant']}/{r['case']}")
            metric_checks.append([r["variant"], r["case"]])
    write_json(out/"summary.json", {"variants": summaries, "records": len(records),
                                   "metrics_equal_clean": metric_checks, "protected_unchanged": True})
    lines = ["# CP-07: независимые эксперименты и сочетания", "",
             "Все варианты собраны из замороженного рабочего исходника MSVC /O2. Рабочая DLL не менялась.",
             "Время включает упаковку запроса и native search; подготовка входа, сборка и воспроизведение плана исключены.",
             "Чистые замеры и отдельные сборки со счётчиками не смешиваются. Поиски выполнялись последовательно.", "",
             "| Вариант | Входов | Отличия от baseline | distant26, медиана с |", "|---|---:|---|---:|"]
    for variant, value in summaries.items():
        big = value["cases"].get("distant26", {}).get("median")
        lines.append(f"| {variant} | {len(value['cases'])} | {', '.join(value['differences']) or 'нет'} | {big if big is not None else '—'} |")
    lines += ["", "## Время на основных входах", "",
              "Медианы секунд; в скобках минимум–максимум. Малые отличия при пересекающихся диапазонах не объявляются выигрышем.", ""]
    for name in manifest["timing"]:
        lines += [f"### {name}", "", "| Вариант | Повторов | Wall time, с | CPU time, с |", "|---|---:|---:|---:|"]
        for variant, value in summaries.items():
            row = value["cases"].get(name)
            if row:
                cpu = f"{row['cpu_median']:.6f} ({row['cpu_count']})" if row["cpu_median"] is not None else "—"
                lines.append(f"| {variant} | {row['count']} | {row['median']:.6f} ({row['min']:.6f}–{row['max']:.6f}) | {cpu} |")
        lines.append("")
    lines += ["## Изменения результата", "", "| Вариант | Потерян непустой план | Изменено конечное состояние |",
              "|---|---|---|"]
    for variant, value in summaries.items():
        lines.append(f"| {variant} | {', '.join(value['lost_plans']) or 'нет'} | {', '.join(value['changed_finals']) or 'нет'} |")
    lines += ["", "## Счётчики отдельной инструментированной сборки", "",
              "copy/apply включают временное оценивание и создание сохраняемых детей; они не равны generated.", "",
              "| Вариант / вход | copy | apply | combo | unsafe | no_improvement | generated |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        if r["metrics"]:
            c = r["counters"]
            lines.append(f"| {r['variant']} / {r['case']} | {c['copy']} | {c['apply']} | {c['combo']} | {c['unsafe']} | {c['no_improvement']} | {r['result']['stats']['generated']} |")
    lines += ["", "## Память процесса", "",
              "Пиковый RSS измерен сразу после поиска, до проверки плана. Это вся память процесса, не размер пула поиска.",
              "В таблице только distant26 (отдельный worker на каждый повтор).", "",
              "| Вариант | Пиковый RSS, медиана MiB |", "|---|---:|"]
    for variant in summaries:
        values = [r["memory_after"]["peak_rss_bytes"]/(1024*1024) for r in records if
                  r["variant"] == variant and r["case"] == "distant26" and not r["metrics"] and
                  "memory_after" in r and "peak_rss_bytes" in r["memory_after"]]
        if values:
            lines.append(f"| {variant} | {statistics.median(values):.2f} |")
    lines += ["", "Для точных вариантов сравниваются все поля и порядок назначений, причина и SearchStats.",
              "Для алгоритмических изменений сравнение с baseline фиксирует изменение поведения, а не автоматическую ошибку.",
              "Законность и предсказанный урон проверяются исполнением каждого плана в штатном ядре, когда сохранён snapshot.",
              "Пустой план проходит проверку законности, но это само по себе не подтверждает его качество.", "",
              "Состав входов, настройки, хеши и все результаты сохранены в manifest.json, inputs/, records/ и summary.json.",
              "Это ограниченные локальные эксперименты. Победы в полных матчах и эквивалентность на всех состояниях не установлены."]
    lines += ["", "## Нативные сценарии", "", "| Вариант | Успешно | Ошибки |", "|---|---:|---|"]
    for path in sorted((out/"variants").glob('*/native-checks.json')):
        checks = json.loads(path.read_text())
        failed = [r["group"] for r in checks if r["returncode"]]
        lines.append(f"| {path.parent.name} | {len(checks)-len(failed)}/{len(checks)} | {', '.join(failed) or 'нет'} |")
    (out/"report.md").write_text('\n'.join(lines)+'\n', encoding="utf-8")
    failed = {variant: value["differences"] for variant, value in summaries.items()
              if value["expected_exact"] and value["differences"]}
    if failed:
        raise ValueError(f"Exact variants changed results: {failed}")
    print(json.dumps({"records": len(records), "variants": len(summaries)}))


def archive(out):
    """Portable inputs plus compact results; no compiler objects or DLLs in git."""
    report(out)
    target = ROOT/"artifacts/combat-ai/experiments-v1"
    target.mkdir(parents=True, exist_ok=True)
    manifest = verify_frozen(out)
    corpus = {name: json.loads((out/"inputs"/f"{name}.json").read_text()) for name in manifest["inputs"]}
    # Retain full payloads in compact JSON; hashes use each original expanded input.
    (target/"inputs.json").write_text(json.dumps(corpus, ensure_ascii=False, separators=(',', ':'))+'\n', encoding="utf-8")
    summary = json.loads((out/"summary.json").read_text())
    results = []
    for path in sorted((out/"records").glob("*.json")):
        for r in json.loads(path.read_text()):
            if r["repeat"] < 0:
                continue
            item = {k: v for k, v in r.items() if k not in ("result", "validation")}
            item["result_sha256"] = digest(r["result"])
            item["final_sha256"] = r["validation"].get("final_sha256")
            results.append(item)
    builds = {p.parent.name: json.loads(p.read_text()) for p in (out/"variants").glob('*/build.json')}
    checks = {p.parent.name: json.loads(p.read_text()) for p in (out/"variants").glob('*/native-checks.json')}
    compact = {"schema": 1, "manifest": manifest, "builds": builds, "native_checks": checks,
               "summary": summary, "runs": results, "driver_sha256": sha(__file__),
               "environment": {"python": sys.version, "platform": platform.platform(),
                               "machine": platform.machine(), "cpu": platform.processor(),
                               "cpu_count": os.cpu_count(), "compiler_flags": "/std:c++17 /EHsc /O2 /LD /utf-8",
                               "scope": "local-search experiments; no production changes or full-match acceptance"}}
    (target/"results.json").write_text(json.dumps(compact, ensure_ascii=False, separators=(',', ':'))+'\n', encoding="utf-8")
    shutil.copyfile(out/"report.md", target/"report.md")
    print(json.dumps({"archived": str(target), "runs": len(results)}))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("stage", choices=("prepare", "build", "run", "worker", "report", "native-checks", "archive"))
    parser.add_argument("--out", type=Path, default=ROOT/"build/cp07-experiments-v1")
    parser.add_argument("--variants", nargs="+", default=["base"])
    parser.add_argument("--suite", choices=("timing", "differential", "metrics"), default="timing")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--repeat", type=int, default=0)
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--inputs", type=Path, default=ROOT/"artifacts/combat-ai/experiments-v1/inputs.json")
    parser.add_argument("--source-root", type=Path,
                        help="Pre-transfer source tree (native/, build/, optionally tests/)")
    args = parser.parse_args()
    out = args.out.resolve()
    if args.stage == "prepare":
        prepare(out, args.inputs, args.source_root)
    elif args.stage == "build":
        build(out, args.variants)
    elif args.stage == "run":
        run(out, args.variants, args.suite, args.repeats, args.names)
    elif args.stage == "worker":
        worker(out, args.variants[0], args.names, args.repeat, args.metrics)
    elif args.stage == "native-checks":
        native_checks(out, args.variants)
    elif args.stage == "archive":
        archive(out)
    else:
        report(out)


if __name__ == "__main__":
    main()
