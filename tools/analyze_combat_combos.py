"""Isolated, reproducible diagnosis of the three experimental combos cases.

Uses the frozen experiment inputs/sources; never edits the production planner.
Tracing changes observation only. Runtime ablations select either filter clause.
"""

import argparse
import ctypes
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from civ5sim import combat_ai as ai
from civ5sim.evaluation import write_json
from tools.experiment_combat_ai import change, load_request, protected, replay_plan

CASES = ("differential14", "differential33", "differential37")
SOURCE = ROOT / "build/cp07-experiments-v1"
OUT = ROOT / "build/cp07-combos-diagnosis"


TRACE = r'''
  FILE* audit_file = nullptr;
  int audit_filter = 0, audit_current = -1;
  const Candidate* audit_candidate = nullptr;
  void audit(const char* event, const TacticalPosition& p, int value=0) {
    if(!audit_file) return;
    auto v=view(*p.branch);
    fprintf(audit_file,"{\"event\":\"%s\",\"parent\":%d,\"value\":%d,\"depth\":%d,\"rank\":%d,\"attack\":%d,\"generated\":%d,\"checked\":%d,\"depth_first\":%d,\"done\":[",event,audit_current,value,p.generation,p.rank,p.attack_score,stats.generated,stats.checked,int(depth_first));
    for(size_t i=0;i<selected.size();++i) fprintf(audit_file,"%s%d",i?",":"",p.done[i]);
    fprintf(audit_file,"],\"moved\":[");
    for(size_t i=0;i<selected.size();++i) fprintf(audit_file,"%s%d",i?",":"",p.moved[i]);
    fprintf(audit_file,"],\"units\":[");
    for(size_t i=0;i<selected.size();++i) {const auto& u=v.units[selected[i]];fprintf(audit_file,"%s[%d,%d,%d,%d,%d]",i?",":"",u.id,u.tile,u.hp,u.moves,u.attacked);}
    fprintf(audit_file,"],\"commands\":[");
    if(audit_candidate) for(size_t i=0;i<audit_candidate->commands.size();++i) {auto c=audit_candidate->commands[i];fprintf(audit_file,"%s[%d,%d,%d]",i?",":"",c.kind,c.unit,c.target);}
    fprintf(audit_file,"],\"choice_score\":%d,\"choice_unit\":%d,\"assignments\":[",audit_candidate?audit_candidate->score:0,audit_candidate?selected[audit_candidate->selected_order]:-1);
    bool comma=false;for(auto a:p.assignments) {fprintf(audit_file,"%s[%d,%d,%d,%d,%d,%d]",comma?",":"",a.kind,a.unit,a.from_tile,a.to_tile,a.command_target,a.score);comma=true;}
    fprintf(audit_file,"]}\n");
  }
'''


def prepare(cp=False):
    directory = OUT / "trace"
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE / "frozen", directory, dirs_exist_ok=True)
    if cp:
        from tools.combat_combos_variant import transform
        transform(directory)
    path = directory / "native/combat_ai/planner.cpp"
    text = path.read_text(encoding="utf-8")
    text = change(text, '#include "planner.h"', '''#include "planner.h"
#include <cstdio>
#include <string>
static int audit_mode=0;
static std::string audit_path;
static std::vector<int> audit_witness_data;
extern "C" __declspec(dllexport) void audit_configure(int mode,const char* path) {
  audit_mode=mode; audit_path=path?path:"";
}
extern "C" __declspec(dllexport) void audit_set_witness(const int* data,int count) {
  audit_witness_data.clear();
  if(count) audit_witness_data.assign(data,data+count);
}''')
    text = change(text, "  float bias = 1.f;", "  float bias = 1.f;\n" + TRACE)
    text = change(text, "    work.clear();", '''    work.clear();
    audit_filter = audit_mode;
    if(!audit_path.empty()) audit_file=std::fopen(audit_path.c_str(),"w");''')
    text = change(text, "    return plan;", "    if(best>=0) {audit_candidate=nullptr;audit(\"best\",*work.positions[best],best);}\n    if(audit_file) std::fclose(audit_file);\n    return plan;")
    text = change(text, "      auto& parent = *work.positions[current];", "      auto& parent = *work.positions[current];\n      audit_current=current; audit_candidate=nullptr; audit(\"expand\",parent);")
    if not cp:
        text = change(text, "          if (j == i || parent.done[j] || parent.moved[j]) continue;", '''          if (j == i || parent.done[j] || parent.moved[j]) continue;
          if((audit_filter&1) && adapter->distance(old.units[selected[j]].tile,freed)>1) continue;
          if((audit_filter&2) && enemy_distance(old,*adapter,freed)>2) continue;''')
    else:
        text = change(text, "        choices.push_back({{command}, gain, i, command.kind, command.target});",
                      '''        choices.push_back({{command}, gain, i, command.kind, command.target});
        {auto previous=audit_candidate;audit_candidate=&choices.back();audit("proposal",parent,old.occupied[command.target]);audit_candidate=previous;}''')
        text = change(text, "    std::vector<Candidate> result;", '''    for(auto& c:overall) {auto previous=audit_candidate;audit_candidate=&c;audit("preferred",parent,c.commands.empty()?-1:old.occupied[c.tile]);audit_candidate=previous;}
    std::vector<Candidate> result;''')
    text = change(text, "      int children = 0;", '''      int children = 0;
      for(auto& c:choices) {audit_candidate=&c;audit("candidate",parent);}''')
    text = change(text, "      for (auto& candidate : choices) {", "      for (auto& candidate : choices) {\n        audit_candidate=&candidate;")
    text = change(text, "          if (candidate.commands.size() > 1) child->moved[candidate.selected_order] = 0;", "          if (candidate.commands.size() > 1) child->moved[candidate.selected_order] = 0;")
    text = change(text, "          ++stats.pruned; continue;\n        }\n        bool finished", "          audit(\"equivalent\",*child); ++stats.pruned; continue;\n        }\n        bool finished")
    text = change(text, "        work.positions.push_back(std::move(child));", "        audit(\"saved\",*child,id);\n        work.positions.push_back(std::move(child));")
    text = change(text, "      if (!acceptable) ++losses;", "      if (!acceptable) {++losses;audit(\"unsafe_unit\",p,id);}")
    text = change(text, "if (losses > config.risk_budget_base + config.risk_budget_per_kill * int(p.killed.size())) return false;", "if (losses > config.risk_budget_base + config.risk_budget_per_kill * int(p.killed.size())) {audit(\"reject_unsafe\",p,losses);return false;}")
    text = change(text, "    return improved;", "    audit(improved?\"complete\":\"reject_no_improvement\",p,score);\n    return improved;")
    text = change(text, "    stats.generated = stats.peak_positions = 1;", r'''    stats.generated = stats.peak_positions = 1;
    if(!audit_witness_data.empty()) {
      auto probe=copy(*work.positions[0]);probe->generation=0;
      size_t offset=0;
      while(offset<audit_witness_data.size()) {
        int count=audit_witness_data[offset++];
        std::vector<Command> commands;
        for(int k=0;k<count;++k) {
          commands.push_back({audit_witness_data[offset],audit_witness_data[offset+1],audit_witness_data[offset+2]});offset+=3;
        }
        depth_first=probe->generation>3;
        auto options=candidates(*probe);
        auto same=[&](const Candidate& c) {
          if(c.commands.size()!=commands.size()) return false;
          for(size_t k=0;k<commands.size();++k)
            if(c.commands[k].kind!=commands[k].kind || c.commands[k].unit!=commands[k].unit || c.commands[k].target!=commands[k].target) return false;
          return true;
        };
        auto found=std::find_if(options.begin(),options.end(),same);
        Candidate manual{commands,0,selected_index[commands[0].unit],commands[0].kind,commands[0].target};
        audit_candidate=&manual;
        audit(found!=options.end()?"witness_present":"witness_missing",*probe);
        auto next=copy(*probe);
        for(auto command:commands) {
          bool ok=apply(*next,command,selected_index[command.unit]);
          if(!ok) {audit("witness_apply_failed",*next);std::fclose(audit_file);return {{},{}};}
        }
        if(commands.size()>1) next->moved[manual.selected_order]=0;
        probe=std::move(next);audit("witness_step",*probe);
      }
      audit_candidate=nullptr;auto v=view(*probe->branch);int score;
      complete(*probe,v,score);std::fclose(audit_file);return {{},{}};
    }''')
    path.write_text(text, encoding="utf-8")
    original = (SOURCE / "variants/base/compile.cmd").read_text()
    old = str(SOURCE / "variants/base")
    lines = original.splitlines()[:4]
    script = "\n".join(lines).replace(old, str(directory)) + "\n"
    (directory / "compile.cmd").write_text(script)
    compiled = subprocess.run(["cmd", "/c", str(directory / "compile.cmd")],
                              capture_output=True, text=True, timeout=120, check=False)
    (directory / "compile.log").write_text(compiled.stdout + compiled.stderr)
    if compiled.returncode:
        raise RuntimeError(compiled.stdout + compiled.stderr)
    write_json(OUT / "build.json", {"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "variant": "combos-cp" if cp else "surrogate-ablation",
               "dll_sha256": hashlib.sha256((directory / "search.dll").read_bytes()).hexdigest()})


def run():
    ai.LIBRARY_PATH = OUT / "trace/search.dll"
    ai._library.cache_clear()
    archive = json.loads((ROOT / "artifacts/combat-ai/experiments-v1/results.json").read_text())
    records = []
    for name in CASES:
        case = json.loads((SOURCE / "inputs" / (name + ".json")).read_text())
        req = load_request(case["request"])
        for mode, label in ((0, "base"), (1, "adjacent"), (2, "enemy"), (3, "combos")):
            for capacity in (180, 360, 720, 1500, 3000, 6000):
                config = dict(case["config"], position_capacity=capacity)
                config["aggression"] = ai.Aggression(config["aggression"])
                trace = OUT / f"{name}-{label}-{capacity}.jsonl"
                with ai.Planner(ai.SearchConfig(**config)) as planner:
                    planner._lib.audit_configure.argtypes = [ctypes.c_int, ctypes.c_char_p]
                    planner._lib.audit_configure(mode, str(trace).encode() if capacity <= 360 else None)
                    result = planner.search(req)
                validation = replay_plan(case, result)
                record = {"case": name, "variant": label, "capacity": capacity,
                          "result": asdict(result), "validation": validation}
                if capacity == 180 and label in ("base", "combos"):
                    expected = archive["summary"]["variants"][label]["cases"][name]["result"]
                    actual = json.loads(json.dumps(record["result"]))
                    assert actual == expected, (name, label, "instrumentation changed output")
                records.append(record)
                print(name, label, capacity, result.reason.name, asdict(result.stats),
                      "damage", validation["damage"], "own", validation["own_damage"], flush=True)
    write_json(OUT / "sweep.json", records)


def witness():
    """Follow the original winner with and without grouping; bypass beam order.

    Missing candidates are reported, then physically applied to inspect later gates.
    This is a reachability probe, not an ordinary search result.
    """
    ai.LIBRARY_PATH = OUT / "trace/search.dll"
    ai._library.cache_clear()
    for name in CASES:
        rows = [json.loads(line) for line in (OUT / f"{name}-base-180.jsonl").read_text().splitlines()]
        saved = {r["value"]: r for r in rows if r["event"] == "saved"}
        node = next(r["value"] for r in rows if r["event"] == "best")
        groups = []
        while node:
            groups.append(saved[node]["commands"])
            node = saved[node]["parent"]
        groups.reverse()
        case = json.loads((SOURCE / "inputs" / (name + ".json")).read_text())
        config = dict(case["config"])
        config["aggression"] = ai.Aggression(config["aggression"])
        for mode, variant in ((0, "base"), (3, "combos")):
            for split in (False, True):
                commands = [[c] for group in groups for c in group] if split else groups
                flat = [x for group in commands for x in [len(group), *(n for c in group for n in c)]]
                packed = (ctypes.c_int * len(flat))(*flat)
                path = OUT / f"{name}-{variant}-witness-{'split' if split else 'grouped'}.jsonl"
                with ai.Planner(ai.SearchConfig(**config)) as planner:
                    planner._lib.audit_configure.argtypes = [ctypes.c_int, ctypes.c_char_p]
                    planner._lib.audit_set_witness.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
                    planner._lib.audit_configure(mode, str(path).encode())
                    planner._lib.audit_set_witness(packed, len(flat))
                    planner.search(load_request(case["request"]))
                    planner._lib.audit_set_witness(None, 0)
                events = [json.loads(line) for line in path.read_text().splitlines()]
                print(path.name, [(r["event"], r["commands"], r["moved"]) for r in events
                                 if r["event"] in ("witness_present", "witness_missing", "witness_apply_failed")])


def trace_cp():
    ai.LIBRARY_PATH = OUT / "trace/search.dll"
    ai._library.cache_clear()
    expected = {}
    for path in (ROOT / "build/cp07-combos-cp-v2/records").glob("*.json"):
        for r in json.loads(path.read_text()):
            if r["variant"] == "combos-cp":
                expected[r["case"]] = r["result"]
    records = []
    for name in CASES:
        case = json.loads((SOURCE / "inputs" / (name + ".json")).read_text())
        for capacity in (180, 360):
            config = dict(case["config"], position_capacity=capacity)
            config["aggression"] = ai.Aggression(config["aggression"])
            trace = OUT / f"{name}-combos-cp-{capacity}.jsonl"
            with ai.Planner(ai.SearchConfig(**config)) as planner:
                planner._lib.audit_configure.argtypes = [ctypes.c_int, ctypes.c_char_p]
                planner._lib.audit_configure(0, str(trace).encode())
                result = planner.search(load_request(case["request"]))
            actual = json.loads(json.dumps(asdict(result)))
            if capacity == 180:
                assert actual == expected[name], (name, "instrumentation changed result")
            records.append({"case": name, "capacity": capacity, "result": actual,
                            "validation": replay_plan(case, result)})
            print(name, capacity, actual["stats"], flush=True)
    write_json(OUT / "results.json", records)


def main():
    global OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "witness", "trace-cp"))
    parser.add_argument("--cp", action="store_true")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out.resolve()
    before = protected()
    if args.stage == "prepare":
        prepare(args.cp)
    elif args.stage == "run":
        run()
    elif args.stage == "witness":
        witness()
    else:
        trace_cp()
    assert protected() == before, "Production inputs changed"


if __name__ == "__main__":
    main()
