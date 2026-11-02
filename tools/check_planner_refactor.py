"""Compare two ABI6 planners on fresh requests and replay every plan."""

import argparse
import hashlib
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from civ5sim import combat_ai as ai
from civ5sim.engine import digest
from civ5sim.evaluation import write_json
from tools.tactical_corpus import fresh_cases, load_request, replay_plan


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(before, after, output):
    corpus = fresh_cases()
    runs = {}
    versions = {}
    for label, dll in (("before", before), ("after", after)):
        ai.LIBRARY_PATH = dll
        ai._library.cache_clear()
        runs[label] = {}
        for name, case in corpus.items():
            config = dict(case["config"])
            config["aggression"] = ai.Aggression(config["aggression"])
            with ai.Planner(ai.SearchConfig(**config)) as planner:
                versions[label] = planner.version
                result = planner.search(load_request(case["request"]))
            runs[label][name] = {"result": asdict(result), "validation": replay_plan(case, result)}
    rows = []
    for name in corpus:
        old, new = runs["before"][name], runs["after"][name]
        same = old["result"] == new["result"]
        row = {"case": name, "identical_result": same,
               "before_sha256": digest(old["result"]), "after_sha256": digest(new["result"]),
               "before_validation": old["validation"], "after_validation": new["validation"]}
        if not same:
            row["before"] = old["result"]
            row["after"] = new["result"]
        rows.append(row)
    write_json(output, {"versions": versions, "before_dll_sha256": sha(before),
                        "after_dll_sha256": sha(after), "inputs_sha256": digest(corpus),
                        "source_sha256": {str(p.relative_to(ROOT)): sha(p)
                                          for p in (ROOT / "native/combat_ai").glob("*") if p.is_file()},
                        "cases": len(rows), "identical": sum(r["identical_result"] for r in rows),
                        "rows": rows})
    print(f'{len(rows)} cases replay-checked; {sum(r["identical_result"] for r in rows)} identical full results')
    print("Changed:", [r["case"] for r in rows if not r["identical_result"]])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, default=ai.LIBRARY_PATH)
    parser.add_argument("--out", type=Path, default=ROOT / "build/planner-refactor-comparison.json")
    args = parser.parse_args()
    check(args.before.resolve(), args.after.resolve(), args.out.resolve())
