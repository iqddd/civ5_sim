"""Measure off/compact/full overhead on the same saved queries, never on new matches."""

import argparse
import gzip
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from civ5sim.combat_ai import Planner, SupportMode
from civ5sim.decision_diagnostics import (
    DecisionRecorder,
    load_package,
    plain,
    provenance,
    unpack_request,
)


def benchmark(package, *, repeats=5, limit=24):
    inputs = [unpack_request(package, q) for q in package["queries"][:limit]]
    if not inputs:
        raise ValueError("Package has no queries")
    rows = []
    reference = []
    for mode in ("off", "compact", "full"):
        times, sizes = [], []
        for _ in range(repeats):
            collector = DecisionRecorder(mode) if mode != "off" else None
            started = time.perf_counter()
            answers = []
            for index, (config, request) in enumerate(inputs):
                query = package["queries"][index]
                with Planner(config) as planner:
                    if collector is not None:
                        collector.begin(index + 1, request.side, {}, object())
                        planner._decision_recorder = collector
                    if query["mode"] == "candidates":
                        answer = planner.attack_candidates(request)
                    elif query["mode"] == "search":
                        answer = planner.search(request)
                    else:
                        answer = planner.support(request, SupportMode(query["mode"]))
                    answers.append(plain(answer))
            times.append(time.perf_counter() - started)
            if mode == "off":
                if reference and answers != reference:
                    raise AssertionError("Baseline queries are not deterministic")
                reference = answers
            elif answers != reference:
                raise AssertionError(f"{mode} changed query results")
            if collector is not None:
                if collector.errors:
                    raise AssertionError(collector.errors)
                payload = {"queries": collector.queries, "states": collector.states, "calls": collector.calls}
                trace_bytes = sum(p.stat().st_size for p in Path(collector.trace_directory.name).glob("*.json.gz")) if collector.trace_directory else 0
                sizes.append(len(gzip.compress(json.dumps(payload).encode("utf-8"))) + trace_bytes)
        rows.append({"mode": mode, "seconds": times, "median_seconds": statistics.median(times),
                     "query_payload_gzip_bytes": sizes, "results_equal": True})
    baseline = rows[0]["median_seconds"]
    for row in rows:
        row["relative_to_off"] = row["median_seconds"] / baseline
    return {"queries": len(inputs), "repeats": repeats, "rows": rows, "provenance": provenance(),
            "timing_scope": "Fresh planner per query; includes request capture and trace decoding. "
                            "Excludes one-time source hashing and disk compression; this is not match timing."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--out", "--output", dest="output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--limit", type=int, default=24)
    args = parser.parse_args()
    if args.output.exists() or args.repeats < 1 or args.limit < 1:
        parser.error("Use a new output file and positive repeats/limit")
    result = benchmark(load_package(args.package), repeats=args.repeats, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for row in result["rows"]:
        print(f"{row['mode']}: {row['median_seconds']:.4f}s, x{row['relative_to_off']:.2f}")


if __name__ == "__main__":
    main()
