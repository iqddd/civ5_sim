"""CP-07: isolated, repeated matches with checked replays and separate telemetry."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from civ5sim import MatchConfig, run_match
from civ5sim.agents import load_agent
from civ5sim.cp_agent import CPAgent
from civ5sim.engine import digest
from civ5sim.evaluation import (
    cases,
    diagnostics,
    distribution,
    load_suite,
    passport,
    process_memory,
    reusable_check,
    search_config,
    timing_summary,
    verify_replay,
    write_json,
    write_report,
)


def run_bounded(command, timeout):
    """Kill the entire worker tree, including Windows venv redirectors."""
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               text=True, start_new_session=os.name != "nt")
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "nt":
                killed = subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                        capture_output=True, check=False, timeout=5)
                if killed.returncode and process.poll() is None:
                    raise OSError(f"taskkill failed: {killed.stderr!r}")
            else:
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)
        except (OSError, subprocess.TimeoutExpired) as cleanup:
            if process.poll() is None:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            # Abort the benchmark: an unconfirmed tree termination invalidates
            # subsequent latency samples. Avoid Popen.__exit__'s unbounded wait.
            raise WorkerCleanupError(f"Could not confirm worker-tree termination: {cleanup}") from cleanup
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


class WorkerCleanupError(Exception):
    """Fatal benchmark error: do not continue timing with a surviving worker."""


def worker(args, suite):
    case = next(c for c in cases(suite) if c["id"] == args.worker)
    agents = [CPAgent(search_config(suite)) if name == "cp" else load_agent(name)
              for name in case["agents"]]
    samples = []
    args.out.mkdir(parents=True, exist_ok=True)
    last_progress = 0

    def collect(sample):
        nonlocal last_progress
        samples.append(sample)
        now = time.perf_counter()
        if now - last_progress >= 10:
            write_json(args.out / "progress.json", {"turn": sample["turn"],
                       "phase": sample["kind"], "elapsed_seconds": now - start})
            last_progress = now

    baseline = process_memory()
    start = time.perf_counter()
    try:
        outcome = run_match(MatchConfig(**case["config"]), agents, metrics=collect)
        elapsed = time.perf_counter() - start
        memory = process_memory()
        outcome.save(args.out)
        if outcome.result["status"] != "completed":
            raise AssertionError(outcome.result["error"])
        counters = diagnostics(outcome.replay, search_config(suite))
        verify_replay(outcome)
        memory["baseline_peak_rss_bytes"] = baseline["peak_rss_bytes"]
        memory["peak_rss_growth_bytes"] = max(0, memory["peak_rss_bytes"] - baseline["peak_rss_bytes"])
        write_json(args.out / "metrics.json", {"samples": samples,
                   "timings": timing_summary(samples), "memory": memory, "match_seconds": elapsed})
        write_json(args.out / "checked.json", {
            "case": case["id"], "replay_verified": True,
            "run_signature": args.signature,
            "deterministic_sha256": digest({"result": outcome.result, "replay": outcome.replay,
                                            "final": outcome.final}),
            "diagnostics": counters})
    finally:
        for agent in agents:
            close = getattr(agent, "close", None)
            if close:
                close()


def tactical_checks(directory):
    suffix = ".exe" if os.name == "nt" else ""
    for name in ("combat_ai_model_test", "combat_ai_planner_test"):
        if not (ROOT / "build" / (name + suffix)).is_file():
            raise RuntimeError("Build tools/build_combat_ai.py before acceptance checks")
    commands = [[sys.executable, "-m", "unittest", "discover", "-s", "tests",
                 "-p", pattern] for pattern in
                ("test_combat_ai_planner.py", "test_cp_agent.py", "test_cp_coordinator.py",
                 "test_combat_ai_evaluation.py")]
    records = []
    for command in commands:
        start = time.perf_counter()
        completed = run_bounded(command, 600)
        records.append({"command": command, "seconds": time.perf_counter()-start,
                        "returncode": completed.returncode,
                        "output": completed.stdout + completed.stderr})
        write_json(directory / "tactical.json", records)
        if completed.returncode:
            raise RuntimeError(records[-1]["output"])
    return records


def aggregate(records):
    groups = defaultdict(list)
    timings = defaultdict(list)
    cp_diagnostics = Counter()
    cancel_reasons = Counter()
    search_reasons = Counter()
    memory = []
    for record in records:
        case, result = record["case"], record["result"]
        for side, name in enumerate(case["agents"]):
            if name != "cp":
                continue
            opponent = case["agents"][1-side]
            for key in (f"opponent/{opponent}", f"field/{case['field']}/{opponent}",
                        f"modes/f{int(case['config']['fog'])}-r{int(case['config']['random_damage'])}/{opponent}",
                        f"side/{side}/{opponent}"):
                groups[key].append({"outcome": "draw" if result["winner"] is None else
                                    ("win" if result["winner"] == side else "loss"),
                                    "alive": result["alive"][side],
                                    "hp": result["remaining_hp"][side], "reason": result["reason"]})
            d = record["checked"]["diagnostics"][side]
            cp_diagnostics.update({k: d[k] for k in ("searches", "support_calls", "empty_searches",
                                                   "empty_support", "replanning_searches")})
            cp_diagnostics.update({"positions_"+k: v for k, v in d["positions"].items()})
            cancel_reasons.update(d["cancellations"])
            search_reasons.update(d["search_reasons"])
        for sample in record["metrics"]["samples"]:
            name = case["agents"][sample["side"]]
            timings[f"{name}/{sample['kind']}_ms"].append(sample["ns"] / 1e6)
        for side, name in enumerate(case["agents"]):
            selected = [s for s in record["metrics"]["samples"] if s["side"] == side]
            # Reconstruct decision and turn samples without merging percentiles.
            pending, decision, turns = 0, [], defaultdict(float)
            for sample in selected:
                kind, ms = sample["kind"], sample["ns"] / 1e6
                if kind.startswith("prepare_"):
                    pending += ms
                elif kind == "act":
                    decision.append(pending + ms)
                    pending = 0
                elif kind == "callback":
                    decision[-1] += ms
                if kind.startswith("prepare_") or kind in ("act", "callback"):
                    turns[sample["turn"]] += ms
            timings[f"{name}/decision_ms"].extend(decision)
            timings[f"{name}/turn_ms"].extend(turns.values())
        memory.append(record["metrics"]["memory"]["peak_rss_bytes"])
    return {"matches": len(records), "outcomes": {key: {
                "count": len(values), "outcomes": dict(Counter(v["outcome"] for v in values)),
                "reasons": dict(Counter(v["reason"] for v in values)),
                "remaining_units": distribution([v["alive"] for v in values]),
                "remaining_hp": distribution([v["hp"] for v in values]),
            } for key, values in sorted(groups.items())},
            "timings": {key: distribution(values) for key, values in sorted(timings.items())},
            "peak_process_rss_bytes": distribution(memory),
            "cp_diagnostics": dict(cp_diagnostics), "cancellations": dict(cancel_reasons),
            "search_reasons": dict(search_reasons),
            "empty_search_fraction": cp_diagnostics["empty_searches"] / max(1, cp_diagnostics["searches"]),
            "replanning_search_fraction": cp_diagnostics["replanning_searches"] / max(1, cp_diagnostics["searches"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "scenarios/combat-ai-v1.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--case", help="Run one exact case ID")
    parser.add_argument("--quick", action="store_true", help="Two small cases; not acceptance")
    parser.add_argument("--resume", action="store_true", help="Reuse checks with identical run signature")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--signature", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.config = args.config.resolve()
    args.out = args.out.resolve()
    suite = load_suite(args.config)
    if args.worker:
        worker(args, suite)
        return
    args.out.mkdir(parents=True, exist_ok=True)
    identity = passport(args.config, suite)
    signature = digest({k: identity[k] for k in ("source_set_sha256", "resolved_suite_sha256",
                                                "binary_sha256", "python", "platform")})
    old = args.out / "passport.json"
    if args.resume and old.exists():
        previous = json.loads(old.read_text(encoding="utf-8"))
        if previous.get("run_signature") != signature:
            parser.error("Cannot resume: sources, suite, binaries or environment changed")
    identity["run_signature"] = signature
    write_json(old, identity)
    selected = cases(suite)
    if args.case:
        selected = [c for c in selected if c["id"] == args.case]
        if not selected:
            parser.error("Unknown case ID")
    elif args.quick:
        selected = [selected[0], selected[2]]
    print(f"Checking tactical regressions; {len(selected)} cases, each repeated", flush=True)
    tactical_checks(args.out)
    records = []
    failures = []
    for index, case in enumerate(selected):
        print(f"[{index+1}/{len(selected)}] {case['id']}", flush=True)
        checks = []
        for repeat in (0, 1):
            directory = args.out / "matches" / case["id"] / str(repeat)
            checked_path = directory / "checked.json"
            reusable = args.resume and reusable_check(directory, signature, case["id"])
            if not reusable:
                directory.mkdir(parents=True, exist_ok=True)
                # A stale completion record must never survive a failed rerun.
                checked_path.unlink(missing_ok=True)
                command = [sys.executable, str(Path(__file__).resolve()), "--config", str(args.config),
                           "--out", str(directory), "--worker", case["id"],
                           "--signature", signature]
                try:
                    completed = run_bounded(command, suite["timeout_seconds"])
                    if completed.returncode:
                        raise RuntimeError(completed.stdout + completed.stderr)
                except (subprocess.TimeoutExpired, RuntimeError) as exc:
                    failure = {"case": case["id"], "repeat": repeat, "error": str(exc)}
                    failures.append(failure)
                    write_json(directory / "failure.json", failure)
                    print(f"  FAILED: {str(exc)[-500:]}", flush=True)
                    break
            checks.append(json.loads(checked_path.read_text(encoding="utf-8")))
        if len(checks) != 2:
            continue
        if checks[0] != checks[1]:
            failures.append({"case": case["id"], "error": "Repeat differs"})
            continue
        directory = args.out / "matches" / case["id"] / "0"
        records.append({"case": {k: v for k, v in case.items() if k != "config"} | {
                           "config": {k: v for k, v in case["config"].items() if k != "map"}},
                        "result": {k: v for k, v in json.loads((directory / "result.json").read_text(
                            encoding="utf-8")).items() if k != "config"},
                        "checked": checks[0],
                        "metrics": json.loads((directory / "metrics.json").read_text(encoding="utf-8"))})
        summary = aggregate(records)
        write_json(args.out / "summary.json", {"acceptance": False, "summary": summary,
                   "failures": failures, "records": records})
    full = not args.quick and args.case is None and len(records) == len(cases(suite))
    current = passport(args.config, load_suite(args.config))
    current_signature = digest({k: current[k] for k in ("source_set_sha256", "resolved_suite_sha256",
                                                        "binary_sha256", "python", "platform")})
    if current_signature != signature:
        failures.append({"error": "Sources, suite, binaries or environment changed during the run"})
    write_json(args.out / "summary.json", {"schema": 1, "acceptance": full and not failures,
               "run_signature": signature, "summary": aggregate(records),
               "failures": failures, "records": records})
    write_report(args.out / "report.md", aggregate(records), full, failures, suite)
    print(json.dumps({"checked": len(records), "failures": len(failures),
                      "acceptance": full and not failures}), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
