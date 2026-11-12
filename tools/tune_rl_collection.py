"""Bounded streaming Collection tuning on a fixed policy and workload."""

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.rl_collection_hardware import hardware
from tools._collection_tuning import (
    DEFAULT_ENV_CAPACITY, TIE_FRACTION, Candidate, Measurements,
    preferred, streaming_candidates,
)
from tools._common import write_json, file_sha256
from tools._resources import collection_resources
from tools._process import stop_tree


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "rollout", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--cpu-budget", type=float)
    parser.add_argument("--budget-seconds", type=float, default=600)
    parser.add_argument(
        "--warmup-episodes",
        type=int,
        default=8,
        help="Cases in each Collection warmup; 0 uses the full workload",
    )
    parser.add_argument("--candidates", help="Explicit list, e.g. 12/12/1,16/16/1")
    parser.add_argument(
        "--scheduler", choices=("ready", "streaming"), default="streaming"
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.scheduler != "streaming" and not args.candidates:
        parser.error("Automatic tuning uses streaming; ready requires --candidates")
    started = perf_counter()
    if (
        args.warmup_episodes < 0
        or not math.isfinite(args.budget_seconds)
        or args.budget_seconds <= 0
        or (
            args.cpu_budget is not None
            and (not math.isfinite(args.cpu_budget) or args.cpu_budget <= 0)
        )
    ):
        parser.error(
            "CPU/time budgets must be positive finite numbers; warmup cases must be nonnegative"
        )
    sys.path.insert(0, str(ROOT))
    from civ5rl.config import load_config
    from civ5rl.experiments.ppo_artifacts import load_rollout

    load_config(args.config)
    machine = hardware()
    try:
        import pynvml

        pynvml.nvmlInit()
        machine["GPUs"] = [
            pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(i))
            for i in range(pynvml.nvmlDeviceGetCount())
        ]
        pynvml.nvmlShutdown()
    except Exception as exc:  # noqa: BLE001 -- GPU metadata may be unavailable in dry-run
        machine["GPU_query_error"] = f"{type(exc).__name__}: {exc}"
    cpu_budget = (
        args.cpu_budget
        if args.cpu_budget is not None
        else machine["effective_cpu_budget"]
    )
    base = max(1, math.ceil(cpu_budget))
    episodes = load_rollout(args.rollout)
    count = (
        args.episodes
        if args.episodes is not None
        else min(len(episodes), 2 * max(DEFAULT_ENV_CAPACITY, 2 * base))
    )
    if count < 1 or count > len(episodes):
        parser.error("Episode count must be positive and fit the saved rollout")
    max_envs = count
    if not args.candidates and count < 2:
        parser.error("Automatic search requires at least two saved episodes")
    try:
        candidates = (
            [
                Candidate(*map(int, part.split("/")), scheduler=args.scheduler)
                for part in args.candidates.split(",")
            ]
            if args.candidates
            else streaming_candidates(cpu_budget, count)
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    candidates = list(dict.fromkeys(candidates))
    if any(c.envs > count for c in candidates):
        parser.error("A candidate cannot have more environments than selected episodes")
    report = {
        "method": "streaming-quick-v1",
        "selection_trials": "all",
        "hardware": machine,
        "cpu_budget": cpu_budget,
        "episodes": count,
        "available_episodes": len(episodes),
        "budget_seconds": args.budget_seconds,
        "warmup_episodes": min(count, args.warmup_episodes or count),
        "warmup_workers": True,
        "collection_scheduler": args.scheduler,
        "tie_fraction": TIE_FRACTION,
        "confirmation_repeats_per_finalist": 1,
        "max_automatic_probes": 5,
        "max_envs": max_envs,
        "initial_env_capacity": candidates[0].envs,
        "initial_refill_expected": count > candidates[0].envs,
        "cpu_stat_enabled": sys.platform == "linux"
        and bool(machine.get("cgroup", {}).get("quota_sources")),
        "initial_candidates": [str(c) for c in candidates],
        "results": [],
    }
    if args.dry_run:
        print(json.dumps(report, indent=2))
        return
    if args.out.exists() and any(args.out.iterdir()):
        parser.error(
            "Output directory must be empty; use a new directory for each search"
        )
    args.out.mkdir(parents=True, exist_ok=True)
    protected = [
        args.config.resolve(),
        args.checkpoint.resolve(),
        args.rollout.resolve(),
    ]
    for name in ("state.json", "status.json"):
        path = args.checkpoint.resolve().parent.parent / name
        if path.exists():
            protected.append(path)
    report["protected_inputs"] = {str(path): file_sha256(path) for path in protected}
    del episodes
    rows = {}
    signature = None
    reference_trace = None
    exhausted = False
    deadline = started + args.budget_seconds

    def save():
        confirmed = len(finalists) == 2 and all(
            r.status == "ok" and len(r.confirmations) == 1 for r in finalists
        )
        best = preferred(finalists if confirmed else rows.values())
        report["elapsed_seconds"] = perf_counter() - started
        report["budget_exhausted"] = exhausted
        report["recommended"] = str(best.candidate) if best else None
        report["preliminary"] = not confirmed or best is None
        report["results"] = [
            {
                "candidate": str(r.candidate),
                **r.candidate.runtime(),
                "status": r.status,
                "median_seconds": r.median,
                "min_seconds": min((t["seconds"] for t in r.trials), default=None),
                "max_seconds": max((t["seconds"] for t in r.trials), default=None),
                "decisions_per_second": (
                    r.trials[-1]["positions"] / r.median
                    if r.median is not None
                    else None
                ),
                "trials": r.trials,
                "confirmations": r.confirmations,
                "sessions": r.sessions,
            }
            for r in rows.values()
        ]
        write_json(args.out / "summary.json", report)
        lines = [
            "# Short Collection tuning",
            "",
            (
                "Same cases and frozen policy; median of all successful trials. "
                "This is a bounded practical selection, not a claim of an optimum."
            ),
            "",
            "| env/workers/reply | Collection median, s | Range, s | Decisions/s | Trials | Status |",
            "|---|---:|---|---:|---:|---|",
        ]
        for item in report["results"]:
            if item["median_seconds"] is None:
                lines.append(
                    f"| {item['candidate']} | — | — | — | 0 | {item['status']} |"
                )
                continue
            lines.append(
                f"| {item['candidate']} | {item['median_seconds']:.3f} | "
                f"{item['min_seconds']:.3f}–{item['max_seconds']:.3f} | "
                f"{item['decisions_per_second']:.1f} | {len(item['trials'])} | {item['status']} |"
            )
        lines += [
            "",
            f"Recommended: {report['recommended']}; preliminary: {report['preliminary']}.",
            (
                f"Elapsed: {report['elapsed_seconds']:.1f}/{args.budget_seconds:g} s. "
                f"Budget exhausted: {exhausted}."
            ),
            "",
            (
                "Per-trial timings, inference batch, CPU/RAM, NVML, quota counters and "
                "verification are in summary.json and the profiler session directories. "
                "CPU pressure or GPU load do not override Collection time in selection."
            ),
        ]
        (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        runtime_path = args.out / "recommended-runtime.json"
        if best:
            write_json(runtime_path, best.candidate.runtime())
        elif runtime_path.exists():
            runtime_path.unlink()

    finalists = []

    def measure(candidate, confirmation=False):
        nonlocal signature, reference_trace, exhausted
        if exhausted or candidate.envs > max_envs:
            return None
        if not confirmation and candidate in rows:
            return rows[candidate]
        row = rows.setdefault(candidate, Measurements(candidate))
        remaining = deadline - perf_counter()
        if remaining <= 0:
            exhausted = True
            if not row.trials:
                row.status = "not_started"
            save()
            return row
        directory = (
            args.out / str(candidate).replace("/", "-") / f"session-{len(row.sessions)}"
        )
        directory.mkdir(parents=True)
        command = [
            sys.executable,
            str(ROOT / "tools/profile_rl_runtime.py"),
            "collection",
            "--config",
            str(args.config.resolve()),
            "--checkpoint",
            str(args.checkpoint.resolve()),
            "--rollout",
            str(args.rollout.resolve()),
            "--out",
            str(directory.resolve()),
            "--episodes",
            str(count),
            "--envs",
            str(candidate.envs),
            "--workers",
            str(candidate.workers),
            "--reply-size",
            str(candidate.reply),
            "--scheduler",
            candidate.scheduler,
            "--inference-min-batch",
            str(candidate.inference_min_batch),
            "--inference-wait-seconds",
            str(candidate.inference_wait_seconds),
            "--warmup",
            "1",
            "--warmup-workers",
            "--warmup-episodes",
            str(args.warmup_episodes),
            "--repeat",
            "1",
        ]
        if report["cpu_stat_enabled"]:
            command.append("--cpu-stat")
        if reference_trace:
            command += ["--reference-trace", str(reference_trace.resolve())]
        print(
            f"{'Confirm' if confirmation else 'Probe'} {candidate}; remaining={remaining:.1f}s",
            flush=True,
        )
        tick = perf_counter()
        session = {"directory": str(directory), "confirmation": confirmation}
        row.sessions.append(session)
        with (directory / "stdout.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
            )
            try:
                process.wait(timeout=max(0.001, deadline - perf_counter()))
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                stop_tree(process)
                exhausted = True
                session["error"] = type(exc).__name__
        session.update(
            wall_seconds=perf_counter() - tick, returncode=process.returncode
        )
        session["collection_resources"] = collection_resources(
            directory / "resources.jsonl"
        )
        path = directory / "result.json"
        if path.exists():
            output = json.loads(path.read_text(encoding="utf-8"))
            session["setup_seconds"] = output.get("setup_seconds")
            session["worker_warmup_seconds"] = output.get("worker_warmup_seconds")
            session["warmup_trials"] = output.get("warmup_trials", [])
            trials = output["trials"]
        else:
            trials = []
        if process.returncode != 0 or not trials:
            if not row.trials:
                row.status = "timeout" if exhausted else "error"
            session.setdefault("error", "Profiler did not finish; see stdout.log")
        else:
            trial = trials[0]
            actual = (trial["commands_hash"], trial["results_hash"])
            if signature is None:
                signature = actual
                reference_trace = directory / "trace.npz"
            if actual != signature:
                row.status = "incomparable"
                session["error"] = (
                    "Commands/outcomes differ from the reference candidate"
                )
            else:
                row.status = "ok"
                trial["resources"] = session["collection_resources"]
                row.trials.append(trial)
                if confirmation:
                    row.confirmations.append(trial)
                print(
                    f"  {trial['seconds']:.3f}s, {trial['positions'] / trial['seconds']:.1f} decisions/s",
                    flush=True,
                )
        save()
        return row

    try:
        for candidate in candidates:
            measure(candidate)
        if not args.candidates and not exhausted:
            best = preferred(rows.values())
            if best:
                candidate = best.candidate
                # Change only capacity. If one env per worker won, try a second;
                # otherwise check whether half the env pool retains the speed.
                envs = (
                    max(candidate.workers, candidate.envs // 2)
                    if candidate.envs > candidate.workers
                    else min(count, 2 * candidate.envs)
                )
                measure(replace(candidate, envs=envs))
            best = preferred(rows.values())
            if best and best.candidate.envs >= 2 * best.candidate.workers:
                measure(replace(best.candidate, reply=2))
        while not exhausted:
            valid = sorted(
                (r for r in rows.values() if r.status == "ok"), key=lambda r: r.median
            )
            best = preferred(valid)
            if best is None:
                break
            # If a finalist fails comparison, confirm its replacement too.
            # Never promote an unconfirmed fallback while time remains.
            finalists = [best, *(r for r in valid if r is not best)][:2]
            pending = [r for r in finalists if not r.confirmations]
            if not pending:
                break
            order = list(rows)
            row = max(pending, key=lambda r: order.index(r.candidate))
            measure(row.candidate, confirmation=True)
            if row.status == "ok" and not row.confirmations:
                row.status = "confirmation_failed"
    finally:
        changed = [
            path
            for path, digest in report["protected_inputs"].items()
            if file_sha256(Path(path)) != digest
        ]
        report["protected_inputs_unchanged"] = not changed
        save()
        if changed:
            raise AssertionError(f"Protected inputs changed: {changed}")
    print("\nenv/workers/reply  median(s)  status")
    for row in rows.values():
        value = f"{row.median:.3f}" if row.median is not None else "-"
        print(f"{row.candidate!s:>17}  {value:>9}  {row.status}")
    print(f"Recommended: {report['recommended']}; preliminary={report['preliminary']}")


if __name__ == "__main__":
    main()
