"""Choose Collection env/workers/reply settings on a fixed policy and workload."""

import argparse
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from time import perf_counter

from rl_collection_hardware import hardware

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, order=True)
class Candidate:
    envs: int
    workers: int
    reply: int = 1
    scheduler: str = "ready"
    inference_min_batch: int = 0
    inference_wait_seconds: float = 0.0

    def __post_init__(self):
        if (
            min(self.envs, self.workers, self.reply) < 1
            or self.workers > self.envs
            or (self.scheduler != "streaming" and self.envs % self.workers)
            or self.reply > math.ceil(self.envs / self.workers)
        ):
            raise ValueError(
                "Require workers <= envs and a reply size that fits worker slots"
            )

    def __str__(self):
        label = f"{self.envs}/{self.workers}/{self.reply}"
        if self.inference_min_batch or self.inference_wait_seconds:
            label += f"@{self.inference_min_batch}/{self.inference_wait_seconds:g}"
        return label

    def runtime(self):
        result = {
            "num_envs": self.envs,
            "cpu_workers": self.workers,
            "worker_reply_size": self.reply,
        }
        if (
            self.scheduler == "streaming"
            or self.inference_min_batch
            or self.inference_wait_seconds
        ):
            result.update(
                collection_scheduler=self.scheduler,
                inference_min_batch=self.inference_min_batch,
                inference_wait_seconds=self.inference_wait_seconds,
            )
        return result


@dataclass
class Measurements:
    candidate: Candidate
    trials: list = field(default_factory=list)
    confirmations: list = field(default_factory=list)
    sessions: list = field(default_factory=list)
    status: str = "pending"

    @property
    def median(self):
        trials = self.confirmations if len(self.confirmations) == 3 else self.trials
        return statistics.median(t["seconds"] for t in trials) if trials else None


def initial_candidates(budget, max_envs):
    base = max(1, math.floor(budget))
    counts = sorted(
        {max(1, math.floor(value + 0.5)) for value in (base / 2, base, 1.5 * base)}
    )
    counts = [value for value in counts if value <= max_envs]
    # Small fixed datasets still need an executable starting point.
    return [Candidate(value, value) for value in counts] or [
        Candidate(max_envs, max_envs)
    ]


def preferred(rows):
    valid = [r for r in rows if r.status == "ok" and r.median is not None]
    if not valid:
        return None
    fastest = min(r.median for r in valid)
    close = [r for r in valid if r.median <= fastest * 1.03]
    return min(
        close,
        key=lambda r: (
            r.candidate.workers,
            r.candidate.envs,
            r.candidate.inference_wait_seconds,
            r.candidate.reply,
            r.candidate.inference_min_batch,
        ),
    )


def _json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collection_resources(path, phase="collection"):
    if not path.exists():
        return {}
    samples = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    selected = [s for s in samples if s["phase"] == phase]
    result = {"samples": len(selected)}
    for name in ("gpu_core_load_percent", "cpu_tree_percent"):
        weighted = duration = 0.0
        for before, after in pairwise(samples):
            if (
                before["phase"] == after["phase"] == phase
                and before.get(name) is not None
            ):
                seconds = after["elapsed_seconds"] - before["elapsed_seconds"]
                weighted += before[name] * seconds
                duration += seconds
        result[f"{name}_mean"] = weighted / duration if duration else None
    for name in ("ram_tree_rss_bytes", "vram_used_bytes"):
        result[f"{name}_peak"] = max(
            (s[name] for s in selected if s.get(name) is not None), default=None
        )
    return result


def stop_tree(process):
    """Give the profiler a chance to close workers and unlink shared memory."""
    import psutil

    if process.poll() is not None:
        return
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            process.wait()
            return
        try:
            process.wait(timeout=5)
            return
        except subprocess.TimeoutExpired:
            pass
    try:
        root = psutil.Process(process.pid)
        descendants = root.children(recursive=True)
        for child in descendants:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        root.kill()
        psutil.wait_procs([*descendants, root], timeout=5)
    except psutil.NoSuchProcess:
        pass
    process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "checkpoint", "rollout", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--cpu-budget", type=float)
    parser.add_argument("--budget-seconds", type=float, default=900)
    parser.add_argument(
        "--warmup-episodes",
        type=int,
        default=8,
        help="Cases in each Collection warmup; 0 uses the full workload",
    )
    parser.add_argument("--candidates", help="Explicit list, e.g. 12/12/1,16/16/1")
    parser.add_argument("--scheduler", choices=("ready", "streaming"), default="ready")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
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

    config = load_config(args.config)
    if config.get("performance", {}).get("collection_scheduler") != "ready":
        parser.error(
            "Reply-size tuning requires a config with collection_scheduler=ready"
        )
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
    base = max(1, math.floor(cpu_budget))
    episodes = load_rollout(args.rollout)
    count = (
        args.episodes
        if args.episodes is not None
        else min(len(episodes), max(32, 4 * base))
    )
    if count < 1 or count > len(episodes):
        parser.error("Episode count must be positive and fit the saved rollout")
    max_envs = (
        count
        if args.candidates or args.scheduler == "streaming"
        else max(1, count // 2)
    )
    if not args.candidates and count < 2:
        parser.error("Automatic search requires at least two saved episodes")
    try:
        candidates = (
            [
                Candidate(*map(int, part.split("/")), scheduler=args.scheduler)
                for part in args.candidates.split(",")
            ]
            if args.candidates
            else (
                [
                    Candidate(count, workers, scheduler="streaming")
                    for workers in sorted(
                        {
                            min(count, max(1, math.floor(value + 0.5)))
                            for value in (base / 2, base, 1.5 * base, 2 * base, count)
                        }
                    )
                ]
                if args.scheduler == "streaming"
                else initial_candidates(cpu_budget, max_envs)
            )
        )
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))
    candidates = list(dict.fromkeys(candidates))
    if any(c.envs > count for c in candidates):
        parser.error("A candidate cannot have more environments than selected episodes")
    report = {
        "hardware": machine,
        "cpu_budget": cpu_budget,
        "episodes": count,
        "available_episodes": len(episodes),
        "budget_seconds": args.budget_seconds,
        "warmup_episodes": min(count, args.warmup_episodes or count),
        "collection_scheduler": args.scheduler,
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
    report["protected_inputs"] = {str(path): _sha(path) for path in protected}
    del episodes
    rows = {}
    signature = None
    reference_trace = None
    exhausted = False
    deadline = started + args.budget_seconds

    def save():
        confirmed = len(finalists) == 2 and all(
            len(r.confirmations) == 3 for r in finalists
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
        _json(args.out / "summary.json", report)
        runtime_path = args.out / "recommended-runtime.json"
        if best:
            _json(runtime_path, best.candidate.runtime())
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
            "--warmup-episodes",
            str(args.warmup_episodes),
            "--repeat",
            "1",
        ]
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
        if not args.candidates and not exhausted and args.scheduler == "streaming":
            best = preferred(rows.values())
            if best:
                workers = best.candidate.workers
                for envs in sorted(
                    {min(count, multiplier * workers) for multiplier in (1, 2, 4)}
                ):
                    measure(Candidate(envs, workers, scheduler="streaming"))
            best = preferred(rows.values())
            if best and best.candidate.envs > best.candidate.workers:
                measure(replace(best.candidate, reply=2))
            best = preferred(rows.values())
            if best:
                for minimum, delay in ((8, 0.0005), (16, 0.001)):
                    measure(
                        replace(
                            best.candidate,
                            inference_min_batch=minimum,
                            inference_wait_seconds=delay,
                        )
                    )
        elif not args.candidates and not exhausted:
            probes = [
                rows[c] for c in candidates if c in rows and rows[c].status == "ok"
            ]
            if len(probes) >= 2:
                fastest = min(probes, key=lambda r: r.median)
                ordered = sorted(probes, key=lambda r: r.candidate.workers)
                if (
                    fastest is ordered[-1]
                    and fastest.median <= ordered[-2].median * 0.95
                ):
                    measure(Candidate(2 * base, 2 * base))
                elif fastest is not ordered[0] and fastest is not ordered[-1]:
                    neighbor = min(
                        (r for r in probes if r is not fastest), key=lambda r: r.median
                    )
                    middle = (
                        fastest.candidate.workers + neighbor.candidate.workers + 1
                    ) // 2
                    measure(Candidate(middle, middle))
            best = preferred(rows.values())
            if best:
                workers = best.candidate.workers
                doubled = measure(Candidate(2 * workers, workers))
                if (
                    doubled
                    and doubled.status == "ok"
                    and doubled.median <= best.median * 0.95
                ):
                    measure(Candidate(3 * workers, workers))
            best = preferred(rows.values())
            if best and best.candidate.envs >= 2 * best.candidate.workers:
                measure(Candidate(best.candidate.envs, best.candidate.workers, 2))
        finalists = sorted(
            (r for r in rows.values() if r.status == "ok"), key=lambda r: r.median
        )[:2]
        for repetition in range(3):
            for row in finalists if repetition % 2 == 0 else list(reversed(finalists)):
                measure(row.candidate, confirmation=True)
    finally:
        changed = [
            path
            for path, digest in report["protected_inputs"].items()
            if _sha(Path(path)) != digest
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
