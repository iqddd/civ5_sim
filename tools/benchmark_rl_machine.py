"""Machine benchmark disabled pending rework; all entrypoints raise NotImplementedError."""

# TODO: Rework this benchmark before enabling any of its functions. The v1
# series is closed. Separate runtime tuning from final machine measurements,
# agree on the fixed workload/refill behavior, validate tuner completion and
# GPU identity, and separate execution provenance from measurement provenance.
# The former implementation is retained below for that rework, but every
# module-defined entrypoint (including helpers and the CLI) is disabled.

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.rl_collection_hardware import hardware
from tools._common import write_json, file_sha256
from tools._resources import collection_resources
from tools._collection_tuning import initial_candidates
from tools._process import stop_tree

METHOD = "cp-collection-v1"
TOOLS = (
    "benchmark_rl_machine.py",
    "profile_rl_runtime.py",
    "rl_cpu_replay.py",
    "tune_rl_collection.py",
    "rl_collection_hardware.py",
)

_REWORK_REASON = (
    "benchmark_rl_machine requires rework and is disabled. "
    "Use tools/tune_rl_collection.py for runtime tuning."
)


def read(path):
    raise NotImplementedError(_REWORK_REASON)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fingerprint(value):
    raise NotImplementedError(_REWORK_REASON)
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def text_sha(path):
    raise NotImplementedError(_REWORK_REASON)
    # Git checkouts can use CRLF on Windows and LF on Linux.
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def source_hashes():
    raise NotImplementedError(_REWORK_REASON)
    paths = [
        p
        for directory in ("civ5rl", "civ5sim", "native")
        for p in (ROOT / directory).rglob("*")
        if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".inc")
    ]
    paths += [
        ROOT / "build" / name
        for name in ("combat_rules.h", "cp_geometry.inc", "combat_rules.json")
    ]
    paths += [
        ROOT / name
        for name in (
            "requirements.txt",
            "requirements-rl.txt",
            "tools/build_combat_ai.py",
        )
    ]
    return {p.relative_to(ROOT).as_posix(): text_sha(p) for p in sorted(paths)}


def tool_hashes():
    raise NotImplementedError(_REWORK_REASON)
    return {name: text_sha(ROOT / "tools" / name) for name in TOOLS}


def chosen_collection(summary):
    raise NotImplementedError(_REWORK_REASON)
    return next(
        (r for r in summary["results"] if r["candidate"] == summary["recommended"]),
        None,
    )


def collection_trials(summary, row):
    raise NotImplementedError(_REWORK_REASON)
    if summary.get("selection_trials") == "all":
        return row["trials"]
    return row["confirmations"] or row["trials"]


def validate_collection(summary, manifest):
    raise NotImplementedError(_REWORK_REASON)
    if summary["episodes"] != manifest["episodes"]:
        raise ValueError("Collection used a different number of episodes")
    protected = set(summary["protected_inputs"].values())
    for name in ("rollout.pt", "checkpoint.pt"):
        if manifest["files"][name] not in protected:
            raise ValueError(f"Collection used a different {name}")
    # The earlier profiler protected the original inherited config, rather than
    # its resolved copy. Both hashes are recorded by prepare.
    if not protected.intersection(manifest["config_input_hashes"]):
        raise ValueError("Collection used a different config")
    row = chosen_collection(summary)
    if row is None or row["status"] != "ok":
        raise ValueError("Collection has no usable recommendation")
    trials = collection_trials(summary, row)
    signatures = {(t["commands_hash"], t["results_hash"]) for t in trials}
    if len(signatures) != 1:
        raise ValueError("Collection commands/outcomes differ between trials")
    actual = dict(zip(("commands_hash", "results_hash"), next(iter(signatures))))
    expected = manifest.get("collection_signature")
    if expected is not None and actual != expected:
        raise ValueError("Collection differs from the frozen workload")
    return row, actual


def prepare(args):
    raise NotImplementedError(_REWORK_REASON)
    sys.path.insert(0, str(ROOT))
    from civ5rl.config import load_config
    from civ5rl.experiments.ppo_artifacts import load_rollout
    from civ5sim.engine import digest

    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Bundle output must be empty")
    episodes = load_rollout(args.rollout)
    if not 2 <= args.episodes <= len(episodes):
        raise ValueError("Episode count must be between 2 and the saved rollout size")
    selected = episodes[: args.episodes]
    args.out.mkdir(parents=True, exist_ok=True)
    # Keep original serialization; the fixed prefix is part of the manifest.
    shutil.copyfile(args.rollout, args.out / "rollout.pt")
    shutil.copyfile(args.checkpoint, args.out / "checkpoint.pt")
    write_json(args.out / "config.json", load_config(args.config))
    coverage = {}
    for label, extract in {
        "map_sizes": lambda ep: (
            f"{ep['spec']['scenario']['map']['width']}x{ep['spec']['scenario']['map']['height']}"
        ),
        "army_sizes": lambda ep: "/".join(str(len(a)) for a in ep["spec"]["armies"]),
        "learner_sides": lambda ep: str(ep["spec"]["learner_side"]),
        "outcomes": lambda ep: str(ep["result"]["winner"]),
        "opponents": lambda ep: str(ep["spec"]["opponent"]),
        "placements": lambda ep: str(ep["spec"].get("placement")),
        "first_contact_plies": lambda ep: str(ep["result"].get("first_contact_turn")),
        "duration_bands": lambda ep: (
            "short_0_10"
            if ep["result"]["rounds"] <= 10
            else "medium_10_20"
            if ep["result"]["rounds"] <= 20
            else "long_20_plus"
        ),
    }.items():
        coverage[label] = dict(Counter(extract(ep) for ep in selected))
    coverage["unit_types"] = dict(
        Counter(
            unit["type"] for ep in selected for unit in ep["spec"]["scenario"]["units"]
        )
    )
    manifest = {
        "schema": 1,
        "name": args.name,
        "method": METHOD,
        "episodes": len(selected),
        "available_episodes": len(episodes),
        "positions": sum(len(ep["chosen"]) for ep in selected),
        "commands": sum(len(ep["joint_commands"]) for ep in selected),
        "specs_hash": digest([ep["spec"] for ep in selected]),
        "cpu_signature": {
            "commands_hash": digest([ep["joint_commands"] for ep in selected]),
            "results_hash": digest(
                [
                    {k: v for k, v in ep["result"].items() if k != "timings"}
                    for ep in selected
                ]
            ),
        },
        "files": {
            name: file_sha256(args.out / name)
            for name in ("config.json", "rollout.pt", "checkpoint.pt")
        },
        "config_input_hashes": sorted(
            {file_sha256(args.config), text_sha(args.config), file_sha256(args.out / "config.json")}
        ),
        "execution_sources": source_hashes(),
        "measurement_tools": tool_hashes(),
        "coverage": coverage,
        "auto_env_limit": max(1, len(selected) // 2),
    }
    if args.collection_reference:
        reference = read(args.collection_reference)
        row, signature = validate_collection(reference, manifest)
        manifest["collection_signature"] = signature
        manifest["collection_reference"] = {
            "summary_sha256": file_sha256(args.collection_reference),
            "median_seconds": row["median_seconds"],
            "preliminary": reference["preliminary"],
            "candidate": reference["recommended"],
        }
        manifest["collection_GPU_class"] = reference["hardware"].get("GPUs")
    manifest["workload_id"] = fingerprint(manifest)
    write_json(args.out / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "bundle": str(args.out),
                "workload_id": manifest["workload_id"],
                "coverage": coverage,
            },
            indent=2,
        )
    )


def verify_bundle(bundle):
    raise NotImplementedError(_REWORK_REASON)
    manifest = read(bundle / "manifest.json")
    if manifest["method"] != METHOD:
        raise ValueError("Unsupported benchmark method")
    if (
        fingerprint({k: v for k, v in manifest.items() if k != "workload_id"})
        != manifest["workload_id"]
    ):
        raise ValueError("Bundle manifest was modified")
    for name, expected in manifest["files"].items():
        if file_sha256(bundle / name) != expected:
            raise ValueError(f"Bundle file changed: {name}")
    if source_hashes() != manifest["execution_sources"]:
        raise ValueError("Execution sources differ; prepare a new workload version")
    if tool_hashes() != manifest["measurement_tools"]:
        raise ValueError("Measurement tools differ; prepare a new workload version")
    return manifest


def summarize(trials, units, unit_name):
    raise NotImplementedError(_REWORK_REASON)
    times = [t["seconds"] for t in trials]
    median = statistics.median(times)
    return {
        "median_seconds": median,
        "min_seconds": min(times),
        "max_seconds": max(times),
        "spread_percent": 100 * (max(times) - min(times)) / median,
        "unstable": (max(times) - min(times)) / median > 0.10,
        "measurements": len(times),
        f"{unit_name}_per_second": units / median,
    }


def run(args):
    raise NotImplementedError(_REWORK_REASON)
    started = perf_counter()
    deadline = started + args.budget_seconds
    manifest = verify_bundle(args.bundle)
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Run output must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    machine = hardware()
    try:
        import pynvml

        pynvml.nvmlInit()
        try:
            machine["GPUs"] = [
                pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(i))
                for i in range(pynvml.nvmlDeviceGetCount())
            ]
            machine["GPU_UUIDs"] = [
                pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(i))
                for i in range(pynvml.nvmlDeviceGetCount())
            ]
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # noqa: BLE001 -- CPU benchmarks also work without NVML
        machine["GPU_query_error"] = f"{type(exc).__name__}: {exc}"
    reference = read(args.reference) if args.reference else None
    if reference and reference["workload_id"] != manifest["workload_id"]:
        raise ValueError("Reference belongs to a different workload version")
    report = {
        "schema": 1,
        "method": METHOD,
        "workload_id": manifest["workload_id"],
        "workload": manifest["name"],
        "machine_label": args.machine_label,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "hardware": machine,
        "budget_seconds": args.budget_seconds,
        "price_per_hour": args.price_per_hour,
        "episodes": manifest["episodes"],
        "coverage": manifest["coverage"],
        "auto_env_limit": manifest["auto_env_limit"],
        "reference": str(args.reference) if args.reference else "this_machine",
        "benchmarks": {},
        "cpu_search": [],
        "sessions": [],
    }

    def save():
        scores = report["benchmarks"]
        for name, row in scores.items():
            base = reference["benchmarks"].get(name) if reference else row
            comparable = base and base.get("signature") == row.get("signature")
            if name == "collection" and reference:
                comparable = comparable and reference["hardware"].get(
                    "GPUs"
                ) == machine.get("GPUs")
            row["comparable_to_reference"] = bool(comparable)
            row["score"] = (
                100 * base["median_seconds"] / row["median_seconds"]
                if comparable and not row["preliminary"] and not base["preliminary"]
                else None
            )
            if args.price_per_hour and row["score"] is not None:
                row["score_per_dollar_per_hour"] = row["score"] / args.price_per_hour
        serial, parallel = scores.get("cp_serial"), scores.get("cp_parallel")
        if serial and parallel:
            parallel["speedup_over_serial"] = (
                serial["median_seconds"] / parallel["median_seconds"]
            )
        report["elapsed_seconds"] = perf_counter() - started
        report["budget_exhausted"] = perf_counter() >= deadline
        report["preliminary"] = len(scores) < 3 or any(
            row["preliminary"] for row in scores.values()
        )
        write_json(args.out / "summary.json", report)
        lines = [
            f"# {manifest['name']} — {args.machine_label or machine['CPU']}",
            "",
            f"Workload: `{manifest['workload_id']}`",
            f"CPU: {machine['CPU']}; quota: {machine['effective_cpu_budget']}; GPU: {machine.get('GPUs')}",
            f"Price/hour: {args.price_per_hour}; preliminary: {report['preliminary']}",
            "",
            "| Benchmark | Median, s | Min–max, s | Score | Preliminary |",
            "|---|---:|---:|---:|---|",
        ]
        for name, row in scores.items():
            score = "—" if row["score"] is None else f"{row['score']:.1f}"
            lines.append(
                f"| {name} | {row['median_seconds']:.3f} | {row['min_seconds']:.3f}–{row['max_seconds']:.3f} | {score} | {row['preliminary']} |"
            )
        if parallel:
            lines += [
                "",
                f"CPU workers: {parallel['workers']}; speedup over serial: {parallel.get('speedup_over_serial'):.2f}x"
                if serial
                else f"CPU workers: {parallel['workers']}",
            ]
        collection = scores.get("collection")
        if collection:
            lines += [
                "",
                f"Collection runtime: {collection['runtime']}; decisions/s: {collection['decisions_per_second']:.1f}",
                f"Collection $/million decisions: {collection.get('dollars_per_million_decisions')}",
            ]
        (args.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def execute(command, directory, limit):
        remaining = min(deadline, limit) - perf_counter()
        if remaining <= 0:
            return None
        directory.mkdir(parents=True)
        tick = perf_counter()
        session = {
            "directory": str(directory.relative_to(args.out)),
            "command": list(map(str, command)),
        }
        report["sessions"].append(session)
        with (directory / "stdout.log").open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(
                list(map(str, command)),
                cwd=ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=os.name != "nt",
            )
            try:
                process.wait(timeout=max(0.001, min(deadline, limit) - perf_counter()))
            except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
                stop_tree(process)
                session["error"] = type(exc).__name__
                if isinstance(exc, KeyboardInterrupt):
                    raise
        session.update(
            returncode=process.returncode, wall_seconds=perf_counter() - tick
        )
        if (directory / "resources.jsonl").exists():
            session["resources"] = collection_resources(
                directory / "resources.jsonl", "cpu"
            )
        save()
        if process.returncode:
            session.setdefault("error", "See stdout.log")
            return None
        return directory

    cpu_deadline = (
        deadline
        if args.reuse_collection
        else min(deadline, started + args.budget_seconds * 0.45)
    )
    cpu_rows = {}
    history_reference = None

    def measure_cpu(workers, repeat=1, confirmation=False):
        nonlocal history_reference
        directory = args.out / "cpu" / f"w{workers}-session{len(report['sessions'])}"
        print(
            f"CPU workers={workers}, repeat={repeat}, confirmation={confirmation}",
            flush=True,
        )
        command = [
            sys.executable,
            ROOT / "tools/profile_rl_runtime.py",
            "cpu",
            "--cpu-benchmark",
            "--rollout",
            args.bundle / "rollout.pt",
            "--episodes",
            manifest["episodes"],
            "--workers",
            workers,
            "--warmup",
            1,
            "--repeat",
            repeat,
            "--out",
            directory,
        ]
        if history_reference:
            command += ["--cpu-history-reference", history_reference]
        completed = execute(command, directory, cpu_deadline)
        if not completed:
            return None
        output = read(directory / "result.json")
        history_reference = history_reference or directory / "result.json"
        report["sessions"][-1]["setup_seconds"] = output["setup_seconds"]
        report["sessions"][-1]["history_verification_seconds"] = output[
            "history_verification_seconds"
        ]
        report["sessions"][-1]["native_library_sha256"] = output[
            "native_library_sha256"
        ]
        trials = output["trials"]
        if any(
            {k: t[k] for k in manifest["cpu_signature"]} != manifest["cpu_signature"]
            for t in trials
        ):
            raise AssertionError("CPU workload differs from the bundle")
        row = cpu_rows.setdefault(
            workers, {"workers": workers, "trials": [], "confirmations": []}
        )
        row["trials"].extend(trials)
        if confirmation or repeat == 3:
            row["confirmations"].extend(trials)
        report["cpu_search"] = list(cpu_rows.values())
        save()
        return row

    def cpu_time(row):
        return statistics.median(t["seconds"] for t in row["trials"])

    def cpu_result(row):
        trials = (
            row["confirmations"] if len(row["confirmations"]) >= 3 else row["trials"]
        )
        return {
            **summarize(trials, manifest["commands"], "commands"),
            "workers": row["workers"],
            "trials": trials,
            "episodes_per_second": manifest["episodes"]
            / statistics.median(t["seconds"] for t in trials),
            "signature": manifest["cpu_signature"],
            "preliminary": len(row["confirmations"]) < 3,
        }

    try:
        save()
        serial = measure_cpu(1, repeat=3)
        if serial:
            report["benchmarks"]["cp_serial"] = cpu_result(serial)
            save()
        counts = sorted(
            {
                c.workers
                for c in initial_candidates(
                    machine["effective_cpu_budget"], manifest["auto_env_limit"]
                )
            }
        )
        for workers in counts:
            if workers not in cpu_rows:
                measure_cpu(workers)
        candidates = [cpu_rows[w] for w in counts if w in cpu_rows]
        if len(candidates) >= 2:
            best = min(candidates, key=cpu_time)
            if (
                best is candidates[-1]
                and cpu_time(best) <= cpu_time(candidates[-2]) * 0.95
            ):
                extra = 2 * max(1, math.floor(machine["effective_cpu_budget"]))
                if extra <= manifest["auto_env_limit"] and extra not in cpu_rows:
                    measure_cpu(extra)
            elif best is not candidates[0] and best is not candidates[-1]:
                neighbor = min((r for r in candidates if r is not best), key=cpu_time)
                middle = (best["workers"] + neighbor["workers"] + 1) // 2
                if middle not in cpu_rows:
                    measure_cpu(middle)
        finalists = sorted(cpu_rows.values(), key=cpu_time)[:2]
        for repetition in range(3):
            for row in finalists if repetition % 2 == 0 else list(reversed(finalists)):
                if len(row["confirmations"]) < 3:
                    measure_cpu(row["workers"], confirmation=True)
        if finalists:
            confirmed = len(finalists) == min(2, len(cpu_rows)) and all(
                len(r["confirmations"]) >= 3 for r in finalists
            )
            pool = finalists if confirmed else list(cpu_rows.values())
            best_time = min(
                statistics.median(
                    t["seconds"] for t in (r["confirmations"] or r["trials"])
                )
                for r in pool
            )
            close = [
                r
                for r in pool
                if statistics.median(
                    t["seconds"] for t in (r["confirmations"] or r["trials"])
                )
                <= best_time * 1.03
            ]
            best = min(close, key=lambda r: r["workers"])
            result = cpu_result(best)
            result["preliminary"] |= not confirmed
            result["at_search_limit"] = best["workers"] == manifest["auto_env_limit"]
            report["benchmarks"]["cp_parallel"] = result
            save()

        if args.reuse_collection:
            collection = read(args.reuse_collection)
            frozen = manifest.get("collection_reference", {})
            if file_sha256(args.reuse_collection) != frozen.get("summary_sha256"):
                raise ValueError("Only the frozen Collection reference may be reused")
            for field in ("CPU", "cpu_affinity", "effective_cpu_budget", "platform"):
                if collection["hardware"][field] != machine[field]:
                    raise ValueError(
                        f"Reused Collection belongs to different hardware: {field}"
                    )
            if collection["hardware"].get("GPUs") != machine.get("GPUs"):
                raise ValueError("Reused Collection belongs to a different GPU class")
            shutil.copyfile(args.reuse_collection, args.out / "collection-reused.json")
            report["collection_reused"] = True
        else:
            remaining = deadline - perf_counter()
            if remaining <= 0:
                return
            directory = args.out / "collection"
            command = [
                sys.executable,
                ROOT / "tools/tune_rl_collection.py",
                "--config",
                args.bundle / "config.json",
                "--checkpoint",
                args.bundle / "checkpoint.pt",
                "--rollout",
                args.bundle / "rollout.pt",
                "--episodes",
                manifest["episodes"],
                "--budget-seconds",
                max(0.001, remaining - 2),
                "--out",
                directory,
            ]
            # The tuner owns its output directory, so place the wrapper log beside it.
            execute(command, args.out / "collection-launch", deadline)
            path = directory / "summary.json"
            if not path.exists() or read(path).get("recommended") is None:
                return
            collection = read(path)
        selected, signature = validate_collection(collection, manifest)
        trials = collection_trials(collection, selected)
        result = {
            **summarize(trials, trials[0]["positions"], "decisions"),
            "signature": signature,
            "preliminary": collection["preliminary"],
            "runtime": {
                k: selected[k]
                for k in (
                    "num_envs",
                    "cpu_workers",
                    "worker_reply_size",
                    "collection_scheduler",
                    "inference_min_batch",
                    "inference_wait_seconds",
                )
                if k in selected
            },
            "at_search_limit": selected["num_envs"]
            == collection.get("max_envs", manifest["auto_env_limit"]),
            "trials": trials,
            "sessions": selected["sessions"],
        }
        expected_gpu = manifest.get("collection_GPU_class")
        result["GPU_class_matches_workload"] = (
            expected_gpu is None or expected_gpu == machine.get("GPUs")
        )
        result["preliminary"] |= not result["GPU_class_matches_workload"]
        if args.price_per_hour:
            result["dollars_per_million_decisions"] = (
                args.price_per_hour * 1e6 / (3600 * result["decisions_per_second"])
            )
        report["benchmarks"]["collection"] = result
        write_json(args.out / "recommended-runtime.json", result["runtime"])
    finally:
        try:
            verify_bundle(args.bundle)
            report["bundle_unchanged"] = True
        finally:
            save()
            print((args.out / "report.md").read_text(encoding="utf-8"), flush=True)


def compare(args):
    raise NotImplementedError(_REWORK_REASON)
    reports = [read(path) for path in args.summaries]
    if len({r["workload_id"] for r in reports}) != 1:
        raise ValueError("Cannot compare different workload versions")
    baseline = reports[0]
    rows = []
    for report in reports:
        row = {
            "machine": report["machine_label"] or report["hardware"]["CPU"],
            "cpu_budget": report["hardware"]["effective_cpu_budget"],
            "price_per_hour": report["price_per_hour"],
            "preliminary": report["preliminary"],
        }
        for name in ("cp_serial", "cp_parallel", "collection"):
            base = baseline["benchmarks"].get(name)
            actual = report["benchmarks"].get(name)
            valid = (
                base
                and actual
                and not base["preliminary"]
                and not actual["preliminary"]
                and base["signature"] == actual["signature"]
            )
            if name == "collection":
                valid = valid and baseline["hardware"].get("GPUs") == report[
                    "hardware"
                ].get("GPUs")
            row[name] = (
                100 * base["median_seconds"] / actual["median_seconds"]
                if valid
                else None
            )
        collection = report["benchmarks"].get("collection", {})
        row["runtime"] = collection.get("runtime")
        row["collection_score_per_dollar_per_hour"] = (
            row["collection"] / report["price_per_hour"]
            if row["collection"] is not None and report["price_per_hour"]
            else None
        )
        rows.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out / "comparison.json",
        {"workload_id": baseline["workload_id"], "machines": rows},
    )
    lines = [
        "| Machine | CPU quota | CP Serial | CP Parallel | Collection | env/workers/reply | $/h | Collection score / ($/h) |",
        "|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:

        def number(value):
            return "—" if value is None else f"{value:.2f}"

        runtime = row["runtime"]
        runtime = (
            "/".join(
                str(runtime[k])
                for k in ("num_envs", "cpu_workers", "worker_reply_size")
            )
            if runtime
            else "—"
        )
        lines.append(
            f"| {row['machine']} | {row['cpu_budget']:.2f} | {number(row['cp_serial'])} | {number(row['cp_parallel'])} | {number(row['collection'])} | {runtime} | {number(row['price_per_hour'])} | {number(row['collection_score_per_dollar_per_hour'])} |"
        )
    output = "\n".join(lines) + "\n"
    (args.out / "comparison.md").write_text(output, encoding="utf-8")
    print(output)


def main():
    raise NotImplementedError(_REWORK_REASON)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser(
        "prepare", help="Freeze inputs once, without collecting data"
    )
    for name in ("config", "checkpoint", "rollout", "out"):
        create.add_argument(f"--{name}", type=Path, required=True)
    create.add_argument("--name", default="combat-cp-v1")
    create.add_argument("--episodes", type=int, default=80)
    create.add_argument("--collection-reference", type=Path)
    execute = sub.add_parser("run", help="Benchmark this machine, without training")
    for name in ("bundle", "out"):
        execute.add_argument(f"--{name}", type=Path, required=True)
    execute.add_argument("--budget-seconds", type=float, default=900)
    execute.add_argument("--price-per-hour", type=float)
    execute.add_argument("--machine-label")
    execute.add_argument(
        "--reference", type=Path, help="First machine's summary.json for score=100"
    )
    execute.add_argument(
        "--reuse-collection",
        type=Path,
        help="Reuse the frozen reference on its original machine",
    )
    comparison = sub.add_parser(
        "compare", help="Compare saved runs; first machine scores 100"
    )
    comparison.add_argument("summaries", nargs="+", type=Path)
    comparison.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "run":
        if not math.isfinite(args.budget_seconds) or args.budget_seconds <= 0:
            parser.error("Budget must be positive and finite")
        if args.price_per_hour is not None and (
            not math.isfinite(args.price_per_hour) or args.price_per_hour <= 0
        ):
            parser.error("Price must be positive and finite")
        args.bundle = args.bundle.resolve()
        args.out = args.out.resolve()
        run(args)
    elif args.command == "prepare":
        prepare(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
