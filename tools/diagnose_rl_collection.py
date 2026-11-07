"""Explicit, bounded 12/80-worker experiment; never starts training or applies runtime settings.

Run only when measurements are authorized. --trace-reason selects a conditional
Windows pass and never contributes to the performance ranking.
"""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from tune_rl_collection import collection_resources, stop_tree

ROOT = Path(__file__).resolve().parents[1]
MODES = {
    "B0": (12, 0, 0),
    "B1": (12, 8, 0.005),
    "B2": (12, 16, 0.010),
    "B3": (12, 16, 0.020),
    "W80": (80, 0, 0),
}
WARMUP = [0, 2, 6, 25, 50, 75, 100, 125]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


class BudgetExhausted(Exception):
    pass


class Series:
    def __init__(self, out, seconds):
        self.out = out
        # Historical inputs are read only. New sources deliberately have a new identity.
        self.historical = ROOT / "artifacts/benchmarks/windows-streaming-refill-200"
        self.reference = read(self.historical / "streaming-80-primary/result.json")
        self.carrier = (
            ROOT / "artifacts/benchmarks/windows-ready-80-200/case-specs-only.pt"
        )
        self.config = ROOT / "artifacts/benchmarks/combat-cp-v1/config.json"
        self.checkpoint = ROOT / "artifacts/benchmarks/combat-cp-v1/checkpoint.pt"
        self.trace = self.historical / "streaming-80-primary/trace.npz"
        expected = self.reference["protected_inputs"]
        for path in (self.carrier, self.config, self.checkpoint):
            if sha(path) != expected[str(path.resolve())]:
                raise ValueError(f"Frozen input changed: {path}")
        if sha(ROOT / "build/combat_ai.dll") != self.reference["native_library_sha256"]:
            raise ValueError("Native CP DLL differs from the historical reference")
        self.protected = {}
        for path in (
            self.carrier,
            self.config,
            self.checkpoint,
            self.trace,
            ROOT / "build/combat_ai.dll",
            ROOT / "artifacts/rl/ppo-after-target-bc/rollouts/latest.pt",
            ROOT / "artifacts/rl/ppo-after-target-bc/state.json",
            ROOT / "artifacts/rl/ppo-after-target-bc/status.json",
            ROOT / "configs/rl-ppo-after-target-bc-runtime.json",
        ):
            if path.exists():
                self.protected[str(path)] = sha(path)
        self.sources = {
            str(path): sha(path)
            for path in (
                ROOT / "civ5rl/core/env.py",
                ROOT / "civ5rl/core/contracts.py",
                ROOT / "civ5rl/engine/collector.py",
                ROOT / "civ5rl/engine/collection_diagnostics.py",
                ROOT / "civ5rl/engine/streaming_collector.py",
                ROOT / "civ5rl/engine/streaming_worker.py",
                ROOT / "tools/profile_rl_runtime.py",
                Path(__file__).resolve(),
            )
        }
        self.out.mkdir(parents=True, exist_ok=False)
        self.started = perf_counter()
        self.deadline = self.started + seconds
        self.rows = []
        self.result = {
            "schema": 1,
            "budget_seconds": seconds,
            "episodes": 200,
            "envs": 80,
            "reply": 1,
            "scheduler": "streaming",
            "protected_inputs": self.protected,
            "source_hashes": self.sources,
            "source_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "rows": self.rows,
            "confirmation_complete": False,
        }

    def save(self):
        self.result["elapsed_seconds"] = perf_counter() - self.started
        write(self.out / "summary.json", self.result)

    def verify_inputs(self):
        for path, expected in {**self.protected, **self.sources}.items():
            if sha(Path(path)) != expected:
                raise AssertionError(f"Protected input/source changed: {path}")

    def reserve(self, runs):
        # Reserve a complete pair before launching its first member. Include
        # spawn, warmup, verification and bounded diagnostic export in estimates.
        duration = max((row["session_seconds"] for row in self.rows), default=75)
        if self.deadline - perf_counter() < runs * duration * 1.10:
            raise BudgetExhausted("Not enough time for the next complete block")

    def measure(self, label, mode, *, ranked=True):
        import numpy as np

        workers, minimum, delay = MODES[mode]
        directory = self.out / label
        command = [
            sys.executable,
            str(ROOT / "tools/profile_rl_runtime.py"),
            "collection",
            "--rollout",
            str(self.carrier),
            "--config",
            str(self.config),
            "--checkpoint",
            str(self.checkpoint),
            "--out",
            str(directory),
            "--episodes",
            "200",
            "--repeat",
            "1",
            "--warmup",
            "1",
            "--warmup-episodes",
            "8",
            "--envs",
            "80",
            "--workers",
            str(workers),
            "--reply-size",
            "1",
            "--scheduler",
            "streaming",
            "--inference-min-batch",
            str(minimum),
            "--inference-wait-seconds",
            str(delay),
            "--collection-diagnostics",
            "--process-priority",
            "normal",
            "--reference-trace",
            str(self.trace),
        ]
        print(
            f"{label}: {mode}, workers={workers}, batch>={minimum}, wait={delay:g}s",
            flush=True,
        )
        started = perf_counter()
        with (self.out / f"{label}.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT
            )
            try:
                process.wait(timeout=max(0.01, self.deadline - perf_counter()))
            except subprocess.TimeoutExpired:
                stop_tree(process)
                raise BudgetExhausted(f"Deadline reached during {label}") from None
            except BaseException:
                stop_tree(process)
                raise
            if process.returncode:
                raise RuntimeError(f"{label} failed; see {log.name}")
        session = perf_counter() - started
        result = read(directory / "result.json")
        if os.name == "nt":
            import psutil

            priority = int(psutil.NORMAL_PRIORITY_CLASS)
            if result["parent_priority"] != priority or any(
                value != priority for value in result["worker_priorities"].values()
            ):
                raise AssertionError(
                    f"{label}: parent/workers must use Normal priority"
                )
        if result["native_library_sha256"] != self.reference["native_library_sha256"]:
            raise AssertionError(
                f"{label}: profiler used a different native CP library"
            )
        trial = result["trials"][0]
        expected = self.reference["trials"][0]
        for key in ("commands_hash", "results_hash", "positions", "commands"):
            if trial[key] != expected[key]:
                raise AssertionError(f"{label}: incomparable {key}")
        if result["warmup_case_indices"] != WARMUP:
            raise AssertionError("Warmup workload changed")
        with (
            np.load(self.trace) as original,
            np.load(directory / "trace.npz") as actual,
        ):
            for name in original.files:
                np.testing.assert_allclose(
                    actual[name],
                    original[name],
                    rtol=1e-5,
                    atol=1e-5,
                    err_msg=f"{label}: {name}",
                )
        diagnostic = read(directory / trial["diagnostics_file"])
        worker_metrics = diagnostic["metrics"]
        parent = diagnostic["parent"]
        row = {
            "label": label,
            "mode": mode,
            "ranked": ranked,
            "workers": workers,
            "inference_min_batch": minimum,
            "inference_wait_seconds": delay,
            "seconds": trial["seconds"],
            "session_seconds": session,
            "resources": collection_resources(directory / "resources.jsonl"),
            "timings": trial["timings"],
            "max_behavior_deviation": trial["max_behavior_deviation"],
            "diagnostics_file": str(directory / trial["diagnostics_file"]),
            "cp_wall_seconds": worker_metrics.get("cp_decision.wall", {}).get(
                "sum_ns", 0
            )
            / 1e9,
            "cp_cpu_seconds": worker_metrics.get("cp_decision.cpu", {}).get("sum_ns", 0)
            / 1e9,
            "worker_process_cpu_seconds": sum(
                worker["process_cpu_ns"] for worker in diagnostic["workers"]
            )
            / 1e9,
            "parent_metrics": parent["metrics"],
            "parent_gauges": parent["gauges"],
            "receiver_gauges": diagnostic["receiver"]["gauges"],
            "verified": True,
            "process_priority": result["process_priority"],
            "parent_priority": result.get("parent_priority"),
            "worker_priorities": result.get("worker_priorities"),
        }
        self.rows.append(row)
        self.verify_inputs()
        self.save()
        print(
            f"  Collection={trial['seconds']:.3f}s; CP wall/CPU={row['cp_wall_seconds']:.2f}/{row['cp_cpu_seconds']:.2f}s",
            flush=True,
        )
        return row

    def medians(self):
        return {
            mode: statistics.median(
                row["seconds"]
                for row in self.rows
                if row["mode"] == mode and row["ranked"]
            )
            for mode in MODES
            if any(row["mode"] == mode and row["ranked"] for row in self.rows)
        }

    def run(self):
        for number, pair in enumerate((("B0", "W80"), ("W80", "B0")), 1):
            self.reserve(2)
            for index, mode in enumerate(pair):
                self.measure(f"control-{number}-{index}-{mode}", mode)
        for mode in ("B1", "B2", "B3"):
            self.reserve(1)
            self.measure(f"primary-{mode}", mode)
        medians = self.medians()
        ordered = sorted((mode for mode in medians if mode != "W80"), key=medians.get)
        best, opponent = ordered[:2]
        if medians[opponent] <= medians[best] * 1.03:
            # Spend at most one additional run on the nearest challenger.
            self.reserve(1)
            self.measure(f"tie-{opponent}", opponent)
            medians = self.medians()
            fastest = min((best, opponent), key=medians.get)
            close = [
                mode
                for mode in (best, opponent)
                if medians[mode] <= medians[fastest] * 1.03
            ]
            best = min(close, key=lambda mode: (MODES[mode][2], MODES[mode][1]))
        self.result["selected_mode"] = best
        confirmations = []
        for number, pair in enumerate(((best, "W80"), ("W80", best)), 1):
            self.reserve(2)
            block = [
                self.measure(f"confirm-{number}-{i}-{mode}", mode)
                for i, mode in enumerate(pair)
            ]
            candidate = next(row for row in block if row["mode"] == best)
            control = next(row for row in block if row["mode"] == "W80")
            confirmations.append(
                {
                    "candidate": candidate,
                    "control": control,
                    "ratio": candidate["seconds"] / control["seconds"],
                }
            )
        best_time = statistics.median(
            pair["candidate"]["seconds"] for pair in confirmations
        )
        control_time = statistics.median(
            pair["control"]["seconds"] for pair in confirmations
        )

        def rss(side):
            return statistics.median(
                pair[side]["resources"]["ram_tree_rss_bytes_peak"]
                for pair in confirmations
            )

        self.result.update(
            confirmation_complete=True,
            confirmation_ratios=[pair["ratio"] for pair in confirmations],
            confirmed_candidate_seconds=best_time,
            confirmed_control_seconds=control_time,
            confirmed_ratio=best_time / control_time,
            candidate_rss_peak_median=rss("candidate"),
            control_rss_peak_median=rss("control"),
            practical_success=best_time <= control_time * 1.03
            and rss("candidate") < rss("control"),
        )

    def report(self):
        if self.result.get("diagnostic_only"):
            lines = [
                "# Windows ETW: приёмник, parent и workers",
                "",
                "Только диагностические проходы streaming 80/12/1 и 80/80/1, batching 0/0, Normal priority.",
                "Время трассируемых запусков не используется для ранжирования.",
                "",
                "| Workers | ETL | Collection UTC start | Collection UTC end | Parent PID/TID | Receiver TID |",
                "|---:|---|---|---|---|---|",
            ]
            for capture in self.result.get("captures", []):
                parent = capture["parent"]
                lines.append(
                    f"| {capture['workers']} | [{capture['etl']}]({capture['etl']}) | "
                    f"{capture['collection_start_utc']} | {capture['collection_end_utc']} | "
                    f"{parent['pid']}/{parent['tid']} | {capture['receiver']['tid']} |"
                )
            lines += [
                "",
                f"Полная пара сохранена: {self.result.get('capture_pair_complete', False)}.",
            ]
            for key in ("trace_unavailable", "budget_exhausted", "error"):
                if self.result.get(key):
                    lines += ["", f"{key}: {self.result[key]}"]
            lines += [
                "",
                "PID/TID всех workers, точные timestamps и QPC/UTC anchors — в summary.json → captures.",
                "Сначала ограничить ETL указанным Collection: не смешивать spawn, warmup, export и сохранение ETL.",
                "CPU (Precise): сопоставить Running/Ready/Waiting приёмника и главного потока с workers.",
                "CPU (Sampled): проверить стеки wait/ReadFile/recv, GIL и сериализации; отдельно искать batch-wait helpers.",
                "Порог >60 применяется к фактически ожидающим handles внутри Python, а не автоматически к числу workers.",
                "Installed wait source сохранён в python-wait-source.txt. Отличие механизма ожидания — гипотеза, не вывод.",
            ]
            (self.out / "report.md").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
            return
        lines = [
            "# Collector: фиксированная диагностика 12/80",
            "",
            "200 случаев; 80 сред; reply=1; одинаковая модель, CP, seeds и precision.",
            "",
            "| Запуск | Режим | Collection, с | CP wall, с | CP CPU, с | Batch | Компиляция, с |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for row in self.rows:
            timing = row["timings"]
            lines.append(
                f"| {row['label']} | {row['mode']} | {row['seconds']:.3f} | {row['cp_wall_seconds']:.2f} | {row['cp_cpu_seconds']:.2f} | {timing['inference_batch_mean']:.2f} | {timing['compilation_seconds']:.3f} |"
            )
        lines += [
            "",
            f"Подтверждение завершено: {self.result['confirmation_complete']}.",
            f"Практический критерий выполнен: {self.result.get('practical_success', 'не определено')}.",
            "",
            "Подробные интервалы, гистограммы, очереди и выборки событий находятся в diagnostics-*.json каждого запуска.",
            "CPU/wall и GPU/parent интервалы перекрываются; их суммы не являются разложением времени Collection.",
            "Разность CP wall и CPU не доказывает вытеснение ОС. RSS дерева может повторно учитывать общие страницы.",
            "ETW, если потребуется, записывается отдельными проходами и не входит в ранжирование.",
        ]
        (self.out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def trace_windows(series, reason):
    """Capture only our own WPR sessions; do not rank their timings."""
    if os.name != "nt":
        raise ValueError("Scheduler traces require Windows WPR")
    import ctypes
    import inspect
    from multiprocessing import connection

    import psutil

    series.result.update(
        diagnostic_only=True,
        trace_reason=reason,
        trace_profile="CPU (CSwitch, ReadyThread, SampledProfile and stacks)",
        process_priority="normal",
        captures=[],
        capture_pair_complete=False,
    )
    if not ctypes.windll.shell32.IsUserAnAdmin():
        series.result["trace_unavailable"] = (
            "WPR requires an elevated session; nothing started"
        )
        return
    psutil.Process().nice(psutil.NORMAL_PRIORITY_CLASS)
    (series.out / "python-wait-source.txt").write_text(
        f"Python: {sys.version}\nSource: {connection.__file__}\n\n"
        + inspect.getsource(connection.wait)
        + "\n"
        + inspect.getsource(connection._exhaustive_wait),
        encoding="utf-8",
    )
    status = subprocess.run(
        ["wpr", "-status"], check=False, capture_output=True, text=True, timeout=10
    )
    if status.returncode or "WPR is not recording" not in status.stdout:
        raise RuntimeError(
            "WPR may already be recording; existing sessions left untouched"
        )
    series.reserve(2)
    for mode in ("B0", "W80"):
        series.reserve(1)
        started = False
        row = None
        instance = "civ5rl-" + uuid4().hex
        original_deadline = series.deadline
        series.result.setdefault("trace_instances", []).append(
            {"mode": mode, "instance": instance}
        )
        series.save()

        def cancel_owned(instance=instance, deadline=original_deadline):
            # Leave the last five seconds for metadata/report finalization.
            timeout = max(0.01, min(10, deadline - perf_counter() - 5))
            subprocess.run(
                ["wpr", "-cancel", "-instancename", instance],
                check=False,
                capture_output=True,
                timeout=timeout,
            )

        try:
            try:
                result = subprocess.run(
                    ["wpr", "-start", "CPU", "-filemode", "-instancename", instance],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=min(15, max(0.01, original_deadline - perf_counter() - 30)),
                )
            except BaseException:
                # A timed-out CLI may already have started recording. A unique
                # instance permits safe cleanup without touching other sessions.
                cancel_owned()
                raise
            if result.returncode:
                cancel_owned()
                raise RuntimeError(
                    f"Cannot start WPR: {result.stdout}\n{result.stderr}"
                )
            started = True
            # Leave time to save/stop this owned trace even if Collection stalls.
            series.deadline = original_deadline - 30
            row = series.measure(f"etw-{mode}", mode, ranked=False)
        finally:
            series.deadline = original_deadline
            if started:
                try:
                    result = subprocess.run(
                        [
                            "wpr",
                            "-stop",
                            str(series.out / f"scheduler-{mode}.etl"),
                            "-instancename",
                            instance,
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        # Reserve ten seconds for an emergency cancel and five
                        # for final metadata inside the same total budget.
                        timeout=max(0.01, original_deadline - perf_counter() - 15),
                    )
                    if result.returncode:
                        raise RuntimeError(
                            f"Cannot save WPR trace: {result.stdout}\n{result.stderr}"
                        )
                    started = False
                    if row is not None:
                        diagnostic = read(Path(row["diagnostics_file"]))
                        parent = diagnostic["parent"]
                        receiver = diagnostic["receiver"]
                        anchor = parent["clock_anchor"]

                        def utc(ns, anchor=anchor):
                            epoch_ns = (
                                anchor["unix_time_ns"] + ns - anchor["perf_counter_ns"]
                            )
                            return datetime.fromtimestamp(
                                epoch_ns / 1e9, UTC
                            ).isoformat()

                        series.result["captures"].append(
                            {
                                "workers": row["workers"],
                                "wpr_instance": instance,
                                "etl": f"scheduler-{mode}.etl",
                                "collection_start_ns": parent["started_ns"],
                                "collection_end_ns": parent["ended_ns"],
                                "collection_start_utc": utc(parent["started_ns"]),
                                "collection_end_utc": utc(parent["ended_ns"]),
                                "scope": "instrumented Collection loop; excludes warmup and diagnostic export",
                                "clock_anchor": anchor,
                                "parent": {
                                    "pid": parent["pid"],
                                    "tid": parent["native_thread_id"],
                                },
                                "receiver": {
                                    "pid": receiver["pid"],
                                    "tid": receiver["native_thread_id"],
                                },
                                "worker_threads": [
                                    {
                                        "pid": worker["pid"],
                                        "tid": worker["native_thread_id"],
                                    }
                                    for worker in diagnostic["workers"]
                                ],
                            }
                        )
                        series.save()
                except BaseException:
                    # start succeeded and no pre-existing WPR recording was present.
                    if started:
                        cancel_owned()
                    raise
    series.result["capture_pair_complete"] = len(series.result["captures"]) == 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=900)
    parser.add_argument(
        "--trace-reason",
        help="Run only the two unranked ETW passes, with this residual-effect explanation",
    )
    args = parser.parse_args()
    limit = 300 if args.trace_reason else 900
    if args.trace_reason and args.budget_seconds == 900:
        args.budget_seconds = 300
    if not 0 < args.budget_seconds <= limit:
        parser.error(f"The agreed budget for this phase is at most {limit} seconds")
    series = Series(args.out.resolve(), args.budget_seconds)
    try:
        if args.trace_reason:
            trace_windows(series, args.trace_reason)
        else:
            series.run()
    except BudgetExhausted as exc:
        series.result["budget_exhausted"] = str(exc)
    except BaseException as exc:
        series.result["error"] = repr(exc)
        raise
    finally:
        series.verify_inputs()
        series.save()
        series.report()
    if args.trace_reason:
        print(
            f"ETW captures: {len(series.result.get('captures', []))}/2; report: {series.out / 'report.md'}",
            flush=True,
        )
        if not series.result.get("capture_pair_complete"):
            raise SystemExit(2)


if __name__ == "__main__":
    main()
