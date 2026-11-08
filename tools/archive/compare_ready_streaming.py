"""Frozen 200-case A/B/C/C/B/A comparison, bounded to 900 seconds."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import psutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools._resources import collection_resources
from tools._process import stop_tree

ORDER = ("A", "B", "C", "C", "B", "A")
MODES = {"A": ("ready", 8), "B": ("streaming", 8), "C": ("streaming", 80)}
PRIORITY = int(psutil.ABOVE_NORMAL_PRIORITY_CLASS) if os.name == "nt" else 0
PRIORITY_NAME = "ABOVE_NORMAL" if os.name == "nt" else "POSIX nice=0"
PROCESS_OPTIONS = (
    {"creationflags": subprocess.ABOVE_NORMAL_PRIORITY_CLASS}
    if os.name == "nt"
    else {"start_new_session": True}
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--series-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    parser.add_argument(
        "--pair-workers",
        type=int,
        help="Compare streaming W/80/80/W with cpu.stat snapshots",
    )
    parser.add_argument(
        "--diagnostic-pair",
        action="store_true",
        help="One diagnostic pass W/80; no performance ranking",
    )
    args = parser.parse_args()
    if args.diagnostic_pair and not args.pair_workers:
        parser.error("--diagnostic-pair requires --pair-workers")
    order = ("B", "C", "C", "B") if args.pair_workers else ORDER
    if args.diagnostic_pair:
        order = ("B", "C")
    modes = {**MODES, "B": ("streaming", args.pair_workers or 8)}
    if args.pair_workers is not None and not 1 <= args.pair_workers <= 80:
        parser.error("--pair-workers must be between 1 and 80")
    started = perf_counter()
    deadline = args.deadline if args.series_worker else started + 900
    if not args.series_worker:
        # An independent watchdog also bounds verification, hashing and report
        # writing, not just the individual Collection subprocesses.
        psutil.Process().nice(PRIORITY)
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--out",
                str(args.out),
                "--series-worker",
                "--deadline",
                str(deadline - 12),
                *(["--diagnostic-pair"] if args.diagnostic_pair else []),
                *(
                    ["--pair-workers", str(args.pair_workers)]
                    if args.pair_workers
                    else []
                ),
            ],
            cwd=ROOT,
            **PROCESS_OPTIONS,
        )
        try:
            process.wait(timeout=max(0.01, deadline - perf_counter() - 12))
        except BaseException as exc:
            stop_tree(process)
            out = args.out.resolve()
            if out.exists():
                path = out / "summary.json"
                try:
                    data = read(path)
                except (OSError, json.JSONDecodeError):
                    data = {"rows": []}
                data.update(
                    complete=False,
                    error=repr(exc),
                    budget_seconds=900,
                    elapsed_seconds=perf_counter() - started,
                )
                path.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                with (out / "report.md").open("a", encoding="utf-8") as report:
                    report.write(
                        "\nСерия остановлена внешним ограничителем; результат неполный.\n"
                    )
            raise
        if process.returncode:
            raise SystemExit(process.returncode)
        path = args.out.resolve() / "summary.json"
        data = read(path)
        data["total_elapsed_seconds"] = perf_counter() - started
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return
    import numpy as np

    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    old = (
        ROOT / "artifacts/benchmarks/windows-streaming-refill-200/streaming-80-primary"
    )
    carrier = ROOT / "artifacts/benchmarks/windows-ready-80-200/case-specs-only.pt"
    config = ROOT / "artifacts/benchmarks/combat-cp-v1/config.json"
    checkpoint = ROOT / "artifacts/benchmarks/combat-cp-v1/checkpoint.pt"
    reference = read(old / "result.json")
    native = (
        ROOT / "build" / ("combat_ai.dll" if os.name == "nt" else "libcombat_ai.so")
    )
    native_sha = sha(native)
    rows = []
    summary = {
        "entrypoint": {
            "path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
            "sha256": sha(Path(__file__).resolve()),
        },
        "order": order,
        "modes": modes,
        "budget_seconds": 900,
        "rows": rows,
        "complete": False,
        "envs": 80,
        "reply_size": 1,
        "priority": PRIORITY_NAME,
        "native_library_sha256": native_sha,
        "inference_min_batch": 0,
        "inference_wait_seconds": 0,
        "diagnostics": args.diagnostic_pair,
        "diagnostic_only": args.diagnostic_pair,
        "etw": False,
    }

    def save():
        summary["elapsed_seconds"] = perf_counter() - started
        lines = [
            "# Ready / Streaming: сопоставимая серия",
            "",
            (
                f"Прежние 200 случаев; {PRIORITY_NAME}; batching 0/0; reply=1. "
                "Каждый worker выполняет один CP-вызов до короткого GPU-прогрева. "
                + (
                    "Диагностический запуск; времена не используются для ранжирования. ETW выключен."
                    if args.diagnostic_pair
                    else "ETW и подробная диагностика выключены."
                )
            ),
            "",
            "| Запуск | Scheduler | Workers | Collection, с | Batch | Inference-вызовы | CPU, лог. ядер | Peak RSS, GiB |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            resources = r["resources"]
            cpu = resources["cpu_tree_percent_mean"] / 100
            rss = resources["ram_tree_rss_bytes_peak"] / 2**30
            lines.append(
                f"| {r['label']} | {r['scheduler']} | {r['workers']} | "
                f"{r['seconds']:.3f} | {r['batch']:.2f} | {r['inference_calls']} | "
                f"{cpu:.2f} | {rss:.2f} |"
            )
        medians = {
            mode: statistics.median(r["seconds"] for r in rows if r["mode"] == mode)
            for mode in dict.fromkeys(order)
            if sum(r["mode"] == mode for r in rows) == 2
        }
        summary["medians_seconds"] = medians
        if args.pair_workers and "B" in medians and "C" in medians:
            retained = 100 * medians["C"] / medians["B"]
            summary["pair_retained_speed_percent"] = retained
            summary["pair_collection_overhead_percent"] = 100 * (
                medians["B"] / medians["C"] - 1
            )
            lines += [
                "",
                (
                    f"B = streaming/{args.pair_workers}; C = streaming/80. "
                    f"Медианы: B={medians['B']:.3f} с; C={medians['C']:.3f} с. "
                    f"B сохраняет {retained:.2f}% скорости C."
                ),
            ]
        if args.pair_workers:
            lines += [
                "",
                (
                    "cpu.stat: разности счётчиков вокруг collector.collect; "
                    "включают всю активность указанной cgroup. Вложенные cgroup не суммируются."
                ),
                "",
                "| Запуск | cgroup | Δnr_periods | Δnr_throttled | Δthrottled_usec | Δthrottled_time, ns (v1) |",
                "|---|---|---:|---:|---:|---:|",
            ]
            for row in rows:
                for path, delta in row["timings"]["cpu_stat"]["delta"].items():
                    lines.append(
                        f"| {row['label']} | {path} | {delta.get('nr_periods', '—')} | "
                        f"{delta.get('nr_throttled', '—')} | {delta.get('throttled_usec', '—')} | "
                        f"{delta.get('throttled_time', '—')} |"
                    )
        if len(medians) == 3:
            gain = 100 * (1 - medians["B"] / medians["A"])
            retained = 100 * medians["C"] / medians["B"]
            summary.update(
                streaming8_reduction_percent=gain,
                streaming8_retained_speed_percent=retained,
            )
            lines += [
                "",
                f"Медианы: A={medians['A']:.3f} с; B={medians['B']:.3f} с; C={medians['C']:.3f} с.",
                (
                    f"B сокращает время A на {gain:.2f}%. "
                    f"При 8 workers сохраняется {retained:.2f}% скорости streaming с 80 workers."
                ),
            ]
        lines += [
            "",
            "| Запуск | Подготовка, с* | Прогрев всех workers, с | Короткий Collection-прогрев, с |",
            "|---|---:|---:|---:|",
        ]
        for r in rows:
            lines.append(
                f"| {r['label']} | {r['setup_seconds']:.3f} | "
                f"{r['worker_warmup_seconds']:.3f} | {r['warmup_seconds']:.3f} |"
            )
        lines += [
            "",
            "*Подготовка включает отдельный прогрев workers; эти интервалы не следует складывать.",
            (
                "CPU — средняя загрузка дерева процессов, делённая на 100%; "
                "RSS может повторно учитывать разделяемые страницы. "
                "Два повтора не подтверждают статистическую устойчивость. "
                "Проверка таймеров отложена."
            ),
            "",
            "Серия завершена." if summary["complete"] else "Серия не завершена.",
        ]
        if "error" in summary:
            lines += ["", f"Причина: {summary['error']}"]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        # Include derived comparisons in the machine-readable artifact too.
        (out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    protected = {}

    def verify_files():
        for filename, expected in protected.items():
            if sha(Path(filename)) != expected:
                raise AssertionError(f"Input or source changed: {filename}")

    try:
        for path in (carrier, config, checkpoint):
            # The frozen reference was written on Windows; compare portable
            # basenames without rewriting its original serialized contents.
            expected = [
                value
                for name, value in reference["protected_inputs"].items()
                if name.replace("\\", "/").rsplit("/", 1)[-1] == path.name
            ]
            if len(expected) != 1 or sha(path) != expected[0]:
                raise ValueError(f"Frozen input changed: {path}")
        if os.name == "nt" and native_sha != reference["native_library_sha256"]:
            raise ValueError("Native library changed")
        paths = [
            carrier,
            config,
            checkpoint,
            native,
            old / "result.json",
            old / "trace.npz",
            ROOT / "artifacts/rl/ppo-after-target-bc/state.json",
            ROOT / "artifacts/rl/ppo-after-target-bc/status.json",
            ROOT / "artifacts/rl/ppo-after-target-bc/rollouts/latest.pt",
            ROOT / "configs/rl-ppo-after-target-bc-runtime.json",
        ]
        paths += [
            p
            for directory in ("civ5rl", "civ5sim", "tools")
            for p in (ROOT / directory).rglob("*.py")
        ]
        protected.update({str(p): sha(p) for p in paths if p.exists()})
        summary["protected_inputs_and_sources"] = protected
        psutil.Process().nice(PRIORITY)
        for index, mode in enumerate(order, 1):
            verify_files()
            remaining = deadline - perf_counter() - 15
            if remaining <= 0:
                raise TimeoutError("Series budget exhausted before next launch")
            scheduler, workers = modes[mode]
            label = f"{index:02d}-{mode}-{scheduler}-w{workers}"
            directory = out / label
            command = [
                sys.executable,
                str(ROOT / "tools/profile_rl_runtime.py"),
                "collection",
                "--config",
                str(config),
                "--checkpoint",
                str(checkpoint),
                "--rollout",
                str(carrier),
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
                "--warmup-workers",
                "--workers",
                str(workers),
                "--envs",
                "80",
                "--reply-size",
                "1",
                "--scheduler",
                scheduler,
                "--inference-min-batch",
                "0",
                "--inference-wait-seconds",
                "0",
                "--process-priority",
                "above_normal" if os.name == "nt" else "normal",
                "--reference-trace",
                str(old / "trace.npz"),
                *(["--cpu-stat"] if args.pair_workers else []),
                *(["--collection-diagnostics"] if args.diagnostic_pair else []),
            ]
            print(
                f"{label}: starting; {remaining:.1f}s available before cleanup reserve",
                flush=True,
            )
            with (out / f"{label}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    **PROCESS_OPTIONS,
                )
                try:
                    process.wait(timeout=max(0.01, deadline - perf_counter() - 15))
                except BaseException:
                    stop_tree(process)
                    raise
                if process.returncode:
                    raise RuntimeError(f"{label} failed; see {log.name}")
            result = read(directory / "result.json")
            trial = result["trials"][0]
            for key in ("commands_hash", "results_hash", "positions", "commands"):
                if trial[key] != reference["trials"][0][key]:
                    raise AssertionError(f"{label}: changed {key}")
            if result["native_library_sha256"] != native_sha:
                raise AssertionError("Native library changed")
            if result["warmup_case_indices"] != [0, 2, 6, 25, 50, 75, 100, 125]:
                raise AssertionError("GPU warmup cases changed")
            pids = set(result["worker_priorities"])
            acknowledgements = result["worker_warmup"]
            if (
                len(acknowledgements) != workers
                or len(pids) != workers
                or {str(a["pid"]) for a in acknowledgements} != pids
                or any(a["cp_calls"] != 1 for a in acknowledgements)
            ):
                raise AssertionError("Not every worker completed CP warmup")
            priority = PRIORITY
            if result["parent_priority"] != priority or any(
                v != priority for v in result["worker_priorities"].values()
            ):
                raise AssertionError("Incorrect process priority")
            with (
                np.load(old / "trace.npz") as expected,
                np.load(directory / "trace.npz") as actual,
            ):
                for name in expected.files:
                    np.testing.assert_allclose(
                        actual[name], expected[name], rtol=1e-5, atol=1e-5
                    )
            rows.append(
                {
                    "label": label,
                    "mode": mode,
                    "scheduler": scheduler,
                    "workers": workers,
                    "seconds": trial["seconds"],
                    "batch": trial["timings"]["inference_batch_mean"],
                    "inference_calls": trial["timings"]["inference_batches"],
                    "resources": collection_resources(directory / "resources.jsonl"),
                    "setup_seconds": result["setup_seconds"],
                    "worker_warmup_seconds": result["worker_warmup_seconds"],
                    "warmup_seconds": result["warmup_trials"][0]["seconds"],
                    "timings": trial["timings"],
                    "verified": True,
                }
            )
            verify_files()
            if perf_counter() > deadline:
                raise TimeoutError("Series budget exhausted during verification")
            save()
            print(f"{label}: Collection={trial['seconds']:.3f}s; verified", flush=True)
        summary["complete"] = True
    except BaseException as exc:
        summary["error"] = repr(exc)
        raise
    finally:
        try:
            verify_files()
        except BaseException as exc:
            summary.update(complete=False, error=repr(exc))
            raise
        finally:
            save()
    print(out / "report.md", flush=True)


if __name__ == "__main__":
    main()
