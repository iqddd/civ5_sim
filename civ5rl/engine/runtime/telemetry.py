"""Out-of-band process-tree/system telemetry. NVIDIA NVML, never D3D Usage."""

import json
import os
import statistics
import threading
import time
from bisect import bisect_right
from collections import defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from uuid import uuid4

import psutil

from civ5rl.engine.runtime import is_int_at_least
from civ5rl.provenance.common import write_json


class CompilationClock:
    """Track compilation intervals and unregister callbacks on every exit."""

    def __init__(self):
        self.intervals = []
        self.active = None
        self.monitor = None

    def __enter__(self):
        from torch._dynamo.callback import callback_handler

        self.handler = callback_handler
        self.handler.register_start_callback(self.start)
        self.handler.register_end_callback(self.end)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            self.handler.remove_start_callback(self.start)
        finally:
            self.handler.remove_end_callback(self.end)
        if self.active is not None:
            self.end(None)

    def start(self, args):
        self.active = time.perf_counter()
        if self.monitor:
            self.previous_phase = dict(self.monitor.state)
            self.monitor.mark("compilation")

    def end(self, args):
        self.intervals.append((self.active, time.perf_counter()))
        self.active = None
        if self.monitor:
            self.monitor.mark(**self.previous_phase)

    def seconds(self, start, stop):
        return sum(max(0, min(stop, b) - max(start, a)) for a, b in self.intervals)


class ResourceMonitor:
    _GPU_FIELDS = (
        "gpu_core_load_percent",
        "gpu_memory_controller_load_percent",
        "vram_used_bytes",
        "vram_total_bytes",
        "gpu_power_watts",
        "gpu_default_power_limit_watts",
        "gpu_enforced_power_limit_watts",
        "gpu_power_percent_tdp",
    )

    def __init__(self, directory, interval=1.0, gpu_index=0):
        self.directory = Path(directory)
        self.interval = interval
        self.stop_event = threading.Event()
        self.state = {"phase": "startup", "update": 0, "decisions": 0}
        self.processes = {}
        self.gpu = None
        self.nvml = None
        self.errors = {}
        self.lock = threading.Lock()
        self.summary = {}
        self.samples = 0
        self.gpu_index = gpu_index
        self.session_id = uuid4().hex
        self.phase_stream = None
        self.last_sample = None

    def record_cuda_memory(self, policy, boundary, *, snapshot=False, **state):
        """Read host counters between updates; failures never mask training errors.

        Ordinary samples must not synchronize CUDA or mutate the allocator (see
        sample_cuda_memory). Boundary hooks surround collection/PPO, rollout
        cleanup and evaluation, never the optimizer's inner loop. NVML values are
        the latest asynchronous background sample with their own timestamp, not a
        synchronous boundary measurement. sampling_seconds excludes serialization
        and disk writes. Only the explicit OOM path requests an expensive layout
        snapshot; allocation history and per-allocation stacks stay disabled.
        """
        started = time.perf_counter()
        try:
            from civ5rl.engine.runtime.cuda_memory import (
                dump_oom_snapshot,
                sample_cuda_memory,
            )

            row = sample_cuda_memory(policy)
            if row is None:
                return
            with self.lock:
                phase = self.state.copy()
            recent = self.last_sample or {}
            row.update(
                schema=1,
                session_id=self.session_id,
                boundary=boundary,
                timestamp_utc=datetime.now(UTC).isoformat(),
                elapsed_seconds=time.monotonic() - self.started,
                **{**phase, **state},
                nvml_sample={
                    key: recent.get(key)
                    for key in (
                        "timestamp_utc",
                        "elapsed_seconds",
                        "vram_used_bytes",
                        "vram_total_bytes",
                    )
                },
            )
            if snapshot:
                path = (
                    self.directory
                    / f"cuda-oom-{self.session_id}-{time.monotonic_ns()}.json.gz"
                )
                try:
                    dump_oom_snapshot(path)
                    row["snapshot"] = path.name
                except Exception as error:
                    row["snapshot_error"] = repr(error)
            row["sampling_seconds"] = time.perf_counter() - started
            with (self.directory / "cuda-memory.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        except Exception as error:
            self.errors["cuda_memory"] = f"{type(error).__name__}: {error}"

    @contextmanager
    def capture_cuda_oom(self, policy):
        import torch

        try:
            yield
        except torch.cuda.OutOfMemoryError:
            self.record_cuda_memory(policy, "oom", snapshot=True)
            raise

    def mark(self, phase, **state):
        with self.lock:
            keys = ("phase", "rollout", "update", "epoch")
            previous = tuple(self.state.get(key) for key in keys)
            self.state.update(phase=phase, **state)
            # Collection reports decision progress very frequently. Only write
            # phase/cycle transitions; progress remains in the sampled state.
            if tuple(self.state.get(key) for key in keys) != previous:
                self._phase_event()

    def _phase_event(self):
        if self.phase_stream is not None:
            self.phase_stream.write(
                json.dumps(
                    {
                        "schema": 1,
                        "session_id": self.session_id,
                        "timestamp_utc": datetime.now(UTC).isoformat(),
                        "elapsed_seconds": time.monotonic() - self.started,
                        **self.state,
                    },
                    allow_nan=False,
                )
                + "\n"
            )

    def _query(self, name, function):
        try:
            return function()
        except Exception as exc:  # noqa: BLE001 -- unsupported NVML fields are explicit nulls
            self.errors[name] = f"{type(exc).__name__}: {exc}"
            return None

    def start(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        try:
            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            self.gpu = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception as exc:  # noqa: BLE001 -- CPU telemetry remains available without NVML
            self.errors["nvml"] = f"{type(exc).__name__}: {exc}"
        self.started = time.monotonic()
        psutil.cpu_percent(None)
        self.stream = (self.directory / "resources.jsonl").open(
            "a", encoding="utf-8", buffering=1
        )
        self.phase_stream = (self.directory / "phases.jsonl").open(
            "a", encoding="utf-8", buffering=1
        )
        with self.lock:
            self._phase_event()
        metadata = {
            "schema": 1,
            "gpu_source": "NVIDIA NVML",
            "scope": "whole_device",
            "cpu_process_percent": "100% = one logical CPU; tree can exceed 100%",
            "cpu_tree_capacity_percent": "tree / logical_cpu_count",
            "gpu_core_load_percent": "nvmlDeviceGetUtilizationRates.gpu: time executing kernels",
            "gpu_memory_controller_load_percent": "nvmlDeviceGetUtilizationRates.memory: time reading/writing device memory, not capacity or bandwidth fraction",
            "gpu_power_percent_tdp": "100 * total board power / NVML default power limit (recorded proxy for board TDP)",
            "interval_seconds": self.interval,
            "logical_cpus": psutil.cpu_count(),
            "unavailable": "null plus errors; no D3D fallback",
            "phase_events": "phases.jsonl: exact monotonic transitions per session",
        }
        if self.gpu:
            nvml = self.nvml
            metadata.update(
                gpu_name=self._query("name", lambda: nvml.nvmlDeviceGetName(self.gpu)),
                gpu_uuid=self._query("uuid", lambda: nvml.nvmlDeviceGetUUID(self.gpu)),
            )
        write_json(self.directory / "resources-schema.json", metadata)
        self.thread = threading.Thread(
            target=self._run, name="rl-resource-monitor", daemon=True
        )
        self.thread.start()
        return self

    def sample(self):
        root = psutil.Process(os.getpid())
        alive = {}
        rss = cpu = 0
        for p in [root, *root.children(recursive=True)]:
            try:
                key = (p.pid, p.create_time())
                tracked = self.processes.get(key, p)
                cpu += tracked.cpu_percent(None)
                rss += tracked.memory_info().rss
                alive[key] = tracked
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        self.processes = alive
        vm = psutil.virtual_memory()
        with self.lock:
            state = self.state.copy()
        row = {
            "schema": 1,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - self.started,
            "session_id": self.session_id,
            **state,
            "cpu_system_percent": psutil.cpu_percent(None),
            "cpu_tree_percent": cpu,
            "cpu_tree_capacity_percent": cpu / (psutil.cpu_count() or 1),
            "ram_tree_rss_bytes": rss,
            "ram_system_used_bytes": vm.total - vm.available,
            "ram_system_available_bytes": vm.available,
            "ram_system_percent": vm.percent,
            "process_count": len(alive),
            **dict.fromkeys(self._GPU_FIELDS, None),
        }
        if self.gpu:
            nvml, gpu_handle = self.nvml, self.gpu
            util = self._query(
                "utilization", lambda: nvml.nvmlDeviceGetUtilizationRates(gpu_handle)
            )
            mem = self._query(
                "memory", lambda: nvml.nvmlDeviceGetMemoryInfo(gpu_handle)
            )
            if util:
                row.update(
                    gpu_core_load_percent=util.gpu,
                    gpu_memory_controller_load_percent=util.memory,
                )
            if mem:
                row.update(vram_used_bytes=mem.used, vram_total_bytes=mem.total)
            for field, fn in (
                ("gpu_power_watts", nvml.nvmlDeviceGetPowerUsage),
                (
                    "gpu_default_power_limit_watts",
                    nvml.nvmlDeviceGetPowerManagementDefaultLimit,
                ),
                (
                    "gpu_enforced_power_limit_watts",
                    nvml.nvmlDeviceGetEnforcedPowerLimit,
                ),
            ):
                value = self._query(field, lambda fn=fn: fn(gpu_handle))
                row[field] = None if value is None else value / 1000
            if (
                row["gpu_default_power_limit_watts"]
                and row["gpu_power_watts"] is not None
            ):
                row["gpu_power_percent_tdp"] = (
                    100 * row["gpu_power_watts"] / row["gpu_default_power_limit_watts"]
                )
        row["errors"] = self.errors.copy()
        return row

    def _run(self):
        while True:
            try:
                row = self.sample()
                self.last_sample = row
                self.stream.write(json.dumps(row, allow_nan=False) + "\n")
                self.samples += 1
                for key, value in row.items():
                    if isinstance(value, (float, int)) and key not in (
                        "schema",
                        "update",
                        "decisions",
                    ):
                        summary = self.summary.setdefault(
                            key, {"count": 0, "sum": 0, "max": value}
                        )
                        summary.update(
                            count=summary["count"] + 1,
                            sum=summary["sum"] + value,
                            max=max(summary["max"], value),
                        )
            except Exception as exc:  # noqa: BLE001 -- never interrupt training from sampler
                self.errors["sampler"] = f"{type(exc).__name__}: {exc}"
            if self.stop_event.wait(self.interval):
                break

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
        with self.lock:
            self.state.update(phase="stopped")
            self._phase_event()
            self.phase_stream.close()
            self.phase_stream = None
        self.stream.close()
        write_json(
            self.directory / "resources-summary.json",
            {
                "samples": self.samples,
                "errors": self.errors,
                "metrics": {
                    k: {"mean": v["sum"] / v["count"], "max": v["max"]}
                    for k, v in self.summary.items()
                },
            },
        )
        if self.nvml:
            self.nvml.nvmlShutdown()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()


def _sample_time(row):
    if "elapsed_seconds" in row:
        return row["elapsed_seconds"]
    return datetime.fromisoformat(row["timestamp_utc"]).timestamp()


def _phase_timelines(events):
    grouped = defaultdict(list)
    for event in events:
        grouped[event.get("session_id")].append(event)
    for rows in grouped.values():
        rows.sort(key=_sample_time)
    return {
        session: ([_sample_time(row) for row in rows], rows)
        for session, rows in grouped.items()
    }


def _phase_pieces(sample, start, stop, timeline):
    """Split one hold interval using exact event times when available."""
    if timeline is None:
        yield sample, start, stop
        return
    boundaries, events = timeline
    index = bisect_right(boundaries, start)
    while start < stop:
        end = min(stop, boundaries[index]) if index < len(boundaries) else stop
        if index and start < end:
            yield events[index - 1], start, end
        start = end
        index += 1


def _sampling_gap(interval):
    return max(5, 3 * interval)


def _sample_pairs(samples, sample_interval):
    max_gap = _sampling_gap(sample_interval)
    for a, b in pairwise(samples):
        session = a.get("session_id")
        begin, end = _sample_time(a), _sample_time(b)
        valid = session == b.get("session_id") and 0 < end - begin <= max_gap
        yield a, session, begin, end, valid


def _contiguous_segments(intervals):
    segment = []
    for interval in intervals:
        if segment and (
            segment[-1][1] != interval[0] or segment[-1][3:] != interval[3:]
        ):
            yield segment
            segment = []
        segment.append(interval)
    if segment:
        yield segment


def _load_windows(segment, window):
    """Time-weighted, sample-and-hold windows over one contiguous segment."""
    left = 0
    weighted = 0.0
    for right, (begin, end, value, *_rest) in enumerate(segment):
        weighted += (end - begin) * value
        start = end - window
        while left < right and segment[left][1] <= start:
            a, b, load, *_rest = segment[left]
            weighted -= (b - a) * load
            left += 1
        first = segment[left]
        if first[0] <= start:
            yield end, (weighted - (start - first[0]) * first[2]) / window


def integrate_gpu_interval(
    samples,
    start_elapsed,
    stop_elapsed,
    *,
    session_id,
    sample_interval=1.0,
    phase_events=None,
    active_phase=None,
):
    """Integrate sample-and-hold NVML load in one session's elapsed clock.

    Bounds and phase events use seconds since ResourceMonitor.start(), not UTC,
    perf_counter, or an absolute monotonic timestamp. Never bridge a session,
    missing value, or sampling gap. Events split samples at exact transitions;
    absent events, sampled phase labels give approximate boundaries.
    """
    if stop_elapsed <= start_elapsed or sample_interval <= 0:
        raise ValueError("Expected positive interval and sample interval")
    timeline = _phase_timelines(phase_events or ()).get(session_id)
    covered = busy = 0.0
    for sample, session, left, right, valid in _sample_pairs(samples, sample_interval):
        if not valid or session != session_id:
            continue
        value = sample.get("gpu_core_load_percent")
        low, high = max(left, start_elapsed), min(right, stop_elapsed)
        if value is None or high <= low:
            continue
        if active_phase is None:
            # Without a phase filter, times before the first event also count.
            pieces = ((sample, low, high),)
        else:
            pieces = _phase_pieces(sample, low, high, timeline)
        for label, begin, end in pieces:
            if active_phase is not None and label.get("phase") != active_phase:
                continue
            duration = end - begin
            covered += duration
            busy += duration * value / 100
    duration = stop_elapsed - start_elapsed
    return {
        "interval_seconds": duration,
        "covered_seconds": covered,
        # This denominator includes any phases filtered out by active_phase.
        "coverage_fraction": covered / duration,
        "coverage_denominator": "entire_query_interval_including_excluded_phases",
        "estimated_busy_seconds": busy if covered else None,
        "estimated_idle_seconds": covered - busy if covered else None,
        "gpu_core_load_percent_mean": 100 * busy / covered if covered else None,
        "phase_boundaries": "events" if timeline else "sampled",
    }


def analyze_telemetry(jsonl_path, active_phase="iterative_training", window=60):
    """Aggregate active NVML samples and time-weighted, uninterrupted windows."""
    if window <= 0:
        raise ValueError("Expected positive load window")
    events = []
    if isinstance(jsonl_path, (str, Path)):
        path = Path(jsonl_path)
        samples = _lines(path)
        events = _lines(path.with_name("phases.jsonl"))
        interval = _sample_interval(path.parent)
    else:
        samples, interval = list(jsonl_path), 1.0
    active = [r for r in samples if r["phase"] == active_phase]
    metrics = {}
    for field in (
        "gpu_memory_controller_load_percent",
        "vram_used_bytes",
        "gpu_power_percent_tdp",
        "cpu_tree_percent",
        "ram_tree_rss_bytes",
    ):
        values = [r[field] for r in active if r.get(field) is not None]
        if values:
            scale = 2**30 if field.endswith("bytes") else 1
            metrics[field] = {
                "mean": sum(values) / len(values) / scale,
                "max": max(values) / scale,
                "unit": "GiB" if scale > 1 else "%",
            }
    timelines = _phase_timelines(events)
    intervals = []
    for sample, session, begin, end, valid in _sample_pairs(samples, interval):
        value = sample.get("gpu_core_load_percent")
        if not valid or value is None:
            continue
        for label, start, stop in _phase_pieces(
            sample, begin, end, timelines.get(session)
        ):
            if label["phase"] == active_phase:
                intervals.append((start, stop, value, session, label.get("rollout")))
    if intervals:
        covered = sum(stop - start for start, stop, *_ in intervals)
        weighted = sum((stop - start) * value for start, stop, value, *_ in intervals)
        metrics["gpu_core_load_percent"] = {
            "mean": weighted / covered,
            "max": max(interval[2] for interval in intervals),
            "unit": "%",
        }
    windows = [
        {"end_elapsed_seconds": end, "core_average": average}
        for segment in _contiguous_segments(intervals)
        for end, average in _load_windows(segment, window)
    ]
    return {"active": active, "metrics": metrics, "windows": windows}


def _lines(path):
    """A live writer may have an unfinished last record, never skip corrupt history."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    result = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return result


def _sample_interval(directory):
    schema = directory / "resources-schema.json"
    return (
        json.loads(schema.read_text()).get("interval_seconds", 1)
        if schema.exists()
        else 1
    )


def _stats(values):
    return (
        {
            "count": len(values),
            "mean_seconds": statistics.mean(values),
            "median_seconds": statistics.median(values),
            "min_seconds": min(values),
            "max_seconds": max(values),
        }
        if values
        else None
    )


def summarize_performance(directory, last=None):
    """Select last N completed updates, not last N sampler rows or active cycles.

    GPU is time weighted by sample-and-hold, split at exact phase transitions
    when available. Gaps/resumes are never interpolated. Old logs use sampled
    phase labels and explicitly report their lower boundary precision.
    """
    if last is not None and not is_int_at_least(last):
        raise ValueError("last must be a positive cycle count")
    directory = Path(directory)
    updates = _lines(directory / "updates.jsonl")
    total = len(updates)
    updates = updates[-last:] if last else updates
    ids = [r["rollout"] for r in updates]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate completed rollout IDs")
    selected = set(ids)
    telemetry = directory / "telemetry"
    if not telemetry.exists():
        telemetry = directory
    samples = _lines(telemetry / "resources.jsonl")
    events = _lines(telemetry / "phases.jsonl")
    interval = _sample_interval(telemetry)
    timelines = _phase_timelines(events)
    tails = {row.get("session_id"): _sample_time(row) for row in samples}
    phase = defaultdict(
        lambda: {
            "seconds": 0.0,
            "sampled_phase_seconds": 0.0,
            "nvml_seconds": 0.0,
            "weighted_gpu": 0.0,
            "samples": 0,
            "cycles": set(),
            "per_cycle": defaultdict(float),
            "intervals": [],
        }
    )

    def duration(row, seconds):
        if row.get("rollout") in selected and row["phase"] != "stopped":
            item = phase[row["phase"]]
            item["seconds"] += seconds
            item["cycles"].add(row["rollout"])
            item["per_cycle"][row["rollout"]] += seconds

    # Durations use monotonic time, independent of GPU sampling and clock changes.
    for session, (_, rows) in timelines.items():
        for a, b in pairwise(rows):
            duration(a, max(0, _sample_time(b) - _sample_time(a)))
        if rows and rows[-1]["phase"] != "stopped" and session in tails:
            duration(rows[-1], max(0, tails[session] - _sample_time(rows[-1])))

    excluded_gaps = 0
    for a, session, begin, end, valid in _sample_pairs(samples, interval):
        if not valid:
            excluded_gaps += 1
            continue
        timeline = timelines.get(session)
        if timeline is None:
            duration(a, end - begin)
        pieces = _phase_pieces(a, begin, end, timeline)
        for label, begin, end in pieces:
            if label.get("rollout") not in selected or label["phase"] == "stopped":
                continue
            item = phase[label["phase"]]
            seconds = end - begin
            item["sampled_phase_seconds"] += seconds
            value = a.get("gpu_core_load_percent")
            if value is not None:
                item["nvml_seconds"] += seconds
                item["weighted_gpu"] += value * seconds
                item["samples"] += 1
                item["intervals"].append(
                    (begin, end, value, session, label.get("rollout"))
                )

    for item in phase.values():
        segments = list(_contiguous_segments(item.pop("intervals")))
        windows = [
            average for segment in segments for _, average in _load_windows(segment, 60)
        ]
        weighted_gpu = item.pop("weighted_gpu")
        item["gpu_core_load_percent_mean"] = (
            weighted_gpu / item["nvml_seconds"] if item["nvml_seconds"] else None
        )
        item["nvml_coverage_fraction"] = (
            min(1.0, item["nvml_seconds"] / item["seconds"])
            if item["seconds"]
            else None
        )
        item["cycle_count"] = len(item.pop("cycles"))
        item["per_cycle_seconds"] = _stats(list(item.pop("per_cycle").values()))
        longest = max((s[-1][1] - s[0][0] for s in segments), default=0.0)
        item["longest_contiguous_nvml_seconds"] = longest
        item["gpu_60s_windows"] = {
            "count": len(windows),
            "min_percent": min(windows) if windows else None,
            "mean_percent": statistics.mean(windows) if windows else None,
            "below_80_count": sum(v < 80 for v in windows),
        }
        item["sustained_10min_observed"] = longest >= 600

    def timing(group, field, skip=False):
        return _stats(
            [
                r[group][field]
                for r in updates
                if field in r.get(group, {})
                and not (skip and r[group].get("PPO_skipped"))
            ]
        )

    return {
        "schema": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "directory": str(directory),
        "total_completed_cycles": total,
        "selected_cycles": len(ids),
        "rollout_ids": ids,
        "latest_sample_utc": samples[-1]["timestamp_utc"] if samples else None,
        "gpu_source": "NVIDIA NVML, whole device; time-weighted sample-and-hold",
        "phase_boundaries": "events"
        if events and all(r.get("session_id") in timelines for r in samples)
        else "sampled_or_mixed",
        "excluded_sample_gaps": excluded_gaps,
        "phases": dict(phase),
        "timings": {
            "collection": timing("collection", "collection_seconds"),
            "ppo_recorded": timing("ppo", "ppo_seconds", True),
            "ppo_operation": timing("ppo", "operation_seconds", True),
            "ppo_operation_without_compilation": timing(
                "ppo", "operation_without_compilation_seconds", True
            ),
            "dynamo_callbacks": timing("ppo", "runtime_compilation_seconds", True),
        },
        "notes": [
            "Only cycles committed to updates.jsonl are selected; the current incomplete cycle is excluded.",
            "Current ppo_seconds starts before backend setup/precheck; operation_seconds is the enclosing runner interval. Historical formats may differ.",
            "Dynamo callbacks cover tracing/backend work, not just GPU code generation; subtract only matching intervals and do not add nested timers.",
            "Phase durations and GPU means are separate: missing NVML/gaps reduce coverage, never imply zero load.",
            "Without phases.jsonl, boundaries are estimated from ~1s samples; NVML itself remains sampled even with exact boundaries.",
        ],
    }


def performance_markdown(result):
    def number(value):
        return "n/a" if value is None else f"{value:.2f}"

    lines = [
        f"Performance: {result['selected_cycles']} / {result['total_completed_cycles']} completed cycles",
        f"Rollouts: {result['rollout_ids']}",
        f"Phase boundaries: {result['phase_boundaries']}; GPU: NVML, time weighted.",
        "",
        "| Timer | N | Mean, s | Median, s | Min, s | Max, s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stats in result["timings"].items():
        if stats:
            values = " | ".join(
                number(stats[f"{key}_seconds"])
                for key in ("mean", "median", "min", "max")
            )
            lines.append(f"| {name} | {stats['count']} | {values} |")
    lines += [
        "",
        "| Phase | Cycles | Mean/cycle, s | Total, s | GPU Core Load, % | NVML coverage, % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in result["phases"].items():
        coverage = item["nvml_coverage_fraction"]
        lines.append(
            f"| {name} | {item['cycle_count']} | {number(item['per_cycle_seconds']['mean_seconds'])} | "
            f"{number(item['seconds'])} | {number(item['gpu_core_load_percent_mean'])} | "
            f"{number(100 * coverage if coverage is not None else None)} |"
        )
    lines += ["", *result["notes"], ""]
    return "\n".join(lines)
