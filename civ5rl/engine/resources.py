"""Out-of-band process-tree/system telemetry. NVIDIA NVML, never D3D Usage."""

import json
import os
import threading
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from uuid import uuid4

import psutil

from civ5rl.provenance.common import write_json


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
        """Read host counters between updates; failures never mask training errors."""
        started = time.perf_counter()
        try:
            from civ5rl.reporting.cuda_memory import dump_oom_snapshot, sample_cuda_memory

            row = sample_cuda_memory(policy)
            if row is None:
                return
            with self.lock:
                phase = self.state.copy()
            recent = self.last_sample or {}
            row.update(
                schema=1, session_id=self.session_id, boundary=boundary,
                timestamp_utc=datetime.now(UTC).isoformat(),
                elapsed_seconds=time.monotonic() - self.started,
                **{**phase, **state},
                nvml_sample={key: recent.get(key) for key in (
                    "timestamp_utc", "elapsed_seconds", "vram_used_bytes", "vram_total_bytes"
                )},
            )
            if snapshot:
                path = self.directory / f"cuda-oom-{self.session_id}-{time.monotonic_ns()}.json.gz"
                try:
                    dump_oom_snapshot(path)
                    row["snapshot"] = path.name
                except Exception as error:
                    row["snapshot_error"] = repr(error)
            row["sampling_seconds"] = time.perf_counter() - started
            with (self.directory / "cuda-memory.jsonl").open("a", encoding="utf-8") as stream:
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


def integrate_gpu_interval(samples, start_elapsed, stop_elapsed, *, session_id,
                           sample_interval=1.0, phase_events=None, active_phase=None):
    """Integrate sample-and-hold NVML load in one session's elapsed clock.

    Bounds and phase events use seconds since ResourceMonitor.start(), not UTC,
    perf_counter, or an absolute monotonic timestamp. Never bridge a session,
    missing value, or sampling gap. Events split samples at exact transitions;
    absent events, sampled phase labels give approximate boundaries.
    """
    if stop_elapsed <= start_elapsed or sample_interval <= 0:
        raise ValueError("Expected positive interval and sample interval")
    rows = list(samples)
    events = sorted(
        (e for e in (phase_events or []) if e.get("session_id") == session_id),
        key=lambda e: e["elapsed_seconds"],
    )
    covered = busy = 0.0
    for a, b in pairwise(rows):
        if a.get("session_id") != session_id or b.get("session_id") != session_id:
            continue
        left, right = a["elapsed_seconds"], b["elapsed_seconds"]
        if not 0 < right - left <= max(5, 3 * sample_interval):
            continue
        value = a.get("gpu_core_load_percent")
        if value is None:
            continue
        low, high = max(left, start_elapsed), min(right, stop_elapsed)
        if high <= low:
            continue
        points = [low, *(e["elapsed_seconds"] for e in events
                         if low < e["elapsed_seconds"] < high), high]
        for begin, end in pairwise(points):
            preceding = [e for e in events if e["elapsed_seconds"] <= begin]
            phase = preceding[-1]["phase"] if preceding else (
                None if events else a.get("phase")
            )
            if active_phase is not None and phase != active_phase:
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
        "phase_boundaries": "events" if events else "sampled",
    }


def analyze_telemetry(jsonl_path, active_phase="iterative_training", window=60):
    """Aggregate active NVML samples and time-weighted, uninterrupted windows."""
    if isinstance(jsonl_path, (str, Path)):
        path = Path(jsonl_path)
        samples = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
        schema = path.with_name("resources-schema.json")
        interval = (
            json.loads(schema.read_text()).get("interval_seconds", 1.0)
            if schema.exists()
            else 1.0
        )
    else:
        samples, interval = list(jsonl_path), 1.0
    active = [r for r in samples if r["phase"] == active_phase]
    metrics = {}
    for field in (
        "gpu_core_load_percent",
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
    windows, segment = [], []
    for row in samples:
        timestamp = row["elapsed_seconds"]
        if (
            row["phase"] != active_phase
            or row.get("gpu_core_load_percent") is None
            or (
                segment
                and not 0 < timestamp - segment[-1]["elapsed_seconds"] <= 2 * interval
            )
        ):
            segment = []
        if row["phase"] != active_phase or row.get("gpu_core_load_percent") is None:
            continue
        segment.append(row)
        start = timestamp - window
        while len(segment) > 1 and segment[1]["elapsed_seconds"] <= start:
            segment.pop(0)
        if segment[0]["elapsed_seconds"] > start:
            continue
        total = sum(
            a["gpu_core_load_percent"]
            * (b["elapsed_seconds"] - max(start, a["elapsed_seconds"]))
            for a, b in pairwise(segment)
        )
        windows.append(
            {"end_elapsed_seconds": timestamp, "core_average": total / window}
        )
    return {"active": active, "metrics": metrics, "windows": windows}
