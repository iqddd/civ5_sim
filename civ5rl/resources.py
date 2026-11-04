"""Out-of-band process-tree/system telemetry. NVIDIA NVML, never D3D Usage."""

import json
import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil


class ResourceMonitor:
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

    def mark(self, phase, **state):
        with self.lock:
            self.state.update(phase=phase, **state)

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
        }
        if self.gpu:
            n = self.nvml
            metadata.update(
                gpu_name=self._query("name", lambda: n.nvmlDeviceGetName(self.gpu)),
                gpu_uuid=self._query("uuid", lambda: n.nvmlDeviceGetUUID(self.gpu)),
            )
        from .common import write_json

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
            **state,
            "cpu_system_percent": psutil.cpu_percent(None),
            "cpu_tree_percent": cpu,
            "cpu_tree_capacity_percent": cpu / (psutil.cpu_count() or 1),
            "ram_tree_rss_bytes": rss,
            "ram_system_used_bytes": vm.total - vm.available,
            "ram_system_available_bytes": vm.available,
            "ram_system_percent": vm.percent,
            "process_count": len(alive),
        }
        for key in (
            "gpu_core_load_percent",
            "gpu_memory_controller_load_percent",
            "vram_used_bytes",
            "vram_total_bytes",
            "gpu_power_watts",
            "gpu_default_power_limit_watts",
            "gpu_enforced_power_limit_watts",
            "gpu_power_percent_tdp",
        ):
            row[key] = None
        if self.gpu:
            n, h = self.nvml, self.gpu
            util = self._query(
                "utilization", lambda: n.nvmlDeviceGetUtilizationRates(h)
            )
            mem = self._query("memory", lambda: n.nvmlDeviceGetMemoryInfo(h))
            if util:
                row.update(
                    gpu_core_load_percent=util.gpu,
                    gpu_memory_controller_load_percent=util.memory,
                )
            if mem:
                row.update(vram_used_bytes=mem.used, vram_total_bytes=mem.total)
            for field, fn in (
                ("gpu_power_watts", n.nvmlDeviceGetPowerUsage),
                (
                    "gpu_default_power_limit_watts",
                    n.nvmlDeviceGetPowerManagementDefaultLimit,
                ),
                ("gpu_enforced_power_limit_watts", n.nvmlDeviceGetEnforcedPowerLimit),
            ):
                value = self._query(field, lambda fn=fn: fn(h))
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
                self.stream.write(json.dumps(row, allow_nan=False) + "\n")
                self.samples += 1
                for key, value in row.items():
                    if isinstance(value, (float, int)) and key not in (
                        "schema",
                        "update",
                        "decisions",
                    ):
                        s = self.summary.setdefault(
                            key, {"count": 0, "sum": 0, "max": value}
                        )
                        s.update(
                            count=s["count"] + 1,
                            sum=s["sum"] + value,
                            max=max(s["max"], value),
                        )
            except Exception as exc:  # noqa: BLE001 -- never interrupt training from sampler
                self.errors["sampler"] = f"{type(exc).__name__}: {exc}"
            if self.stop_event.wait(self.interval):
                break

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=10)
        self.stream.close()
        from .common import write_json

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
