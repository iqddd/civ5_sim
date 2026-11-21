"""Kineto trace summaries and verified trace archives used by RL measurements."""

from __future__ import annotations

import gzip
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import torch

from civ5rl.provenance.common import read_json
from civ5rl.reporting.storage import pack_verified


def top_cuda_operators(trace, limit=12):
    """Summarize inclusive profiler operator timings, preserving the legacy metric."""
    rows = [
        {"name": row.key, "cuda_us": row.device_time_total, "count": row.count}
        for row in trace.key_averages()
        if row.device_time_total
    ]
    return sorted(rows, key=lambda row: row["cuda_us"], reverse=True)[:limit]


def top_cuda_kernels(trace, limit=12):
    """Aggregate device events, excluding inclusive CPU/ATen operator timings."""
    grouped = defaultdict(lambda: {"cuda_us": 0.0, "count": 0})
    for event in trace.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            row = grouped[event.name]
            row["cuda_us"] += event.device_time_total
            row["count"] += 1
    return sorted(
        [dict(name=name, **row) for name, row in grouped.items()],
        key=lambda row: row["cuda_us"],
        reverse=True,
    )[:limit]


def store_trace(out):
    """Keep large Kineto JSON in verified PPMd storage, outside measured timing."""
    out = Path(out).resolve()
    if not (out / "trace.json").exists():
        return None
    raw = out / "trace"
    if raw.resolve().parent != out or raw.exists():
        raise ValueError("Trace staging path is occupied")
    raw.mkdir()
    (out / "trace.json").rename(raw / "trace.json")
    pack_verified(
        raw,
        out / "trace.storage",
        token="turn-fit-trace",
        temporary_prefix="verify-turn-trace-",
        changed_message="Trace changed during verification",
    )
    if raw.resolve().parent != out or raw.is_symlink():
        raise ValueError("Unsafe trace cleanup")
    shutil.rmtree(raw)
    return "trace.storage"


def trace_summary(path):
    events = read_json(path)["traceEvents"]
    kernels = [event for event in events if event.get("cat") == "kernel" and event.get("dur", 0) > 0]
    if not kernels:
        return {"error": "No CUDA kernels recorded"}
    intervals = sorted((event["ts"], event["ts"] + event["dur"]) for event in kernels)
    end = intervals[0][0]
    busy = 0.0
    gaps = []
    for start, stop in intervals:
        if start > end:
            gaps.append(start - end)
            busy += stop - start
        elif stop > end:
            busy += stop - end
        end = max(end, stop)
    times = Counter()
    runtime = Counter()
    copies = {}
    for event in events:
        if event.get("cat") == "kernel":
            times[event["name"]] += event.get("dur", 0) / 1000
        elif "cuda_runtime" in event.get("cat", ""):
            runtime[event["name"]] += event.get("dur", 0) / 1000
        elif event.get("cat") == "gpu_memcpy":
            row = copies.setdefault(event["name"], {"count": 0, "bytes": 0, "ms": 0.0})
            row["count"] += 1
            row["bytes"] += event.get("args", {}).get("bytes", 0)
            row["ms"] += event.get("dur", 0) / 1000
    window = end - intervals[0][0]
    return {
        "kernel_count": len(kernels),
        "kernel_window_ms": window / 1000,
        "kernel_union_ms": busy / 1000,
        "kernel_busy_in_window_percent": 100 * busy / window,
        "gaps_over_1ms": sum(g > 1000 for g in gaps),
        "largest_gap_ms": max(gaps, default=0) / 1000,
        "top_kernels": [{"name": n, "ms": t} for n, t in times.most_common(12)],
        "top_runtime": [{"name": n, "ms": t} for n, t in runtime.most_common(8)],
        "copies": copies,
    }


def _merge_intervals(intervals):
    if not intervals:
        return 0.0, []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    total = sum(end - start for start, end in merged)
    return total, merged


def summarize_kineto_trace(trace_path, wall_seconds):
    opener = gzip.open if str(trace_path).endswith(".gz") else open
    with opener(trace_path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    events = payload.get("traceEvents", [])

    kernel_intervals = []
    memcpy_intervals = []
    memset_intervals = []
    kernel_by_name = {}
    memcpy_by_name = {}
    cuda_runtime_by_name = {}
    annotations = {}

    for ev in events:
        if ev.get("ph") != "X":
            continue
        dur = float(ev.get("dur", 0.0))
        ts = float(ev.get("ts", 0.0))
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        if cat == "kernel":
            kernel_intervals.append((ts, ts + dur))
            rec = kernel_by_name.setdefault(name, {"count": 0, "us": 0.0})
            rec["count"] += 1
            rec["us"] += dur
        elif cat == "gpu_memcpy":
            memcpy_intervals.append((ts, ts + dur))
            rec = memcpy_by_name.setdefault(name, {"count": 0, "us": 0.0})
            rec["count"] += 1
            rec["us"] += dur
        elif cat == "gpu_memset":
            memset_intervals.append((ts, ts + dur))
        elif cat == "cuda_runtime":
            rec = cuda_runtime_by_name.setdefault(name, {"count": 0, "us": 0.0})
            rec["count"] += 1
            rec["us"] += dur
        elif cat == "user_annotation":
            rec = annotations.setdefault(name, {"count": 0, "us": 0.0})
            rec["count"] += 1
            rec["us"] += dur

    kernel_us, merged_kernels = _merge_intervals(kernel_intervals)
    memcpy_us, _ = _merge_intervals(memcpy_intervals)
    memset_us, _ = _merge_intervals(memset_intervals)
    any_gpu_us, merged_gpu = _merge_intervals([*kernel_intervals, *memcpy_intervals, *memset_intervals])

    gaps_us = [
        merged_gpu[i + 1][0] - merged_gpu[i][1]
        for i in range(len(merged_gpu) - 1)
        if merged_gpu[i + 1][0] > merged_gpu[i][1]
    ]
    wall_us = wall_seconds * 1e6

    top_kernels = sorted(
        ({"name": k, "count": v["count"], "ms": v["us"] / 1000.0} for k, v in kernel_by_name.items()),
        key=lambda item: item["ms"],
        reverse=True,
    )[:20]
    top_runtime = sorted(
        ({"name": k, "count": v["count"], "ms": v["us"] / 1000.0} for k, v in cuda_runtime_by_name.items()),
        key=lambda item: item["ms"],
        reverse=True,
    )[:20]
    memcpy_summary = sorted(
        ({"name": k, "count": v["count"], "ms": v["us"] / 1000.0} for k, v in memcpy_by_name.items()),
        key=lambda item: item["ms"],
        reverse=True,
    )
    annotation_summary = sorted(
        ({"name": k, "count": v["count"], "ms": v["us"] / 1000.0} for k, v in annotations.items()),
        key=lambda item: item["ms"],
        reverse=True,
    )

    return {
        "wall_ms": wall_us / 1000.0,
        "gpu_kernel_busy_ms": kernel_us / 1000.0,
        "gpu_kernel_busy_pct_of_wall": 100.0 * kernel_us / wall_us if wall_us else 0.0,
        "gpu_memcpy_ms": memcpy_us / 1000.0,
        "gpu_memset_ms": memset_us / 1000.0,
        "gpu_any_engine_busy_ms": any_gpu_us / 1000.0,
        "gpu_any_engine_busy_pct_of_wall": 100.0 * any_gpu_us / wall_us if wall_us else 0.0,
        "gpu_idle_gaps": {
            "count": len(gaps_us),
            "total_ms": sum(gaps_us) / 1000.0,
            "gaps_over_100us_count": sum(1 for g in gaps_us if g >= 100.0),
            "gaps_over_100us_ms": sum(g for g in gaps_us if g >= 100.0) / 1000.0,
            "gaps_over_1ms_count": sum(1 for g in gaps_us if g >= 1000.0),
            "gaps_over_1ms_ms": sum(g for g in gaps_us if g >= 1000.0) / 1000.0,
            "max_gap_ms": max(gaps_us, default=0.0) / 1000.0,
        },
        "user_annotations": annotation_summary,
        "memcpy_breakdown": memcpy_summary,
        "top_cuda_runtime": top_runtime,
        "top_kernels": top_kernels,
    }
