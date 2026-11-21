"""Analyze saved phase events, resource samples and training updates."""

from __future__ import annotations

import json
import statistics
from bisect import bisect_right
from collections import defaultdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

from civ5rl.engine.runtime import is_int_at_least


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
    return {session: ([_sample_time(row) for row in rows], rows) for session, rows in grouped.items()}


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
        if segment and (segment[-1][1] != interval[0] or segment[-1][3:] != interval[3:]):
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
        samples = read_live_jsonl(path)
        events = read_live_jsonl(path.with_name("phases.jsonl"))
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
        for label, start, stop in _phase_pieces(sample, begin, end, timelines.get(session)):
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


def read_live_jsonl(path):
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
    return json.loads(schema.read_text()).get("interval_seconds", 1) if schema.exists() else 1


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
    updates = read_live_jsonl(directory / "updates.jsonl")
    total = len(updates)
    updates = updates[-last:] if last else updates
    ids = [r["rollout"] for r in updates]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate completed rollout IDs")
    selected = set(ids)
    telemetry = directory / "telemetry"
    if not telemetry.exists():
        telemetry = directory
    samples = read_live_jsonl(telemetry / "resources.jsonl")
    events = read_live_jsonl(telemetry / "phases.jsonl")
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
                item["intervals"].append((begin, end, value, session, label.get("rollout")))

    for item in phase.values():
        segments = list(_contiguous_segments(item.pop("intervals")))
        windows = [average for segment in segments for _, average in _load_windows(segment, 60)]
        weighted_gpu = item.pop("weighted_gpu")
        item["gpu_core_load_percent_mean"] = (
            weighted_gpu / item["nvml_seconds"] if item["nvml_seconds"] else None
        )
        item["nvml_coverage_fraction"] = (
            min(1.0, item["nvml_seconds"] / item["seconds"]) if item["seconds"] else None
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
                if field in r.get(group, {}) and not (skip and r[group].get("PPO_skipped"))
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
            "ppo_operation_without_compilation": timing("ppo", "operation_without_compilation_seconds", True),
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
