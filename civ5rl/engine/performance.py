"""Read-only performance summaries of completed PPO cycles and NVML phases."""

import json
import statistics
from bisect import bisect_right
from collections import defaultdict
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path


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


def _time(row):
    return datetime.fromisoformat(row["timestamp_utc"]).timestamp()


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
    if last is not None and (type(last) is not int or last < 1):
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
    schema = telemetry / "resources-schema.json"
    interval = (
        json.loads(schema.read_text()).get("interval_seconds", 1)
        if schema.exists()
        else 1
    )
    max_gap = max(5, 3 * interval)
    by_session = defaultdict(list)
    for event in events:
        by_session[event["session_id"]].append(event)
    starts = {key: [_time(r) for r in rows] for key, rows in by_session.items()}
    tails = {}
    for row in samples:
        tails[row.get("session_id")] = _time(row)
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
    for session, rows in by_session.items():
        for a, b in pairwise(rows):
            duration(a, max(0, b["elapsed_seconds"] - a["elapsed_seconds"]))
        if rows and rows[-1]["phase"] != "stopped" and session in tails:
            duration(rows[-1], max(0, tails[session] - _time(rows[-1])))

    excluded_gaps = 0
    for a, b in pairwise(samples):
        begin, end = _time(a), _time(b)
        session = a.get("session_id")
        elapsed = b.get("elapsed_seconds", end) - a.get("elapsed_seconds", begin)
        if (
            session != b.get("session_id")
            or not 0 < end - begin <= max_gap
            or elapsed <= 0
        ):
            excluded_gaps += 1
            continue
        boundaries = starts.get(session)
        if boundaries:
            index = bisect_right(boundaries, begin) - 1
            pieces = []
            cursor = begin
            while cursor < end:
                stop = (
                    min(end, boundaries[index + 1])
                    if index + 1 < len(boundaries)
                    else end
                )
                if index >= 0 and stop > cursor:
                    pieces.append((by_session[session][index], cursor, stop))
                cursor = stop
                index += 1
        else:
            pieces = [(a, begin, end)]
            duration(a, end - begin)
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
        segments = []
        for a, b, value, session, rollout in item.pop("intervals"):
            if (
                not segments
                or segments[-1][-1][1] != a
                or segments[-1][-1][3:] != (session, rollout)
            ):
                segments.append([])
            segments[-1].append((a, b, value, session, rollout))
        windows = []
        for segment in segments:
            left = 0
            weighted = 0.0
            for right, (a, b, value, *_rest) in enumerate(segment):
                weighted += (b - a) * value
                start = b - 60
                while left < right and segment[left][1] <= start:
                    x, y, v, *_ = segment[left]
                    weighted -= (y - x) * v
                    left += 1
                if segment[left][0] <= start:
                    windows.append(
                        (weighted - (start - segment[left][0]) * segment[left][2]) / 60
                    )
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
        if events and all(r.get("session_id") in by_session for r in samples)
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
