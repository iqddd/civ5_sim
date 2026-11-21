"""Report on telemetry saved by RL runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from civ5rl.engine.runtime.telemetry_analysis import analyze_telemetry
from civ5rl.provenance.common import write_json


def write_ppo_gpu_analysis(out: Path) -> dict[str, Any] | None:
    telemetry = out / "telemetry/resources.jsonl"
    if not telemetry.is_file():
        return None
    analysis = analyze_telemetry(telemetry, active_phase="ppo")
    write_json(out / "gpu-analysis.json", {key: value for key, value in analysis.items() if key != "active"})
    return analysis


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
            values = " | ".join(number(stats[f"{key}_seconds"]) for key in ("mean", "median", "min", "max"))
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
