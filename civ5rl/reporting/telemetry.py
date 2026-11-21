"""Report on telemetry saved by RL runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from civ5rl.engine.runtime.telemetry import analyze_telemetry
from civ5rl.provenance.common import write_json


def write_ppo_gpu_analysis(out: Path) -> dict[str, Any] | None:
    telemetry = out / "telemetry/resources.jsonl"
    if not telemetry.is_file():
        return None
    analysis = analyze_telemetry(telemetry, active_phase="ppo")
    write_json(out / "gpu-analysis.json", {key: value for key, value in analysis.items() if key != "active"})
    return analysis
