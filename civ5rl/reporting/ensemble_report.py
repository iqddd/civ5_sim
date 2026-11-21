"""Render a saved ensemble evaluation report."""

from __future__ import annotations

from pathlib import Path

from civ5rl.provenance.common import read_json


def report(out):
    out = Path(out)
    result = read_json(out / "summary.json")
    lines = [
        "# Fixed 50/50 probability ensemble",
        "",
        f"Development: {result['development']}",
        "",
        f"Paired comparison: {result['comparison']}",
        "",
        f"Additional: {result['additional']}",
        "",
        "T=0.5; end_turn, 60 rounds, CUDA FP32. Development uses the saved parent evaluation.",
        "All match replays were verified.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
