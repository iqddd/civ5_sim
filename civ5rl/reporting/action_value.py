"""Render saved action-value fitting results."""

from __future__ import annotations

from pathlib import Path

from civ5rl.provenance.common import read_json


def report(out):
    out = Path(out)
    if not (out / "training.json").exists():
        return
    training = read_json(out / "training.json")
    lines = [
        "# Action-value fitting",
        "",
        f"Completed epochs: {training['epoch']}; steps: {training['steps']}; training: {training['seconds']:.2f}s.",
        "",
        f"Best selection: {training['best']}.",
    ]
    if (out / "result.json").exists():
        result = read_json(out / "result.json")
        confirmation = result["confirmation"]
        bootstrap = result["bootstrap"]
        lines += ["", "| Confirmation rule | Mean win fraction |", "|---|---:|"]
        lines += [f"| {k} | {confirmation[k]:.6f} |" for k in ("head", "parent", "alternative")]
        lines += [
            "",
            f"G: {bootstrap['G']:+.6f}; CI95: {bootstrap['ci95']}; support for a limited policy experiment: {bootstrap['supported']}.",
            "",
            f"Preparation: {result['preparation_seconds']:.2f}s; confirmation: {result['confirmation_seconds']:.2f}s.",
            "",
            "This assesses the new head on held-out training worlds; u171 has used these worlds. It does not establish full-battle policy improvement. Working policy: u171.",
        ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
