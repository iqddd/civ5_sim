"""Compact artifact-only summary for the rollout comparison protocol."""

from pathlib import Path

from civ5rl.provenance.common import read_json


def report(out):
    out = Path(out)
    sections = []
    keys = (
        "status", "phase", "seconds", "overshoot_seconds", "source_matches",
        "selected_forks", "completed_forks", "fork_statuses", "accepted_pairs",
        "labels", "rejection_reasons", "rejections", "continuations",
        "training_steps", "stop_reason", "changed", "selected", "evaluations_skipped",
        "attempts", "rejected_attempts", "lr", "last_full_kl", "anchor_cursor",
    )
    for name in ("collection.json", "dataset.json", "training.json", "result.json"):
        if not (out / name).exists():
            continue
        value = read_json(out / name)
        compact = {key: value[key] for key in keys if key in value}
        if name == "collection.json":
            compact.update(
                selected_forks=len(value["forks"]),
                completed_forks=sum(fork["status"] == "completed" for fork in value["forks"]),
            )
        sections.extend([f"## {name}", "", str(compact), ""])
    if (out / "result.json").exists() and read_json(out/'run.json')['config'].get('evaluation_protocol')!='paired_development_only':
        sections.extend([
            "Historical Development comparison uses the recorded 361 wins of update171; GPU/runtime differences limit attribution. Additional, when admitted, runs both policies through one local evaluator.",
            "",
        ])
    (out / "report.md").write_text(
        "# Rollout comparison\n\n" + "\n".join(sections), encoding="utf-8"
    )
