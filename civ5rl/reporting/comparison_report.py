"""Compact artifact-only summary for the rollout comparison protocol."""

from __future__ import annotations

from pathlib import Path

from civ5rl.provenance.common import read_json


def report(out):
    out = Path(out)
    sections = []
    keys = (
        "status",
        "phase",
        "seconds",
        "overshoot_seconds",
        "source_matches",
        "selected_forks",
        "completed_forks",
        "fork_statuses",
        "accepted_pairs",
        "labels",
        "rejection_reasons",
        "rejections",
        "continuations",
        "training_steps",
        "stop_reason",
        "changed",
        "selected",
        "evaluations_skipped",
        "attempts",
        "rejected_attempts",
        "lr",
        "last_full_kl",
        "anchor_cursor",
        "feature_parity",
        "inherited_steps",
        "inherited_seconds",
        "parent_evaluation_reused",
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
    training = read_json(out / "training.json") if (out / "training.json").exists() else {}
    if training.get("learning"):
        baseline = training["baseline_learning"]
        current = training["learning"]
        sections.extend(
            [
                "## Pair learning",
                "",
                "| Group | Pairs | Loss baseline → accepted | Preferred baseline → accepted | Better argmax baseline → accepted | Preference gained / lost | Argmax gained / lost |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, row in current["pairs"].items():
            before = baseline["pairs"][label]
            sections.append(
                f"| {label} | {row['count']} | {before['pair_loss']:.6f} → {row['pair_loss']:.6f} | "
                f"{before['preferred']} → {row['preferred']} | {before['better_argmax']} → {row['better_argmax']} | "
                f"{row['preference_gained']} / {row['preference_lost']} | {row['argmax_gained']} / {row['argmax_lost']} |"
            )
        sections.extend(
            ["", "## Anchor changes", "", "| Positions | Changed argmax | Fraction |", "|---|---:|---:|"]
        )
        for label, row in current["anchors"].items():
            sections.append(f"| {label} | {row['changed']} / {row['count']} | {row['fraction']:.2%} |")
        sections.append("")
    result = read_json(out / "result.json") if (out / "result.json").exists() else {}
    if "parent_development" in result and "development" in result:
        sections.extend(["## Paired Development", "", "| Policy | W / D / L | Win rate |", "|---|---:|---:|"])
        for label, key in (("Parent", "parent_development"), ("Candidate", "development")):
            row = result[key]["runs"]["argmax"]["cp"]["summary"]
            sections.append(f"| {label} | {row['W']} / {row['D']} / {row['L']} | {row['win_rate']:.2%} |")
        comparison = result["comparison"]
        lo, hi = comparison["CI95"]
        sections.extend(
            [
                "",
                f"Win-rate difference: {100 * comparison['delta']:+.2f} pp; paired world-bootstrap CI95 [{100 * lo:+.2f}, {100 * hi:+.2f}] pp.",
                "",
            ]
        )
    (out / "report.md").write_text("# Rollout comparison\n\n" + "\n".join(sections), encoding="utf-8")
