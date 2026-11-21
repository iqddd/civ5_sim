"""Read saved experiment records; never launch evaluation or modify scientific files."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from civ5rl.engine.runtime.telemetry_analysis import read_live_jsonl, summarize_performance
from civ5rl.provenance.common import write_json
from civ5rl.reporting.markdown import format_wdl
from civ5rl.reporting.output_paths import derived_output
from civ5rl.reporting.plot import pyplot
from civ5rl.reporting.telemetry import performance_markdown


def _read(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def build_snapshot(directory, *, block_size=10, last=None):
    """Aggregate saved training blocks and distinct Development records.

    Evaluation records are keyed by (purpose, checkpoint); conflicting W/D/L
    values are errors. Historical records without purpose default to development.
    Prefer a duplicate with decision coordinates over one without them.
    """
    if (
        type(block_size) is not int
        or block_size < 1
        or (last is not None and (type(last) is not int or last < 1))
    ):
        raise ValueError("block_size and last must be positive integers")
    directory = Path(directory)
    state = _read(directory / "state.json", {})
    status = _read(directory / "status.json", {})
    record = _read(directory / "run.json", {})
    config = _read(directory / "resolved-config.json", record.get("config", {}))
    path = directory / "updates.jsonl"
    rows = read_live_jsonl(path)
    if last:
        rows = rows[-last:]
    blocks = []
    for i in range(0, len(rows), block_size):
        chunk = rows[i : i + block_size]
        counts = {k: sum(row.get(k, 0) for row in chunk) for k in ("W", "D", "L")}
        total = sum(counts.values())
        block = {
            "first_update": chunk[0].get("rollout", chunk[0].get("update")),
            "last_update": chunk[-1].get("rollout", chunk[-1].get("update")),
            "total_decisions": chunk[-1].get("total_decisions"),
            "updates": len(chunk),
            **counts,
            "matches": total,
            "win_rate": counts["W"] / total if total else None,
        }
        for field in (
            "critic_explained_variance",
            "critic_mse",
            "entropy",
            "normalized_entropy",
            "sampled_kl",
            "full_kl_free",
        ):
            values = [r.get("ppo", {}).get(field) for r in chunk]
            values = [v for v in values if v is not None]
            block[field + "_mean"] = statistics.mean(values) if values else None
        block["attempted_steps"] = sum(r.get("ppo", {}).get("attempted_optimizer_steps", 0) for r in chunk)
        block["rejected_steps"] = sum(r.get("ppo", {}).get("rejected_optimizer_steps", 0) for r in chunk)
        blocks.append(block)
    points = {}
    evaluated = state.get("evaluated", {})
    sources = list(evaluated.values()) if isinstance(evaluated, dict) else list(evaluated)
    extra = _read(directory / "evaluations.json", {})
    sources += list(extra.values()) if isinstance(extra, dict) else list(extra)
    for item in sources:
        if not isinstance(item, dict) or item.get("purpose", "development") not in (
            "development",
            "development_512",
        ):
            continue
        purpose = item.get("purpose", "development")
        checkpoint = item.get("checkpoint")
        summary = item.get("runs", {}).get("argmax", {}).get("cp", {}).get("summary")
        if checkpoint and summary and all(k in summary for k in ("W", "D", "L")):
            total = sum(summary[k] for k in ("W", "D", "L"))
            point = {
                "purpose": purpose,
                "checkpoint": checkpoint,
                "decisions": item.get("decisions"),
                "epoch": item.get("epoch"),
                **{k: summary[k] for k in ("W", "D", "L")},
                "matches": total,
                "win_rate": summary["W"] / total if total else None,
            }
            key = (purpose, checkpoint)
            if key in points and any(points[key][k] != point[k] for k in ("W", "D", "L")):
                raise ValueError(f"Conflicting {purpose} results for {checkpoint}")
            if key not in points or (points[key]["decisions"] is None and point["decisions"] is not None):
                points[key] = point
    evals = list(points.values())
    evals.sort(
        key=lambda p: (
            p["decisions"] is None,
            p["decisions"] or 0,
            p["epoch"] or 0,
            p["purpose"],
            p["checkpoint"],
        )
    )
    performance = None
    if path.exists() and all("rollout" in row for row in rows):
        performance = summarize_performance(directory, last=last)
    return {
        "run_directory": str(directory.resolve()),
        "status": status,
        "state_boundary": {
            k: state.get(k)
            for k in (
                "phase",
                "latest",
                "updates",
                "rollouts",
                "epoch",
                "epoch_completed",
                "decisions",
                "active_seconds",
                "optimizer_steps",
                "attempted_steps",
                "rejected_steps",
                "stop_reason",
            )
            if k in state
        },
        "budget": config.get("experiment", {}),
        "training_blocks": blocks,
        "training_aggregation": "W/D/L pooled over episodes; numeric PPO fields are unweighted means over updates",
        "development_argmax_cp": evals,
        "confirmation": _read(directory / "confirmation.json", None),
        "performance": performance,
    }


def _missing_coordinates(result):
    return sum(
        point["decisions"] is None and point["epoch"] is None for point in result["development_argmax_cp"]
    )


def render_snapshot(result):
    boundary, budget = result["state_boundary"], result["budget"]

    def shown(value):
        return "n/a" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)

    lines = [
        "# Derived run snapshot",
        "",
        f"Status: {result['status'].get('status', 'unknown')}",
        (
            f"Decisions: {shown(boundary.get('decisions'))} / {shown(budget.get('decisions'))}; "
            f"active time: {shown(boundary.get('active_seconds'))} / {shown(budget.get('active_seconds'))} s."
        ),
        (
            f"Adam steps: {shown(boundary.get('optimizer_steps'))} accepted, "
            f"{shown(boundary.get('attempted_steps'))} attempted, "
            f"{shown(boundary.get('rejected_steps'))} rejected."
        ),
        "",
        result["training_aggregation"],
        "",
        "| Last update | Decisions | W/D/L | Pooled win rate |",
        "|---:|---:|---:|---:|",
    ]
    for block in result["training_blocks"]:
        rate = f"{block['win_rate']:.2%}" if block["win_rate"] is not None else "n/a"
        lines.append(
            f"| {block['last_update']} | {block['total_decisions']} | {format_wdl(block)} | {rate} |"
        )
    if result["training_blocks"]:

        def metric(value):
            return "n/a" if value is None else f"{value:.4g}"

        lines += [
            "",
            "PPO metrics below are means per update; step counts are totals per block.",
            "",
            "| Last update | Critic EV | Critic MSE | Entropy (nats) | Sampled KL | Full KL (free) | Attempted | Rejected |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for block in result["training_blocks"]:
            fields = (
                "critic_explained_variance_mean",
                "critic_mse_mean",
                "entropy_mean",
                "sampled_kl_mean",
                "full_kl_free_mean",
            )
            values = " | ".join(metric(block[field]) for field in fields)
            lines.append(
                f"| {block['last_update']} | {values} | {block['attempted_steps']} | {block['rejected_steps']} |"
            )
    lines += [
        "",
        "Development: fixed cases, argmax policy against CP.",
        "",
        "| Purpose | Checkpoint | Decisions | Epoch | W/D/L | Win rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for point in result["development_argmax_cp"]:
        rate = f"{point['win_rate']:.2%}" if point["win_rate"] is not None else "n/a"
        lines.append(
            f"| {point['purpose']} | {point['checkpoint']} | {point['decisions']} | {point['epoch']} | {format_wdl(point)} | {rate} |"
        )
    no_coordinates = _missing_coordinates(result)
    if no_coordinates:
        lines += [
            "",
            f"{no_coordinates} development point(s) have no decisions or epoch and cannot be plotted.",
        ]
    if result["performance"] is not None:
        lines += [
            "",
            "## Timing and GPU utilization",
            "",
            performance_markdown(result["performance"]).rstrip(),
        ]
    lines += ["", "![Learning curves](learning-curves-detailed.png)", ""]
    return "\n".join(lines)


def _plot_series(axis, series_points, field):
    for purpose in sorted({p["purpose"] for p in series_points}):
        series = sorted(
            (p for p in series_points if p["purpose"] == purpose and p[field] is not None),
            key=lambda p: p[field],
        )
        if series:
            axis.plot(
                [p[field] for p in series],
                [p["win_rate"] for p in series],
                marker="s",
                label=f"{purpose}: fixed cases, argmax vs CP",
            )


def _plot_snapshot(result, output):
    plt = pyplot()
    blocks = result["training_blocks"]
    points = result["development_argmax_cp"]
    x = [b["total_decisions"] for b in blocks]
    training_has_x = bool(blocks) and all(v is not None for v in x)
    has_decisions = training_has_x or any(p["decisions"] is not None for p in points)
    has_epoch_only = any(p["decisions"] is None and p["epoch"] is not None for p in points)
    mixed = has_decisions and has_epoch_only
    fig, axes = plt.subplots(3 if mixed else 2, 2, figsize=(12, 11 if mixed else 8), constrained_layout=True)
    if training_has_x:
        axes[0, 0].plot(x, [b["win_rate"] for b in blocks], marker="o", label="Training: pooled W / episodes")
        for ax, fields in (
            (axes[0, 1], ("critic_explained_variance_mean", "critic_mse_mean")),
            (axes[1, 0], ("entropy_mean",)),
            (axes[1, 1], ("sampled_kl_mean", "full_kl_free_mean")),
        ):
            for field in fields:
                ax.plot(
                    x, [b[field] for b in blocks], marker="o", label=field.replace("_mean", " (mean/update)")
                )
    x_field = "decisions" if has_decisions else "epoch"
    x_label = "Cumulative learner decisions" if has_decisions else "Completed training epoch"

    _plot_series(axes[0, 0], points, x_field)
    for ax, title, ylabel in zip(
        axes.flat[:4],
        ("Win rate", "Critic", "Policy entropy", "KL divergence"),
        ("Fraction", "Explained variance / MSE", "Entropy (nats)", "KL (nats)"),
    ):
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize="small")
    if mixed:
        epoch_axis = axes[2, 0]
        _plot_series(epoch_axis, [p for p in points if p["decisions"] is None], "epoch")
        epoch_axis.set(
            title="Win rate (epoch-only points)", xlabel="Completed training epoch", ylabel="Fraction"
        )
        epoch_axis.grid(True, alpha=0.3)
        if epoch_axis.get_legend_handles_labels()[0]:
            epoch_axis.legend(fontsize="small")
        axes[2, 1].set_visible(False)
    no_coordinates = _missing_coordinates(result)
    if no_coordinates:
        fig.suptitle(f"{no_coordinates} point(s) without decisions or epoch not plotted", fontsize="small")
    fig.savefig(output / "learning-curves-detailed.png", dpi=150)
    plt.close(fig)


def write_snapshot(directory, output=None, *, block_size=10, last=None):
    """Write tables and curves without inventing missing evaluation coordinates.

    If decision-based and epoch-only points coexist, plot the latter on a
    separate epoch axis. Points with neither coordinate remain in the table
    with a missing-coordinate notice and are excluded from curves.
    """
    directory = Path(directory).resolve()
    output = derived_output(directory, output if output else directory / "derived" / "snapshot")
    result = build_snapshot(directory, block_size=block_size, last=last)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "snapshot.json", result)
    (output / "snapshot.md").write_text(render_snapshot(result), encoding="utf-8")
    _plot_snapshot(result, output)
    return result
