"""Read saved experiment records; never launch evaluation or modify scientific files."""

import json
import statistics
from pathlib import Path

from civ5rl.engine.performance import (
    _lines,
    performance_markdown,
    summarize_performance,
)
from civ5rl.provenance.common import write_json


def _read(path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def build_snapshot(directory, *, block_size=10, last=None):
    if type(block_size) is not int or block_size < 1 or (last is not None and (type(last) is not int or last < 1)):
        raise ValueError("block_size and last must be positive integers")
    directory = Path(directory)
    state = _read(directory / "state.json", {})
    status = _read(directory / "status.json", {})
    record = _read(directory / "run.json", {})
    config = _read(directory / "resolved-config.json", record.get("config", {}))
    path = directory / "updates.jsonl"
    rows = _lines(path)
    if last:
        rows = rows[-last:]
    blocks = []
    for i in range(0, len(rows), block_size):
        chunk = rows[i:i + block_size]
        counts = {k: sum(row.get(k, 0) for row in chunk) for k in ("W", "D", "L")}
        total = sum(counts.values())
        block = {"first_update": chunk[0].get("rollout", chunk[0].get("update")),
                 "last_update": chunk[-1].get("rollout", chunk[-1].get("update")),
                 "total_decisions": chunk[-1].get("total_decisions"),
                 "updates": len(chunk), **counts, "matches": total,
                 "win_rate": counts["W"] / total if total else None}
        for field in ("critic_explained_variance", "critic_mse", "entropy", "normalized_entropy", "sampled_kl", "full_kl_free"):
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
        if not isinstance(item, dict) or item.get("purpose", "development") != "development":
            continue
        checkpoint = item.get("checkpoint")
        summary = item.get("runs", {}).get("argmax", {}).get("cp", {}).get("summary")
        if checkpoint and summary and all(k in summary for k in ("W", "D", "L")):
            total = sum(summary[k] for k in ("W", "D", "L"))
            point = {"checkpoint": checkpoint, "decisions": item.get("decisions"),
                     "epoch": item.get("epoch"), **{k: summary[k] for k in ("W", "D", "L")},
                     "matches": total, "win_rate": summary["W"] / total if total else None}
            if checkpoint in points and any(points[checkpoint][k] != point[k] for k in ("W", "D", "L")):
                raise ValueError(f"Conflicting development results for {checkpoint}")
            if checkpoint not in points or (points[checkpoint]["decisions"] is None and point["decisions"] is not None):
                points[checkpoint] = point
    evals = list(points.values())
    evals.sort(key=lambda p: (p["decisions"] is None, p["decisions"] or 0, p["epoch"] or 0, p["checkpoint"]))
    performance = None
    if path.exists() and all("rollout" in row for row in rows):
        performance = summarize_performance(directory, last=last)
    return {"run_directory": str(directory.resolve()), "status": status, "state_boundary": {k: state.get(k) for k in ("phase", "latest", "updates", "rollouts", "epoch", "epoch_completed", "decisions", "active_seconds", "optimizer_steps", "attempted_steps", "rejected_steps", "stop_reason") if k in state},
            "budget": config.get("experiment", {}), "training_blocks": blocks,
            "training_aggregation": "W/D/L pooled over episodes; numeric PPO fields are unweighted means over updates",
            "development_argmax_cp": evals,
            "confirmation": _read(directory / "confirmation.json", None), "performance": performance}


def write_snapshot(directory, output=None, *, block_size=10, last=None):
    directory = Path(directory).resolve()
    output = Path(output).resolve() if output else directory / "derived" / "snapshot"
    if output == directory or directory.is_relative_to(output):
        raise ValueError("Snapshot output must be separate from the run root")
    if output.is_relative_to(directory) and output.relative_to(directory).parts[0] not in ("derived", "snapshots", "interim-snapshot"):
        raise ValueError("Inside a run, use derived/, snapshots/ or interim-snapshot/")
    result = build_snapshot(directory, block_size=block_size, last=last)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "snapshot.json", result)
    boundary, budget = result["state_boundary"], result["budget"]
    def shown(value):
        return "n/a" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)

    lines = ["# Derived run snapshot", "", f"Status: {result['status'].get('status', 'unknown')}",
             (f"Decisions: {shown(boundary.get('decisions'))} / {shown(budget.get('decisions'))}; "
             f"active time: {shown(boundary.get('active_seconds'))} / {shown(budget.get('active_seconds'))} s."),
             (f"Adam steps: {shown(boundary.get('optimizer_steps'))} accepted, "
             f"{shown(boundary.get('attempted_steps'))} attempted, "
             f"{shown(boundary.get('rejected_steps'))} rejected."),
             "", result["training_aggregation"], "", "| Last update | Decisions | W/D/L | Pooled win rate |",
             "|---:|---:|---:|---:|"]
    for b in result["training_blocks"]:
        rate = f"{b['win_rate']:.2%}" if b['win_rate'] is not None else "n/a"
        lines.append(f"| {b['last_update']} | {b['total_decisions']} | {b['W']}/{b['D']}/{b['L']} | {rate} |")
    if result["training_blocks"]:
        def metric(value):
            return "n/a" if value is None else f"{value:.4g}"

        lines += ["", "PPO metrics below are means per update; step counts are totals per block.", "",
                  "| Last update | Critic EV | Critic MSE | Entropy (nats) | Sampled KL | Full KL (free) | Attempted | Rejected |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for b in result["training_blocks"]:
            fields = ("critic_explained_variance_mean", "critic_mse_mean", "entropy_mean",
                      "sampled_kl_mean", "full_kl_free_mean")
            values = " | ".join(metric(b[field]) for field in fields)
            lines.append(f"| {b['last_update']} | {values} | {b['attempted_steps']} | {b['rejected_steps']} |")
    lines += ["", "Development: fixed cases, argmax policy against CP.", "", "| Checkpoint | Decisions | W/D/L | Win rate |", "|---|---:|---:|---:|"]
    for p in result["development_argmax_cp"]:
        rate = f"{p['win_rate']:.2%}" if p['win_rate'] is not None else "n/a"
        lines.append(f"| {p['checkpoint']} | {p['decisions']} | {p['W']}/{p['D']}/{p['L']} | {rate} |")
    if result["performance"] is not None:
        lines += ["", "## Timing and GPU utilization", "", performance_markdown(result["performance"]).rstrip()]
    lines += ["", "![Learning curves](learning-curves-detailed.png)", ""]
    (output / "snapshot.md").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    blocks = result["training_blocks"]
    x = [b["total_decisions"] for b in blocks]
    if blocks and all(v is not None for v in x):
        axes[0, 0].plot(x, [b["win_rate"] for b in blocks], marker="o", label="Training: pooled W / episodes")
        for ax, fields in ((axes[0, 1], ("critic_explained_variance_mean", "critic_mse_mean")),
                           (axes[1, 0], ("entropy_mean",)),
                           (axes[1, 1], ("sampled_kl_mean", "full_kl_free_mean"))):
            for field in fields:
                ax.plot(x, [b[field] for b in blocks], marker="o", label=field.replace("_mean", " (mean/update)"))
    points = [p for p in result["development_argmax_cp"] if p["decisions"] is not None]
    x_label = "Cumulative learner decisions"
    if points:
        axes[0, 0].plot([p["decisions"] for p in points], [p["win_rate"] for p in points], marker="s", label="Development: fixed cases, argmax vs CP")
    elif not blocks:
        epoch_points = sorted(
            (p for p in result["development_argmax_cp"] if p["epoch"] is not None),
            key=lambda p: p["epoch"],
        )
        if epoch_points:
            x_label = "Completed training epoch"
            axes[0, 0].plot([p["epoch"] for p in epoch_points],
                            [p["win_rate"] for p in epoch_points], marker="s",
                            label="Development: fixed cases, argmax vs CP")
    for ax, title, ylabel in zip(axes.flat, ("Win rate", "Critic", "Policy entropy", "KL divergence"), ("Fraction", "Explained variance / MSE", "Entropy (nats)", "KL (nats)")):
        ax.set_title(title); ax.set_xlabel(x_label); ax.set_ylabel(ylabel); ax.grid(True, alpha=.3)
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize="small")
    fig.savefig(output / "learning-curves-detailed.png", dpi=150)
    plt.close(fig)
    return result
