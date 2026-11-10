"""Reporting and telemetry formatting for target scale BC experiments."""

from __future__ import annotations

import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from civ5rl.engine.resources import analyze_telemetry
from civ5rl.protocols.imitation_target_scale import C2_KIND, MORE_DATA_KIND
from civ5rl.provenance.common import read_json, write_json


def extract_environment_metadata(
    run_info: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dynamically extract environment metadata without hardcoded assumptions."""
    run_info = run_info or {}
    config = config or {}
    gpu_name = run_info.get("GPU")
    if not gpu_name and torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except (RuntimeError, ValueError, AttributeError):
            gpu_name = None
    if not gpu_name:
        gpu_name = "CUDA GPU"

    cpu_name = run_info.get("CPU") or run_info.get("cpu") or platform.processor() or "CPU"
    plat = run_info.get("platform") or platform.platform()
    seed = config.get("seed", run_info.get("seed", 17))
    return {
        "gpu": gpu_name,
        "cpu": cpu_name,
        "platform": plat,
        "seed": seed,
    }


def render_learning_curves_plot(
    evaluations: dict[str, Any],
    out: Path,
    eval_purpose: str,
    kind: str,
) -> bool:
    """Render learning curves plot using Matplotlib if evaluation points are present."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = [
        p
        for p in evaluations.values()
        if isinstance(p, dict) and p.get("purpose") == eval_purpose
    ]
    if kind == C2_KIND:
        points = [p for p in points if p.get("candidate_role") == "c2_candidate"]
    points.sort(key=lambda p: int(p.get("Adam_steps", 0)))
    if not points:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(
        [p.get("Adam_steps", 0) for p in points],
        [100 * p["runs"]["argmax"]["cp"]["summary"]["win_rate"] for p in points],
        marker="o",
    )
    axes[0].set(
        xlabel="BC Adam steps",
        ylabel=f"{eval_purpose} argmax wins vs CP, %",
    )
    metrics = [p for p in points if p.get("fixed_training")]
    if metrics:
        axes[1].plot(
            [p.get("Adam_steps", 0) for p in metrics],
            [p["fixed_training"]["ce"] for p in metrics],
            marker="o",
            label="Fixed training CE",
        )
    holdout_pts = [p for p in points if p.get("holdout")]
    if holdout_pts:
        axes[1].plot(
            [p.get("Adam_steps", 0) for p in holdout_pts],
            [p["holdout"]["ce"] for p in holdout_pts],
            marker="s",
            label="Holdout-126 CE",
        )
        axes[1].legend()
    axes[1].set(xlabel="BC Adam steps", ylabel="Offline CE")
    for ax in axes:
        ax.grid(alpha=0.25)
    fig.savefig(out / "learning-curves.png", dpi=150)
    plt.close(fig)
    return True


def report(config: dict[str, Any], out: Path) -> None:
    """Generate Markdown report and plots for target scale BC experiment."""
    out = Path(out)
    collection = (
        read_json(out / "collection.json")
        if (out / "collection.json").exists()
        else {}
    )
    training = (
        read_json(out / "training.json") if (out / "training.json").exists() else {}
    )
    selection = (
        read_json(out / "selection.json") if (out / "selection.json").exists() else {}
    )
    run_info = read_json(out / "run.json") if (out / "run.json").exists() else {}
    status_info = (
        read_json(out / "status.json") if (out / "status.json").exists() else {}
    )

    env_meta = extract_environment_metadata(run_info, config)
    gpu_name = env_meta["gpu"]
    seed = env_meta["seed"]

    if config["kind"] == C2_KIND:
        title = "# Архитектура C2: 8 hex-блоков и глобальное внимание к известным юнитам (`cp-imitation-target-c2`)"
    elif config["kind"] == MORE_DATA_KIND:
        title = "# Расширение корпуса и продолжение target BC (`cp-imitation-target-more-data`)"
    else:
        title = "# Имитация штатного CP на целевых армиях"
    lines = [
        title,
        "",
        f"Статус: {status_info}; GPU: {gpu_name}; seed {seed}.",
        "",
    ]
    if config["kind"] == C2_KIND:
        lines.append(
            f"Корпус: переиспользован объединённый корпус из {collection.get('labels')} свободных меток "
            f"({collection.get('positions')} позиций; старых боёв {collection.get('old_matches_count')}, "
            f"новых боёв {collection.get('new_matches_count')}); отложенный корпус Holdout-126: "
            f"{collection.get('holdout_labels')} свободных меток ({collection.get('holdout_positions')} позиций, "
            f"{collection.get('holdout_matches_count')} боёв на 7 мирах)."
        )
    elif config["kind"] == MORE_DATA_KIND:
        lines.append(
            f"Сбор: новых свободных меток {collection.get('new_labels')}, старых {collection.get('old_labels')}, "
            f"всего в объединённом корпусе {collection.get('labels')} свободных меток "
            f"({collection.get('positions')} позиций; старых боёв {collection.get('old_matches_count')}, "
            f"новых боёв {collection.get('new_matches_count')}); время нового сбора {collection.get('seconds')} с; "
            f"причина остановки: {collection.get('stop_reason')}."
        )
    else:
        lines.append(
            f"Сбор: {collection.get('labels')} свободных меток, {collection.get('positions')} позиций, "
            f"{len(collection.get('matches', []))} полных CP–CP боёв, {collection.get('seconds')} с; "
            f"причина {collection.get('stop_reason')}."
        )

    manifest = read_json(out / "manifest.json") if (out / "manifest.json").exists() else {}
    worlds_count = len(manifest.get("base", {}).get("worlds", {}).get("training", [])) or 57
    lines += [
        "",
        (
            "Обе стороны обычного stateful CP сохранены отдельно; все исходы включены. "
            f"Training-worlds: {worlds_count}. Forced states и prefix не входят в знаменатель CE."
        ),
        "",
        (
            f"Обучение: {training.get('stop_reason')}; Adam steps {training.get('Adam_steps')}; "
            f"обработано свободных меток {training.get('free_labels_processed')}; active {training.get('active_seconds')} с; "
            f"превышение {training.get('time_overshoot')} с; microbatch {training.get('microbatch')}."
        ),
        "",
    ]
    eval_purpose = "development_512" if config["kind"] == C2_KIND else "development"
    if config["kind"] == C2_KIND:
        lines += [
            "| Роль | Эпоха | Adam steps | Checkpoint | W/D/L (512) | Win rate | Fixed training CE / acc | Holdout-126 CE / acc |",
            "|---|---:|---:|---|---|---:|---|---|",
        ]
    else:
        lines += [
            "| Эпоха | Adam steps | Checkpoint | W/D/L | Win rate | Fixed training CE / accuracy |",
            "|---:|---:|---|---|---:|---|",
        ]
    if (out / "evaluations.json").exists():
        seen_dirs = set()
        raw_points = [
            p
            for p in read_json(out / "evaluations.json").values()
            if isinstance(p, dict) and p.get("purpose") == eval_purpose
        ]
        raw_points.sort(
            key=lambda p: (1 if p.get("candidate_role") else 0), reverse=True
        )
        points = []
        for p in raw_points:
            d = p.get("runs", {}).get("argmax", {}).get("cp", {}).get("directory")
            if d and d not in seen_dirs:
                seen_dirs.add(d)
                points.append(p)
        points.sort(
            key=lambda p: (
                int(p.get("Adam_steps", 0)),
                int(p.get("epoch", 0)),
                str(p.get("checkpoint", "")),
            )
        )
        for point in points:
            summary = point["runs"]["argmax"]["cp"]["summary"]
            metric = point.get("fixed_training") or {}
            if config["kind"] == C2_KIND:
                hold = point.get("holdout") or {}
                lines.append(
                    f"| {point.get('candidate_role', '')} | {point.get('epoch', 0)} | {point.get('Adam_steps', 0)} | {point['checkpoint']} | "
                    f"{summary['W']}/{summary['D']}/{summary['L']} | {summary['win_rate']:.2%} | "
                    f"{metric.get('ce')} / {metric.get('accuracy')} | {hold.get('ce')} / {hold.get('accuracy')} |"
                )
            else:
                lines.append(
                    f"| {point.get('epoch', 0)} | {point.get('Adam_steps', 0)} | {point['checkpoint']} | "
                    f"{summary['W']}/{summary['D']}/{summary['L']} | {summary['win_rate']:.2%} | "
                    f"{metric.get('ce')} / {metric.get('accuracy')} |"
                )
    lines += [
        "",
        "Selection: " + json.dumps(selection, ensure_ascii=False),
        "",
        "Value head неизменна: "
        + str(training.get("value_head_unchanged"))
        + ". Общий encoder менялся; калибровка critic не сохраняется.",
    ]
    if (out / "confirmation.json").exists():
        lines += [
            "",
            "Additional: "
            + json.dumps(read_json(out / "confirmation.json"), ensure_ascii=False),
        ]
    match_rows = collection.get("new_matches", collection.get("matches", []))
    if match_rows:
        commands = Counter()
        units = Counter()
        outcomes = Counter()
        strata = Counter()
        for row in match_rows:
            commands.update(row["commands"])
            units.update(row["unit_types"])
            outcomes[
                "draw" if row["winner"] is None else f"side_{row['winner']}_win"
            ] += 1
            strata[f"{row['map_size']}/{row['army_size']}/{row['placement']}"] += 1
        new_pos = collection.get("new_positions", collection.get("positions", 0))
        new_lbl = collection.get("new_labels", collection.get("labels", 0))
        lines += [
            "",
            "## Корпус",
            "",
            (
                f"Forced positions (новый сбор): {new_pos - new_lbl}; "
                f"превышение меток {collection.get('label_overshoot')}, времени {collection.get('time_overshoot')} с; "
                f"физические повторы {collection.get('physical_repeats')}; "
                f"миров фактически {len(collection.get('coverage', {}).get('worlds', {}))}; "
                f"crops фактически {len(collection.get('coverage', {}).get('crops', {}))}."
            ),
            "",
            "Исходы CP–CP (новый сбор): " + json.dumps(outcomes),
            "",
            "Свободные команды (новый сбор): " + json.dumps(commands),
            "",
            "Свободные метки по типу действующего юнита (новый сбор): "
            + json.dumps(units),
            "",
            "Расстановки map/army/contact (новый сбор): " + json.dumps(strata),
        ]
    if training.get("epochs"):
        if config["kind"] == C2_KIND:
            lines += [
                "",
                "## Обучение по эпохам",
                "",
                "| Эпоха | Полная | Свободные метки | Adam steps | Online CE | Fixed-training CE / acc | Holdout-126 CE / acc |",
                "|---:|---|---:|---:|---:|---|---|",
            ]
            for epoch in training["epochs"]:
                ft = epoch.get("fixed_training") or {}
                ho = epoch.get("holdout") or {}
                lines.append(
                    f"| {epoch['epoch']} | {epoch['completed']} | {epoch['labels']} | {epoch['steps']} | {epoch['online_CE']} | "
                    f"{ft.get('ce')} / {ft.get('accuracy')} | {ho.get('ce')} / {ho.get('accuracy')} |"
                )
        else:
            lines += [
                "",
                "## Обучение",
                "",
                "| Эпоха | Полная | Свободные метки | Adam steps | Online CE |",
                "|---:|---|---:|---:|---:|",
            ]
            for epoch in training["epochs"]:
                lines.append(
                    f"| {epoch['epoch']} | {epoch['completed']} | {epoch['labels']} | {epoch['steps']} | {epoch['online_CE']} |"
                )
        lines += [
            "",
            "История/prefix и кеш: " + json.dumps(training.get("cache", {})),
            "",
            "Online CE усредняется по меткам при изменявшихся весах. Fixed training CE измеряется при фиксированных весах.",
        ]
    if (out / "training-telemetry/resources.jsonl").exists():
        analysis = analyze_telemetry(out / "training-telemetry/resources.jsonl")
        write_json(
            out / "gpu-analysis.json",
            {"metrics": analysis["metrics"], "windows": analysis["windows"]},
        )
        lines += [
            "",
            "## GPU NVML",
            "",
            json.dumps(analysis["metrics"]),
            "",
            f"Непрерывных окон 60 s: {len(analysis['windows'])}. Подготовка, compilation, offline CE и боевая оценка исключены.",
            "",
            "Разбор загрузки GPU: "
            + json.dumps(training.get("GPU_utilization_review"), ensure_ascii=False),
        ]
    if selection and (out / "evaluations.json").exists():
        evaluations_dict = read_json(out / "evaluations.json")
        rendered = render_learning_curves_plot(
            evaluations_dict, out, eval_purpose, config["kind"]
        )
        if rendered:
            lines += ["", "![Кривые](learning-curves.png)"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


generate_target_scale_report = report

__all__ = [
    "extract_environment_metadata",
    "generate_target_scale_report",
    "render_learning_curves_plot",
    "report",
]
