"""Reporting and telemetry formatting for target scale BC experiments."""

from __future__ import annotations

import json
import platform
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from civ5rl.engine.runtime.telemetry import analyze_telemetry
from civ5rl.experiments._shared.markdown import Alignment, MarkdownReport, format_wdl
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

    points = [p for p in evaluations.values() if isinstance(p, dict) and p.get("purpose") == eval_purpose]
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


def _format_ce_accuracy(metrics: dict[str, Any]) -> str:
    return f"{metrics.get('ce')} / {metrics.get('accuracy')}"


def _append_evaluation_table(
    lines: MarkdownReport, evaluation_path: Path, purpose: str, *, is_c2: bool
) -> None:
    headers = [
        "Эпоха",
        "Adam steps",
        "Checkpoint",
        "W/D/L",
        "Win rate",
        "Fixed training CE / accuracy",
    ]
    alignments: list[Alignment] = ["right", "right", "left", "left", "right", "left"]
    if is_c2:
        headers[3] = "W/D/L (512)"
        headers[5] = "Fixed training CE / acc"
        headers = ["Роль", *headers, "Holdout-126 CE / acc"]
        alignments = ["left", *alignments, "left"]

    rows: list[list[object]] = []
    if evaluation_path.exists():
        seen_dirs = set()
        raw_points = [
            p
            for p in read_json(evaluation_path).values()
            if isinstance(p, dict) and p.get("purpose") == purpose
        ]
        raw_points.sort(key=lambda p: 1 if p.get("candidate_role") else 0, reverse=True)
        points = []
        for p in raw_points:
            directory = p.get("runs", {}).get("argmax", {}).get("cp", {}).get("directory")
            if directory and directory not in seen_dirs:
                seen_dirs.add(directory)
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
            row = [point.get("candidate_role", "")] if is_c2 else []
            row.extend(
                (
                    point.get("epoch", 0),
                    point.get("Adam_steps", 0),
                    point["checkpoint"],
                    format_wdl(summary),
                    f"{summary['win_rate']:.2%}",
                    _format_ce_accuracy(metric),
                )
            )
            if is_c2:
                row.append(_format_ce_accuracy(point.get("holdout") or {}))
            rows.append(row)
    lines.table(headers, alignments, rows)


def _append_epoch_table(lines: MarkdownReport, epochs: list[dict[str, Any]], *, is_c2: bool) -> None:
    title = "## Обучение по эпохам" if is_c2 else "## Обучение"
    headers = ["Эпоха", "Полная", "Свободные метки", "Adam steps", "Online CE"]
    alignments: list[Alignment] = ["right", "left", "right", "right", "right"]
    if is_c2:
        headers.extend(("Fixed-training CE / acc", "Holdout-126 CE / acc"))
        alignments.extend(("left", "left"))
    rows: list[list[object]] = []
    for epoch in epochs:
        row = [
            epoch["epoch"],
            epoch["completed"],
            epoch["labels"],
            epoch["steps"],
            epoch["online_CE"],
        ]
        if is_c2:
            row.extend(
                (
                    _format_ce_accuracy(epoch.get("fixed_training") or {}),
                    _format_ce_accuracy(epoch.get("holdout") or {}),
                )
            )
        rows.append(row)
    lines.extend(("", title, ""))
    lines.table(headers, alignments, rows)


def _write_gpu_analysis(out: Path) -> dict[str, Any] | None:
    telemetry_path = out / "training-telemetry/resources.jsonl"
    if not telemetry_path.exists():
        return None
    analysis: dict[str, Any] = analyze_telemetry(telemetry_path)
    write_json(
        out / "gpu-analysis.json",
        {"metrics": analysis["metrics"], "windows": analysis["windows"]},
    )
    return analysis


def report(config: dict[str, Any], out: Path) -> None:
    """Generate Markdown report and plots for target scale BC experiment."""
    out = Path(out)
    collection = read_json(out / "collection.json") if (out / "collection.json").exists() else {}
    training = read_json(out / "training.json") if (out / "training.json").exists() else {}
    selection = read_json(out / "selection.json") if (out / "selection.json").exists() else {}
    run_info = read_json(out / "run.json") if (out / "run.json").exists() else {}
    status_info = read_json(out / "status.json") if (out / "status.json").exists() else {}

    env_meta = extract_environment_metadata(run_info, config)
    gpu_name = env_meta["gpu"]
    seed = env_meta["seed"]

    is_c2 = config["kind"] == C2_KIND
    if is_c2:
        title = "# Архитектура C2: 8 hex-блоков и глобальное внимание к известным юнитам (`cp-imitation-target-c2`)"
    elif config["kind"] == MORE_DATA_KIND:
        title = "# Расширение корпуса и продолжение target BC (`cp-imitation-target-more-data`)"
    else:
        title = "# Имитация штатного CP на целевых армиях"
    lines = MarkdownReport(
        [
            title,
            "",
            f"Статус: {status_info}; GPU: {gpu_name}; seed {seed}.",
            "",
        ]
    )
    if is_c2:
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
    eval_purpose = "development_512" if is_c2 else "development"
    _append_evaluation_table(lines, out / "evaluations.json", eval_purpose, is_c2=is_c2)
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
            "Additional: " + json.dumps(read_json(out / "confirmation.json"), ensure_ascii=False),
        ]
    match_rows = collection.get("new_matches", collection.get("matches", []))
    if match_rows:
        commands: Counter[str] = Counter()
        units: Counter[str] = Counter()
        outcomes: Counter[str] = Counter()
        strata: Counter[str] = Counter()
        for row in match_rows:
            commands.update(row["commands"])
            units.update(row["unit_types"])
            outcomes["draw" if row["winner"] is None else f"side_{row['winner']}_win"] += 1
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
            "Свободные метки по типу действующего юнита (новый сбор): " + json.dumps(units),
            "",
            "Расстановки map/army/contact (новый сбор): " + json.dumps(strata),
        ]
    if training.get("epochs"):
        _append_epoch_table(lines, training["epochs"], is_c2=is_c2)
        lines += [
            "",
            "История/prefix и кеш: " + json.dumps(training.get("cache", {})),
            "",
            "Online CE усредняется по меткам при изменявшихся весах. Fixed training CE измеряется при фиксированных весах.",
        ]
    analysis = _write_gpu_analysis(out)
    if analysis is not None:
        lines += [
            "",
            "## GPU NVML",
            "",
            json.dumps(analysis["metrics"]),
            "",
            f"Непрерывных окон 60 s: {len(analysis['windows'])}. Подготовка, compilation, offline CE и боевая оценка исключены.",
            "",
            "Разбор загрузки GPU: " + json.dumps(training.get("GPU_utilization_review"), ensure_ascii=False),
        ]
    if selection and (out / "evaluations.json").exists():
        evaluations_dict = read_json(out / "evaluations.json")
        rendered = render_learning_curves_plot(evaluations_dict, out, eval_purpose, config["kind"])
        if rendered:
            lines += ["", "![Кривые](learning-curves.png)"]
    lines.write(out / "report.md")


generate_target_scale_report = report

__all__ = [
    "extract_environment_metadata",
    "generate_target_scale_report",
    "render_learning_curves_plot",
    "report",
]
