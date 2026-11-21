"""Saved target-scale PPO report."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from civ5rl.provenance.common import read_json
from civ5rl.provenance.jsonl import read_jsonl
from civ5rl.reporting.markdown import MarkdownReport, format_wdl


def _report_update_rows(rows: list[dict[str, Any]]) -> Iterator[tuple[object, ...]]:
    for row in rows:
        ppo = row["ppo"]
        yield (
            row["update"],
            format_wdl(row),
            ppo.get("final_accepted_kl"),
            f"{ppo.get('critic_mse')}/{ppo.get('critic_explained_variance')}",
            ppo.get("gradient_norm_mean"),
            ppo.get("epochs_completed"),
            f"{ppo['optimizer_steps']}/{ppo['rejected_optimizer_steps']}",
            f"{row['collection']['collection_seconds']:.2f}/{ppo.get('ppo_seconds', 0):.2f}",
        )


def _report_evaluation_rows(
    state: dict[str, Any],
) -> Iterator[tuple[object, ...]]:
    for point in state["evaluated"].values():
        summary = point["runs"]["argmax"]["cp"]["summary"]
        yield (
            point["decisions"],
            point["checkpoint"],
            format_wdl(summary),
            f"{summary['win_rate']:.2%}",
        )


def _render_report(config: dict[str, Any], out: Path, state: dict[str, Any]) -> MarkdownReport:
    lines = MarkdownReport(
        [
            "# PPO target scale: локальный перенос на армии 5–13",
            "",
            f"Статус: {read_json(out / 'status.json')['status']}; причина: {state['stop_reason']}.",
            (
                f"Новые решения {state['decisions']}; rollouts {state['updates']}; active {state['active_seconds']:.2f} s; "
                f"accepted/rejected {state['optimizer_steps']}/{state['rejected_steps']}; Adam {state['origin_Adam_counter'] + state['optimizer_steps']}."
            ),
            "",
            f"Параллелизм: {config['collection']['num_envs']} сред / {config['collection']['cpu_workers']} workers; microbatch {config['ppo']['sequences_per_microbatch']}; CPU rollout.",
            "",
        ]
    )
    lines.table(
        ("Новые решения", "Checkpoint", "W/D/L CP argmax", "Win rate"),
        ("right", "left", "left", "right"),
        _report_evaluation_rows(state),
    )
    lines += ["", f"Выбран: {state['best']}."]
    for name in (
        "confirmation.json",
        "small-control.json",
        "evaluation-cost.json",
        "memory-calibration.json",
        "preparation.json",
    ):
        if (out / name).exists():
            lines += [
                "",
                f"## {name}",
                "",
                json.dumps(read_json(out / name), ensure_ascii=False),
            ]
    updates_path = out / "updates.jsonl"
    if updates_path.exists():
        rows = read_jsonl(updates_path, skip_blank=False, encoding=None)
        lines += ["", "## Оптимизация", ""]
        lines.table(
            (
                "Rollout",
                "W/D/L",
                "KL final",
                "Critic MSE/EV",
                "Gradient",
                "Epochs",
                "Adam accept/reject",
                "Collection/PPO s",
            ),
            ("right", "left", "right", "left", "right", "right", "left", "left"),
            _report_update_rows(rows),
        )
    lines += [
        "",
    ]
    return lines


def report(config: dict[str, Any], out: Path) -> None:
    _render_report(config, out, read_json(out / "state.json")).write(out / "report.md")


def write_invalid_dataset_report(out: Path, error: Exception) -> None:
    (out / "report.md").write_text(
        "# PPO target scale: подготовка остановлена\n\n"
        + str(error)
        + "\n\nОсновное обучение и оценочные бои не запускались. "
        "Карта, состав и категория не заменены. Исчерпание попыток генератора не доказывает "
        "невозможность расстановки другим алгоритмом. Диагностика: invalid-dataset.json.\n",
        encoding="utf-8",
    )
