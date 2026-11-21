"""Saved PPO-after-imitation report."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from civ5rl.protocols.ppo_after_imitation import controlled
from civ5rl.provenance.common import read_json
from civ5rl.provenance.jsonl import read_jsonl
from civ5rl.reporting.markdown import MarkdownReport, format_wdl
from civ5rl.reporting.plot import pyplot
from civ5rl.reporting.ppo_controlled_checks import extend_report
from civ5rl.reporting.telemetry import write_ppo_gpu_analysis


def _render_report(
    config: dict[str, Any],
    out: Path,
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    final: dict[str, Any] | None,
) -> MarkdownReport:
    lines = MarkdownReport(
        [
            "# PPO с контролем принятого изменения политики"
            if controlled(config)
            else "# PPO после имитации CP",
            "",
            f"Статус: {read_json(out / 'status.json')['status']}; причина остановки: {state['stop_reason']}.",
            f"Решения: {state['decisions']}; rollouts: {state['updates']}; Adam steps: {state['optimizer_steps']}; активное время: {state['active_seconds']:.2f} с.",
            f"Evaluation: {state['evaluation_seconds']:.2f} с; подготовка: {state['preparation_seconds']:.2f} с; сохранение checkpoints: {state['save_seconds']:.2f} с.",
            "",
            "Основной режим — argmax; sampling и greedy — отдельные контрольные показатели.",
            "",
        ]
    )
    lines.table(
        (
            "Update",
            "Решения",
            "Adam",
            "Active, с",
            "Train W/D/L",
            "Win rate",
            "Free/forced",
            "KL free / all",
            "Stop",
        ),
        ("left", "right", "right", "right", "right", "right", "right", "left", "left"),
        (
            (
                update["update"],
                update["total_decisions"],
                update["total_Adam_steps"],
                f"{update['active_seconds']:.1f}",
                format_wdl(update),
                f"{update['win_rate']:.1%}",
                f"{update['free_decisions']}/{update['forced_decisions']}",
                f"{update['ppo'].get('sampled_kl_free')} / {update['ppo'].get('sampled_kl')}",
                update["ppo"]["stop_reason"],
            )
            for update in rows
        ),
    )
    lines += ["", "## Development", ""]
    lines.table(
        ("Update", "Режим", "Соперник", "W/D/L", "Win rate"),
        ("left", "left", "left", "right", "right"),
        (
            (
                number,
                mode,
                enemy,
                format_wdl(record["summary"]),
                f"{record['summary']['win_rate']:.1%}",
            )
            for number, evaluation in state["evaluated"].items()
            for mode, enemies in evaluation["runs"].items()
            for enemy, record in enemies.items()
        ),
    )
    lines += ["", f"Выбранный кандидат: {state['best']}."]
    if final:
        lines += [
            "",
            "## Дополнительные 256 матчей argmax против CP",
            "",
            f"Baseline: {final['baseline']['runs']['argmax']['cp']['summary']}",
            f"Кандидат: {final['candidate']['runs']['argmax']['cp']['summary']}",
            f"Парный прирост и 95% CI: {final['comparison']}",
            f"Подтверждённый прирост: {final['confirmed_improvement']}; ориентир +10 п.п.: {final['practical_10pp']}.",
        ]
    if rows:
        update = rows[-1]
        lines += [
            "",
            f"Последний rollout: {update['decisions']} решений; превышение общего лимита: {update['decision_overshoot']}; превышение времени: {update['active_time_overshoot_seconds']:.3f} с.",
        ]
    gpu = write_ppo_gpu_analysis(out)
    if gpu is not None:
        lines += [
            "",
            "## NVML",
            "",
            f"PPO samples: {gpu['metrics']}; непрерывных 60-секундных окон: {len(gpu['windows'])}.",
            "CPU collection и evaluation исключены. Компиляция внутри update учитывается в активном бюджете; начальная подготовка/прогрев отделены.",
        ]
    lines += [
        "",
        "## Границы вывода",
        "",
        "Дополнительная проверка повторно использует 256 validation-матчей прежнего PPO; это открытый validation, не test."
        if controlled(config)
        else "Дополнительная проверка использует новые расстановки знакомых validation-worlds. Test закрыт; 75% — цель проекта, не автоматическое подтверждение обобщения.",
    ]
    if controlled(config):
        lines += extend_report(config, out, state, rows, final)
    return lines


def _plot_learning_curves(out: Path, rows: list[dict[str, Any]]) -> None:
    plt = pyplot()

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(
        [r["total_decisions"] for r in rows],
        [r["win_rate"] for r in rows],
        label="Train P(win)",
    )
    axes[1].plot(
        [r["total_decisions"] for r in rows],
        [r["ppo"].get("sampled_kl_free") or 0 for r in rows],
        label="Post-update sampled KL (free)",
    )
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
        ax.set_xlabel("Learner decisions")
    fig.savefig(out / "learning-curves.png", dpi=150)
    plt.close(fig)


def report(config: dict[str, Any], out: Path) -> dict[str, Any]:
    state: dict[str, Any] = read_json(out / "state.json")
    rows = read_jsonl(out / "updates.jsonl", missing_ok=True, skip_blank=False, encoding=None)
    final_path = out / "confirmation.json"
    final = read_json(final_path) if final_path.exists() else None
    _render_report(config, out, state, rows, final).write(out / "report.md")
    if rows:
        _plot_learning_curves(out, rows)
    return state
