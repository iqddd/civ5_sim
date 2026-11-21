"""Saved PPO scale-up report."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from civ5rl.provenance.common import read_json
from civ5rl.provenance.jsonl import read_jsonl
from civ5rl.reporting.markdown import MarkdownReport, format_wdl
from civ5rl.reporting.plot import pyplot
from civ5rl.reporting.telemetry import write_ppo_gpu_analysis


def _report_update_rows(rows: list[dict[str, Any]]) -> Iterator[tuple[object, ...]]:
    for row in rows:
        ppo = row["ppo"]
        yield (
            row["update"],
            row["decisions"],
            format_wdl(row),
            f"{ppo['attempted_optimizer_steps']}/{ppo['optimizer_steps']}/{ppo['rejected_optimizer_steps']}",
            f"{ppo.get('max_attempted_kl')}/{ppo.get('final_accepted_kl')}",
            f"{ppo.get('policy_loss')}/{ppo.get('value_loss')}",
            f"{ppo.get('critic_mse')}/{ppo.get('critic_explained_variance')}",
            f"{ppo.get('gradient_norm_mean')}/{ppo.get('gradient_norm_max')}",
        )


def _render_report(
    out: Path,
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    final: dict[str, Any] | None,
    parent: str,
    origin: int,
) -> MarkdownReport:
    lines = MarkdownReport(
        [
            "# PPO scale-up на RTX 4090",
            "",
            f"Статус: {read_json(out / 'status.json')['status']}; причина: {state['stop_reason']}.",
            (
                f"Новые решения: {state['decisions']}; rollouts: {state['updates']}; attempted/accepted/rejected: "
                f"{state['attempted_steps']}/{state['optimizer_steps']}/{state['rejected_steps']}; "
                f"Adam counter: {origin + state['optimizer_steps']}."
            ),
            f"Active: {state['active_seconds']:.2f} с; development evaluation: {state['evaluation_seconds']:.2f} с; warmup: {state['preparation_seconds']:.2f} с.",
            "",
            f"Родитель — {parent} с обученным critic и Adam {origin}. Основная пара оценивается одним Linux evaluator.",
            "",
        ]
    )
    lines.table(
        ("Новые решения", "Rollout", "Режим", "Соперник", "W/D/L", "Win rate"),
        ("right", "right", "left", "left", "left", "right"),
        (
            (
                point["decisions"],
                number,
                mode,
                enemy,
                format_wdl(result["summary"]),
                f"{result['summary']['win_rate']:.2%}",
            )
            for number, point in state["evaluated"].items()
            for mode, enemies in point["runs"].items()
            for enemy, result in enemies.items()
        ),
    )
    lines += ["", f"Выбранный checkpoint: {state['best']}."]
    if final:
        lines += [
            "",
            "## Дополнительный validation, 256 argmax-матчей против CP",
            "",
            f"Linux baseline {parent}: {final['baseline']['runs']['argmax']['cp']['summary']}",
            f"Кандидат: {final['candidate']['runs']['argmax']['cp']['summary']}",
            f"Парный прирост/95% CI: {final['comparison']}",
            f"Срезы: {final['slices']}",
            "Продолжение не прошло development-отбор; одинаковые веса не дают проверки эквивалентности."
            if final["baseline_selected"]
            else "Bootstrap: 10 000 выборок, seed 20261103, кластер — parent-world. Дополнительная оценка не участвовала в выборе.",
            f"Additional evaluation: {final['baseline']['seconds'] + (0 if final['baseline_selected'] else final['candidate']['seconds']):.2f} с.",
        ]
    if rows:
        lines += [
            "",
            "## Стоимость обучения",
            "",
            (
                f"Collection: {sum(r['collection']['collection_seconds'] for r in rows):.2f} с; "
                f"PPO including KL: {sum(r['ppo'].get('ppo_seconds', 0) for r in rows):.2f} с; "
                f"KL: {sum(r['ppo'].get('kl_seconds', 0) for r in rows):.2f} с; "
                f"runtime compilation: {sum(r['ppo'].get('runtime_compilation_seconds', 0) for r in rows):.2f} с (входит в бюджет)."
            ),
            f"Последний rollout: {rows[-1]['decisions']} решений; превышения: {rows[-1]['decision_overshoot']} решений, {rows[-1]['active_time_overshoot_seconds']:.3f} с.",
            "",
        ]
        lines.table(
            (
                "Rollout",
                "Decisions",
                "W/D/L",
                "Attempt/accept/reject",
                "Full KL max/final",
                "Policy/value loss",
                "Critic MSE/EV",
                "Gradient mean/max",
            ),
            ("right", "right", "left", "left", "left", "left", "left", "left"),
            _report_update_rows(rows),
        )
    if (out / "coverage.json").exists():
        coverage = read_json(out / "coverage.json")
        lines += [
            "",
            (
                f"Покрытие: {coverage['episodes']} эпизодов; {len(coverage['worlds'])} миров, {len(coverage['crops'])} различных карт; "
                f"уникальных физических стартов {coverage['physical_unique']}, повторов {coverage['physical_repeats']}. Историческое физическое пересечение не измерялось."
            ),
        ]
    gpu = write_ppo_gpu_analysis(out)
    if gpu is not None:
        lines += [
            "",
            f"NVML (PPO, без CPU collection/evaluation и compilation): {gpu['metrics']}; 60s windows: {len(gpu['windows'])}.",
        ]
    lines += [
        "",
    ]
    return lines


def _plot_learning_curves(out: Path, points: list[dict[str, Any]]) -> None:
    plt = pyplot()

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for mode in ("argmax", "sample"):
        ax.plot(
            [point["decisions"] for point in points],
            [point["runs"][mode]["cp"]["summary"]["win_rate"] for point in points],
            marker="o",
            label=mode + " vs CP",
        )
    ax.set_xlabel("New learner decisions")
    ax.set_ylabel("Development win rate")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(out / "learning-curves.png", dpi=150)
    plt.close(fig)


def report(config: dict[str, Any], out: Path) -> None:
    state = read_json(out / "state.json")
    origin = config["experiment"]["parent_optimizer_steps"]
    parent = config["experiment"].get("parent_label", Path(config["experiment"]["parent"]).stem)
    rows = read_jsonl(out / "updates.jsonl", missing_ok=True, skip_blank=False, encoding=None)
    final_path = out / "confirmation.json"
    final = read_json(final_path) if final_path.exists() else None
    _render_report(out, state, rows, final, parent, origin).write(out / "report.md")
    if state["evaluated"]:
        _plot_learning_curves(out, list(state["evaluated"].values()))
