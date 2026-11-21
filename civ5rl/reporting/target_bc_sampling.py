"""Saved target BC sampling reports."""

from __future__ import annotations

from civ5rl.provenance.common import write_json
from civ5rl.reporting.markdown import MarkdownReport, format_wdl


def write_incomplete_report(out, setting, rows):
    write_json(
        out / "summary.json",
        {
            "status": "incomplete",
            "mode": setting["label"],
            "completed": len(rows),
        },
    )
    (out / "report.md").write_text(
        f"# Неполная оценка\n\n{setting['label']}: {len(rows)}/128; итоговое сравнение не готово.\n",
        encoding="utf-8",
    )


def _sampling_behavior_cells(label, behavior):
    def formatted_pair(field):
        summary = behavior[field]
        return "—" if summary["mean"] is None else f"{summary['mean']:.4f}/{summary['median']:.4f}"

    return (
        label,
        behavior["free_decisions"],
        formatted_pair("p_max"),
        f"{behavior['argmax_selected']['mean']:.2%}"
        if behavior["argmax_selected"]["mean"] is not None
        else "—",
        formatted_pair("normalized_entropy"),
        formatted_pair("legal_actions"),
    )


def render_sampling_report(out, result, *, workers=4):
    modes = result["modes"]
    comparison = result["T05_minus_T1"]
    lines = MarkdownReport(
        [
            "# Target BC: режим выбора действий",
            "",
            f"CPU, {workers} процессов, один поток Torch на процесс. Stateful CP, 128 боёв Development на режим.",
            "",
        ]
    )
    lines.table(
        ("Режим", "W/D/L", "Победы", "Round mean/median", "Ничьи по лимиту", "Wall, с"),
        ("---", "---", "---:", "---", "---:", "---:"),
        (
            (
                label,
                format_wdl(row),
                f"{row['win_rate']:.2%}",
                f"{row['mean_rounds']:.2f}/{row['median_rounds']:.1f}",
                f"{row['turn_limit_draw_rate']:.2%}",
                f"{row['wall_seconds']:.1f}" if row["wall_seconds"] is not None else "—",
            )
            for label, row in modes.items()
        ),
    )
    lines += [
        "",
        f"T=0.5 − T=1: {comparison['delta']:+.2%}; 95% CI [{comparison['CI95'][0]:+.2%}, {comparison['CI95'][1]:+.2%}]. Bootstrap {comparison['bootstrap_samples']}, seed {comparison['bootstrap_seed']}, {result['clusters']} миров × {result['cases_per_world']} матчей.",
        "",
    ]
    lines.table(
        (
            "Режим",
            "Free decisions",
            "p_max mean/median",
            "Argmax share",
            "Entropy/log(N) mean/median",
            "Legal N mean/median",
        ),
        ("---", "---:", "---", "---:", "---", "---"),
        (_sampling_behavior_cells(label, row["behavior"]) for label, row in modes.items()),
    )
    lines += [
        "",
        "Forced исключены из метрик уверенности. Для argmax распределение метрик — T=1. Посещённые состояния расходятся; эти метрики не являются парной оценкой качества действий.",
        "",
        f"T=0.5 ≥13 побед: {result['T05_at_least_13_wins']}; положительная нижняя граница CI: {result['positive_lower_CI']}.",
        "Ориентир 13 побед — минимальный сигнал, не гарантия полезности PPO. Выбор сделан на development; дополнительного подтверждения обобщения нет. Новый PPO, температуры и seeds автоматически не запускаются.",
        "",
        "Полные срезы: [summary.json](summary.json). Парные случаи и seeds: [cases.json](cases.json). Условия и fingerprint: [run.json](run.json).",
    ]
    lines.write(out / "report.md")
