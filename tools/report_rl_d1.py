"""Render bounded D1 transfer results without interpreting first shots as wins."""

import json
from pathlib import Path

from civ5rl.common import read_json


def report_d1(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    run = directory / "seed-17"
    if not (run / "run-result.json").exists():
        return
    result = read_json(run / "run-result.json")
    metrics = (
        [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        if (run / "metrics.jsonl").exists()
        else []
    )
    text = [
        "# D1 — перенос из одношагового D0, seed 17",
        "",
        f"Статус: **{result['status']}**, причина `{result.get('reason', 'pause')}`.",
        "",
        f"D1: {result['update']} updates, {result['decisions']} новых решений; обучение {result['training_seconds']:.2f} с, post-update evaluation {result['evaluation_seconds']:.2f} с. Базовые оценки {result.get('baseline_evaluation_seconds', 0):.2f} с; паузы {result.get('pause_seconds', 0):.2f} с. Предыдущие затраты D0: {result.get('previous_decisions', 49152)} решений.",
        "",
        "Инициализация — все веса policy/value update 6, новый Adam и RNG, нулевая память эпизодов. LR/entropy отсчитываются от нуля решений D1. Критерий: ≥80% побед, ≥20 п.п. над случайной сетью и нижняя граница парного 95% bootstrap >0 на двух последовательных post-update оценках.",
        "",
        "| Оценка | W/D/L | Победы | Прирост над random | 95% CI | P атаки / выбор при доступности | Урон / потери / раунды | Gate |",
        "|---|---:|---:|---:|---|---:|---:|---|",
    ]
    points = []
    names = ["random-D1", "transfer-D1"] + sorted(
        p.name for p in (run / "evaluation").glob("update-*")
    )
    for name in names:
        path = run / "evaluation" / name / "summary.json"
        if not path.exists():
            continue
        s = read_json(path)
        a = s["actions"]["attack"]
        g = s.get("gate", s.get("comparison_to_random", {}))
        delta = f"{g['win_rate_delta']:+.2%}" if "win_rate_delta" in g else "—"
        ci = ", ".join(f"{v:+.2%}" for v in g.get("paired_bootstrap_ci95", [])) or "—"
        text.append(
            f"| {name} | {s['wins']}/{s['draws']}/{s['losses']} | {s['win_rate']:.2%} | {delta} | {ci} | {a['probability_when_available']:.2%} / {a['chosen_when_available']:.2%} | {s['damage_dealt_mean']:.2f} / {s['units_lost_mean']:.2f} / {s['rounds_mean']:.2f} | {s.get('gate', {}).get('passed', 'baseline')} |"
        )
        if name == "transfer-D1" or name.startswith("update-"):
            points.append((0 if name == "transfer-D1" else int(name.split("-")[1]), s))
    text += [
        "",
        "| Update | Rollout / overflow | Эпох начато / завершено | Adam steps / tokens | KL final / max | Grad max / Δweights L2 | Loss policy / value | Collection / PPO / KL, с | Stop |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for m in metrics:
        p = m["ppo"]
        text.append(
            f"| {m['update']} | {m['rollout_decisions']} / {m['overflow']} | {p['epochs_started']} / {p['epochs_completed']} | {p['optimizer_steps']} / {p['objective_valid_tokens']} | {p['kl']:.6f} / {p['max_kl']:.6f} | {p['gradient_norm_max']:.4f} / {m['weight_delta_l2']:.4f} | {p['policy_loss']:.5f} / {p['value_loss']:.5f} | {sum(v for k, v in m['timings'].items() if k == 'collection_seconds'):.2f} / {p['ppo_seconds']:.2f} / {p['kl_seconds']:.2f} | {p['stop_reason']} |"
        )
    text += [
        "",
        "Полный шаг Adam — 1024 valid tokens, последний неполный шаг нормирован по фактическому размеру. KL проверяется по всему rollout после каждого Adam.step(); превышающий шаг сохраняется. Это ранняя остановка, не жёсткая гарантия KL ≤0,015.",
        "",
        "Подробные частоты/вероятности melee и ranged, пропуски атак, end_turn при доступной атаке/движении, LR/entropy, нормы градиентов до clipping и каждый шаг Adam записаны в metrics.jsonl и evaluation/*/summary.json. Первый выстрел D1 не считается гарантированной победой.",
    ]
    resources = run / "resources-summary.json"
    if resources.exists():
        values = read_json(resources)["metrics"]
        text += [
            "",
            "| Ресурс, включая evaluation/прогрев | Среднее | Максимум |",
            "|---|---:|---:|",
        ]
        for key in (
            "cpu_tree_capacity_percent",
            "ram_tree_rss_bytes",
            "vram_used_bytes",
            "gpu_core_load_percent",
            "gpu_memory_controller_load_percent",
            "gpu_power_percent_tdp",
        ):
            if key in values:
                scale = 2**30 if key.endswith("bytes") else 1
                text.append(
                    f"| {key} {'GiB' if scale > 1 else '%'} | {values[key]['mean'] / scale:.2f} | {values[key]['max'] / scale:.2f} |"
                )
    if points:
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout="constrained")
        axes[0].plot(
            [n for n, _ in points],
            [s["win_rate"] for _, s in points],
            "o-",
            label="D1 win rate",
        )
        for label, style in (("random-D1", ":"), ("transfer-D1", "--")):
            path = run / "evaluation" / label / "summary.json"
            if path.exists():
                axes[0].axhline(
                    read_json(path)["win_rate"], linestyle=style, label=label
                )
        axes[0].axhline(0.8, color="gray", linestyle="--", label="80%")
        axes[1].plot(
            [n for n, _ in points],
            [s["actions"]["attack"]["probability_when_available"] for _, s in points],
            "o-",
            label="Attack probability when available",
        )
        for ax in axes:
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.2)
            ax.legend()
        axes[1].set_xlabel("D1 update")
        fig.savefig(directory / "progress.png", dpi=140)
        plt.close(fig)
        text += ["", "![Прогресс](progress.png)"]
    text += [
        "",
        "Вывод ограничен диагностическими боями D1 против attack-or-end-turn. Обычные карты, преимущество GRU и победа над greedy не проверялись. Исходные D0 сохранены; seed 42 и закрытые test не запускались.",
        "",
    ]
    (directory / "report.md").write_text("\n".join(text), encoding="utf-8")
