"""Render all D0-M outcomes, optimizer diagnostics and public firing sequences."""

import json
from pathlib import Path

from civ5rl.common import read_json


def fmt(value):
    return "null" if value is None else f"{value:.2%}"


def report(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    run = directory / "seed-17"
    if not (run / "run-result.json").exists():
        if (directory / "suite-result.json").exists():
            result = read_json(directory / "suite-result.json")
            (directory / "report.md").write_text(
                f"# D0-M\n\nСтатус: {result['status']}. Причина: {result.get('reason')}.\n",
                encoding="utf-8",
            )
        return
    result = read_json(run / "run-result.json")
    text = [
        "# D0-M — победа несколькими выстрелами",
        "",
        f"Статус: **{result['status']}**, причина `{result.get('reason', 'pause')}`.",
        "",
        f"Обновлений: {result['update']}; новых решений: {result['decisions']}; прошлые затраты D0: 49152 решения. Обучение: {result['training_seconds']:.2f} с; post-update evaluation: {result['evaluation_seconds']:.2f} с; baselines: {result.get('baseline_evaluation_seconds', 0):.2f} с; прогрев: {result['warmup_seconds']:.2f} с; паузы: {result['pause_seconds']:.2f} с.",
        "",
        "Один лучник 100 HP против неподвижного воина 40 HP, обычные действия и лечение, четыре раунда. Только терминальная награда +1/−1/0. Policy/value перенесены из D0-one-decision update 6, Adam и расписания начаты заново. Критерий обучения: ≥122/128 побед в двух последовательных post-update оценках.",
        "",
        "| Оценка | W/D/L | Победы | Выстрелы mean | Решения mean | Урон / лечение | P ranged / выбор |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    names = ["random", "transfer", "reference", "passive"] + sorted(
        p.name for p in (run / "evaluation").glob("update-*")
    )
    points = []
    summaries = []
    for name in names:
        path = run / "evaluation" / name / "summary.json"
        if not path.exists():
            continue
        s = read_json(path)
        summaries.append((name, s))
        shot_mean = sum(int(k) * v for k, v in s["shots"].items()) / s["games"]
        a = s["actions"]["ranged"]
        text.append(
            f"| {name} | {s['wins']}/{s['draws']}/{s['losses']} | {s['win_rate']:.2%} | {shot_mean:.2f} | {s['decisions']['mean']:.2f} | {s['damage_dealt']['mean']:.2f} / {s['enemy_healing']['mean']:.2f} | {fmt(a['probability_when_available'])} / {fmt(a['chosen_when_available'])} |"
        )
        if name == "transfer" or name.startswith("update-"):
            points.append((0 if name == "transfer" else int(name.split("-")[1]), s))
    text += [
        "",
        "| Оценка | Возможность / первые выстрелы | Немедленное использование / возможности | P использования | Причины отсутствия |",
        "|---|---:|---:|---:|---|",
    ]
    for name, s in summaries:
        a, b = s["contact_opportunity_reached"], s["contact_immediate_use"]
        text.append(
            f"| {name} | {a['numerator']}/{a['denominator']} ({fmt(a['value'])}) | {b['numerator']}/{b['denominator']} ({fmt(b['value'])}) | {fmt(s['contact_probability']['value'])} | {json.dumps(s['contact_absence_reasons'], ensure_ascii=False)} |"
        )
    text += [
        "",
        "Огневой контакт проверяется на первом решении следующего своего хода: те же живые юниты, неизменные позиции, цель видима во всех промежуточных наблюдениях, ranged легален. Расход атаки в предыдущем ходу игнорируется; отсутствие возможности не считается отказом стрелять. Видимость между наблюдениями не утверждается.",
        "",
        "| Update | Rollout / overflow | Эпох начато / завершено | Adam / gradient tokens | KL final / max | Grad max / Δweights | Policy / value loss | Return / value / raw adv mean | Collection / PPO / KL, с |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = (
        [json.loads(line) for line in (run / "metrics.jsonl").read_text().splitlines()]
        if (run / "metrics.jsonl").exists()
        else []
    )
    for m in rows:
        p = m["ppo"]
        text.append(
            f"| {m['update']} | {m['rollout_decisions']} / {m['overflow']} | {p['epochs_started']} / {p['epochs_completed']} | {p['optimizer_steps']} / {p['objective_valid_tokens']} | {p['kl']:.6f} / {p['max_kl']:.6f} | {p['gradient_norm_max']:.4f} / {m['weight_delta_l2']:.4f} | {p['policy_loss']:.5f} / {p['value_loss']:.5f} | {p['return_mean']:.4f} / {p['value_mean']:.4f} / {p['raw_advantage_mean']:.4f} | {m['timings']['collection_seconds']:.2f} / {p['ppo_seconds']:.2f} / {p['kl_seconds']:.2f} |"
        )
    text += [
        "",
        f"Фактические шаги Adam: {sum(m['ppo']['optimizer_steps'] for m in rows)}. Предъявления решений для градиента: {sum(m['ppo']['objective_valid_tokens'] for m in rows)}. Полный minibatch — 1024 valid tokens, остаток нормирован по фактическому размеру; KL-stop после каждого шага, превышающий шаг сохранён.",
        "",
        "Полные распределения return/value/raw и normalized advantage, explained variance (null при нулевой дисперсии return), условные числители/знаменатели, HP/лечение и последовательности находятся в summaries/outcomes и сохранённых episodes.pt.gz. Готовые rollout с хешами сохранены до PPO. Отсутствующие категории реплеев обозначены null.",
        "",
        "Прирост над random и transfer — парный percentile bootstrap по 64 расстановкам, обе стороны связаны, 10000 samples, seed 20261103. Это описательные интервалы; повторные оценки на одном dev не независимы.",
    ]
    for name, s in summaries:
        if name.startswith("update-"):
            text += [
                "",
                f"{name}: random {s.get('comparison_to_random')}; transfer {s.get('comparison_to_transfer')}.",
            ]
    if (run / "initial-rollout/summary.json").exists():
        initial = read_json(run / "initial-rollout/summary.json")
        text += [
            "",
            f"Первый training rollout: {initial['wins']}/{initial['draws']}/{initial['losses']}, {initial['actions']['decisions']} решений. Return: {initial['token_return']}; old value: {initial['old_value']}; raw advantage: {initial['raw_advantage']}; normalized advantage: {initial['normalized_advantage']}. Adam не запускался, веса и optimizer остались исходными.",
        ]
    resources = run / "resources-summary.json"
    if resources.exists():
        text += [
            "",
            "| Ресурс (включая оценки и прогрев) | Среднее | Максимум |",
            "|---|---:|---:|",
        ]
        for key, values in read_json(resources)["metrics"].items():
            if key in (
                "cpu_tree_capacity_percent",
                "ram_tree_rss_bytes",
                "vram_used_bytes",
                "gpu_core_load_percent",
                "gpu_memory_controller_load_percent",
                "gpu_power_percent_tdp",
            ):
                scale = 2**30 if key.endswith("bytes") else 1
                text.append(
                    f"| {key} {'GiB' if scale > 1 else '%'} | {values['mean'] / scale:.2f} | {values['max'] / scale:.2f} |"
                )
    if points:
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout="constrained")
        for ax, field, label in (
            (axes[0], "win_rate", "Dev win rate"),
            (axes[1], "contact_immediate_use", "Immediate repeated shot | opportunity"),
        ):
            ax.plot(
                [n for n, _ in points],
                [
                    s[field] if field == "win_rate" else s[field]["value"]
                    for _, s in points
                ],
                "o-",
                label=label,
            )
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.2)
            ax.legend()
        axes[0].axhline(122 / 128, linestyle="--", color="gray")
        axes[1].set_xlabel("D0-M update")
        fig.savefig(directory / "progress.png", dpi=140)
        plt.close(fig)
        text += ["", "![Прогресс](progress.png)"]
    text += [
        "",
        "Вывод относится только к D0-M. Старые D0 и D1 сохранены. Манёвр, полезность GRU, critic ablation, seed 42 и закрытые test не проверялись. Автоматического следующего этапа и продления бюджета нет.",
        "",
    ]
    (directory / "report.md").write_text("\n".join(text), encoding="utf-8")
