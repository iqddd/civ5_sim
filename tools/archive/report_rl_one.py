"""Render the one-decision diagnostic report and plot."""

import json
from pathlib import Path

from civ5rl.common import read_json


def report_one(directory, seed=17):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    run = directory / f"seed-{seed}"
    if not (run / "run-result.json").exists():
        return
    result = read_json(run / "run-result.json")
    metrics = (
        [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
        if (run / "metrics.jsonl").exists()
        else []
    )
    text = [
        "# D0-one-decision",
        "",
        "Отдельный опыт с горизонтом одна команда learner. Исходный D0 и его формальный статус не изменены.",
        "",
        "Начальные веса побитово совпадают с прежним initial.pt; Adam, RNG обучения и счётчики созданы заново. Награда +1 за естественную победу, 0 при завершении без уничтожения. Native-ошибки остаются ошибками.",
        "",
        f"Статус: **{result['status']}**, причина `{result.get('reason', 'pause')}`. Updates: {result['update']}; решений: {result['decisions']}; время обучения: {result['training_seconds']:.2f} с; post-update оценки: {result['evaluation_seconds']:.2f} с.",
        "",
        "Критерий: не менее 122 побед из 128 в двух последовательных оценках после update. Максимум 8 updates и 1800 секунд обучения. D1 и test-наборы не запускались.",
        "",
        "| Оценка | W/D/L | Победы | P выстрела | Gate |",
        "|---|---:|---:|---:|---|",
    ]
    points = []
    for path in sorted((run / "evaluation").glob("*/summary.json")):
        s = read_json(path)
        text.append(
            f"| {path.parent.name} | {s['wins']}/{s['draws']}/{s['losses']} | {s['win_rate']:.3%} | {s['first_winning_attack_probability']:.3%} | {s.get('gate', {}).get('passed', 'baseline')} |"
        )
        if path.parent.name.startswith("update-"):
            points.append((int(path.parent.name.split("-")[1]), s))
    text += [
        "",
        "| Update | Решений/эпизодов | PPO epochs | Adam steps | KL | Норма градиента max | Изменение весов L2 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for m in metrics:
        p = m["ppo"]
        text.append(
            f"| {m['update']} | {m['rollout_decisions']}/{m['episodes']} | {p['epochs']} | {p['optimizer_steps']} | {p['kl']:.5f} | {p['gradient_norm_max']:.4f} | {m['weight_delta_l2']:.4f} |"
        )
    fourth = next((s for n, s in points if n == 4), None)
    text += [
        "",
        f"**Update 4:** {fourth['wins']}/128 побед ({fourth['win_rate']:.3%})."
        if fourth
        else "**Update 4:** не достигнут; опыт завершился раньше.",
        "",
        "Последовательности имеют длину 1; minibatch 8 последовательностей содержит 8 решений. При 8192 решениях одна полная эпоха даёт 1024 optimizer steps. Это не равный объём градиентной работы с четырьмя updates прежнего D0; LR, KL-stop и размер minibatch не подбирались.",
        "",
        f"Raw outcomes, вероятность/частота пропуска атак, loss, telemetry и реплеи находятся в seed-{seed}. Telemetry — CPU/RAM и NVML VRAM/Core/Memory Controller/Power % default TDP. Данные подтверждают только одношаговую диагностику; перенос и полезность памяти не проверялись.",
    ]
    resources = run / "resources-summary.json"
    if resources.exists():
        values = read_json(resources)["metrics"]
        text += [
            "",
            "| Ресурс всего запуска, включая оценки/прогрев | Среднее | Максимум |",
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
                text += [
                    f"| {key} {'(GiB)' if scale > 1 else '(%)'} | {values[key]['mean'] / scale:.2f} | {values[key]['max'] / scale:.2f} |"
                ]
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    initial = read_json(run / "evaluation/initial-D0/summary.json")
    x = [0] + [n for n, _ in points]
    wins = [initial["win_rate"]] + [s["win_rate"] for _, s in points]
    probs = [initial["first_winning_attack_probability"]] + [
        s["first_winning_attack_probability"] for _, s in points
    ]
    ax.plot(x, wins, "o-", label="Dev win rate")
    ax.plot(x, probs, "o-", label="First shot probability")
    ax.axhline(0.95, color="gray", linestyle="--", label="Win threshold")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("PPO update")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(directory / "progress.png", dpi=140)
    plt.close(fig)
    text += ["", "![Прогресс](progress.png)", ""]
    (directory / "report.md").write_text("\n".join(text), encoding="utf-8")
