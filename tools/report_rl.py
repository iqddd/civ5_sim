"""Produce a factual RL run report and resource/training plots."""

import argparse
import json
from pathlib import Path


def lines(path):
    return (
        [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if path.exists()
        else []
    )


def report(directory):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    resources = lines(directory / "resources.jsonl")
    metrics = lines(directory / "metrics.jsonl")
    if resources:
        fig, axes = plt.subplots(
            4, 1, figsize=(12, 10), sharex=True, constrained_layout=True
        )
        plots = [
            (
                "CPU (%)",
                [
                    ("cpu_system_percent", "system"),
                    ("cpu_tree_capacity_percent", "training tree / CPU capacity"),
                ],
            ),
            (
                "Memory (GiB)",
                [
                    ("ram_tree_rss_bytes", "process tree RSS"),
                    ("vram_used_bytes", "device VRAM"),
                ],
            ),
            (
                "NVML utilization (%)",
                [
                    ("gpu_core_load_percent", "GPU kernels"),
                    ("gpu_memory_controller_load_percent", "memory controller"),
                ],
            ),
            (
                "Board power (% default limit)",
                [("gpu_power_percent_tdp", "power / default limit")],
            ),
        ]
        for axis, (ylabel, fields) in zip(axes, plots):
            for key, label in fields:
                pairs = [
                    (r["elapsed_seconds"] / 60, r[key])
                    for r in resources
                    if r.get(key) is not None
                ]
                if pairs:
                    x, y = zip(*pairs)
                    if key.endswith("bytes"):
                        y = [v / 2**30 for v in y]
                    axis.plot(x, y, label=label, linewidth=0.8)
            axis.set_ylabel(ylabel)
            axis.grid(alpha=0.2)
            axis.legend(loc="upper right")
        axes[-1].set_xlabel("Elapsed minutes (each invocation)")
        fig.savefig(directory / "resources.png", dpi=140)
        plt.close(fig)
    if metrics:
        fig, axes = plt.subplots(
            3, 1, figsize=(11, 9), sharex=True, constrained_layout=True
        )
        decisions = [r["decisions"] for r in metrics]
        for key in ("policy_loss", "value_loss"):
            axes[0].plot(decisions, [r[key] for r in metrics], label=key)
        axes[1].plot(decisions, [r["kl"] for r in metrics], label="KL")
        axes[1].plot(
            decisions,
            [r["normalized_entropy"] for r in metrics],
            label="normalized entropy",
        )
        for opponent in ("random", "greedy"):
            points = [
                (r["decisions"], r["validation"][opponent]["score"])
                for r in metrics
                if "validation" in r
            ]
            if points:
                x, y = zip(*points)
                axes[2].plot(x, y, marker="o", label=f"validation vs {opponent}")
        for axis in axes:
            axis.grid(alpha=0.2)
            axis.legend()
        axes[-1].set_xlabel("Learner decisions")
        fig.savefig(directory / "training.png", dpi=140)
        plt.close(fig)
    result_path = directory / "run-result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else {}
    text = [
        f"# RL-01 — {directory.name}",
        "",
        f"Статус: {result.get('status', 'incomplete')}. Решений: {result.get('decisions', 0):,}.",
        "",
        "Преимущество памяти этим запуском не доказано. Выполненные оценки перечислены ниже по фактическим артефактам.",
        "",
    ]
    memory = list(directory.glob("memory-dev*/summary.json"))
    text.extend([f"Memory-dev: {'результаты найдены' if memory else 'не выполнена в этом каталоге'}. Существующие пары проверяют историю, но не изолируют GRU.", ""])
    if metrics:
        speed = sum(r["rollout_decisions"] for r in metrics) / sum(
            r["update_seconds"] for r in metrics
        )
        text.extend(
            [
                f"Полный цикл обучения: {speed:.1f} решений/с (без validation и записи checkpoints). Updates: {len(metrics)}.",
                "",
                f"Лучший checkpoint по validation: `{result.get('best_checkpoint')}`.",
                "",
            ]
        )
    for path in sorted((directory / "evaluation").glob("*/summary.json")):
        value = json.loads(path.read_text())
        text.extend(
            [
                f"## Validation — {path.parent.name}",
                "",
                "| Соперник | W/D/L | Score | 95% CI по мирам |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, r in value["metrics"].items():
            text.append(
                f"| {name} | {r['wins']}/{r['draws']}/{r['losses']} | {r['score']:.3f} | {r['ci95'][0]:.3f}–{r['ci95'][1]:.3f} |"
            )
        text.append("")
    if resources:
        text.extend(
            [
                "## Ресурсы",
                "",
                "Источник GPU: NVIDIA NVML, вся плата. D3D Usage не используется. RSS — сумма по дереву процессов.",
                "",
                "| Показатель | Среднее | Максимум |",
                "|---|---:|---:|",
            ]
        )
        for key in (
            "cpu_system_percent",
            "cpu_tree_capacity_percent",
            "ram_tree_rss_bytes",
            "vram_used_bytes",
            "gpu_core_load_percent",
            "gpu_memory_controller_load_percent",
            "gpu_power_watts",
            "gpu_power_percent_tdp",
        ):
            values = [r[key] for r in resources if r.get(key) is not None]
            if values:
                scale = 2**30 if key.endswith("bytes") else 1
                label = key.replace("_bytes", " (GiB)")
                text.append(
                    f"| {label} | {sum(values) / len(values) / scale:.2f} | {max(values) / scale:.2f} |"
                )
        errors = {k: v for r in resources for k, v in r.get("errors", {}).items()}
        text.extend(
            [
                "",
                f"Ошибки/недоступные счётчики: `{errors}`.",
                "",
                "![Ресурсы](resources.png)",
                "",
            ]
        )
    if metrics:
        text.extend(["![Обучение](training.png)", ""])
    (directory / "report.md").write_text("\n".join(text), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    args = p.parse_args()
    report(args.directory)


if __name__ == "__main__":
    main()
