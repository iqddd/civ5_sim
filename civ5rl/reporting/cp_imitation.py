"""Small cost report first; later phases append data, BC and battle strength."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from civ5rl.provenance.common import read_json, write_json
from civ5rl.reporting.evaluation_audit import classify_outcome, summarize_wdl
from civ5rl.reporting.markdown import MarkdownReport, format_wdl


def _evaluation_cost(sample: dict[str, Any], latency: dict[str, Any]) -> dict[str, Any]:
    choices = [
        (measurement["samples"]["all"]["mean_ms"], device)
        for device, measurement in latency.items()
        if not measurement.get("unavailable")
    ]
    latency_ms, fastest = min(choices)
    rows = sample["matches"]
    cp_calls = sum(row["agent_calls"][row["side"]] for row in rows) / 8
    # 192 matches use CP. The other 128 use the measured greedy cost.
    greedy_seconds = sum(row["agent_act_seconds"][1 - row["side"]] for row in rows) / 8
    cp_matches = sample["pool_wall_seconds"] / 8 * 192
    network_seconds = latency_ms / 1000 * cp_calls * 256 / sample["workers"]
    extra_greedy = greedy_seconds * 128 / sample["workers"]
    observed = cp_matches + network_seconds + extra_greedy
    mean_rounds = sum(row["rounds"] for row in rows) / 8
    factor = max(1, 30 / max(1, mean_rounds))
    return {
        "evaluation_observed_trajectory_seconds": observed,
        "evaluation_round_scaled_seconds": observed * factor,
        "evaluation_network_device": fastest,
        "evaluation_matches": 320,
        "evaluation_CP_matches": 192,
        "evaluation_assumption": "teacher-like decision counts; round-scaled scenario is not an upper bound",
    }


def _compute_and_save_cost_estimate(
    config: dict[str, Any], out: Path, sample: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    collection = (
        (config["training_labels"] + config["holdout_labels"]) / sample["labels_per_second"]
        if sample["labels"]
        else None
    )
    cost = {
        "collection_total_seconds": collection,
        "collection_remaining_seconds": max(0, collection - sample["pool_wall_seconds"])
        if collection
        else None,
        "collection_target_labels": 22457,
        "extrapolation_sample_matches": 8,
    }
    latency_path = out / "benchmark/latency.json"
    latency = read_json(latency_path) if latency_path.exists() else None
    if latency is not None:
        cost.update(_evaluation_cost(sample, latency))
    write_json(out / "benchmark/cost-estimate.json", cost)
    return cost, latency


def _append_collection_benchmark(text, sample, cost):
    rows = sample["matches"]
    outcomes = summarize_wdl(rows)
    text += [
        "",
        f"**Замер завершён:** {outcomes['matches']} матчей, {sample['workers']} CPU-процесса, {sample['pool_wall_seconds']:.2f} с wall time (включая старт пула). Меток с альтернативами: {sample['labels']}, вынужденных команд: {sample['forced']}; {sample['labels_per_second']:.2f} пригодных меток/с. CP W/D/L: {format_wdl(outcomes)}.",
        "",
    ]
    text.table(
        (
            "Case",
            "Сторона CP",
            "W/D/L",
            "Раунды",
            "Метки / вынужденные",
            "Wall с",
            "CP act с",
            "Planning input с",
        ),
        ("---", "---:", "---", "---:", "---:", "---:", "---:", "---:"),
        (
            (
                row["case_id"],
                row["side"],
                classify_outcome(row["winner"], row["side"]),
                row["rounds"],
                f"{row['labels']} / {row['forced']}",
                f"{row['seconds']:.2f}",
                f"{row['agent_act_seconds'][row['side']]:.2f}",
                f"{row['timings'].get('prepare_planning', 0):.2f}",
            )
            for row in rows
        ),
    )
    collection = cost["collection_total_seconds"]
    text += [
        "",
        f"Прогноз всего сбора 20 000 training + 2457 holdout: **{collection / 60:.2f} мин** при той же скорости."
        if collection
        else "В пилоте нет пригодных меток; прогноз сбора невозможен.",
        "Это экстраполяция восьми матчей, а не гарантированный срок: состав посещённых миров, длины боёв и доля вынужденных команд меняются. Лимит будущего сбора — час, обучение — до 30 минут. На произвольной недобранной выборке обучение не запускается.",
    ]


def _append_latency(text, latency, cost):
    text.append("")
    text.table(
        ("RLAgent device", "Тип", "N", "Mean мс", "Median мс", "p95 мс", "Прогрев с"),
        ("---", "---", "---:", "---:", "---:", "---:", "---:"),
        (
            (
                device,
                kind,
                sample["count"],
                f"{sample['mean_ms']:.3f}",
                f"{sample['median_ms']:.3f}",
                f"{sample['p95_ms']:.3f}",
                f"{measurement['warmup_seconds']:.2f}",
            )
            for device, measurement in latency.items()
            if not measurement.get("unavailable")
            for kind, sample in measurement["samples"].items()
            if sample
        ),
    )
    observed = cost["evaluation_observed_trajectory_seconds"]
    fastest = cost["evaluation_network_device"]
    text += [
        "",
        f"Прогноз 320 матчей оценки: **{observed / 60:.2f} мин** при частоте команд и длительности пилотных траекторий; **{cost['evaluation_round_scaled_seconds'] / 60:.2f} мин** при линейном масштабировании до 30 раундов. Для сетевой части использована меньшая измеренная средняя latency ({fastest}). Это два сценария расчёта, не доверительный интервал и не верхняя граница. Initial/BC могут ходить и сражаться иначе; GPU contention нескольких процессов здесь не измерен.",
        "",
        "Во время генерации CP CPU — основной вычислительный ресурс; сеть в этой фазе не работает. BC-эпохи используют GPU без вызовов CP. Менять N можно через --workers; автоматической настройки пула нет.",
    ]


def _append_phase_status(text, out):
    for phase in ("collection", "training"):
        if (out / f"{phase}.json").exists():
            result = read_json(out / f"{phase}.json")
            text += ["", f"## {phase}", "", f"Статус: {result['status']}."]
            if phase == "training":
                text += [f"Initial holdout: {result['initial_holdout']}"]
                for row in result["epochs"]:
                    text.append(
                        f"Эпоха {row['epoch']}: completed={row['completed']}, CE={row['ce']}, labels={row['labels']}, Adam steps={row['Adam_steps_total']}; holdout={row['holdout']}."
                    )


def _append_evaluation_report(text, out):
    if not (out / "evaluation.json").exists():
        return
    text += ["", "## Сила в бою", ""]
    text.table(
        (
            "Модель / соперник",
            "W/D/L",
            "Win rate",
            "Move / melee / ranged",
            "Ничьи по лимиту",
        ),
        ("---", "---:", "---:", "---:", "---:"),
    )
    groups = {}
    for group in (
        "initial-cp",
        "initial-greedy",
        "final-cp",
        "final-greedy",
        "teacher-greedy",
    ):
        rows = [read_json(path) for path in sorted((out / "evaluation" / group).glob("*/metrics.json"))]
        if len(rows) != 64:
            raise ValueError("Incomplete imitation evaluation group")
        outcomes = summarize_wdl(rows)
        kinds = sum((Counter(row["actions_by_kind"]) for row in rows), Counter())
        groups[group] = outcomes["win_rate"]
        turn_limits = sum(row["reason"] == "turn_limit" for row in rows)
        text.append(
            f"| {group} | {format_wdl(outcomes)} | {groups[group]:.2%} | "
            f"{kinds['move']}/{kinds['melee']}/{kinds['ranged']} | {turn_limits} |"
        )
    for opponent in ("cp", "greedy"):
        gain = groups["final-" + opponent] - groups["initial-" + opponent]
        text.append(f"Прирост win rate против {opponent}: {gain:+.2%}.")


def _append_telemetry(text, out):
    telemetry = out / "benchmark/telemetry/resources-summary.json"
    if telemetry.exists():
        text.append("")
        text.table(("Ресурс генерации", "Среднее", "Max"), ("---", "---:", "---:"))
        for key, measurements in read_json(telemetry)["metrics"].items():
            if key in (
                "cpu_tree_capacity_percent",
                "ram_tree_rss_bytes",
                "gpu_core_load_percent",
                "gpu_memory_controller_load_percent",
                "vram_used_bytes",
                "gpu_power_percent_tdp",
            ):
                scale = 2**30 if key.endswith("bytes") else 1
                text.append(
                    f"| {key} {'GiB' if scale > 1 else '%'} | {measurements['mean'] / scale:.2f} | {measurements['max'] / scale:.2f} |"
                )


def _render_report(out, manifest, sample, cost, latency):
    text = MarkdownReport(
        [
            "# Имитация CP — данные → обучение → сила в бою",
            "",
            "Первый этап ограничен восемью CP–greedy матчами и измерением latency. Полный сбор, исследовательское обучение и 320 оценочных матчей автоматически не запускаются.",
            "",
            f"Training: {len(manifest['worlds']['training'])} миров; offline holdout: {len(manifest['worlds']['holdout'])}; validation: {len(manifest['worlds']['validation'])}. Закрытый test не используется. Seed 17, свежая модель C. Учитель использует зафиксированный SearchConfig с privileged_cp_queries=True; в policy поступают только публичные наблюдения и legal list.",
        ]
    )
    if sample is not None:
        _append_collection_benchmark(text, sample, cost)
        if latency is not None:
            _append_latency(text, latency, cost)
    _append_phase_status(text, out)
    _append_evaluation_report(text, out)
    _append_telemetry(text, out)
    text += [
        "",
        "Артефакты: manifest.json, паспорт и sources.zip; checkpoints/initial.pt; data/training/match-*/demonstration.json.gz и полные replay/final/result. Вынужденные команды сохраняются для истории, но не входят в CE и целевой объём.",
        "",
        "После benchmark выполнение остановлено для обсуждения стоимости. Seed 42, закрытый test и PPO не запускались. Имитация привилегированного учителя может иметь неоднозначные публичные метки; accuracy не заменяет силу в бою.",
        "",
    ]
    text.write(out / "report.md", trailing_newline=False)


def report(config, out):
    out = Path(out)
    manifest = read_json(out / "manifest.json")
    sample_path = out / "benchmark/collection.json"
    sample = read_json(sample_path) if sample_path.exists() else None
    cost, latency = (None, None) if sample is None else _compute_and_save_cost_estimate(config, out, sample)
    _render_report(out, manifest, sample, cost, latency)
