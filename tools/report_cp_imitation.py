"""Small cost report first; later phases append data, BC and battle strength."""

from collections import Counter
from pathlib import Path

from civ5rl.provenance.common import read_json, write_json


def report(config, out):
    out = Path(out)
    manifest = read_json(out / "manifest.json")
    text = [
        "# Имитация CP — данные → обучение → сила в бою",
        "",
        "Первый этап ограничен восемью CP–greedy матчами и измерением latency. Полный сбор, исследовательское обучение и 320 оценочных матчей автоматически не запускаются.",
        "",
        f"Training: {len(manifest['worlds']['training'])} миров; offline holdout: {len(manifest['worlds']['holdout'])}; validation: {len(manifest['worlds']['validation'])}. Закрытый test не используется. Seed 17, свежая модель C. Учитель использует зафиксированный SearchConfig с privileged_cp_queries=True; в policy поступают только публичные наблюдения и legal list.",
    ]
    cost = {}
    path = out / "benchmark/collection.json"
    if path.exists():
        sample = read_json(path)
        rows = sample["matches"]
        wdl = Counter(
            "draw"
            if r["winner"] is None
            else "win"
            if r["winner"] == r["side"]
            else "loss"
            for r in rows
        )
        text += [
            "",
            f"**Замер завершён:** {len(rows)} матчей, {sample['workers']} CPU-процесса, {sample['pool_wall_seconds']:.2f} с wall time (включая старт пула). Меток с альтернативами: {sample['labels']}, вынужденных команд: {sample['forced']}; {sample['labels_per_second']:.2f} пригодных меток/с. CP W/D/L: {wdl['win']}/{wdl['draw']}/{wdl['loss']}.",
            "",
            "| Case | Сторона CP | W/D/L | Раунды | Метки / вынужденные | Wall с | CP act с | Planning input с |",
            "|---|---:|---|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            text.append(
                f"| {r['case_id']} | {r['side']} | {'D' if r['winner'] is None else 'W' if r['winner'] == r['side'] else 'L'} | {r['rounds']} | {r['labels']} / {r['forced']} | {r['seconds']:.2f} | {r['agent_act_seconds'][r['side']]:.2f} | {r['timings'].get('prepare_planning', 0):.2f} |"
            )
        collection = (
            (config["training_labels"] + config["holdout_labels"])
            / sample["labels_per_second"]
            if sample["labels"]
            else None
        )
        cost.update(
            collection_total_seconds=collection,
            collection_remaining_seconds=max(
                0, collection - sample["pool_wall_seconds"]
            )
            if collection
            else None,
            collection_target_labels=22457,
            extrapolation_sample_matches=8,
        )
        text += [
            "",
            f"Прогноз всего сбора 20 000 training + 2457 holdout: **{collection / 60:.2f} мин** при той же скорости."
            if collection
            else "В пилоте нет пригодных меток; прогноз сбора невозможен.",
            "Это экстраполяция восьми матчей, а не гарантированный срок: состав посещённых миров, длины боёв и доля вынужденных команд меняются. Лимит будущего сбора — час, обучение — до 30 минут. На произвольной недобранной выборке обучение не запускается.",
        ]
        latency_path = out / "benchmark/latency.json"
        if latency_path.exists():
            measured = read_json(latency_path)
            text += [
                "",
                "| RLAgent device | Тип | N | Mean мс | Median мс | p95 мс | Прогрев с |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
            for device, values in measured.items():
                if values.get("unavailable"):
                    continue
                for kind, v in values["samples"].items():
                    if v:
                        text.append(
                            f"| {device} | {kind} | {v['count']} | {v['mean_ms']:.3f} | {v['median_ms']:.3f} | {v['p95_ms']:.3f} | {values['warmup_seconds']:.2f} |"
                        )
            choices = [
                (v["samples"]["all"]["mean_ms"], k)
                for k, v in measured.items()
                if not v.get("unavailable")
            ]
            latency_ms, fastest = min(choices)
            cp_calls = sum(r["agent_calls"][r["side"]] for r in rows) / 8
            # 192 matches contain CP: 128 network-vs-CP plus 64 teacher-vs-greedy.
            # Remaining 128 network-vs-greedy matches use the measured greedy cost.
            greedy_seconds = (
                sum(r["agent_act_seconds"][1 - r["side"]] for r in rows) / 8
            )
            cp_matches = sample["pool_wall_seconds"] / 8 * 192
            network_seconds = latency_ms / 1000 * cp_calls * 256 / sample["workers"]
            extra_greedy = greedy_seconds * 128 / sample["workers"]
            observed = cp_matches + network_seconds + extra_greedy
            mean_rounds = sum(r["rounds"] for r in rows) / 8
            factor = max(1, 30 / max(1, mean_rounds))
            cost.update(
                evaluation_observed_trajectory_seconds=observed,
                evaluation_round_scaled_seconds=observed * factor,
                evaluation_network_device=fastest,
                evaluation_matches=320,
                evaluation_CP_matches=192,
                evaluation_assumption="teacher-like decision counts; round-scaled scenario is not an upper bound",
            )
            text += [
                "",
                f"Прогноз 320 матчей оценки: **{observed / 60:.2f} мин** при частоте команд и длительности пилотных траекторий; **{observed * factor / 60:.2f} мин** при линейном масштабировании до 30 раундов. Для сетевой части использована меньшая измеренная средняя latency ({fastest}). Это два сценария расчёта, не доверительный интервал и не верхняя граница. Initial/BC могут ходить и сражаться иначе; GPU contention нескольких процессов здесь не измерен.",
                "",
                "Во время генерации CP CPU — основной вычислительный ресурс; сеть в этой фазе не работает. BC-эпохи используют GPU без вызовов CP. Менять N можно через --workers; автоматической настройки пула нет.",
            ]
        write_json(out / "benchmark/cost-estimate.json", cost)
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
    if (out / "evaluation.json").exists():
        text += [
            "",
            "## Сила в бою",
            "",
            "| Модель / соперник | W/D/L | Win rate | Move / melee / ranged | Ничьи по лимиту |",
            "|---|---:|---:|---:|---:|",
        ]
        groups = {}
        for group in (
            "initial-cp",
            "initial-greedy",
            "final-cp",
            "final-greedy",
            "teacher-greedy",
        ):
            rows = [
                read_json(p)
                for p in sorted((out / "evaluation" / group).glob("*/metrics.json"))
            ]
            if len(rows) != 64:
                raise ValueError("Incomplete imitation evaluation group")
            counts = Counter(
                "draw"
                if r["winner"] is None
                else "win"
                if r["winner"] == r["side"]
                else "loss"
                for r in rows
            )
            kinds = sum((Counter(r["actions_by_kind"]) for r in rows), Counter())
            groups[group] = counts["win"] / 64
            text.append(
                f"| {group} | {counts['win']}/{counts['draw']}/{counts['loss']} | {groups[group]:.2%} | {kinds['move']}/{kinds['melee']}/{kinds['ranged']} | {sum(r['reason'] == 'turn_limit' for r in rows)} |"
            )
        for opponent in ("cp", "greedy"):
            text.append(
                f"Прирост win rate против {opponent}: {groups['final-' + opponent] - groups['initial-' + opponent]:+.2%}."
            )
    telemetry = out / "benchmark/telemetry/resources-summary.json"
    if telemetry.exists():
        text += ["", "| Ресурс генерации | Среднее | Max |", "|---|---:|---:|"]
        for key, v in read_json(telemetry)["metrics"].items():
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
                    f"| {key} {'GiB' if scale > 1 else '%'} | {v['mean'] / scale:.2f} | {v['max'] / scale:.2f} |"
                )
    text += [
        "",
        "Артефакты: manifest.json, паспорт и sources.zip; checkpoints/initial.pt; data/training/match-*/demonstration.json.gz и полные replay/final/result. Вынужденные команды сохраняются для истории, но не входят в CE и целевой объём.",
        "",
        "После benchmark выполнение остановлено для обсуждения стоимости. Seed 42, закрытый test и PPO не запускались. Имитация привилегированного учителя может иметь неоднозначные публичные метки; accuracy не заменяет силу в бою.",
        "",
    ]
    (out / "report.md").write_text("\n".join(text), encoding="utf-8")
