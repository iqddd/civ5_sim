"""Saved CP imitation continuation report."""

from __future__ import annotations

from civ5rl.engine.evaluation.replay import verify_replay
from civ5rl.engine.runtime.telemetry_analysis import analyze_telemetry
from civ5rl.provenance.common import ROOT, read_json, write_json
from civ5rl.reporting.evaluation_audit import collect_summary
from civ5rl.reporting.markdown import format_wdl


def report(config, out, parent, *, verify_checkpoint):
    training = read_json(out / "training.json")
    evaluation = read_json(out / "evaluations.json")
    if "final_sample" not in evaluation["labels"] or "final_argmax" not in evaluation["labels"]:
        raise ValueError("Missing final evaluation")
    initial_metrics = training["offline_metrics"][0]
    chosen = next(m for m in training["offline_metrics"] if m["epoch"] == training["best"]["epoch"])
    lines = [
        "# CP imitation: продолжение обучения",
        "",
        "2026-11-04. Seed 17.",
        "",
        f"Стоп: **{training['stop_reason']}**. Последняя полная эпоха {training['last_completed_epoch']}; выбран checkpoint эпохи **{training['best']['epoch']}** по минимальному holdout CE. Sampling — основной результат, argmax — отдельное сравнение.",
        "",
        "## Фиксированные данные и обучение",
        "",
        "Родитель: CP imitation epoch-03, 60 Adam steps. Восстановлены model, Adam moments/groups и Python/NumPy/Torch/CUDA/shuffle RNG. Training: 20 018 меток из 860 эпизодов, 57 миров; holdout: 2485 из 104 эпизодов, 7 миров. Данные не добирались. Validation: прежние 16 миров; test закрыт.",
        "",
        "Модель C, LR=1e−4, прежний Adam и CE, clipping 0,5, minibatch 1024 свободные метки, microbatch ≤8 фрагментов ≤128 решений. BF16 spatial block 32, FP32 GRU/головы/loss. Forced состояния проходят через историю без веса CE; value head заморожена.",
        "",
        f"Новых шагов Adam: **{training['new_Adam_steps']}**, суммарно **{training['Adam_steps_total']}**. Новый optimizer work: **{training['new_optimizer_seconds']:.3f} с**. Начальная подготовка: {training['preparation_seconds']:.3f} с; прогрев и измерения родителя: {training['warmup_and_parent_metrics_seconds']:.3f} с. Компиляция по callbacks: {training['compile_seconds']:.3f} с, не входит в optimizer work. Offline-метрики: {sum(m.get('wall_seconds', 0) for m in training['offline_metrics']):.3f} с. Боевая оценка учитывается отдельно.",
        "",
        "## CE и accuracy фиксированных весов",
        "",
        "Training subset: целые эпизоды в порядке ID, до первого достижения ≥2000 свободных меток; точные ID и хеши — `fixed-training-subset.json`. Training CE эпохи считается на меняющихся весах и не заменяет этот контроль.",
        "",
        "| Epoch | Fixed train CE | Fixed train accuracy | Holdout CE | Holdout accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in training["offline_metrics"]:
        t, h = row["fixed_training"], row["holdout"]
        lines.append(
            f"| {row['epoch']} | {t['ce']:.6f} | {100 * t['accuracy']:.3f}% | {h['ce']:.6f} | {100 * h['accuracy']:.3f}% |"
        )
    lines += [
        "",
        "| Команда | Parent train CE / accuracy | Best train CE / accuracy | Parent holdout CE / accuracy | Best holdout CE / accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for kind in ("move", "ranged", "melee", "skip", "swap"):
        values = []
        for record, split in [
            (initial_metrics, "fixed_training"),
            (chosen, "fixed_training"),
            (initial_metrics, "holdout"),
            (chosen, "holdout"),
        ]:
            entry = record[split]["by_kind"].get(kind)
            values.append(f"{entry['ce']:.4f} / {100 * entry['accuracy']:.2f}%" if entry else "—")
        lines.append("| " + kind + " | " + " | ".join(values) + " |")
    lines += [
        "",
        "Accuracy означает точный выбор команды вместе с целевой клеткой/юнитом, а не только совпадение типа команды. Минимальный фактический CE определяет best; уменьшение 0,01 применяется только к plateau. Ни LR, ни веса классов по ходу не менялись.",
        "",
        "## Бои",
        "",
        "Все оценки: CPU / 4 процесса, прежние случаи/стороны/seeds и SearchConfig. CP сохраняет привилегированные запросы; policy видит только публичные входы.",
        "",
        "| Кандидат / режим | Соперник | W/D/L | Win rate | Изменение к epoch-03 | Move / ranged / melee / skip |",
        "|---|---|---:|---:|---:|---:|",
    ]
    baselines = {
        enemy: collect_summary(parent / "evaluation" / ("final-" + enemy)) for enemy in ("cp", "greedy")
    }
    for enemy, row in baselines.items():
        kinds = row["actions"]
        lines.append(
            f"| epoch-03 / sample | {enemy} | {format_wdl(row)} | {100 * row['win_rate']:.3f}% | — | "
            + " / ".join(str(kinds.get(k, 0)) for k in ("move", "ranged", "melee", "skip"))
            + " |"
        )
    for label, key in evaluation["labels"].items():
        result = evaluation["runs"][key]
        for enemy, row in result["opponents"].items():
            kinds = row["actions"]
            lines.append(
                f"| {label}: epoch-{result['epoch']:02} / {result['mode']} | {enemy} | {format_wdl(row)} | {100 * row['win_rate']:.3f}% | {100 * (row['win_rate'] - baselines[enemy]['win_rate']):+.3f} п.п. | "
                + " / ".join(str(kinds.get(k, 0)) for k in ("move", "ranged", "melee", "skip"))
                + " |"
            )
    lines += [
        "",
        "При совпадении checkpoint оценки повторно не выполнялись. Initial и CP–greedy взяты из родительского опыта. Argmax не участвует в выборе checkpoint и не заменяет sampling. Все причины ничьих и точные каталоги результатов — `evaluations.json`.",
        "",
    ]
    for label in ("final_sample", "final_argmax"):
        result = evaluation["runs"][evaluation["labels"][label]]
        wins = sum(v["W"] for v in result["opponents"].values())
        lines.append(f"{label}: {wins}/128 побед, checkpoint epoch-{result['epoch']}.")
    lines += [
        "",
        f"Новых матчей: {sum(r['new_matches'] for r in evaluation['runs'].values())}; wall time уникальных боевых оценок: {sum(r['wall_seconds'] for r in evaluation['runs'].values()):.3f} с.",
        "",
        "## Проверка реплеев и состояний",
        "",
    ]
    verified = []
    for label in ("final_sample", "final_argmax"):
        result = evaluation["runs"][evaluation["labels"][label]]
        for enemy, directory in result["directories"].items():
            paths = sorted((ROOT / directory).glob("*/metrics.json"))
            for desired in ("W", "D", "L"):
                chosen_path = next(
                    (
                        p
                        for p in paths
                        if (
                            "D"
                            if read_json(p)["winner"] is None
                            else "W"
                            if read_json(p)["winner"] == read_json(p)["side"]
                            else "L"
                        )
                        == desired
                    ),
                    None,
                )
                if chosen_path is None:
                    lines.append(f"- {label} / {enemy}: исход {desired} отсутствует.")
                    continue
                m = read_json(chosen_path)
                verification = verify_replay(chosen_path.parent)
                verification.update(label=label, enemy=enemy, outcome=desired, rounds=m["rounds"])
                available = verification["attack_available_decisions"]
                selected = verification["attacks_chosen"]
                omitted = verification["non_attack_when_available"]
                verified.append(verification)
                lines.append(
                    f"- {label} / {enemy}, {chosen_path.parent.name}: {desired}, {m['rounds']} раундов; {verification['commands']} команд, все результаты и state hashes совпали; атак выбрано {selected}, решений с атакой {available}, иных команд при доступной атаке {omitted}."
                )
    write_json(out / "replay-verification.json", verified)
    verify_checkpoint(config, out, parent)
    telemetry_analysis = analyze_telemetry(out / "telemetry/resources.jsonl")
    iterative, windows = telemetry_analysis["active"], telemetry_analysis["windows"]
    lines += [
        "",
        "## Телеметрия и границы вывода",
        "",
        f"Итеративных NVML samples: {len(iterative)}. Прогрев, compilation, offline metrics, сохранение и боевая оценка помечены отдельно. CPU tree 100% = одно логическое ядро. WDDM D3D Usage не использовался.",
        "",
    ]
    for field, stats in telemetry_analysis["metrics"].items():
        lines.append(f"- {field}: mean={stats['mean']:.3f}, max={stats['max']:.3f} {stats['unit']}")
    if training["new_optimizer_seconds"] < 600:
        lines.append(
            "Итеративное обучение суммарно короче 10 минут: требование Core Load ≥80% не применяется."
        )
    else:
        lines.append(
            "Для длительной итеративной части GPU Core Load и 60-секундные окна сохранены в gpu-windows.json. При низкой загрузке обоснование продолжения — training.json: GPU_below_target_justification; параметры ради загрузки не менялись."
        )
    write_json(
        out / "gpu-windows.json",
        {
            "contiguous_wall_windows": windows,
            "note": "Windows exclude other phases, gaps, and process resets.",
        },
    )
    lines += [
        "",
        "",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        out / "status.json",
        {
            "status": "complete",
            "stop_reason": training["stop_reason"],
            "last_completed_epoch": training["last_completed_epoch"],
            "best_epoch": training["best"]["epoch"],
            "new_Adam_steps": training["new_Adam_steps"],
            "new_evaluation_matches": sum(r["new_matches"] for r in evaluation["runs"].values()),
            "closed_test_evaluated": False,
        },
    )
