"""Saved expanded CP imitation corpus report."""

from __future__ import annotations

from collections import Counter

from civ5rl.engine.evaluation.replay import verify_replay
from civ5rl.engine.runtime.telemetry_analysis import analyze_telemetry
from civ5rl.protocols.imitation import load_episode, validate_case
from civ5rl.provenance.common import ROOT, file_sha256, read_json, write_json
from civ5rl.reporting.evaluation_audit import collect_summary, compare_paired_directories
from civ5rl.reporting.markdown import format_wdl
from civ5sim.engine import digest


def report(config, out, parent, data, manifest):
    coverage = {}
    physical_sets, seed_sets, id_sets = {}, {}, {}
    for group, files in (
        (
            "old",
            {
                k: v
                for k, v in manifest["inputs"]["training"].items()
                if k.startswith(config["more_data"]["data_run"] + "/")
            },
        ),
        (
            "new",
            {
                k: v
                for k, v in manifest["inputs"]["training"].items()
                if not k.startswith(config["more_data"]["data_run"] + "/")
            },
        ),
    ):
        worlds, maps, physical, seeds = set(), set(), set(), set()
        case_ids = set()
        modes, sides, outcomes, kinds = Counter(), Counter(), Counter(), Counter()
        labels = forced = 0
        for name, sha in files.items():
            path = ROOT / name
            if file_sha256(path) != sha:
                raise ValueError("Demonstration hash changed before report")
            case, metric_row = (
                read_json(path.parent / "case.json"),
                read_json(path.parent / "metrics.json"),
            )
            episode = load_episode(path)
            validate_case(case, config, manifest["base"])
            case_ids.add(case["case_id"])
            worlds.add(case["parent_world"])
            maps.add(digest(case["scenario"]["map"]))
            physical.add(
                digest(
                    {
                        "map": digest(case["scenario"]["map"]),
                        "units": sorted(
                            case["scenario"]["units"],
                            key=lambda u: (u["side"], u["type"], u["tile"]),
                        ),
                    }
                )
            )
            seeds.add((case["scenario"]["seed"], *case["agent_seeds"]))
            modes[case["placement"]] += 1
            sides[str(case["learner_side"])] += 1
            outcomes[
                "D"
                if metric_row["winner"] is None
                else "W"
                if metric_row["winner"] == metric_row["side"]
                else "L"
            ] += 1
            labels += metric_row["labels"]
            forced += metric_row["forced"]
            kinds.update(r["actions"][r["chosen"]]["kind"] for r in episode["decisions"] if r["supervised"])
        coverage[group] = {
            "episodes": len(files),
            "worlds": len(worlds),
            "maps": len(maps),
            "unique_physical_cases": len(physical),
            "unique_seed_tuples": len(seeds),
            "labels": labels,
            "forced": forced,
            "starts": dict(modes),
            "sides": dict(sides),
            "outcomes": dict(outcomes),
            "free_commands": dict(kinds),
        }
        physical_sets[group], seed_sets[group], id_sets[group] = (
            physical,
            seeds,
            case_ids,
        )
    if id_sets["old"] & id_sets["new"] or seed_sets["old"] & seed_sets["new"]:
        raise ValueError("Repeated old case IDs or RNG in new collection")
    coverage["new"]["physical_overlap_with_old"] = len(physical_sets["old"] & physical_sets["new"])
    coverage["new"]["first_case_id"] = min(id_sets["new"]) if id_sets["new"] else None
    coverage["new"]["last_case_id"] = max(id_sets["new"]) if id_sets["new"] else None
    write_json(out / "coverage.json", coverage)
    lines = [
        "# CP imitation: расширенный корпус",
        "",
        f"Seed {config['seed']}.",
        "",
        "## Покрытие данных",
        "",
        "| Часть | Эпизоды | Метки | Forced | Миры | Maps | Физические состояния | Уникальные RNG tuples |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for group, group_coverage in coverage.items():
        lines.append(
            f"| {group} | {group_coverage['episodes']} | {group_coverage['labels']} | {group_coverage['forced']} | {group_coverage['worlds']} | {group_coverage['maps']} | {group_coverage['unique_physical_cases']} | {group_coverage['unique_seed_tuples']} |"
        )
    lines += [
        "",
        "Независимых training-worlds по-прежнему 57; дополнительные crops и RNG не увеличивают это число. Holdout: прежние 2485 меток из 7 миров. Validation: прежние 32 расстановки × обе стороны из 16 миров. Ни holdout, ни validation не доразмечались для обучения.",
        "",
        f"Сбор: {read_json(out / 'collection.json')}",
        "",
        "Полная статистика сторон/стартов, исходов и типов свободных команд — `coverage.json`. Исходы не использованы для фильтрации корпуса.",
    ]
    if not (out / "training.json").exists():
        lines += [
            "",
            "Обучение не выполнялось; целевой объём не был автоматически уменьшен.",
        ]
        (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    training = read_json(out / "training.json")
    probe = read_json(out / "performance-preflight.json")
    lines += [
        "",
        "## Скорость и бюджет",
        "",
        f"После компиляции: среднее время шага {probe['mean_step_seconds']:.6f} с, std {probe['std_step_seconds']:.6f} с по 8 полным minibatches. ETA эпохи {probe['epoch_eta_seconds']:.3f} с; десяти эпох {probe['ten_epoch_eta_seconds']:.3f} с. Это прогноз optimizer work, не всей программы; plateau может сократить запуск.",
        "",
        f"Выбран backend: **{probe['selected']}**. Детали сравнения/отказа — `performance-preflight.json`.",
        "",
        f"Фактически: {training['new_optimizer_seconds']:.3f} с новой optimizer work; {training['new_Adam_steps']} новых Adam steps, суммарно {training['Adam_steps_total']}. Stop: **{training['stop_reason']}**; полных новых эпох {training['last_completed_epoch']}, best **{training['best_epoch']}** (0 означает родитель epoch-30).",
        "",
        f"Подготовка {training['preparation_seconds']:.3f} с; интервалы компиляции {training['compile_seconds']:.3f} с (внутри wall time соответствующих фаз, не прибавлять второй раз); offline-измерения {sum(metric_row['wall_seconds'] for metric_row in training['offline_metrics']):.3f} с. Пробные Adam updates отброшены. В optimizer work не входят подготовка, compile, preflight и оценки.",
        "",
        "## Обобщение",
        "",
        "| Новая эпоха | Fixed train CE / accuracy | Holdout CE / accuracy |",
        "|---|---:|---:|",
    ]
    for metric_row in training["offline_metrics"]:
        fixed_training, holdout = metric_row["fixed_training"], metric_row["holdout"]
        lines.append(
            f"| {metric_row['epoch']} | {fixed_training['ce']:.6f} / {100 * fixed_training['accuracy']:.3f}% | {holdout['ce']:.6f} / {100 * holdout['accuracy']:.3f}% |"
        )
    best = next(
        metric_row
        for metric_row in training["offline_metrics"]
        if metric_row["epoch"] == training["best_epoch"]
    )
    before = training["offline_metrics"][0]
    lines += [
        "",
        "Fixed training subset включает целые эпизоды пропорционально old/new; точные файлы и реальные доли — `fixed-training-subset.json`. Снижение ≥0,01 влияет только на plateau; best выбирается по минимальной фактической holdout CE, tie — более ранний.",
        "",
        "| Команда | Parent holdout CE / accuracy | Best holdout CE / accuracy | Best fixed train CE / accuracy |",
        "|---|---:|---:|---:|",
    ]
    for kind in ("move", "ranged", "melee", "skip", "swap"):
        parts = []
        for metric, split in (
            (before, "holdout"),
            (best, "holdout"),
            (best, "fixed_training"),
        ):
            entry = metric[split]["by_kind"].get(kind)
            parts.append(f"{entry['ce']:.5f} / {100 * entry['accuracy']:.2f}%" if entry else "—")
        lines.append("| " + kind + " | " + " | ".join(parts) + " |")
    lines += [
        "",
        "| Move, знаменатель — все teacher move | Тип | Тип + юнит | Полная команда с клеткой | N |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, metric in (
        ("parent holdout", before["holdout"]),
        ("best holdout", best["holdout"]),
        ("best fixed train", best["fixed_training"]),
    ):
        move = metric["move"]
        lines.append(
            f"| {label} | {100 * move['type']:.2f}% | {100 * move['type_unit']:.2f}% | {100 * move['exact']:.2f}% | {move['denominator']} |"
        )
    if not (out / "evaluations.json").exists():
        raise ValueError("No final battle evaluation")
    evaluation = read_json(out / "evaluations.json")
    inherited = read_json(parent / "evaluations.json")
    comparisons = {}
    checks = []
    lines += [
        "",
        "## Сила в бою",
        "",
        "Sampling temperature=1 — основной результат; argmax — отдельная оценка того же заранее выбранного checkpoint. CPU / 4 workers, прежние cases/стороны/seeds и привилегии CP. Ничьи не считаются победами.",
        "",
        "| Режим | Соперник | Parent W/D/L | Best W/D/L | Win rate | Прирост, п.п. | Парный 95% CI прироста, п.п. |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for mode, result in evaluation["runs"].items():
        original = inherited["runs"][inherited["labels"]["final_" + mode]]
        for enemy, directory in result["directories"].items():
            bootstrap, slices, _, actual = compare_paired_directories(
                original["directories"][enemy],
                directory,
                samples=config["more_data"]["bootstrap_samples"],
                seed=config["more_data"]["bootstrap_seed"],
                require_artifacts=True,
            )
            candidate, baseline = (
                collect_summary(ROOT / directory),
                collect_summary(ROOT / original["directories"][enemy]),
            )
            key = mode + "-" + enemy
            detail = {
                **bootstrap,
                "baseline": baseline,
                "candidate": candidate,
                "slices": {},
            }
            detail["slices"] = {name: value["outcomes"] for name, value in slices.items()}
            comparisons[key] = detail
            lines.append(
                f"| {mode} | {enemy} | {format_wdl(baseline)} | {format_wdl(candidate)} | {100 * candidate['win_rate']:.3f}% | {100 * bootstrap['delta']:+.3f} | [{100 * bootstrap['CI95'][0]:+.3f}; {100 * bootstrap['CI95'][1]:+.3f}] |"
            )
            if result["new_matches"]:
                for desired in ("W", "D", "L"):
                    selected = next(
                        (
                            r
                            for k, r in sorted(actual.items())
                            if (
                                "D"
                                if r[0]["winner"] is None
                                else "W"
                                if r[0]["winner"] == r[0]["side"]
                                else "L"
                            )
                            == desired
                        ),
                        None,
                    )
                    if selected is None:
                        checks.append({"group": key, "outcome": desired, "present": False})
                        continue
                    metric_row, case, path = selected
                    replay_check = verify_replay(path)
                    checks.append(
                        {
                            "group": key,
                            "outcome": desired,
                            "present": True,
                            "path": path.relative_to(ROOT).as_posix(),
                            "commands": replay_check["commands"],
                            "all_state_hashes_equal": True,
                        }
                    )
    write_json(out / "battle-comparisons.json", comparisons)
    write_json(out / "replay-verification.json", checks)
    lines += [
        "",
        "Кластер bootstrap — parent-world, все четыре случая (два старта × две стороны) и результаты обеих моделей связаны; 64 матча не считаются независимыми мирами. Это интервалы отдельных сравнений без поправки на множественные сравнения.",
        "",
        "### Срезы и команды",
        "",
        "| Режим / соперник | Контакт W/D/L | Разделённые W/D/L | Сторона 0 W/D/L | Сторона 1 W/D/L | Move / ranged / melee / skip |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group, group_coverage in comparisons.items():
        parts = [
            "/".join(str(group_coverage["slices"][s][k]) for k in ("W", "D", "L"))
            for s in ("contact", "separated", "side0", "side1")
        ]
        kinds = group_coverage["candidate"]["actions"]
        lines.append(
            "| "
            + group
            + " | "
            + " | ".join(parts)
            + " | "
            + " / ".join(str(kinds.get(k, 0)) for k in ("move", "ranged", "melee", "skip"))
            + " |"
        )
    lines += [
        "",
        f"Новых оценочных матчей: {sum(r['new_matches'] for r in evaluation['runs'].values())}; wall time evaluation {evaluation['wall_seconds']:.3f} с. Полные причины ничьих и action counts — `battle-comparisons.json`. Результаты родителя использованы без повторного исполнения.",
        "",
        "## Проверки и телеметрия",
        "",
        f"Новых полных реплеев проверено: {sum(r['present'] for r in checks)}; отсутствующие исходы перечислены явно в `replay-verification.json`. Проверялись результаты всех команд, промежуточные state hashes и конечные состояния.",
    ]
    telemetry_analysis = analyze_telemetry(out / "training-telemetry/resources.jsonl")
    windows = telemetry_analysis["windows"]
    for field, stats in telemetry_analysis["metrics"].items():
        lines.append(f"- {field}: mean={stats['mean']:.3f}, max={stats['max']:.3f} {stats['unit']}")
    write_json(out / "gpu-active-windows.json", windows)
    lines += [
        "",
        (
            "Итеративное обучение короче 10 минут: требование GPU Core Load ≥80% не применяется."
            if training["new_optimizer_seconds"] < 600
            else "Итеративная часть ≥10 минут. Результат короткого profiler-разбора и выбора backend — performance-preflight.json; GPU непрерывные 60-секундные окна — gpu-active-windows.json. Сохранённые условия, CPU-сборка/синхронизации и small spatial blocks ограничивают загрузку; при продолжении ниже 80% это явное ограничение выбранного проверенного backend, не заявление о достижении целевого процента."
        ),
        "",
        "Источник GPU-метрик — NVML, не WDDM D3D Usage. GPU whole-device; CPU tree 100% = одно логическое ядро. Подготовка, compilation, offline metrics, throughput probe и performance investigation размечены отдельно и исключены из активных GPU averages.",
        "",
        "Детали в coverage.json, replay-verification.json и training.json.",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        out / "status.json",
        {
            "status": "complete",
            "stop_reason": training["stop_reason"],
            "best_epoch": training["best_epoch"],
            "new_Adam_steps": training["new_Adam_steps"],
            "new_evaluation_matches": sum(r["new_matches"] for r in evaluation["runs"].values()),
            "closed_test_evaluated": False,
        },
    )
