"""Saved own-state CP imitation report."""

from __future__ import annotations

from civ5rl.engine.evaluation.replay import verify_replay
from civ5rl.protocols.imitation import load_episode
from civ5rl.provenance.common import ROOT, file_sha256, read_json, write_json
from civ5rl.reporting.evaluation_audit import compare_paired_directories
from civ5rl.reporting.markdown import format_wdl
from civ5rl.reporting.plot import pyplot


def relative(path):
    return path.relative_to(ROOT).as_posix()


def _compare_battles(config, before_path, after_path):
    settings = config["own_states"]
    result, slices, _, _ = compare_paired_directories(
        before_path,
        after_path,
        samples=settings["bootstrap_samples"],
        seed=settings["bootstrap_seed"],
    )
    return {
        **result,
        "slices": {
            name.replace("side0", "side-0").replace("side1", "side-1"): {
                key: value[key] for key in ("matches", "baseline_wins", "candidate_wins")
            }
            for name, value in slices.items()
        },
    }


def report(config, out, parent, *, verify_checkpoint):
    state = read_json(out / "state.json")
    lines = [
        "# CP advice на собственных траекториях",
        "",
        "Sampling против CP — основной результат. Validation используется для разработки; test закрыт.",
        "",
        "## Сбор",
        "",
        "| Набор | Статус | Метки | Матчи | Секунды |",
        "|---|---|---:|---:|---:|",
    ]
    for split in ("dev", "cycle-01", "cycle-02"):
        path = out / split / "collection.json"
        if path.exists():
            row = read_json(path)
            lines.append(
                f"| {split} | {row['status']} | {row['labels']} | {row['matches']} | {row['seconds']:.1f} |"
            )
    lines += [
        "",
        "## Обучение и выбор",
        "",
        f"Фактически выполнено {state['steps_performed']} Adam steps, {state['training_seconds']:.2f} с работы updates. Шаг содержит 512 старых и 512 новых свободных меток. Value head frozen.",
        "",
        "| Цикл / шаг | Old CE | Own-state dev CE | Composite | Adam counter |",
        "|---|---:|---:|---:|---:|",
    ]
    comparisons, selected_replays, all_measurements = {}, [], []
    for cycle in (1, 2):
        directory = out / f"cycle-{cycle:02}"
        if (directory / "training.json").exists():
            training = read_json(directory / "training.json")
            for measurement in training["measurements"]:
                all_measurements.append((cycle, measurement))
                lines.append(
                    f"| {cycle}/{measurement['cycle_step']} | {measurement['old_holdout']['ce']:.6f} | {measurement['own_state_dev']['ce']:.6f} | {measurement['score']:.6f} | {measurement['Adam_counter']} |"
                )
    if all_measurements:
        initial = read_json(out / "initial-metrics.json")
        chosen = (
            initial
            if state["best"]["cycle"] == 0
            else next(
                measurement
                for _, measurement in all_measurements
                if measurement.get("checkpoint") == state["best"]["path"]
            )
        )
        lines += [
            "",
            "### Accuracy выбранной модели",
            "",
            "| Dev | Тип | Initial accuracy | Selected accuracy |",
            "|---|---|---:|---:|",
        ]
        for dataset in ("old_holdout", "own_state_dev"):
            before, after = initial[dataset], chosen[dataset]
            lines.append(
                f"| {dataset} | all | {100 * before['accuracy']:.2f}% | {100 * after['accuracy']:.2f}% |"
            )
            for kind in sorted(before["by_kind"].keys() | after["by_kind"].keys()):
                aa, bb = (
                    before["by_kind"].get(kind, {}).get("accuracy"),
                    after["by_kind"].get(kind, {}).get("accuracy"),
                )
                lines.append(f"| {dataset} | {kind} | {100 * aa:.2f}% | {100 * bb:.2f}% |")
            for detail in ("type", "type_unit", "exact"):
                lines.append(
                    f"| {dataset} | move {detail} | {100 * before['move'][detail]:.2f}% | {100 * after['move'][detail]:.2f}% |"
                )
        plt = pyplot()

        figure, axes = plt.subplots(1, 2, figsize=(11, 4))
        for dataset, label in (
            ("old_holdout", "Old holdout"),
            ("own_state_dev", "Own-state dev"),
        ):
            for cycle_number in (1, 2):
                points = [measurement for cycle, measurement in all_measurements if cycle == cycle_number]
                if points:
                    axes[0].plot(
                        [measurement["steps_performed"] for measurement in points],
                        [measurement[dataset]["ce"] for measurement in points],
                        marker="o",
                        label=f"{label}, cycle {cycle_number}",
                    )
                    axes[1].plot(
                        [measurement["steps_performed"] for measurement in points],
                        [100 * measurement[dataset]["accuracy"] for measurement in points],
                        marker="o",
                        label=f"{label}, cycle {cycle_number}",
                    )
        for ax, ylabel in zip(axes, ("CE", "Accuracy, %")):
            ax.set_xlabel("Actually performed Adam steps")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(out / "learning-curves.png", dpi=150)
        plt.close(figure)
        lines += ["", "![Learning curves](learning-curves.png)"]
    lines += [
        "",
        "## Боевая сила",
        "",
        "| Цикл | Режим | Противник | Baseline W/D/L | Candidate W/D/L | Прирост п.п. | 95% CI п.п. |",
        "|---|---|---|---|---|---:|---|",
    ]
    baseline = read_json(parent / "evaluations.json")["runs"]
    for cycle in (1, 2):
        path = out / f"cycle-{cycle:02}" / "evaluation.json"
        if not path.exists():
            continue
        evaluated = read_json(path)
        for mode, run in evaluated["runs"].items():
            for enemy, directory in run["directories"].items():
                previous = baseline[mode]
                comparison = _compare_battles(config, previous["directories"][enemy], directory)
                comparisons[f"cycle-{cycle}-{mode}-{enemy}"] = comparison
                before, after = previous["opponents"][enemy], run["opponents"][enemy]
                low, high = comparison["CI95"]
                lines.append(
                    f"| {cycle} | {mode} | {enemy} | {format_wdl(before)} | {format_wdl(after)} | {100 * comparison['delta']:.2f} | [{100 * low:.2f}; {100 * high:.2f}] |"
                )
                if not run.get("reused"):
                    metrics = [(p, read_json(p)) for p in sorted((ROOT / directory).glob("*/metrics.json"))]
                    for outcome in ("W", "D", "L"):
                        item = next(
                            (
                                (p, row)
                                for p, row in metrics
                                if (
                                    "D"
                                    if row["winner"] is None
                                    else "W"
                                    if row["winner"] == row["side"]
                                    else "L"
                                )
                                == outcome
                            ),
                            None,
                        )
                        if item:
                            selected_replays.append(item[0].parent)
    write_json(out / "battle-comparisons.json", comparisons)
    for split in ("dev", "cycle-01", "cycle-02"):
        choices = sorted((out / split).glob("match-*/metrics.json"))
        if choices:
            selected_replays.append(choices[0].parent)
            disagreement = next((p.parent for p in choices if read_json(p)["disagreements"]), None)
            if disagreement:
                selected_replays.append(disagreement)
    checks = [
        verify_replay(path, demonstration=(path / "demonstration.json.gz").exists())
        for path in dict.fromkeys(selected_replays)
    ]
    write_json(out / "replay-verification.json", checks)
    lines += [
        "",
        "## Итог",
        "",
        f"Выбранная модель: `{state['best']['path'] if state['best'] else 'обучение не начиналось'}`.",
        f"Статус выполнения: `{read_json(out / 'status.json')['status'] if (out / 'status.json').exists() else 'in_progress'}`.",
        "Положительный результат требует прироста sampling win rate против CP и нижней границы CI >0. Если CI пересекает ноль, результат неопределённый. Argmax и greedy — дополнительные результаты.",
        "",
        f"Полностью воспроизведено {len(checks)} характерных боёв с проверкой результатов команд, публичных входов и хешей состояний. Детальные метрики типов/move, времён, расхождений и покрытия находятся в JSON соответствующих фаз.",
    ]
    lines += ["", "## Примеры сохранённых боёв", ""]
    for path in list(dict.fromkeys(selected_replays))[:12]:
        result = read_json(path / "result.json")
        case = read_json(path / "case.json")
        replay = read_json(path / "replay.json")
        side = case["learner_side"]
        active, actions = 0, []
        for step in replay["steps"]:
            action = step["action"]
            if active == side:
                actions.append(f"{action['kind']}({action['unit']},{action['target']})")
            if action["kind"] == "end_turn":
                active = 1 - active
        outcome = "draw" if result["winner"] is None else "win" if result["winner"] == side else "loss"
        lines.append(
            f"- `{relative(path)}`: {outcome}, {result['completed_rounds']} раундов, HP {result['remaining_hp']}. Первые команды learner: {' → '.join(actions[:10])}."
        )
        if (path / "demonstration.json.gz").exists():
            ep = load_episode(path / "demonstration.json.gz")
            differing = next(
                (
                    (i, row)
                    for i, row in enumerate(ep["decisions"])
                    if row["teacher_action"] != row["executed_action"]
                ),
                None,
            )
            if differing:
                i, row = differing
                lines.append(
                    f"  Решение {i}: CP советовал `{row['actions'][row['teacher_action']]}`, сеть исполнила `{row['actions'][row['executed_action']]}`. Это наблюдаемое расхождение, не доказательство причин исхода."
                )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    verify_checkpoint(state)
    write_json(
        out / "verification.json",
        {
            "replays": len(checks),
            "selected_Adam_counter_verified": bool(state["best"]),
            "source_hashes": read_json(out / "passport.json")["source_hashes"],
            "full_suite": False,
            "resume_tests": False,
        },
    )
    hashes = {
        relative(p): file_sha256(p)
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "artifact-hashes.json" and not p.name.endswith(".tmp")
    }
    write_json(out / "artifact-hashes.json", hashes)
