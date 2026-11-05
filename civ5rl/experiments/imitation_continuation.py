"""Bounded, explicitly parented continuation of the fixed CP imitation experiment."""

import json
import random
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from civ5rl.engine.imitation import (
    DemonstrationCache,
    backward_minibatch,
    label_minibatches,
    teacher_metrics,
)
from civ5rl.engine.resources import ResourceMonitor, analyze_telemetry
from civ5rl.engine.training import FrozenParamsCheck, bc_optimizer_step
from civ5rl.experiments.evaluation import run_validation_cases, verify_replay
from civ5rl.experiments.imitation_helpers import (
    CompilationClock,
    assert_equal,
    collect_summary,
    isolated_rng,
    restore_rng,
    update_selection,
)
from civ5rl.protocols.imitation import prepare_manifest
from civ5rl.protocols.imitation_invariants import (
    CONTINUATION_BUDGET,
    CONTINUATION_CHANGES,
    require_changes,
    require_values,
)
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import ROOT, file_sha256, read_json, write_json
from civ5rl.provenance.lineage import current_parent_lineage, verify_existing_run


def initialize(config, out):
    settings = config["continuation"]
    parent = ROOT / settings["parent_run"]
    parent_config = read_json(parent / "resolved-config.json")
    require_changes(config, parent_config, CONTINUATION_CHANGES)
    require_values(config, CONTINUATION_BUDGET, "continuation")
    path = parent / settings["parent_checkpoint"]
    if file_sha256(path) != settings["parent_checkpoint_sha256"]:
        raise ValueError("Parent checkpoint hash mismatch")
    if (
        file_sha256(parent / "training-inputs.json")
        != settings["parent_training_inputs_sha256"]
    ):
        raise ValueError("Parent training-inputs hash mismatch")
    manifest = prepare_manifest(config)
    if manifest != read_json(parent / "manifest.json"):
        raise ValueError("Continuation corpus differs from parent")
    allowed = {
        "tools/run_cp_imitation.py",
        "civ5rl/protocols/imitation.py",
        "civ5rl/engine/imitation_continuation.py",
        "configs/rl-cp-imitation-continuation.json",
    }
    current, changes = current_parent_lineage(
        parent, config, manifest, allowed, same_corpus=True
    )
    if (out / "passport.json").exists():
        identity = verify_existing_run(out, config, current)
    else:
        identity = checkpoint.passport(out, config, manifest)
        write_json(out / "manifest.json", manifest)
        write_json(
            out / "lineage.json",
            {
                "parent_checkpoint": str(path.relative_to(ROOT)),
                "parent_checkpoint_sha256": file_sha256(path),
                "parent_training_inputs_sha256": file_sha256(
                    parent / "training-inputs.json"
                ),
                "parent_passport_sha256": file_sha256(parent / "passport.json"),
                "changed_execution_sources": changes,
                "parent_epochs": 3,
                "parent_Adam_steps": 60,
                "new_optimizer_work_limit_seconds": 1800,
                "total_epoch_limit": 100,
            },
        )
    return parent, manifest, identity


def evaluate_candidate(config, manifest, out, parent, candidate, workers, mode, label):
    path = Path(candidate["path"])
    key = file_sha256(path) + ":" + mode
    recorded = (
        read_json(out / "evaluations.json")
        if (out / "evaluations.json").exists()
        else {"runs": {}, "labels": {}}
    )
    if key in recorded["runs"]:
        recorded["labels"][label] = key
        write_json(out / "evaluations.json", recorded)
        return recorded["runs"][key]
    started = perf_counter()
    if candidate["epoch"] == 3 and mode == "sample":
        directories = {
            enemy: parent / "evaluation" / ("final-" + enemy)
            for enemy in ("cp", "greedy")
        }
        new_matches = 0
    else:
        base = out / "evaluation" / mode / f"epoch-{candidate['epoch']:03}"
        directories = {enemy: base / enemy for enemy in ("cp", "greedy")}
        new_matches = run_validation_cases(
            config, manifest, path.resolve(), mode, directories, workers
        )
    result = {
        "epoch": candidate["epoch"],
        "checkpoint_sha256": file_sha256(path),
        "mode": mode,
        "new_matches": new_matches,
        "wall_seconds": perf_counter() - started,
        "directories": {k: str(v.relative_to(ROOT)) for k, v in directories.items()},
        "opponents": {
            enemy: collect_summary(directory)
            for enemy, directory in directories.items()
        },
    }
    recorded["runs"][key], recorded["labels"][label] = result, key
    write_json(out / "evaluations.json", recorded)
    print(json.dumps({"evaluation": label, **result}), flush=True)
    return result


def train(config, out, parent, manifest, identity, workers, device):
    from civ5rl.experiments.imitation_helpers import adam, demonstrations, load_policy
    from civ5rl.provenance.common import configure

    if (out / "training.json").exists():
        raise ValueError(
            "Continuation training already recorded; do not repeat optimization"
        )
    if device != "cuda":
        raise ValueError("This continuation protocol requires CUDA/BF16")
    configure(17, device)
    settings = config["continuation"]
    preparation = perf_counter()
    episodes, train_files = demonstrations(config, manifest, parent, "training")
    holdout, holdout_files = demonstrations(config, manifest, parent, "holdout")
    inputs = {"training": train_files, "holdout": holdout_files}
    if inputs != read_json(parent / "training-inputs.json"):
        raise ValueError("Continuation demonstration set changed")
    counts = {
        split: sum(r["supervised"] for ep in data for r in ep["decisions"])
        for split, data in [("training", episodes), ("holdout", holdout)]
    }
    if counts != {"training": 20018, "holdout": 2485}:
        raise ValueError(f"Unexpected continuation label counts: {counts}")
    write_json(out / "training-inputs.json", inputs)
    subset, subset_files, subset_labels = [], {}, 0
    for ep, (name, sha) in zip(episodes, train_files.items()):
        subset.append(ep)
        subset_files[name] = sha
        subset_labels += sum(r["supervised"] for r in ep["decisions"])
        if subset_labels >= settings["fixed_training_labels_minimum"]:
            break
    write_json(
        out / "fixed-training-subset.json",
        {"files": subset_files, "labels": subset_labels, "episodes": len(subset)},
    )
    source = parent / settings["parent_checkpoint"]
    saved = torch.load(source, weights_only=False, map_location="cpu")
    if saved["config"] != read_json(
        parent / "resolved-config.json"
    ) or not checkpoint.same_execution_identity(
        saved["identity"], read_json(parent / "passport.json")
    ):
        raise ValueError("Parent checkpoint does not match parent experiment identity")
    if saved["state"]["epoch"] != 3 or saved["state"]["Adam_steps"] != 60:
        raise ValueError("Unexpected parent optimization boundary")
    policy = load_policy(config, source, device)
    optimizer = adam(policy, config, device)
    optimizer.load_state_dict(saved["optimizer"])
    shuffle = random.Random()
    cache, dev_cache, fixed_cache = [
        DemonstrationCache(data, device) for data in (episodes, holdout, subset)
    ]
    frozen_value = FrozenParamsCheck(policy.value)
    clock = CompilationClock()

    preparation_seconds = perf_counter() - preparation
    warm_start = perf_counter()
    with clock, ResourceMonitor(out / "telemetry") as monitor:
        clock.monitor = monitor
        monitor.mark("warmup")
        # Compile representative gradient and no-grad paths without an optimizer step.
        spans, count = next(label_minibatches(cache.chunks(), cache.masks))
        optimizer.zero_grad(set_to_none=True)
        backward_minibatch(policy, cache, spans, count)
        optimizer.zero_grad(set_to_none=True)
        parent_metrics = {
            "holdout": teacher_metrics(policy, dev_cache),
            "fixed_training": teacher_metrics(policy, fixed_cache),
        }
        torch.cuda.synchronize()
        warmup_seconds = perf_counter() - warm_start
        restore_rng(saved, shuffle)
        assert_equal(policy.state_dict(), saved["model"])
        assert_equal(optimizer.state_dict(), saved["optimizer"])
        assert_equal(random.getstate(), saved["python_rng"])
        assert_equal(np.random.get_state(), saved["numpy_rng"])
        assert_equal(torch.get_rng_state(), saved["torch_rng"])
        assert_equal(torch.cuda.get_rng_state_all(), saved["cuda_rng"])
        assert_equal(shuffle.getstate(), saved["shuffle_rng"])
        write_json(
            out / "initialization-check.json",
            {
                "model_bitwise_equal": True,
                "Adam_moments_and_groups_equal": True,
                "all_RNG_and_shuffle_equal": True,
                "Adam_steps_before": 60,
                "warmup_Adam_steps": 0,
                "parent_checkpoint_sha256": file_sha256(source),
                "parent_metrics": parent_metrics,
            },
        )
        historical_ce = read_json(parent / "training.json")["epochs"][-1]["holdout"][
            "ce"
        ]
        state = {
            "best_epoch": 3,
            "best_ce": historical_ce,
            "anchor_ce": historical_ce,
            "plateau_count": 0,
            "plateau": False,
        }
        best = {"epoch": 3, "ce": historical_ce, "path": str(source.resolve())}
        history, metrics, steps, elapsed, evaluation_seconds = (
            [],
            [{"epoch": 3, **parent_metrics}],
            60,
            0.0,
            0.0,
        )
        reason, last_complete = None, 3
        utilization_justification = None
        checkpoint_dir = out / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)
        shutil.copy2(source, checkpoint_dir / "best.pt")

        def persist():
            write_json(
                out / "training.json",
                {
                    "status": "running" if reason is None else "complete",
                    "stop_reason": reason,
                    "epochs": history,
                    "offline_metrics": metrics,
                    "selection": state,
                    "best": best,
                    "last_completed_epoch": last_complete,
                    "Adam_steps_total": steps,
                    "new_Adam_steps": steps - 60,
                    "new_optimizer_seconds": elapsed,
                    "preparation_seconds": preparation_seconds,
                    "warmup_and_parent_metrics_seconds": warmup_seconds,
                    "compile_seconds": sum(b - a for a, b in clock.intervals),
                    "battle_evaluation_seconds": evaluation_seconds,
                    "GPU_below_target_justification": utilization_justification,
                },
            )

        def measure(epoch, candidate_path):
            nonlocal state, best
            monitor.mark("offline_metrics", epoch=epoch)
            start = perf_counter()
            with isolated_rng(shuffle):
                result = {
                    "epoch": epoch,
                    "holdout": teacher_metrics(policy, dev_cache),
                    "fixed_training": teacher_metrics(policy, fixed_cache),
                }
            result["wall_seconds"] = perf_counter() - start
            metrics.append(result)
            state = update_selection(state, epoch, result["holdout"]["ce"], settings)
            if state["best_epoch"] == epoch:
                best = {
                    "epoch": epoch,
                    "ce": state["best_ce"],
                    "path": str(candidate_path.resolve()),
                }
                shutil.copy2(candidate_path, checkpoint_dir / "best.pt")

        for epoch in range(4, config["epochs"] + 1):
            if elapsed >= config["training_seconds"]:
                reason = "optimizer_time_limit"
                break
            monitor.mark("iterative_training", epoch=epoch, Adam_steps=steps)
            order = cache.chunks()
            shuffle.shuffle(order)
            records, loss_sum, labels, complete = [], 0.0, 0, True
            for spans, count in label_minibatches(order, cache.masks):
                if elapsed >= config["training_seconds"]:
                    complete = False
                    break
                torch.cuda.synchronize()
                start = perf_counter()
                ce, norm = bc_optimizer_step(
                    policy,
                    optimizer,
                    backward_minibatch,
                    cache,
                    spans,
                    count,
                    max_gradient_norm=config["max_gradient_norm"],
                )
                torch.cuda.synchronize()
                stop = perf_counter()
                compile_seconds = clock.seconds(start, stop)
                elapsed += stop - start - compile_seconds
                steps += 1
                if elapsed >= 600 and utilization_justification is None:
                    recent_windows = analyze_telemetry(
                        out / "telemetry/resources.jsonl"
                    )["windows"]
                    utilization_justification = {
                        "reason": "Fixed-condition continuation: CPU batch assembly, synchronous host/device checks and small spatial blocks constrain throughput; do not change minibatch or numerics mid-experiment.",
                        "investigation": "Inspected DemonstrationCache.batch, backward_minibatch and fixed spatial 32; candidate optimizations are resident inputs and fewer host synchronizations, deferred to preserve this experiment.",
                        "optimizer_seconds_at_review": elapsed,
                        "last_60_seconds_core_average": recent_windows[-1][
                            "core_average"
                        ]
                        if recent_windows
                        else None,
                    }
                    print(
                        json.dumps(
                            {"GPU_utilization_review": utilization_justification}
                        ),
                        flush=True,
                    )
                if steps == 61:
                    assert {int(v["step"]) for v in optimizer.state.values()} == {61}
                    write_json(
                        out / "first-step-check.json",
                        {
                            "Adam_steps_before": 60,
                            "Adam_steps_after": 61,
                            "all_Adam_counters_equal": True,
                            "labels": count,
                            "spans": spans,
                            "pre_step_state": "initialization-check.json",
                            "ce": ce,
                        },
                    )
                records.append(
                    {
                        "Adam_step": steps,
                        "labels": count,
                        "ce": ce,
                        "gradient_norm_pre_clip": float(norm),
                        "wall_seconds": stop - start,
                        "compile_seconds": compile_seconds,
                    }
                )
                loss_sum += ce * count
                labels += count
            frozen_value.verify()
            row = {
                "epoch": epoch,
                "completed": complete,
                "labels": labels,
                "training_ce": loss_sum / labels if labels else None,
                "Adam_steps_total": steps,
                "optimizer_steps": records,
            }
            history.append(row)
            if not complete:
                checkpoint.save(
                    checkpoint_dir / "partial.pt",
                    policy,
                    optimizer,
                    {
                        "epoch": last_complete,
                        "partial_epoch": epoch,
                        "Adam_steps": steps,
                    },
                    shuffle,
                    config,
                    identity,
                )
                reason = "optimizer_time_limit"
                break
            last_complete = epoch
            monitor.mark("checkpoint", epoch=epoch)
            checkpoint.save(
                checkpoint_dir / "last-complete.pt",
                policy,
                optimizer,
                {"epoch": epoch, "Adam_steps": steps, "new_optimizer_seconds": elapsed},
                shuffle,
                config,
                identity,
            )
            if epoch % settings["holdout_interval"] == 0:
                candidate_path = checkpoint_dir / f"epoch-{epoch:03}.pt"
                shutil.copy2(checkpoint_dir / "last-complete.pt", candidate_path)
                measure(epoch, candidate_path)
            persist()
            print(
                json.dumps(
                    {
                        "epoch": epoch,
                        "Adam_steps": steps,
                        "new_optimizer_seconds": elapsed,
                        "training_ce": row["training_ce"],
                        "best_epoch": best["epoch"],
                        "best_holdout_ce": best["ce"],
                        "plateau_count": state["plateau_count"],
                    }
                ),
                flush=True,
            )
            if epoch == settings["battle_epoch"]:
                monitor.mark("battle_evaluation", epoch=epoch)
                with isolated_rng(shuffle):
                    result = evaluate_candidate(
                        config,
                        manifest,
                        out,
                        parent,
                        best,
                        workers,
                        "sample",
                        "epoch30",
                    )
                evaluation_seconds += result["wall_seconds"]
            if state["plateau"]:
                reason = "holdout_plateau"
                break
            if elapsed >= config["training_seconds"]:
                reason = "optimizer_time_limit"
                break
        if reason is None:
            reason = "epoch_limit"
        if metrics[-1]["epoch"] != last_complete and last_complete > 3:
            # A partial next epoch must not contaminate measurements of the last full one.
            last = torch.load(
                checkpoint_dir / "last-complete.pt",
                weights_only=False,
                map_location="cpu",
            )
            policy.load_state_dict(last["model"])
            candidate_path = checkpoint_dir / f"epoch-{last_complete:03}.pt"
            shutil.copy2(checkpoint_dir / "last-complete.pt", candidate_path)
            measure(last_complete, candidate_path)
        # Freeze final selection before any final battle results are available.
        best["path"] = str((checkpoint_dir / "best.pt").resolve())
        best["checkpoint_sha256"] = file_sha256(checkpoint_dir / "best.pt")
        persist()


def evaluate(config, out, parent, manifest, workers):
    training = read_json(out / "training.json")
    if training["status"] != "complete":
        raise ValueError("Cannot evaluate an unfinished continuation")
    best = training["best"]
    if file_sha256(best["path"]) != best["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed after selection")
    evaluate_candidate(
        config, manifest, out, parent, best, workers, "sample", "final_sample"
    )
    if config["continuation"]["final_argmax"]:
        evaluate_candidate(
            config, manifest, out, parent, best, workers, "argmax", "final_argmax"
        )


def run(phase, config, out, workers, device):
    if phase not in ("prepare", "train", "evaluate", "report"):
        raise ValueError("Continuation does not allow collection or benchmark")
    parent, manifest, identity = initialize(config, out)
    started = perf_counter()
    try:
        if phase == "train":
            train(config, out, parent, manifest, identity, workers, device)
        elif phase == "evaluate":
            evaluate(config, out, parent, manifest, workers)
        elif phase == "report":
            report(config, out, parent)
        write_json(
            out / f"{phase}-execution.json",
            {"wall_seconds": perf_counter() - started, "status": "complete"},
        )
    except Exception as exc:
        write_json(
            out / "status.json",
            {
                "status": "technical_error",
                "phase": phase,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise


def report(config, out, parent):
    training = read_json(out / "training.json")
    evaluation = read_json(out / "evaluations.json")
    if (
        "final_sample" not in evaluation["labels"]
        or "final_argmax" not in evaluation["labels"]
    ):
        raise ValueError("Missing final evaluation")
    initial_metrics = training["offline_metrics"][0]
    chosen = next(
        m
        for m in training["offline_metrics"]
        if m["epoch"] == training["best"]["epoch"]
    )
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
            values.append(
                f"{entry['ce']:.4f} / {100 * entry['accuracy']:.2f}%" if entry else "—"
            )
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
        enemy: collect_summary(parent / "evaluation" / ("final-" + enemy))
        for enemy in ("cp", "greedy")
    }
    for enemy, row in baselines.items():
        kinds = row["actions"]
        lines.append(
            f"| epoch-03 / sample | {enemy} | {row['W']}/{row['D']}/{row['L']} | {100 * row['win_rate']:.3f}% | — | "
            + " / ".join(
                str(kinds.get(k, 0)) for k in ("move", "ranged", "melee", "skip")
            )
            + " |"
        )
    for label, key in evaluation["labels"].items():
        result = evaluation["runs"][key]
        for enemy, row in result["opponents"].items():
            kinds = row["actions"]
            lines.append(
                f"| {label}: epoch-{result['epoch']:02} / {result['mode']} | {enemy} | {row['W']}/{row['D']}/{row['L']} | {100 * row['win_rate']:.3f}% | {100 * (row['win_rate'] - baselines[enemy]['win_rate']):+.3f} п.п. | "
                + " / ".join(
                    str(kinds.get(k, 0)) for k in ("move", "ranged", "melee", "skip")
                )
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
                verification.update(
                    label=label, enemy=enemy, outcome=desired, rounds=m["rounds"]
                )
                available = verification["attack_available_decisions"]
                selected = verification["attacks_chosen"]
                omitted = verification["non_attack_when_available"]
                verified.append(verification)
                lines.append(
                    f"- {label} / {enemy}, {chosen_path.parent.name}: {desired}, {m['rounds']} раундов; {verification['commands']} команд, все результаты и state hashes совпали; атак выбрано {selected}, решений с атакой {available}, иных команд при доступной атаке {omitted}."
                )
    write_json(out / "replay-verification.json", verified)
    saved = torch.load(
        out / "checkpoints/best.pt", weights_only=False, map_location="cpu"
    )
    parent_saved = torch.load(
        parent / config["continuation"]["parent_checkpoint"],
        weights_only=False,
        map_location="cpu",
    )
    for key, value in parent_saved["model"].items():
        if key.startswith("value."):
            assert_equal(value, saved["model"][key])
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
        lines.append(
            f"- {field}: mean={stats['mean']:.3f}, max={stats['max']:.3f} {stats['unit']}"
        )
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
        "Причины низкой GPU-загрузки по коду: CPU-сборка NumPy minibatches, копирования CPU→GPU и синхронизации `.item()`/`.any()` при небольшом spatial block 32. Это кандидаты оптимизации, не доказанный profiler-разбор. Условия опыта ради процента загрузки не менялись.",
        "",
        "Проверки: initialization-check.json подтверждает побитовое восстановление parameters/Adam/RNG; first-step-check.json подтверждает 60→61. Value head неизменна. Целевые существующие BC-проверки и несколько числовых примеров selection выполнены перед запуском; full suite и resume-тесты не запускались.",
        "",
        "Этот опыт проверяет объём оптимизации BC на фиксированных данных. Даже улучшение argmax не доказывает освоения тактики. Новых seeds, PPO, данных и закрытых test не было; бюджет не продлевался.",
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
            "new_evaluation_matches": sum(
                r["new_matches"] for r in evaluation["runs"].values()
            ),
            "closed_test_evaluated": False,
        },
    )
