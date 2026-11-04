"""Prepare and execute the preregistered bounded D0 -> D1 protocol."""

import argparse
import json
import os
import random
import sys
import tempfile
import traceback
from copy import deepcopy
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def evaluate_stage(policy, collector, dataset, stage, directory, monitor):
    from civ5rl.common import read_json, write_json
    from civ5rl.diagnostic import episode_rows, outcome_summary
    from civ5rl.provenance import bound_cases

    directory = Path(directory)
    manifest = dataset[stage]["dev"]
    if (directory / "summary.json").exists():
        summary, rows = (
            read_json(directory / "summary.json"),
            read_json(directory / "outcomes.json"),
        )
        if (
            summary.get("manifest_hash") != manifest["hash"]
            or len(rows) != 128
            or len({(r["layout_id"], r["side"]) for r in rows}) != 128
        ):
            raise ValueError("Cached evaluation manifest/completeness mismatch")
        return summary, rows
    start = perf_counter()
    monitor.mark("diagnostic_evaluation", diagnostic_stage=stage)
    episodes, _, timings = collector.collect(
        policy, None, 0, fixed_specs=bound_cases(manifest), trusted_manifest=manifest
    )
    if len(episodes) != 128:
        raise ValueError("Incomplete diagnostic evaluation")
    summary = outcome_summary(episodes)
    rows = episode_rows(episodes)
    summary.update(
        seconds=perf_counter() - start, timings=timings, manifest_hash=manifest["hash"]
    )
    from civ5rl.evaluation import save_training_replay

    selected = []
    for reward in (1, 0, -1):
        episode = next(
            (ep for ep in episodes if ep["result"]["reward"] == reward), None
        )
        if episode is not None:
            save_training_replay(episode, directory / "replays" / str(reward))
            selected.append(episode["spec"]["episode_id"])
    summary["saved_episode_ids"] = selected
    write_json(directory / "outcomes.json", rows)
    write_json(directory / "summary.json", summary)  # completion marker committed last
    return summary, rows


def run_seed(base_config, dataset, directory, seed, device="cuda", resume=False):
    import torch

    from civ5rl import checkpoint
    from civ5rl.collector import Collector
    from civ5rl.common import configure, read_json, write_json
    from civ5rl.control import TrainingControl
    from civ5rl.diagnostic import (
        DiagnosticFactory,
        gate,
        may_start_update,
        next_state,
        outcome_summary,
    )
    from civ5rl.diagnostic_one import (
        gate_one,
        is_one,
        load_initial_weights,
        next_state_one,
    )
    from civ5rl.model import Policy
    from civ5rl.ppo import update
    from civ5rl.resources import ResourceMonitor
    from civ5rl.validation import validate_config

    directory = Path(directory)
    config = deepcopy(base_config)
    config["seed"] = seed
    validate_config(config)
    one = is_one(config)
    if (directory / "latest.json").exists() and not resume:
        raise ValueError("Run exists; use --resume")
    configure(seed, device)
    with TrainingControl(directory) as control:
        if (directory / "passport.json").exists():
            identity = read_json(directory / "passport.json")
            with tempfile.TemporaryDirectory() as tmp:
                current = checkpoint.passport(tmp, config, dataset)
            if not checkpoint.same_execution_identity(identity, current):
                raise ValueError("Diagnostic resume source/environment mismatch")
        else:
            identity = checkpoint.passport(directory, config, dataset)
        policy = Policy("C").to(device)
        initialization = load_initial_weights(policy, config) if one else None
        policy.enable_amp(device.startswith("cuda"), spatial_batch_size=32)
        if device.startswith("cuda"):
            policy.enable_compile()
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=config["optimizer"]["lr_schedule"]["initial"],
            betas=tuple(config["optimizer"]["betas"]),
            eps=config["optimizer"]["eps"],
            weight_decay=config["optimizer"]["weight_decay"],
            fused=device.startswith("cuda"),
        )
        shuffle = random.Random(seed + 1)
        if (
            initialization is not None
            and not (directory / "initialization-verification.json").exists()
        ):
            initialization.update(
                new_optimizer_state_entries=len(optimizer.state),
                training_seed=seed,
                counters_reset=True,
                episode_memory_initial_state="zero",
            )
            write_json(directory / "initialization-verification.json", initialization)
        state = {
            "update": 0,
            "decisions": 0,
            "next_episode": 0,
            "stage": "D0",
            "stage_updates": 0,
            "consecutive": 0,
            "training_seconds": 0.0,
            "evaluation_seconds": 0.0,
            "status": "running",
            "pending_evaluation": None,
        }
        if (directory / "latest.json").exists():
            state = checkpoint.restore(
                directory
                / "checkpoints"
                / read_json(directory / "latest.json")["checkpoint"],
                policy,
                optimizer,
                shuffle,
                config,
                identity,
            )
        if "last_metrics" in state:
            metric_path = directory / "metrics.jsonl"
            present = (
                [
                    json.loads(line)["update"]
                    for line in metric_path.read_text().splitlines()
                ]
                if metric_path.exists()
                else []
            )
            if state["last_metrics"]["update"] not in present:
                with metric_path.open("a", encoding="utf-8") as stream:
                    stream.write(
                        json.dumps(state["last_metrics"], allow_nan=False) + "\n"
                    )

        def save(name):
            checkpoint.save(
                directory / "checkpoints" / name,
                policy,
                optimizer,
                state.copy(),
                shuffle,
                config,
                identity,
            )
            write_json(directory / "latest.json", {"checkpoint": name, **state})

        if not (directory / "latest.json").exists():
            save("initial.pt")
        write_json(
            directory / "execution.json",
            {
                "kind": "diagnostic",
                "suite_directory": str(directory.parent.resolve()),
                "budget": None,
                "device": device,
                "variant": "C",
            },
        )
        factory = DiagnosticFactory(dataset, seed)
        try:
            with (
                ResourceMonitor(directory) as monitor,
                Collector(config, factory, device, directory / "errors") as collector,
            ):
                # Baselines always use the immutable initial model, including on resume.
                original = deepcopy(policy.state_dict())
                policy.load_state_dict(
                    torch.load(
                        directory / "checkpoints/initial.pt",
                        map_location=device,
                        weights_only=False,
                    )["model"]
                )
                baselines = {}
                for stage in ("D0",) if one else ("D0", "D1"):
                    baselines[stage] = evaluate_stage(
                        policy,
                        collector,
                        dataset,
                        stage,
                        directory / "evaluation" / ("initial-" + stage),
                        monitor,
                    )
                policy.load_state_dict(original)
                del original
                if not one and baselines["D1"][0]["wins"] > 102:
                    state.update(
                        status="invalid_dataset",
                        reason="D1_improvement_mathematically_impossible",
                    )
                    save(read_json(directory / "latest.json")["checkpoint"])
                while state["status"] == "running":
                    if state["pending_evaluation"] is not None:
                        pending = state["pending_evaluation"]
                        control.mark(
                            "diagnostic_evaluation",
                            update=state["update"],
                            stage=pending["stage"],
                        )
                        summary, rows = evaluate_stage(
                            policy,
                            collector,
                            dataset,
                            pending["stage"],
                            directory
                            / "evaluation"
                            / f"update-{state['update']:06}-{pending['stage']}",
                            monitor,
                        )
                        decision = (
                            gate_one(summary)
                            if one
                            else gate(
                                pending["stage"], summary, rows, baselines["D1"][1]
                            )
                        )
                        summary["gate"] = decision
                        write_json(
                            directory
                            / "evaluation"
                            / f"update-{state['update']:06}-{pending['stage']}"
                            / "summary.json",
                            summary,
                        )
                        state["evaluation_seconds"] += summary["seconds"]
                        state = (
                            next_state_one(state, decision["passed"])
                            if one
                            else next_state(state, decision["passed"])
                        )
                        state["pending_evaluation"] = None
                        save(f"update-{state['update']:06}.pt")
                        print(
                            json.dumps(
                                {
                                    "seed": seed,
                                    "update": state["update"],
                                    "evaluated_stage": pending["stage"],
                                    "wins": summary["wins"],
                                    "first_attack": summary[
                                        "first_winning_attack_rate"
                                    ],
                                    "gate": decision,
                                    "status": state["status"],
                                }
                            ),
                            flush=True,
                        )
                        continue
                    if control.stopping():
                        break
                    if not may_start_update(state):
                        state.update(status="failed", reason="training_time_limit")
                        save(read_json(directory / "latest.json")["checkpoint"])
                        break
                    control.mark(
                        "collection",
                        update=state["update"] + 1,
                        stage=state["stage"],
                        decisions=state["decisions"],
                    )
                    monitor.mark(
                        "collection",
                        update=state["update"] + 1,
                        diagnostic_stage=state["stage"],
                    )
                    start = perf_counter()
                    episodes, next_episode, timings = collector.collect(
                        policy, state["stage"], state["next_episode"], monitor
                    )
                    if any(ep["result"]["reward"] not in (-1, 0, 1) for ep in episodes):
                        raise ValueError("Invalid terminal reward")
                    if one and any(
                        len(ep["chosen"]) != 1
                        or ep["result"]["learner_decisions"] != 1
                        or ep["old_hidden"] != [None]
                        for ep in episodes
                    ):
                        raise ValueError(
                            "One-decision rollout contains repeated decisions or carried memory"
                        )
                    before = {
                        k: v.detach().clone() for k, v in policy.state_dict().items()
                    }
                    control.mark(
                        "ppo", update=state["update"] + 1, stage=state["stage"]
                    )
                    monitor.mark("ppo")
                    try:
                        stats = update(
                            policy,
                            optimizer,
                            episodes,
                            config,
                            state["decisions"],
                            device,
                            shuffle,
                            monitor,
                        )
                    except BaseException:
                        (directory / "errors").mkdir(exist_ok=True)
                        torch.save(
                            episodes,
                            directory / "errors" / f"rollout-{state['update'] + 1}.pt",
                        )
                        raise
                    delta = (
                        sum(
                            (v - before[k]).double().square().sum()
                            for k, v in policy.state_dict().items()
                        )
                        .sqrt()
                        .item()
                    )
                    count = sum(len(ep["chosen"]) for ep in episodes)
                    stage = state["stage"]
                    elapsed = perf_counter() - start
                    state.update(
                        update=state["update"] + 1,
                        decisions=state["decisions"] + count,
                        next_episode=next_episode,
                        stage_updates=state["stage_updates"] + 1,
                        training_seconds=state["training_seconds"] + elapsed,
                        pending_evaluation={"stage": stage},
                    )
                    name = f"update-{state['update']:06}.pt"
                    row = {
                        "update": state["update"],
                        "stage": stage,
                        "stage_update": state["stage_updates"],
                        "decisions": state["decisions"],
                        "rollout_decisions": count,
                        "overflow": count - 8192,
                        "episodes": len(episodes),
                        "weight_delta_l2": delta,
                        "training_seconds": state["training_seconds"],
                        "outcomes": outcome_summary(episodes),
                        "timings": timings,
                        "ppo": stats,
                        "terminal_rewards": {
                            str(reward): sum(
                                ep["result"]["reward"] == reward for ep in episodes
                            )
                            for reward in (-1, 0, 1)
                        },
                    }
                    # Store diagnostics in checkpoint before evaluation for restart idempotence.
                    state["last_metrics"] = row
                    save(name)
                    state["training_seconds"] += perf_counter() - start - elapsed
                    row["training_seconds"] = state["training_seconds"]
                    save(name)
                    with (directory / "metrics.jsonl").open(
                        "a", encoding="utf-8"
                    ) as stream:
                        stream.write(json.dumps(row, allow_nan=False) + "\n")
                    del episodes, before
                paused = state["status"] == "running"
                result = {
                    **state,
                    "status": "paused" if paused else state["status"],
                    "seed": seed,
                }
                write_json(directory / "run-result.json", result)
                control.mark(
                    result["status"],
                    status=result["status"],
                    update=state["update"],
                    decisions=state["decisions"],
                )
                return result
        except BaseException:
            write_json(
                directory / "failure.json",
                {
                    "traceback": traceback.format_exc(),
                    "last_completed": read_json(directory / "latest.json"),
                },
            )
            raise


def report(directory):
    import matplotlib

    from civ5rl.common import read_json

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    from civ5rl.diagnostic_one import is_one, report_one

    saved_config = directory / "seed-17/resolved-config.json"
    if saved_config.exists() and is_one(read_json(saved_config)):
        return report_one(directory)
    lines = [
        "# Диагностический цикл D0 → D1",
        "",
        "Только терминальная награда; новая модель C на каждом seed. Закрытые test-наборы не оценивались.",
        "",
        "Критерии — `docs/rl-diagnostic-protocol.md`. Преимущество памяти, разведки и над greedy этим опытом не проверяется.",
        "",
    ]
    control_dir = directory / "previous-model-control"
    if control_dir.exists():
        lines += [
            "## Прежний checkpoint — дополнительный контроль",
            "",
            "| Ступень | W/D/L | Победы | Первый выстрел |",
            "|---|---:|---:|---:|",
        ]
        for stage in ("D0", "D1"):
            path = control_dir / stage / "summary.json"
            if path.exists():
                value = read_json(path)
                lines += [
                    f"| {stage} | {value['wins']}/{value['draws']}/{value['losses']} | {value['win_rate']:.3f} | {value['first_winning_attack_rate']:.3f} |"
                ]
        lines += [
            "",
            "Этот checkpoint использовался только для оценки; исходный пилот не продолжался.",
            "",
        ]
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), layout="constrained")
    for run in sorted(directory.glob("seed-*")):
        if not (run / "run-result.json").exists():
            continue
        result = read_json(run / "run-result.json")
        lines += [
            f"## {run.name}",
            "",
            f"Статус: **{result['status']}**; причина: `{result.get('reason', 'pause')}`.",
            f"Решений: {result['decisions']}; updates: {result['update']}; время обучения: {result['training_seconds']:.1f} с; post-update оценки: {result['evaluation_seconds']:.1f} с.",
            "",
            "| Оценка | W/D/L | Победы | Первый выстрел | P первого выстрела | Gate |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for path in sorted((run / "evaluation").glob("*/summary.json")):
            s = read_json(path)
            lines += [
                f"| {path.parent.name} | {s['wins']}/{s['draws']}/{s['losses']} | {s['win_rate']:.3f} | {s['first_winning_attack_rate']:.3f} | {s['first_winning_attack_probability']:.3f} | {s.get('gate', {}).get('passed', 'baseline')} |"
            ]
        metrics = (
            [json.loads(l) for l in (run / "metrics.jsonl").read_text().splitlines()]
            if (run / "metrics.jsonl").exists()
            else []
        )
        updates = []
        wins = []
        probs = []
        for path in sorted((run / "evaluation").glob("update-*/summary.json")):
            s = read_json(path)
            updates.append(int(path.parent.name.split("-")[1]))
            wins.append(s["win_rate"])
            probs.append(s["actions"]["attack"]["probability_when_available"])
        axes[0].plot(updates, wins, "o-", label=run.name)
        axes[1].plot(updates, probs, "o-", label=run.name)
        if metrics:
            lines += [
                "",
                f"Изменение весов L2 по updates: {', '.join(f'{r['weight_delta_l2']:.4g}' for r in metrics)}.",
                f"Максимальная норма градиента: {max(r['ppo']['gradient_norm_max'] for r in metrics):.4g}.",
                "Проверка behavior/replay, исходные вероятности, terminal reward и ненулевые градиенты записаны в metrics.jsonl.",
            ]
        if result["status"] != "passed":
            lines += [
                "",
                "Порог не пройден в заданном бюджете. Победы после ожидания и первый победный выстрел оцениваются отдельно; награда не различает время победы.",
                "Автоматического продолжения, смены награды и подбора параметров не выполнялось.",
            ]
        lines.append("")
    for ax, title in zip(axes, ("Dev win rate", "Attack probability when available")):
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        if ax.lines:
            ax.legend()
    axes[1].set_xlabel("PPO update")
    fig.savefig(directory / "progress.png", dpi=140)
    plt.close(fig)
    lines += [
        "![Прогресс](progress.png)",
        "",
        "Полные outcomes и характерные реплеи находятся в evaluation каждого запуска. Соперник D1 — диагностический attack-or-end-turn; его нельзя приравнивать к greedy.",
    ]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _run_suite(config, directory, device="cuda", resume=False, prepare_only=False):
    from civ5rl.common import write_json
    from civ5rl.diagnostic import prepare
    from civ5rl.diagnostic_one import is_one, prepare_one
    from civ5rl.validation import validate_config

    validate_config(config)
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    one = is_one(config)
    dataset = (
        prepare_one(directory / "manifests", config)
        if one
        else prepare(directory / "manifests", config)
    )
    if prepare_only:
        return
    try:
        # Historical model is a frozen control, not a training initialization.
        control_dir = directory / "previous-model-control"
        if not one and not (control_dir / "D1/summary.json").exists():
            from civ5rl.agent import RLAgent
            from civ5rl.collector import Collector
            from civ5rl.diagnostic import DiagnosticFactory
            from civ5rl.resources import ResourceMonitor

            agent = RLAgent.from_checkpoint(
                ROOT / "artifacts/rl/training-result/checkpoints/update-000188.pt",
                device,
            )
            if device.startswith("cuda"):
                agent.policy.enable_compile()
            with (
                ResourceMonitor(control_dir) as monitor,
                Collector(
                    config,
                    DiagnosticFactory(dataset, 17),
                    device,
                    control_dir / "errors",
                ) as collector,
            ):
                for stage in ("D0", "D1"):
                    evaluate_stage(
                        agent.policy,
                        collector,
                        dataset,
                        stage,
                        control_dir / stage,
                        monitor,
                    )
        results = []
        for seed in (17,) if one else (17, 42):
            result = run_seed(
                config, dataset, directory / f"seed-{seed}", seed, device, resume
            )
            results.append(
                {
                    "seed": seed,
                    "status": result["status"],
                    "decisions": result["decisions"],
                    "reason": result.get("reason"),
                }
            )
            write_json(
                directory / "suite-result.json",
                {
                    "status": "passed"
                    if len(results) == (1 if one else 2)
                    and all(r["status"] == "passed" for r in results)
                    else result["status"],
                    "seeds": results,
                    "closed_test_evaluated": False,
                },
            )
            report(directory)
            if result["status"] != "passed":
                break
    except BaseException:
        write_json(directory / "failure.json", {"traceback": traceback.format_exc()})
        report(directory)
        raise


def run_suite(config, directory, device="cuda", resume=False, prepare_only=False):
    from civ5rl.control import RunLock

    with RunLock(directory):
        return _run_suite(config, directory, device, resume, prepare_only)


def main():
    from civ5rl.common import read_json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rl-diagnostic.json")
    p.add_argument("--out")
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    args = p.parse_args()
    from civ5rl.diagnostic_one import is_one

    config = read_json(args.config)
    out = args.out or (
        "artifacts/rl/diagnostic-one-decision"
        if is_one(config)
        else "artifacts/rl/diagnostic-results"
    )
    run_suite(config, out, args.device, args.resume, args.prepare_only)


if __name__ == "__main__":
    main()
