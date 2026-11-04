"""Train/resume RL-01; pilot never opens final test or diagnostic-test."""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def run(config, manifest, directory, **kwargs):
    from civ5rl.control import TrainingControl

    with TrainingControl(directory) as control:
        return _run(config, manifest, directory, control=control, **kwargs)


def _run(
    config,
    manifest,
    directory,
    *,
    device="cuda",
    budget=32768,
    resume=None,
    evaluate_every=True,
    max_updates=None,
    variant="C",
    replays=True,
    stage_cycle=None,
    control=None,
):
    import torch

    from civ5rl import checkpoint
    from civ5rl.collector import Collector
    from civ5rl.common import configure, read_json, write_json
    from civ5rl.corpus import ScenarioFactory, validate_corpus
    from civ5rl.evaluation import (
        evaluate,
        save_replays,
        save_training_replay,
        summarize,
        validation_specs,
    )
    from civ5rl.model import Policy
    from civ5rl.ppo import update
    from civ5rl.resources import ResourceMonitor

    directory = Path(directory).resolve()
    from civ5rl.validation import validate_config
    validate_config(config)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "checkpoints").exists() and resume is None:
        raise ValueError("Run already exists; use --resume or a new --out")
    configure(config["seed"], device)
    validate_corpus(manifest)
    if resume == "latest":
        latest = directory / "latest.json"
        resume = (
            directory
            / "checkpoints"
            / (read_json(latest)["checkpoint"] if latest.exists() else "initial.pt")
        )
    if resume:
        identity = read_json(directory / "passport.json")
        # Verify source/environment without replacing the run's original passport.
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            current = checkpoint.passport(tmp, config, manifest)
        if not checkpoint.same_execution_identity(current, identity):
            raise ValueError("Environment/source changed since checkpoint")
    else:
        identity = checkpoint.passport(directory, config, manifest)
    policy = Policy(variant).to(device)
    performance = config.get("performance", {})
    policy.enable_amp(
        config["numerics"].get("amp", False) and device.startswith("cuda"),
        spatial_batch_size=performance.get("spatial_batch_size", 0),
    )
    if performance.get("compile", False) and device.startswith("cuda"):
        policy.enable_compile(
            dynamic=performance.get("compile_dynamic", True),
            mode=performance.get("compile_mode", "default"),
            recompile_limit=performance.get("compile_recompile_limit", 32),
        )
    if (
        variant == "C"
        and sum(p.numel() for p in policy.parameters())
        != config["model"]["expected_parameter_count"]
    ):
        raise ValueError("Model parameter count mismatch")
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=config["optimizer"]["lr_schedule"]["initial"],
        betas=tuple(config["optimizer"]["betas"]),
        eps=config["optimizer"]["eps"],
        weight_decay=config["optimizer"]["weight_decay"],
        fused=performance.get("fused_adam", False) and device.startswith("cuda"),
    )
    shuffle_rng = random.Random(config["seed"] + 1)
    state = {
        "update": 0,
        "decisions": 0,
        "next_episode": 0,
        "best_score": None,
        "best_checkpoint": None,
        "next_validation": config["evaluation"]["interval_decisions"],
        "total_training_seconds": 0.0,
    }
    if resume:
        state = checkpoint.restore(
            resume, policy, optimizer, shuffle_rng, config, identity
        )
    factory = ScenarioFactory(config, manifest)
    specs = validation_specs(factory, config) if evaluate_every else []
    if specs and not resume:
        write_json(directory / "validation-manifest.json", specs)
    checkpoints = directory / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    if not resume:
        checkpoint.save(
            checkpoints / "initial.pt",
            policy,
            optimizer,
            state.copy(),
            shuffle_rng,
            config,
            identity,
        )
        write_json(directory / "latest.json", {"checkpoint": "initial.pt", **state})
    starting_update = state["update"]
    write_json(
        directory / "execution.json",
        {
            "budget": budget,
            "device": device,
            "variant": variant,
            "max_updates": max_updates,
            "stage_cycle": stage_cycle,
            "evaluate": evaluate_every,
            "replays": replays,
        },
    )
    try:
        with (
            ResourceMonitor(
                directory,
                interval=config.get("telemetry", {}).get("interval_seconds", 1),
                gpu_index=config.get("telemetry", {}).get("gpu_index", 0),
            ) as monitor,
            Collector(config, factory, device, directory / "errors") as collector,
        ):
            if specs and state["best_score"] is None and not control.stopping():
                control.mark(
                    "validation", update=state["update"], decisions=state["decisions"]
                )
                baseline = evaluate(
                    policy,
                    collector,
                    specs,
                    config,
                    directory / "evaluation/initial",
                    monitor,
                )
                state.update(
                    best_score=baseline["greedy"]["score"], best_checkpoint="initial.pt"
                )
                checkpoint.save(
                    checkpoints / "initial.pt",
                    policy,
                    optimizer,
                    state.copy(),
                    shuffle_rng,
                    config,
                    identity,
                )
                write_json(
                    directory / "latest.json", {"checkpoint": "initial.pt", **state}
                )
            elif (
                specs
                and resume
                and state["decisions"] >= state["next_validation"]
                and not control.stopping()
            ):
                control.mark(
                    "validation", update=state["update"], decisions=state["decisions"]
                )
                summary = evaluate(
                    policy,
                    collector,
                    specs,
                    config,
                    directory / "evaluation" / Path(resume).stem,
                    monitor,
                )
                if summary["greedy"]["score"] > state["best_score"]:
                    state.update(
                        best_score=summary["greedy"]["score"],
                        best_checkpoint=Path(resume).name,
                    )
                while state["next_validation"] <= state["decisions"]:
                    state["next_validation"] += config["evaluation"][
                        "interval_decisions"
                    ]
                checkpoint.save(
                    resume,
                    policy,
                    optimizer,
                    state.copy(),
                    shuffle_rng,
                    config,
                    identity,
                )
            while (
                state["decisions"] < budget
                and (
                    max_updates is None
                    or state["update"] - starting_update < max_updates
                )
                and not control.stopping()
            ):
                start = time.perf_counter()
                stage_index = max(
                    i
                    for i, s in enumerate(config["curriculum"])
                    if s["start_decisions"] <= state["decisions"]
                )
                if stage_cycle:
                    stage_index = stage_cycle[state["update"] % len(stage_cycle)]
                stage = config["curriculum"][stage_index]
                monitor.mark(
                    "collection",
                    update=state["update"] + 1,
                    decisions=state["decisions"],
                )
                control.mark(
                    "collection",
                    update=state["update"] + 1,
                    decisions=state["decisions"],
                )
                episodes, next_episode, timings = collector.collect(
                    policy, stage, state["next_episode"], monitor
                )
                if not all(e["result"]["reward"] in (-1, 0, 1) for e in episodes):
                    raise RuntimeError("Invalid terminal reward")
                count = sum(len(e["chosen"]) for e in episodes)
                monitor.mark("ppo")
                control.mark("ppo")
                try:
                    stats = update(
                        policy,
                        optimizer,
                        episodes,
                        config,
                        state["decisions"],
                        device,
                        shuffle_rng,
                        monitor,
                    )
                except BaseException:
                    (directory / "errors").mkdir(exist_ok=True)
                    torch.save(
                        episodes,
                        directory
                        / "errors"
                        / f"rollout-update-{state['update'] + 1}.pt",
                    )
                    raise
                elapsed = time.perf_counter() - start
                state.update(
                    update=state["update"] + 1,
                    decisions=state["decisions"] + count,
                    next_episode=next_episode,
                    total_training_seconds=state["total_training_seconds"] + elapsed,
                    last_rollout_decisions=count,
                )
                row = dict(
                    schema=1,
                    **state,
                    stage=stage_index,
                    rollout_decisions=count,
                    episodes=len(episodes),
                    overflow=count
                    - config["collection"]["min_learner_decisions_per_update"],
                    episode_decisions_mean=sum(len(e["chosen"]) for e in episodes)
                    / len(episodes),
                    episode_decisions_max=max(len(e["chosen"]) for e in episodes),
                    **timings,
                    **stats,
                    update_seconds=elapsed,
                    decisions_per_second=count / elapsed,
                    outcomes=summarize(episodes, samples=100),
                    torch_vram_allocated_peak_bytes=torch.cuda.max_memory_allocated()
                    if device.startswith("cuda")
                    else None,
                )
                row["simulation_seconds"] = sum(
                    e["result"]["timings"]["simulation_seconds"] for e in episodes
                )
                from civ5rl.diagnostic_metrics import summarize_decisions
                row["decisions_diagnostics"] = summarize_decisions(episodes)
                row["opponent_seconds"] = sum(
                    e["result"]["timings"]["opponent_seconds"] for e in episodes
                )
                if performance.get("compile", False) and device.startswith("cuda"):
                    from torch._dynamo.utils import counters

                    row["compile_stats"] = {
                        "unique_graphs": counters["stats"]["unique_graphs"],
                        "calls_captured": counters["stats"]["calls_captured"],
                        "frames_total": counters["frames"]["total"],
                    }
                name = f"update-{state['update']:06}.pt"
                # Commit training state before evaluation; interruption always has a safe boundary.
                monitor.mark("checkpoint", decisions=state["decisions"])
                checkpoint.save(
                    checkpoints / name,
                    policy,
                    optimizer,
                    state.copy(),
                    shuffle_rng,
                    config,
                    identity,
                )
                write_json(directory / "latest.json", {"checkpoint": name, **state})
                if (
                    specs
                    and not control.stopping()
                    and (
                        state["decisions"] >= state["next_validation"]
                        or state["decisions"] >= budget
                    )
                ):
                    control.mark(
                        "validation",
                        update=state["update"],
                        decisions=state["decisions"],
                    )
                    summary = evaluate(
                        policy,
                        collector,
                        specs,
                        config,
                        directory / "evaluation" / name.removesuffix(".pt"),
                        monitor,
                    )
                    score = summary["greedy"]["score"]
                    if state["best_score"] is None or score > state["best_score"]:
                        state.update(best_score=score, best_checkpoint=name)
                    while state["next_validation"] <= state["decisions"]:
                        state["next_validation"] += config["evaluation"][
                            "interval_decisions"
                        ]
                    checkpoint.save(
                        checkpoints / name,
                        policy,
                        optimizer,
                        state.copy(),
                        shuffle_rng,
                        config,
                        identity,
                    )
                    row["validation"] = summary
                # Replay sampling is deterministic and does not consume training RNG.
                selected = [
                    e
                    for e in episodes
                    if e["spec"]["episode_id"]
                    % config["checkpoints"]["training_replay_every_episodes"]
                    == 0
                ]
                if replays and selected:
                    for episode in selected:
                        save_training_replay(
                            episode,
                            directory
                            / "training_replays"
                            / f"episode-{episode['spec']['episode_id']:08}",
                        )
                with (directory / "metrics.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                write_json(directory / "latest.json", {"checkpoint": name, **state})
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "update",
                                "decisions",
                                "episodes",
                                "decisions_per_second",
                                "policy_loss",
                                "value_loss",
                                "kl",
                            )
                        }
                    ),
                    flush=True,
                )
                protected = {name, state["best_checkpoint"], "initial.pt"}
                files = sorted(checkpoints.glob("update-*.pt"))
                protected.update(
                    p.name for p in files[-config["checkpoints"]["keep_last"] :]
                )
                for p in files:
                    saved = torch.load(p, map_location="cpu", weights_only=False)
                    previous_decisions = (
                        saved["state"]["decisions"]
                        - saved["state"]["last_rollout_decisions"]
                    )
                    if any(
                        previous_decisions < milestone <= saved["state"]["decisions"]
                        for milestone in config["checkpoints"]["milestone_decisions"]
                    ):
                        protected.add(p.name)
                    if p.name not in protected:
                        p.unlink()
            paused = control.stopping() and state["decisions"] < budget
            if replays and specs and not paused:
                best = torch.load(
                    checkpoints / state["best_checkpoint"],
                    map_location=device,
                    weights_only=False,
                )
                policy.load_state_dict(best["model"])
                monitor.mark("selected_replays")
                outcome_file = (
                    directory
                    / "evaluation"
                    / Path(state["best_checkpoint"]).stem
                    / "outcomes.json"
                )
                save_replays(
                    policy,
                    specs,
                    device,
                    directory / "selected_replays",
                    outcomes=read_json(outcome_file),
                )
            write_json(
                directory / "run-result.json",
                dict(
                    status="paused" if paused else "completed",
                    **state,
                    budget=budget,
                    estimated_pilot_training_hours=2000000
                    / (state["decisions"] / state["total_training_seconds"])
                    / 3600
                    if state["decisions"]
                    else None,
                ),
            )
            control.mark(
                "paused" if paused else "completed",
                status="paused" if paused else "completed",
                update=state["update"],
                decisions=state["decisions"],
                checkpoint=read_json(directory / "latest.json")["checkpoint"],
            )
    except BaseException:
        import traceback

        write_json(
            directory / "failure.json",
            {
                "traceback": traceback.format_exc(),
                "last_completed_state": read_json(directory / "latest.json")
                if (directory / "latest.json").exists()
                else None,
            },
        )
        raise
    return state


def main():
    from civ5rl.common import read_json

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/rl-v1.json")
    p.add_argument("--corpus", default="artifacts/rl/corpus/manifest.json")
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--phase", choices=("smoke", "pilot"), default="smoke")
    p.add_argument("--resume")
    p.add_argument("--variant", choices=("A", "B", "C"), default="C")
    p.add_argument("--max-updates", type=int)
    p.add_argument("--no-evaluation", action="store_true")
    args = p.parse_args()
    config = read_json(args.config)
    run(
        config,
        read_json(args.corpus),
        args.out,
        device=args.device,
        budget=config["budgets"][f"{args.phase}_decisions"],
        resume=args.resume,
        variant=args.variant,
        max_updates=args.max_updates,
        evaluate_every=not args.no_evaluation,
    )


if __name__ == "__main__":
    main()
