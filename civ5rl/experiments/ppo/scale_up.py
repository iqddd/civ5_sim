"""A bounded continuation experiment with one run record, not environment certification."""

from __future__ import annotations

import gc
import importlib.metadata
import json
import platform
import random
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from civ5rl.core.model import batch_inputs
from civ5rl.engine.collection.collector import Collector
from civ5rl.engine.evaluation.matches import run_policy_match
from civ5rl.engine.evaluation.replay import verify_replay
from civ5rl.engine.ppo.adam import optimizer_for
from civ5rl.engine.ppo.core import encode_episode, update
from civ5rl.engine.ppo.storage import load_rollout, save_rollout
from civ5rl.engine.runtime.control import TrainingCounters
from civ5rl.engine.runtime.telemetry import CompilationClock, ResourceMonitor
from civ5rl.experiments._shared.lifecycle import experiment_run_guard
from civ5rl.experiments.evaluation.paired import run_chunked_pool
from civ5rl.experiments.imitation.helpers import (
    assert_equal,
    isolated_rng,
    load_policy,
    pool,
)
from civ5rl.experiments.ppo.after_imitation import (
    summarize_rollout,
)
from civ5rl.experiments.ppo.checkpoints import prune_checkpoints
from civ5rl.protocols.corpus import validate_corpus
from civ5rl.protocols.ppo_scale_up import KIND, ScaleUpFactory
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import (
    ROOT,
    configure,
    file_sha256,
    read_json,
    seed_for,
    write_json,
)
from civ5rl.provenance.identity import policy_hash
from civ5rl.provenance.jsonl import append_jsonl
from civ5rl.reporting.evaluation_audit import (
    collect_paired_outcomes,
    match_summary,
    paired_world_bootstrap,
    select_outcome_representatives,
    verify_match_artifacts,
)
from civ5rl.reporting.ppo_scale_up import report
from civ5sim.engine import digest


def make_policy(config, device):
    policy = load_policy(config, ROOT / config["experiment"]["parent"], device)
    for parameter in policy.value.parameters():
        parameter.requires_grad_(True)
    return policy


def load_parent(config, device="cpu"):
    saved = torch.load(ROOT / config["experiment"]["parent"], map_location="cpu", weights_only=False)
    policy = make_policy(config, device)
    optimizer = optimizer_for(policy, config, device)
    optimizer.load_state_dict(saved["optimizer"])
    assert_equal(policy.state_dict(), saved["model"])
    assert_equal(optimizer.state_dict(), saved["optimizer"])
    counters = {int(v["step"]) for v in optimizer.state.values()}
    if counters != {config["experiment"]["parent_optimizer_steps"]}:
        raise ValueError(f"Parent Adam counters differ: {counters}")
    if policy_hash(policy.state_dict()) != config["experiment"]["parent_policy_hash"]:
        raise ValueError("Parent policy differs")
    return policy, optimizer


def prepare(config, out):
    experiment = config["experiment"]
    for name, expected in (
        (experiment["parent"], experiment["parent_sha256"]),
        (experiment["parent_manifest"], experiment["parent_manifest_sha256"]),
    ):
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f"Key input differs: {name}")
    previous = read_json(ROOT / experiment["parent_manifest"])
    base = previous["base"]
    worlds = base["worlds"]
    if not worlds["training"] or set(worlds["training"]) & (
        set(worlds["validation"]) | set(worlds["holdout"])
    ):
        raise ValueError("Training/validation worlds overlap")
    maps = [m for m in base["maps"] if m["split"] == "train" and m["parent_seed"] in worlds["training"]]
    if {crop["parent_seed"] for crop in maps} != set(worlds["training"]):
        raise ValueError("Training crops do not cover the training worlds")
    validate_corpus({"maps": maps})
    plan = {
        "schema": 1,
        "kind": KIND,
        "base": {**base, "maps": maps},
        "development": previous["development"],
        "confirmation": previous["confirmation"],
    }
    if not plan["development"] or not plan["confirmation"]:
        raise ValueError("Missing evaluation cases")
    for purpose in ("development", "confirmation"):
        for case in plan[purpose]:
            if case["split"] != "validation" or case["parent_world"] not in worlds["validation"]:
                raise ValueError("Closed/untrusted evaluation case")
            if digest({k: v for k, v in case.items() if k != "case_hash"}) != case["case_hash"]:
                raise ValueError("Evaluation case changed")
    plan["hash"] = digest(plan)
    seed = config.get("seed", 17)
    configure(seed, "cpu")
    policy, optimizer = load_parent(config)
    run_record = {
        "schema": 1,
        "kind": KIND,
        "config": config,
        "parent_sha256": experiment["parent_sha256"],
        "parent_policy_hash": experiment["parent_policy_hash"],
        "input_manifest_sha256": experiment["parent_manifest_sha256"],
        "manifest_hash": plan["hash"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
        "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "CPU": platform.processor(),
        "seed": seed,
        "training_namespace": experiment["training_namespace"],
        "shuffle_seed": seed_for(seed, 0, "shuffle", namespace=experiment["shuffle_namespace"]),
        "initial_model_Adam_bitwise_equal": True,
        "critic_preserved": True,
        "parent_Adam_counter": experiment["parent_optimizer_steps"],
        "native_build": read_json(ROOT / "build/remote-build.json")
        if (ROOT / "build/remote-build.json").exists()
        else None,
    }
    run_record["run_id"] = digest(
        {
            k: run_record[k]
            for k in (
                "config",
                "parent_sha256",
                "input_manifest_sha256",
            )
        }
    )
    counters = TrainingCounters()
    state = {
        **counters.to_state_dict(protocol=KIND),
        "origin_Adam_counter": experiment["parent_optimizer_steps"],
        "decisions": 0,
        "next_episode": experiment["start_episode"],
        "zero_accepted_streak": 0,
        "active_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "preparation_seconds": 0.0,
        "evaluated": {},
        "latest": "checkpoints/initial.pt",
        "pending_evaluation": True,
        "best": {"path": "checkpoints/initial.pt", "wins": -1, "update": 0},
        "stop_reason": None,
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", plan)
    write_json(out / "run.json", run_record)
    write_json(
        out / "packages.json",
        {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
    )
    shuffle = random.Random(run_record["shuffle_seed"])
    checkpoint.save(
        out / state["latest"],
        policy,
        optimizer,
        state,
        shuffle,
        config,
        {"run_id": run_record["run_id"]},
    )
    write_json(out / "state.json", state)
    write_json(out / "status.json", {"status": "prepared"})
    print(
        f"Prepared: model/Adam equal to parent; Adam={experiment['parent_optimizer_steps']}; "
        f"new decisions=0; episode={experiment['start_episode']}",
        flush=True,
    )
    return plan


def evaluation_task(task):
    config, case, path, mode, enemy, directory = task
    if case["split"] != "validation":
        raise ValueError("Evaluation requires open validation")
    return run_policy_match(config, case, path, mode, enemy, directory)


def evaluate_checkpoint(config, plan, out, path, purpose="development"):
    fingerprint = policy_hash(torch.load(out / path, map_location="cpu", weights_only=False)["model"])
    cache_path = out / "evaluations.json"
    cache = read_json(cache_path) if cache_path.exists() else {}
    key = purpose + ":" + fingerprint
    if key in cache:
        return cache[key]
    started = perf_counter()
    runs = {}
    with pool(4) as executor:
        for mode in ("argmax", "sample") if purpose == "development" else ("argmax",):
            runs[mode] = {}
            for enemy in ("cp", "greedy") if purpose == "development" else ("cp",):
                relative = f"evaluation/{purpose}/{fingerprint[:16]}/{mode}/{enemy}"
                directory = out / relative
                tasks = []
                for case in plan[purpose]:
                    target = directory / f"case-{case['case_id']:03}-side-{case['learner_side']}"
                    if not (target / "metrics.json").exists():
                        tasks.append((config, case, str(out / path), mode, enemy, str(target)))
                run_chunked_pool(executor, evaluation_task, tasks, 4)
                runs[mode][enemy] = {
                    "directory": relative,
                    "summary": match_summary(directory, len(plan[purpose])),
                }
                print(
                    f"Evaluation {purpose}/{mode}/{enemy}: {runs[mode][enemy]['summary']}",
                    flush=True,
                )
    result = {
        "policy_hash": fingerprint,
        "checkpoint": path,
        "purpose": purpose,
        "runs": runs,
        "seconds": perf_counter() - started,
    }
    cache[key] = result
    write_json(cache_path, cache)
    return result


def development(config, plan, out, state, shuffle):
    started = perf_counter()
    with isolated_rng(shuffle):
        result = evaluate_checkpoint(config, plan, out, state["latest"])
    state["evaluation_seconds"] += perf_counter() - started
    state["evaluated"][str(state["updates"])] = {
        **result,
        "decisions": state["decisions"],
    }
    wins = result["runs"]["argmax"]["cp"]["summary"]["W"]
    if wins > state["best"]["wins"]:
        state["best"] = {
            "path": state["latest"],
            "wins": wins,
            "update": state["updates"],
            "policy_hash": result["policy_hash"],
        }
    state["pending_evaluation"] = False
    write_json(out / "state.json", state)
    with isolated_rng(shuffle):
        report(config, out)


def train(
    config,
    plan,
    out,
    device,
    *,
    factory_type=ScaleUpFactory,
    development_fn=development,
):
    state = read_json(out / "state.json")
    if state["stop_reason"]:
        return
    protocol = config.get("kind", KIND)
    counters = TrainingCounters.from_dict(state, protocol=protocol)
    state.update(counters.to_state_dict(protocol=protocol))
    configure(config.get("seed", 17), device)
    record = read_json(out / "run.json")
    identity = {"run_id": record["run_id"]}
    policy = make_policy(config, device)
    optimizer = optimizer_for(policy, config, device)
    shuffle = random.Random(record["shuffle_seed"])
    saved = checkpoint.restore(out / state["latest"], policy, optimizer, shuffle, config, identity)
    if saved["decisions"] != state["decisions"] or saved["optimizer_steps"] != state["optimizer_steps"]:
        raise ValueError("Checkpoint and progress disagree")
    if config.get("kind") == "rl01.ppo_target_scale":
        loaded = torch.load(out / state["latest"], map_location="cpu", weights_only=False)
        assert_equal(policy.state_dict(), loaded["model"])
        assert_equal(optimizer.state_dict(), loaded["optimizer"])
        assert_equal(random.getstate(), loaded["python_rng"])
        assert_equal(np.random.get_state(), loaded["numpy_rng"])
        assert_equal(torch.get_rng_state(), loaded["torch_rng"])
        assert_equal(shuffle.getstate(), loaded["shuffle_rng"])
        if device.startswith("cuda") and loaded["cuda_rng"]:
            assert_equal(torch.cuda.get_rng_state_all(), loaded["cuda_rng"])
        write_json(
            out / "resume-verification.json",
            {
                "checkpoint": state["latest"],
                "model_Adam_RNG_equal": True,
                "Adam_counters": sorted({int(s["step"]) for s in optimizer.state.values()}),
                "decisions": state["decisions"],
                "next_episode": state["next_episode"],
                "device": device,
            },
        )
        del loaded
    if state["pending_evaluation"]:
        development_fn(config, plan, out, state, shuffle)
    factory = factory_type(config, plan["base"])
    with (
        ResourceMonitor(out / "telemetry") as monitor,
        Collector(config, factory, device, out / "errors") as collector,
    ):
        warmup_rollout = config["experiment"].get("warmup_rollout")
        warmup_path = (
            ROOT / warmup_rollout
            if warmup_rollout
            else out
            / ("mixed-preparation-rollout" if protocol == "rl01.ppo_target_scale" else "rollouts")
            / "first.pt"
        )
        if warmup_rollout or warmup_path.is_file():
            monitor.mark("initial_warmup")
            prep = perf_counter()
            with isolated_rng(shuffle), torch.no_grad():
                episode = load_rollout(warmup_path)[0]
                encoded = encode_episode(episode, policy.variant)[:20]
                policy.forward_sequences(batch_inputs(encoded, device), [len(encoded)])
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            state["preparation_seconds"] += perf_counter() - prep

        def inspect_first_step(step):
            if (
                config.get("kind") == "rl01.ppo_target_scale"
                and step["accepted"]
                and not (out / "first-resumed-step.json").exists()
            ):
                counters = sorted({int(s["step"]) for s in optimizer.state.values()})
                expected = (
                    config["experiment"]["parent_optimizer_steps"]
                    + state["optimizer_steps"]
                    + step["accepted_step"]
                )
                if counters != [expected]:
                    raise ValueError(f"Resumed Adam counter differs: {counters} != {expected}")
                write_json(
                    out / "first-resumed-step.json",
                    {
                        "Adam_counters": counters,
                        "checkpoint": saved["latest"],
                        "restored_optimizer_steps": saved["optimizer_steps"],
                        "step": step,
                    },
                )
            if step["accepted"] and not (out / "first-accepted-step.json").exists():
                counters = sorted({int(s["step"]) for s in optimizer.state.values()})
                expected = config["experiment"]["parent_optimizer_steps"] + 1
                if counters != [expected]:
                    raise ValueError(f"First accepted Adam counter is {counters}, expected {expected}")
                write_json(
                    out / "first-accepted-step.json",
                    {"Adam_counters": counters, "step": step},
                )

        experiment = config["experiment"]
        while not state["stop_reason"]:
            if (out / "pause.request").exists():
                write_json(
                    out / "status.json",
                    {"status": "paused", "checkpoint": state["latest"]},
                )
                return
            if state["active_seconds"] >= experiment["active_seconds"]:
                state["stop_reason"] = "active_time_limit"
                break
            if state["decisions"] >= experiment["decisions"]:
                state["stop_reason"] = "decision_limit"
                break
            started = perf_counter()
            deadline = started + experiment["active_seconds"] - state["active_seconds"]
            number = state["rollouts_collected"] + 1
            monitor.mark("collection", update=number)
            episodes, following, timings = collector.collect(
                policy,
                None,
                state["next_episode"],
                monitor,
                deadline=deadline,
                max_decisions=experiment["decisions"] - state["decisions"],
            )
            if not episodes:
                state["active_seconds"] += perf_counter() - started
                state["stop_reason"] = "active_time_limit"
                break
            summary = summarize_rollout(episodes)
            before_decisions = state["decisions"]
            state["decisions"] += summary["decisions"]
            state["next_episode"] = following
            monitor.mark("rollout_save")
            artifact = save_rollout(episodes, out / "rollouts", number)
            metrics = {
                "optimizer_steps": 0,
                "attempted_optimizer_steps": 0,
                "rejected_optimizer_steps": 0,
                "stop_reason": "active_time_limit",
                "update_complete": False,
            }
            delta = 0.0
            if perf_counter() < deadline:
                monitor.mark("ppo", update=number)
                before = [parameter.detach().clone() for parameter in policy.parameters()]
                with CompilationClock() as compiler:
                    compiler.monitor = monitor
                    metrics = update(
                        policy,
                        optimizer,
                        episodes,
                        config,
                        before_decisions,
                        device,
                        shuffle,
                        deadline=deadline,
                        on_step=inspect_first_step,
                    )
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                metrics["runtime_compilation_seconds"] = sum(max(0, b - a) for a, b in compiler.intervals)
                delta = float(
                    torch.sqrt(
                        sum(
                            (parameter.detach() - previous).float().square().sum()
                            for parameter, previous in zip(policy.parameters(), before)
                        )
                    )
                )
                del before
            del episodes
            gc.collect()
            state["active_seconds"] += perf_counter() - started
            counters = TrainingCounters.from_dict(state, protocol=protocol)
            counters = counters.replace(
                rollouts_collected=number,
                rollouts_with_accepted_steps=counters.rollouts_with_accepted_steps
                + (1 if metrics["optimizer_steps"] else 0),
                adam_steps_attempted=counters.adam_steps_attempted + metrics["attempted_optimizer_steps"],
                adam_steps_accepted=counters.adam_steps_accepted + metrics["optimizer_steps"],
            )
            state.update(counters.to_state_dict(protocol=protocol))
            state["updates"] = number
            state["zero_accepted_streak"] = (
                0 if metrics["optimizer_steps"] else state["zero_accepted_streak"] + 1
            )
            state["latest"] = f"checkpoints/update-{number:04}.pt"
            if (
                state["active_seconds"] >= experiment["active_seconds"]
                or metrics["stop_reason"] == "active_time_limit"
            ):
                state["stop_reason"] = "active_time_limit"
            elif state["decisions"] >= experiment["decisions"]:
                state["stop_reason"] = "decision_limit"
            elif state["zero_accepted_streak"] >= experiment.get("max_zero_accepted_rollouts", 3):
                state["stop_reason"] = "no_accepted_updates"
            due = any(before_decisions < m <= state["decisions"] for m in experiment["milestones"])
            state["pending_evaluation"] = bool(due or state["stop_reason"])
            servicing = perf_counter()
            monitor.mark("checkpoint")
            checkpoint.save(
                out / state["latest"],
                policy,
                optimizer,
                state,
                shuffle,
                config,
                identity,
            )
            prune_checkpoints(out, state)
            write_json(out / "coverage.json", factory.summary())
            state["active_seconds"] += perf_counter() - servicing
            if state["active_seconds"] >= experiment["active_seconds"]:
                state["stop_reason"] = "active_time_limit"
                state["pending_evaluation"] = True
            write_json(out / "state.json", state)
            row = {
                "update": number,
                **summary,
                "total_decisions": state["decisions"],
                "total_Adam_steps": state["optimizer_steps"],
                "Adam_counter": experiment["parent_optimizer_steps"] + state["optimizer_steps"],
                "active_seconds": state["active_seconds"],
                "weight_delta_l2": delta,
                "decision_overshoot": max(0, state["decisions"] - experiment["decisions"]),
                "active_time_overshoot_seconds": max(
                    0, state["active_seconds"] - experiment["active_seconds"]
                ),
                "collection": timings,
                "rollout_artifact": artifact,
                "ppo": metrics,
            }
            append_jsonl(out / "updates.jsonl", row)
            print(
                f"Rollout {number}: decisions={state['decisions']}; W/D/L={summary['W']}/{summary['D']}/{summary['L']}; "
                f"new Adam={state['optimizer_steps']}; active={state['active_seconds']:.1f}s; {metrics['stop_reason']}",
                flush=True,
            )
            if number == 3 or due:
                eta = {
                    "decisions": state["decisions"],
                    "active_seconds": state["active_seconds"],
                    "estimated_remaining_active_seconds": max(
                        0.0,
                        min(
                            experiment["active_seconds"] - state["active_seconds"],
                            (experiment["decisions"] - state["decisions"])
                            * state["active_seconds"]
                            / state["decisions"],
                        ),
                    ),
                }
                write_json(out / "eta.json", eta)
                print("ETA " + json.dumps(eta), flush=True)
            if (out / "pause.request").exists():
                # Persist all counters/RNG after update servicing, before any
                # pending evaluation; resume evaluates before collecting again.
                checkpoint.save(
                    out / state["latest"],
                    policy,
                    optimizer,
                    state,
                    shuffle,
                    config,
                    identity,
                )
                write_json(out / "state.json", state)
                write_json(
                    out / "status.json",
                    {"status": "paused", "checkpoint": state["latest"]},
                )
                return
            if state["pending_evaluation"]:
                monitor.mark("evaluation")
                development_fn(config, plan, out, state, shuffle)
        monitor.mark("stopped")
        if policy_hash(policy.state_dict()) not in {v["policy_hash"] for v in state["evaluated"].values()}:
            state["pending_evaluation"] = True
        if state["pending_evaluation"]:
            development_fn(config, plan, out, state, shuffle)
    write_json(out / "state.json", state)
    write_json(
        out / "status.json",
        {"status": "training_complete", "stop_reason": state["stop_reason"]},
    )


def confirmation(config, plan, out):
    state = read_json(out / "state.json")
    if not state["stop_reason"] or state["pending_evaluation"]:
        raise ValueError("Training/selection is incomplete")
    write_json(out / "selected.json", state["best"])
    baseline = evaluate_checkpoint(config, plan, out, "checkpoints/initial.pt", "confirmation")
    candidate = evaluate_checkpoint(config, plan, out, state["best"]["path"], "confirmation")
    same = baseline["policy_hash"] == candidate["policy_hash"]

    def rows(evaluation):
        directory = out / evaluation["runs"]["argmax"]["cp"]["directory"]
        return collect_paired_outcomes(directory, check_duplicates=True)

    before, cases, before_outcomes = rows(baseline)
    after, paired, after_outcomes = rows(candidate)
    if cases != paired:
        raise ValueError("Unpaired additional validation")
    ci = (
        None
        if same
        else paired_world_bootstrap(
            before, after, samples=10000, seed=20261103, cases=cases, cases_per_world=16
        )
    )
    slices = {}
    for name, predicate in (
        ("contact", lambda c: c["placement"] == "contact"),
        ("separated", lambda c: c["placement"] == "separated"),
        ("side0", lambda c: c["learner_side"] == 0),
        ("side1", lambda c: c["learner_side"] == 1),
    ):
        keys = [k for k, c in cases.items() if predicate(c)]
        slices[name] = {
            "matches": len(keys),
            "baseline_wins": sum(before[k] for k in keys),
            "candidate_wins": sum(after[k] for k in keys),
            "baseline_WDL": dict(Counter(before_outcomes[k] for k in keys)),
            "candidate_WDL": dict(Counter(after_outcomes[k] for k in keys)),
        }
    write_json(
        out / "confirmation.json",
        {
            "baseline": baseline,
            "candidate": candidate,
            "comparison": ci,
            "slices": slices,
            "baseline_selected": same,
            "selected_before_confirmation": True,
        },
    )


def verify_results(config, out):
    state = read_json(out / "state.json")
    saved = torch.load(out / state["best"]["path"], map_location="cpu", weights_only=False)
    counters = sorted({int(v["step"]) for v in saved["optimizer"]["state"].values()})
    expected = config["experiment"]["parent_optimizer_steps"] + saved["state"]["optimizer_steps"]
    if counters != [expected]:
        raise ValueError("Selected Adam counters differ")
    games = 0
    for evaluation in read_json(out / "evaluations.json").values():
        count = 64 if evaluation["purpose"] == "development" else 256
        for enemies in evaluation["runs"].values():
            for entry in enemies.values():
                directory = out / entry["directory"]
                match_summary(directory, count)
                for path in directory.glob("*/metrics.json"):
                    if not verify_match_artifacts(
                        path.parent,
                        ("case.json", "replay.json", "result.json", "final.json"),
                    ):
                        raise ValueError(f"Missing battle artifact: {path.parent}")
                games += count
    final = read_json(out / "confirmation.json")
    directory = out / final["candidate"]["runs"]["argmax"]["cp"]["directory"]
    representatives = select_outcome_representatives(directory)
    checks = {k: verify_replay(path) for k, path in representatives.items()}
    write_json(
        out / "verification.json",
        {
            "selected_Adam_counter": expected,
            "evaluation_matches": games,
            "replays": checks,
        },
    )


def run(phase, config, out, device):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with experiment_run_guard(
        out,
        phase=phase,
        on_error_report=lambda: report(config, out) if (out / "state.json").exists() else None,
    ):
        if not (out / "run.json").exists():
            if phase not in ("prepare", "run"):
                raise ValueError("Prepare the experiment first")
            prepare(config, out)
        elif read_json(out / "run.json")["config"] != config:
            raise ValueError("This run uses a different training configuration")
        plan = read_json(out / "manifest.json")
        if phase in ("train", "run"):
            write_json(out / "status.json", {"status": "training"})
            train(config, plan, out, device)
        if phase in ("evaluate", "run"):
            confirmation(config, plan, out)
            verify_results(config, out)
            write_json(
                out / "status.json",
                {
                    "status": "complete",
                    "stop_reason": read_json(out / "state.json")["stop_reason"],
                },
            )
        report(config, out)
