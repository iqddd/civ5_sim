"""Bounded on-policy P(win) fine-tuning from the frozen own-state BC policy."""

from __future__ import annotations

import gc
import random
from collections import Counter
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch

from civ5rl.engine.collection.collector import Collector
from civ5rl.engine.evaluation.matches import run_policy_match
from civ5rl.engine.evaluation.replay import verify_replay
from civ5rl.engine.ppo.adam import optimizer_for
from civ5rl.engine.ppo.core import update
from civ5rl.engine.ppo.storage import save_rollout
from civ5rl.engine.runtime.control import TrainingCounters
from civ5rl.engine.runtime.telemetry import CompilationClock, ResourceMonitor
from civ5rl.experiments._shared.lifecycle import experiment_run_guard
from civ5rl.experiments._shared.provenance_checks import verify_manifest_digest
from civ5rl.experiments.checks import ppo_after_imitation as after_imitation_checks
from civ5rl.experiments.checks import ppo_controlled as controlled_checks
from civ5rl.experiments.evaluation.paired import run_chunked_pool
from civ5rl.experiments.imitation.helpers import (
    isolated_rng,
    pool,
)
from civ5rl.experiments.ppo.checkpoints import prune_checkpoints
from civ5rl.experiments.ppo.policy import create_policy
from civ5rl.experiments.ppo.rollout_summary import summarize_rollout
from civ5rl.protocols.corpus import validate_corpus
from civ5rl.protocols.ppo_after_imitation import (
    CONTROLLED_KIND,
    KIND,
    TrainingFactory,
    advance_controlled_rollout,
    controlled,
    prepare_cases,
)
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import ROOT, configure, read_json, write_json
from civ5rl.provenance.common import file_sha256 as file_hash
from civ5rl.provenance.identity import policy_hash
from civ5rl.provenance.jsonl import append_jsonl
from civ5rl.provenance.validation import validate_config
from civ5rl.reporting.evaluation_audit import (
    collect_paired_outcomes,
    match_summary,
    paired_world_bootstrap,
    select_outcome_representatives,
)
from civ5rl.reporting.ppo_after_imitation import report
from civ5sim.engine import digest

if TYPE_CHECKING:
    from civ5rl.core.model import Policy


def check_identity(config, out, *, for_execution=False):
    """Validate frozen artifacts for analysis; opt in to live-code checks to run."""
    out = Path(out)
    plan = read_json(out / "manifest.json")
    verify_manifest_digest(plan)
    if not for_execution:
        checkpoint.verify_frozen_passport(out, config=config, manifest=plan, sources=False)
        return plan
    validate_config(config)
    checkpoint.verify_execution_passport(out, config, plan)
    if file_hash(ROOT / config["experiment"]["parent"]) != config["experiment"]["parent_sha256"]:
        raise ValueError("Parent changed")
    # Validate only open terrain named by the frozen plan, without loading closed maps.
    validate_corpus({"maps": plan["base"]["maps"]})
    if controlled(config):
        parent = ROOT / config["experiment"]["parent_run"]
        if file_hash(parent / "manifest.json") != plan["reused_manifest_sha256"]:
            raise ValueError("Reused manifest changed")
        if prepare_cases(config) != plan:
            raise ValueError("Controlled cases differ from frozen parent")
    return plan


def prepare(config, out):
    validate_config(config)
    if (out / "passport.json").exists():
        return check_identity(config, out, for_execution=True)
    configure(config.get("seed", 17), "cuda" if torch.cuda.is_available() else "cpu")
    out.mkdir(parents=True, exist_ok=True)
    if file_hash(ROOT / config["experiment"]["parent"]) != config["experiment"]["parent_sha256"]:
        raise ValueError("Parent checkpoint differs")
    parent = ROOT / config["experiment"]["parent_run"]
    checkpoint.verify_frozen_passport(parent, sources=False)
    plan = prepare_cases(config)
    write_json(out / "manifest.json", plan)
    current = checkpoint.passport(out, config, plan)
    saved = torch.load(ROOT / config["experiment"]["parent"], map_location="cpu", weights_only=False)
    policy = create_policy(config, "cpu")
    for name, value in saved["model"].items():
        if (controlled(config) or name not in ("value.2.weight", "value.2.bias")) and not torch.equal(
            value, policy.state_dict()[name]
        ):
            raise ValueError(f"Unexpected initial change: {name}")
    assert policy_hash(saved["model"]) == policy_hash(policy.state_dict())
    optimizer = optimizer_for(policy, config, "cpu")
    counters = TrainingCounters()
    state = {
        **counters.to_state_dict(protocol=config["kind"]),
        **({"zero_accepted_streak": 0, "first_accepted_evaluated": False} if controlled(config) else {}),
        "decisions": 0,
        "next_episode": 0,
        "active_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "save_seconds": 0.0,
        "preparation_seconds": 0.0,
        "evaluated": {},
        "best": {"path": str(out / "checkpoints/initial.pt"), "wins": -1, "update": 0},
        "latest": "checkpoints/initial.pt",
        "pending_evaluation": True,
        "stop_reason": None,
    }
    rng = random.Random(config.get("seed", 17))
    checkpoint.save(out / state["latest"], policy, optimizer, state, rng, config, current)
    write_json(out / "state.json", state)
    parent_counters = {int(value["step"]) for value in saved["optimizer"]["state"].values()}
    if len(parent_counters) > 1:
        raise ValueError("Parent Adam counters disagree")
    parent_counter = next(iter(parent_counters), 0)
    parent_lineage = parent / "lineage.json"
    bc_origin = (
        read_json(parent_lineage).get("BC_origin_Adam_counter")
        if controlled(config) and parent_lineage.exists()
        else parent_counter
    )
    write_json(
        out / "lineage.json",
        {
            "parent": config["experiment"]["parent"],
            "parent_sha256": config["experiment"]["parent_sha256"],
            "parent_Adam_counter": parent_counter,
            "BC_origin_Adam_counter": bc_origin,
            "reused_training_streams": controlled(config),
            "initial_model_bitwise_equal": controlled(config),
            "PPO_Adam_counter": 0,
            "initial_policy_equal": True,
            "value_last_layer_zero": True,
        },
    )
    write_json(out / "status.json", {"status": "prepared"})
    return plan


def evaluation_task(task):
    config, manifest_path, purpose, index, checkpoint_path, mode, enemy, directory = task
    plan = read_json(manifest_path)
    if (
        plan["kind"] != config["kind"]
        or plan["kind"] not in (KIND, CONTROLLED_KIND)
        or purpose not in ("development", "confirmation")
    ):
        raise ValueError("Closed or untrusted evaluation")
    if digest({k: v for k, v in plan.items() if k != "hash"}) != plan["hash"]:
        raise ValueError("Evaluation manifest hash changed")
    case = plan[purpose][index]
    if case["split"] != "validation" or case["parent_world"] not in plan["base"]["worlds"]["validation"]:
        raise ValueError("Closed evaluation case")
    if digest({k: v for k, v in case.items() if k != "case_hash"}) != case["case_hash"]:
        raise ValueError("Evaluation case changed")
    return run_policy_match(config, case, checkpoint_path, mode, enemy, directory)


def evaluate_checkpoint(config, plan, out, path, purpose="development"):
    model = torch.load(path, map_location="cpu", weights_only=False)["model"]
    fingerprint = policy_hash(model)
    cache_path = out / "evaluations.json"
    cache = read_json(cache_path) if cache_path.exists() else {}
    key = purpose + ":" + fingerprint
    if key in cache:
        return cache[key]
    started = perf_counter()
    runs = {}
    modes = ("argmax", "sample") if purpose == "development" else ("argmax",)
    enemies = ("cp", "greedy") if purpose == "development" else ("cp",)
    with pool(config["workers"]) as executor:
        for mode in modes:
            runs[mode] = {}
            for enemy in enemies:
                directory = out / f"evaluation/{purpose}/{fingerprint[:16]}/{mode}/{enemy}"
                tasks = []
                for case_index, case in enumerate(plan[purpose]):
                    case_directory = directory / f"case-{case['case_id']:03}-side-{case['learner_side']}"
                    if not (case_directory / "metrics.json").exists():
                        tasks.append(
                            (
                                config,
                                str(out / "manifest.json"),
                                purpose,
                                case_index,
                                str(path),
                                mode,
                                enemy,
                                str(case_directory),
                            )
                        )
                total = len(tasks)
                run_chunked_pool(
                    executor,
                    evaluation_task,
                    tasks,
                    4,
                    lambda count, label=f"evaluate {purpose}/{mode}/{enemy}", total=total: print(
                        f"{label}: {count}/{total}", flush=True
                    ),
                )
                runs[mode][enemy] = {
                    "directory": str(directory),
                    "summary": match_summary(directory, len(plan[purpose])),
                }
    result = {
        "policy_hash": fingerprint,
        "checkpoint": str(path),
        "purpose": purpose,
        "runs": runs,
        "seconds": perf_counter() - started,
    }
    cache[key] = result
    write_json(cache_path, cache)
    return result


def save_boundary(config, out, policy, optimizer, state, rng, identity):
    protocol = config["kind"]
    has_updates = "updates" in state
    saved_updates = state.get("updates")
    counters = TrainingCounters.from_dict(state, protocol=protocol)
    state.update(counters.to_state_dict(protocol=protocol))
    if not controlled(config) and has_updates:
        state["updates"] = saved_updates
    start = perf_counter()
    checkpoint.save(out / state["latest"], policy, optimizer, state, rng, config, identity)
    state["save_seconds"] += perf_counter() - start
    write_json(out / "state.json", state)
    prune_checkpoints(out, state)


def development(config, plan, out, state):
    with isolated_rng(random.Random(0)):
        result = evaluate_checkpoint(config, plan, out, out / state["latest"])
    state["evaluation_seconds"] += result["seconds"]
    state["evaluated"][str(state["updates"])] = result
    wins = result["runs"]["argmax"]["cp"]["summary"]["W"]
    if wins > state["best"]["wins"]:
        state["best"] = {
            "path": str(out / state["latest"]),
            "wins": wins,
            "update": state["updates"],
            "policy_hash": result["policy_hash"],
        }
    state["pending_evaluation"] = False
    write_json(out / "state.json", state)


@dataclass
class _PpoTraining:
    state: dict[str, Any]
    protocol: str
    policy: Policy
    optimizer: torch.optim.Optimizer
    rng: random.Random
    identity: dict[str, Any]


def _restore_ppo_training(
    config: dict[str, Any],
    plan: dict[str, Any],
    out: Path,
    device: str | torch.device,
    state: dict[str, Any],
) -> _PpoTraining:
    """Restore accepted model and optimizer state before opening collection workers."""
    protocol = config["kind"]
    has_updates = "updates" in state
    saved_updates = state.get("updates")
    counters = TrainingCounters.from_dict(state, protocol=protocol)
    state.update(counters.to_state_dict(protocol=protocol))
    if not controlled(config) and has_updates:
        state["updates"] = saved_updates
    if not (out / "preflight.json").exists():
        # Both diagnostic modules import the policy and optimizer factories above.
        if controlled(config):
            preflight = controlled_checks.preflight
        else:
            preflight = after_imitation_checks.preflight
        preflight(config, plan, out)
    if not read_json(out / "preflight.json")["passed"]:
        raise RuntimeError("Acceptance checks failed; training is disabled")
    configure(config.get("seed", 17), device)
    policy = create_policy(config, device)
    optimizer = optimizer_for(policy, config, device)
    rng = random.Random(config.get("seed", 17))
    identity = read_json(out / "passport.json")
    restored = checkpoint.restore(out / state["latest"], policy, optimizer, rng, config, identity)
    # state.json can include a completed evaluation after the model checkpoint was saved.
    assert (
        restored["decisions"] == state["decisions"]
        and restored["optimizer_steps"] == state["optimizer_steps"]
    )
    if state["pending_evaluation"]:
        development(config, plan, out, state)
    return _PpoTraining(state, protocol, policy, optimizer, rng, identity)


def _warmup_ppo_training(
    training: _PpoTraining, monitor: ResourceMonitor, device: str | torch.device
) -> None:
    # One no-update warmup uses saved public inputs; RNG is restored before collection.
    monitor.mark("initial_warmup")
    prep = perf_counter()

    with isolated_rng(training.rng):
        after_imitation_checks.warmup(training.policy, device)
    training.state["preparation_seconds"] += perf_counter() - prep


def _active_time_limit_reached(config: dict[str, Any], state: dict[str, Any]) -> bool:
    return state["active_seconds"] >= config["experiment"]["active_seconds"]


def _decision_limit_reached(config: dict[str, Any], state: dict[str, Any]) -> bool:
    return state["decisions"] >= config["experiment"]["decisions"]


def _no_accepted_updates_limit_reached(config: dict[str, Any], state: dict[str, Any]) -> bool:
    return (
        controlled(config)
        and state["zero_accepted_streak"] >= config["experiment"]["max_zero_accepted_rollouts"]
    )


def _advance_ppo_state(
    config: dict[str, Any],
    state: dict[str, Any],
    protocol: str,
    metrics: dict[str, Any],
    *,
    number: int,
    before_decisions: int,
) -> bool:
    """Commit counters and stop/evaluation flags for a completed collection attempt."""
    counters = TrainingCounters.from_dict(state, protocol=protocol)
    accepted_steps = int(metrics["optimizer_steps"])
    attempted_steps = int(metrics.get("attempted_optimizer_steps", accepted_steps))
    counters = counters.replace(
        rollouts_collected=number,
        rollouts_with_accepted_steps=counters.rollouts_with_accepted_steps + (1 if accepted_steps > 0 else 0),
        adam_steps_attempted=counters.adam_steps_attempted + attempted_steps,
        adam_steps_accepted=counters.adam_steps_accepted + accepted_steps,
    )
    first_accepted = False
    if controlled(config):
        first_accepted = advance_controlled_rollout(state, metrics, number)
        state.update(counters.to_state_dict(protocol=protocol))
        state["updates"] = number
        state["latest"] = f"checkpoints/update-{number:04}.pt"
    else:
        has_updates = "updates" in state
        saved_updates = state.get("updates")
        state.update(counters.to_state_dict(protocol=protocol))
        if accepted_steps > 0:
            state["updates"] = number
            state["latest"] = f"checkpoints/update-{number:04}.pt"
        elif has_updates:
            state["updates"] = saved_updates
    timed = _active_time_limit_reached(config, state) or metrics["stop_reason"] == "active_time_limit"
    if timed:
        state["stop_reason"] = "active_time_limit"
    elif _decision_limit_reached(config, state):
        state["stop_reason"] = "decision_limit"
    elif _no_accepted_updates_limit_reached(config, state):
        state["stop_reason"] = "no_accepted_updates"
    due = any(before_decisions < m <= state["decisions"] for m in config["experiment"]["milestones"])
    state["pending_evaluation"] = bool(
        (due or first_accepted or state["stop_reason"])
        if controlled(config)
        else metrics["optimizer_steps"] and (due or state["stop_reason"])
    )
    return first_accepted


def _run_ppo_rollout_cycle(
    config: dict[str, Any],
    plan: dict[str, Any],
    out: Path,
    device: str | torch.device,
    training: _PpoTraining,
    collector: Collector,
    monitor: ResourceMonitor,
    records_path: Path,
) -> None:
    """Collect, update, publish evidence, and evaluate at one rollout boundary."""
    state = training.state
    policy, optimizer = training.policy, training.optimizer
    rng, identity = training.rng, training.identity
    tick = perf_counter()
    deadline = tick + config["experiment"]["active_seconds"] - state["active_seconds"]
    number = state["rollouts_collected"] + 1
    monitor.mark("collection", update=number)
    episodes, next_episode, timings = collector.collect(
        policy,
        None,
        state["next_episode"],
        monitor,
        deadline=deadline,
        max_decisions=config["experiment"]["decisions"] - state["decisions"],
    )
    if not episodes:
        state["stop_reason"] = "active_time_limit"
        return
    summary = summarize_rollout(episodes)
    before_decisions = state["decisions"]
    state["decisions"] += summary["decisions"]
    state["next_episode"] = next_episode
    monitor.mark("rollout_save")
    rollout_record = save_rollout(episodes, out / "rollouts", number)
    metrics: dict[str, Any] = {
        "optimizer_steps": 0,
        "attempted_optimizer_steps": 0,
        "rejected_optimizer_steps": 0,
        "stop_reason": "active_time_limit",
        "update_complete": False,
    }
    delta = 0.0
    if perf_counter() < deadline:
        monitor.mark("ppo", update=number)
        before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        with CompilationClock() as compiler:
            compiler.monitor = monitor
            metrics = update(
                policy,
                optimizer,
                episodes,
                config,
                before_decisions,
                device,
                rng,
                deadline=deadline,
            )
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        metrics["runtime_compilation_seconds"] = sum(max(0, b - a) for a, b in compiler.intervals)
        delta = float(
            torch.sqrt(sum((v - before[k]).float().square().sum() for k, v in policy.state_dict().items()))
        )
        del before
    rollout_hash = rollout_record["sha256"]
    if controlled(config):
        # Buffer cleanup services this update, so charge it before evaluation.
        del episodes
        gc.collect()
    state["active_seconds"] += perf_counter() - tick
    first_accepted = _advance_ppo_state(
        config,
        state,
        training.protocol,
        metrics,
        number=number,
        before_decisions=before_decisions,
    )
    row: dict[str, Any] = {
        "update": number,
        **(
            {
                "total_attempted_steps": state["attempted_steps"],
                "total_rejected_steps": state["rejected_steps"],
                "accepted_updates": state["accepted_updates"],
                "zero_accepted_streak": state["zero_accepted_streak"],
            }
            if controlled(config)
            else {}
        ),
        "trained": bool(metrics["optimizer_steps"]),
        **summary,
        "total_decisions": state["decisions"],
        "total_Adam_steps": state["optimizer_steps"],
        "active_seconds": state["active_seconds"],
        "active_time_overshoot_seconds": max(
            0, state["active_seconds"] - config["experiment"]["active_seconds"]
        ),
        "decision_overshoot": max(0, state["decisions"] - config["experiment"]["decisions"]),
        "rollout_sha256": rollout_hash,
        "rollout_artifact": rollout_record,
        "weight_delta_l2": delta,
        "collection": timings,
        "ppo": metrics,
    }
    servicing_started = perf_counter()
    monitor.mark("checkpoint")
    if metrics["optimizer_steps"] or controlled(config):
        save_boundary(config, out, policy, optimizer, state, rng, identity)
    else:
        write_json(out / "state.json", state)
    if controlled(config):
        servicing_seconds = perf_counter() - servicing_started
        state["active_seconds"] += servicing_seconds
        row["checkpoint_servicing_seconds"] = servicing_seconds
        row["active_seconds"] = state["active_seconds"]
        row["active_time_overshoot_seconds"] = max(
            0, state["active_seconds"] - config["experiment"]["active_seconds"]
        )
        if _active_time_limit_reached(config, state):
            state["stop_reason"] = "active_time_limit"
            state["pending_evaluation"] = True
        write_json(out / "state.json", state)
    append_jsonl(records_path, row)
    print(
        f"update {number}: {summary['W']}/{summary['D']}/{summary['L']}; decisions={state['decisions']}; Adam={state['optimizer_steps']}; active={state['active_seconds']:.1f}s; {metrics['stop_reason']}",
        flush=True,
    )
    if number == 3:
        write_json(
            out / "early-eta.json",
            {
                "decisions": state["decisions"],
                "active_seconds": state["active_seconds"],
                "estimated_total_active_seconds": config["experiment"]["decisions"]
                * state["active_seconds"]
                / state["decisions"],
            },
        )
    if state["pending_evaluation"]:
        monitor.mark("evaluation")
        development(config, plan, out, state)
        if first_accepted:
            state["first_accepted_evaluated"] = True
            write_json(out / "state.json", state)
    # Heavy public input buffers are local; the current rollout is never reused by another update.
    if not controlled(config):
        del episodes
        gc.collect()


def train(
    config: dict[str, Any], plan: dict[str, Any], out: Path, device: str | torch.device
) -> dict[str, Any]:
    state = read_json(out / "state.json")
    if state["stop_reason"]:
        return state
    training = _restore_ppo_training(config, plan, out, device, state)
    with (
        ResourceMonitor(out / "telemetry") as monitor,
        Collector(config, TrainingFactory(config, plan["base"]), device, out / "errors") as collector,
    ):
        _warmup_ppo_training(training, monitor, device)
        records_path = out / "updates.jsonl"
        while not state["stop_reason"]:
            if _active_time_limit_reached(config, state):
                state["stop_reason"] = "active_time_limit"
                break
            if _decision_limit_reached(config, state):
                state["stop_reason"] = "decision_limit"
                break
            _run_ppo_rollout_cycle(config, plan, out, device, training, collector, monitor, records_path)
        monitor.mark("stopped")
        evaluated_hashes = {evaluation["policy_hash"] for evaluation in state["evaluated"].values()}
        if policy_hash(training.policy.state_dict()) not in evaluated_hashes:
            state["pending_evaluation"] = True
        if state["pending_evaluation"]:
            development(config, plan, out, state)
        write_json(out / "state.json", state)
        write_json(
            out / "status.json",
            {"status": "training_complete", "stop_reason": state["stop_reason"]},
        )
    return state


def confirmation(config, plan, out):
    state = read_json(out / "state.json")
    if not state["stop_reason"] or state["pending_evaluation"]:
        raise ValueError("Training/selection not complete")
    selected = out / "selected.json"
    if not selected.exists():
        write_json(selected, state["best"])
    elif read_json(selected) != state["best"]:
        raise ValueError("Selected candidate changed")
    results = {}
    for name, path in (
        ("baseline", out / "checkpoints/initial.pt"),
        ("candidate", Path(state["best"]["path"])),
    ):
        results[name] = evaluate_checkpoint(config, plan, out, path, "confirmation")
    before, after = results["baseline"], results["candidate"]

    def rows(result):
        values, cases, _ = collect_paired_outcomes(
            result["runs"]["argmax"]["cp"]["directory"],
            side_source="metrics",
            check_duplicates=True,
            sort=False,
        )
        return values, cases

    b, bc = rows(before)
    a, ac = rows(after)
    if bc != ac:
        raise ValueError("Confirmation pairing differs")
    baseline_selected = controlled(config) and before["policy_hash"] == after["policy_hash"]
    ci = (
        None
        if baseline_selected
        else paired_world_bootstrap(
            b,
            a,
            samples=config["experiment"]["bootstrap_samples"],
            seed=config["experiment"]["bootstrap_seed"],
            cases=ac,
            cases_per_world=16,
        )
    )
    result = {
        "baseline": before,
        "candidate": after,
        "comparison": ci,
        "baseline_selected": baseline_selected,
        "selection_result": "PPO-кандидат не прошёл development-отбор"
        if baseline_selected
        else "new_candidate",
        "confirmed_improvement": False if ci is None else ci["CI95"][0] > 0,
        "practical_10pp": False if ci is None else ci["delta"] >= 0.1,
        "selected_before_confirmation": True,
    }
    write_json(out / "confirmation.json", result)
    return result


def _audit_replay_selection(cache: dict[str, Any]) -> list[Path]:
    selected: dict[Path, None] = {}
    for evaluation in cache.values():
        if evaluation.get("reused_parent"):
            continue
        for enemies in evaluation["runs"].values():
            for value in enemies.values():
                representatives = select_outcome_representatives(value["directory"], wdl_order=True)
                selected.update(dict.fromkeys(representatives.values()))
    return list(selected)


def audit_replays(config: dict[str, Any], out: Path) -> None:
    selected = _audit_replay_selection(read_json(out / "evaluations.json"))
    if controlled(config):
        # One representative of each observed outcome is sufficient here.
        representatives = select_outcome_representatives(selected)
        selected = list(representatives.values())
    checks = [verify_replay(directory) for directory in selected]
    write_json(out / "replay-verification.json", checks)
    if controlled(config):
        descriptions = []
        for directory, check in zip(selected, checks):
            replay = read_json(directory / "replay.json")
            descriptions.append(
                {
                    "directory": str(directory),
                    "metrics": read_json(directory / "metrics.json"),
                    "commands": dict(Counter(step["action"]["kind"] for step in replay["steps"])),
                    "verification": check,
                }
            )
        write_json(out / "representative-battles.json", descriptions)
    hashes = {
        path.relative_to(out).as_posix(): file_hash(path)
        for path in out.rglob("*")
        if path.is_file()
        and path.name not in ("artifact-hashes.json",)
        and path.name != "run.lock"
        and "__pycache__" not in path.parts
        and path.suffix not in (".log",)
    }
    write_json(out / "artifact-hashes.json", hashes)


def _report_run_failure(config, out):
    if (out / "state.json").exists():
        with suppress(Exception):
            report(config, out)


def run(phase, config, out, device):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with experiment_run_guard(out, phase=phase, on_error_report=lambda: _report_run_failure(config, out)):
        if phase == "prepare":
            return prepare(config, out)
        plan = check_identity(config, out, for_execution=phase in ("train", "evaluate", "run"))
        if phase in ("train", "run"):
            train(config, plan, out, device)
        if phase in ("evaluate", "run"):
            confirmation(config, plan, out)
            write_json(
                out / "status.json",
                {
                    "status": "complete",
                    "stop_reason": read_json(out / "state.json")["stop_reason"],
                },
            )
        report(config, out)
        if phase in ("evaluate", "run"):
            audit_replays(config, out)
