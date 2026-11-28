"""Behavior-cloning epoch execution and checkpoint metrics."""

from __future__ import annotations

import random
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from civ5rl.core.model import Policy
from civ5rl.engine.evaluation.runtime import EVALUATOR_VERSION
from civ5rl.engine.imitation.core import label_minibatches, teacher_metrics
from civ5rl.engine.imitation.pipeline import (
    run_bc_minibatch_stream,
)
from civ5rl.engine.imitation.storage import PreparedDemonstrations
from civ5rl.engine.runtime.budget import ActiveBudget
from civ5rl.engine.runtime.control import (
    TrainingCounters,
    pause_due,
)
from civ5rl.engine.runtime.telemetry import CompilationClock, ResourceMonitor
from civ5rl.engine.runtime.telemetry_analysis import analyze_telemetry
from civ5rl.engine.runtime.transactions import FrozenParamsCheck
from civ5rl.experiments._shared.lifecycle import (
    ActiveSessionClock,
)
from civ5rl.experiments.evaluation.matches import evaluate_target_scale
from civ5rl.experiments.imitation.helpers import (
    isolated_rng,
)
from civ5rl.experiments.imitation.restoration import (
    EpochBatches,
    PreparedEpochState,
    restore_epoch_cursor_and_batches,
)
from civ5rl.experiments.imitation.selection import (
    refresh_selection,
)
from civ5rl.protocols.imitation_target_scale import (
    C2_KIND,
)
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import (
    read_json,
    write_json,
)
from civ5rl.provenance.identity import policy_hash


def fixed_metrics(policy, cache, micro):
    before = cache.mode
    cache.mode = "diagnostic"
    try:
        return teacher_metrics(policy, cache, micro_size=micro, by_unit_type=True)
    finally:
        cache.mode = before


def measure_checkpoint(config, plan, out, path, epoch, steps, policy, fixed, micro, shuffle):
    fingerprint = policy_hash(policy.state_dict())
    cache = read_json(out / "evaluations.json") if (out / "evaluations.json").exists() else {}
    previous = cache.get("development:" + fingerprint)
    if previous and "fixed_training" in previous and previous.get("runtime", {}).get("evaluator_version") == EVALUATOR_VERSION:
        refresh_selection(config, out)
        return previous
    with isolated_rng(shuffle):
        diagnostic = fixed_metrics(policy, fixed, micro)
        point = evaluate_target_scale(config, plan, out, path, "development")
    point = {
        **point,
        "checkpoint": path,
        "epoch": epoch,
        "Adam_steps": steps,
        "fixed_training": diagnostic,
    }
    evaluations = read_json(out / "evaluations.json")
    evaluations["development:" + point["policy_hash"]] = point
    write_json(out / "evaluations.json", evaluations)
    refresh_selection(config, out)
    return point


def _prepare_epoch(config, cache, shuffle, epoch_state, checkpoint_state, epoch):
    size = config["labels_per_minibatch"]
    if epoch_state.resume_mid_epoch and epoch == epoch_state.start_epoch:
        return restore_epoch_cursor_and_batches(
            checkpoint_state, epoch_state.history, epoch_state.previous_records, cache.masks, size=size
        )
    order = cache.chunks(config["sequence_length"])
    shuffle.shuffle(order)
    return EpochBatches(order, 0, 0, iter(label_minibatches(order, cache.masks, size=size)), [])


def _epoch_result(epoch, records, completed):
    labels = sum(row["labels"] for row in records)
    return {
        "epoch": epoch,
        "completed": completed,
        "steps": len(records),
        "labels": labels,
        "online_CE": sum(row["ce"] * row["labels"] for row in records) / labels,
    }


def _offline_epoch_metrics(policy, fixed, holdout, micro):
    metrics = {"fixed_training": fixed_metrics(policy, fixed, micro)}
    if holdout is not None:
        metrics["holdout"] = fixed_metrics(policy, holdout, micro)
    return metrics


def _checkpoint_stop_reason(config, budget, epoch, completed, reason, step_limit_reached):
    terminal_step = (
        (not completed and reason != "paused")
        or budget.expired()
        or step_limit_reached
        or (completed and epoch == config["epochs"])
    )
    if not terminal_step:
        return None
    if step_limit_reached:
        return "max_adam_steps"
    if not completed or budget.expired():
        return "active_time_limit"
    return "epoch_limit"


def _write_utilization_review(out, cache, *, records=None):
    analysis = analyze_telemetry(out / "training-telemetry/resources.jsonl")
    windows = analysis["windows"]
    utilization = {"metrics": analysis["metrics"], "last_60s": windows[-1] if windows else None}
    if records is None:
        utilization.update(
            windows_count=len(windows),
            min_60s_core_average=min((window["core_average"] for window in windows), default=None),
            max_60s_core_average=max((window["core_average"] for window in windows), default=None),
            measured_cache=cache.statistics(),
        )
    else:
        utilization.update(
            measured_cache=cache.statistics(),
            mean_step_seconds=sum(row["seconds"] for row in records) / len(records),
        )
    utilization.update(
        prefix_seconds_not_separately_profiled=True,
        continuation_reason=(
            "Bounded CPU episode input, full current-weight GRU-prefix and small fixed spatial blocks constrain "
            "dispatch. The configured label objective and precision are preserved."
        ),
    )
    write_json(out / "GPU-utilization-review.json", utilization)
    return utilization


def run_bc_epochs(
    config: dict[str, Any],
    plan: dict[str, Any],
    out: Path,
    policy: Policy,
    optimizer: torch.optim.Optimizer,
    cache: PreparedDemonstrations,
    fixed: PreparedDemonstrations,
    holdout: PreparedDemonstrations | None,
    shuffle: random.Random,
    monitor: ResourceMonitor,
    compilation: CompilationClock,
    frozen: FrozenParamsCheck,
    epoch_state: PreparedEpochState,
    *,
    checkpoint_state: dict[str, Any],
    corpus_labels: int,
    origin_steps: int,
    selected: int,
    source: str,
    resumed: bool,
) -> bool:
    """Run restored or fresh epochs; return whether training reached its final boundary."""
    kind = config["kind"]
    steps = epoch_state.steps
    labels = epoch_state.labels
    history = epoch_state.history
    clock = ActiveSessionClock(epoch_state.clock_offset)
    budget = ActiveBudget(out, elapsed=clock.seconds)
    reason = "epoch_limit"
    last_path = source
    utilization = (
        read_json(out / "GPU-utilization-review.json")
        if (out / "GPU-utilization-review.json").exists()
        else None
    )
    max_steps = config.get("max_adam_steps")
    evaluate_milestones = kind != C2_KIND and not config.get("skip_milestone_evaluations", False)
    check_steps = {1}
    if resumed:
        check_steps.add(origin_steps + 1)

    def save_checkpoint(
        path: str,
        epoch: int,
        work: EpochBatches,
        complete: bool,
        stop_reason: str | None = None,
    ) -> None:
        """Save the drained minibatch stream before publishing its state pointer."""
        monitor.mark("checkpoint", epoch=epoch, Adam_steps=steps)
        serial_order = [[int(episode), int(start), int(stop)] for episode, start, stop in work.order]
        step_counters = TrainingCounters(
            rollouts_collected=0,
            rollouts_with_accepted_steps=0,
            adam_steps_attempted=steps,
            adam_steps_accepted=steps,
        )
        state = {
            **step_counters.to_state_dict(protocol=kind),
            "epoch": epoch,
            "epoch_completed": complete,
            "batch_cursor": work.cursor,
            "epoch_order": serial_order,
            "Adam_steps": steps,
            "free_labels_processed": labels,
            "active_seconds_at_save_start": clock.seconds(),
            "active_seconds_limit": budget.limit(),
            "budget_revision_applied": budget.revision,
            "history_positions": cache.counters["history_positions"],
            "prefix_positions": cache.counters["prefix_positions"],
            "epochs_history": deepcopy(history),
            "stop_reason": stop_reason,
        }
        frozen.verify()
        checkpoint.save(
            out / path,
            policy,
            optimizer,
            state,
            shuffle,
            config,
            {"run_id": read_json(out / "run.json")["run_id"]},
        )
        write_json(
            out / "state.json",
            {**state, "active_seconds": clock.seconds(), "latest": path},
        )

    def evaluate_milestone(epoch):
        monitor.mark("evaluation")
        with clock.pause():
            measure_checkpoint(config, plan, out, last_path, epoch, steps, policy, fixed, selected, shuffle)

    def training_progress(status):
        return {
            "status": status,
            "epochs": history,
            "Adam_steps": steps,
            "free_labels_processed": labels,
            "active_seconds": clock.seconds(),
            "latest": last_path,
        }

    for epoch in range(epoch_state.start_epoch, config["epochs"] + 1):
        if budget.expired():
            reason = "active_time_limit"
            break
        if max_steps is not None and steps >= max_steps:
            reason = "max_adam_steps"
            break
        work = _prepare_epoch(config, cache, shuffle, epoch_state, checkpoint_state, epoch)
        completed = True
        monitor.mark("iterative_training", epoch=epoch, Adam_steps=steps)
        stop_hit = None

        def _check_stop() -> str | None:
            nonlocal stop_hit
            if budget.expired():
                stop_hit = "active_time_limit"
                return stop_hit
            return None

        def _on_step(
            record: dict[str, Any],
            completed_steps: int,
            current_epoch: int = epoch,
            current_work: EpochBatches = work,
        ) -> None:
            nonlocal steps, labels, utilization
            steps = completed_steps
            labels += record["labels"]
            current_work.cursor += 1
            current_work.epoch_labels += record["labels"]
            current_work.records.append(record)
            if steps == 1:
                write_json(
                    out / "first-step-check.json",
                    {"before": 0, "after": 1, "all_counters_equal": True},
                )
            if resumed and steps == origin_steps + 1:
                write_json(
                    out / "first-restarted-step-check.json",
                    {
                        "before": origin_steps,
                        "after": steps,
                        "all_counters_equal": True,
                    },
                )
            if steps % 10 == 0:
                print(
                    f"BC epoch={current_epoch}, Adam={steps}, labels={labels}, CE={record['ce']:.4f}, active={clock.seconds():.1f}s",
                    flush=True,
                )
            if clock.seconds() >= 600 and utilization is None:
                utilization = _write_utilization_review(out, cache, records=current_work.records)

        run_bc_minibatch_stream(
            policy=policy,
            optimizer=optimizer,
            cache=cache,
            batches=work.batches,
            micro_size=selected,
            max_gradient_norm=0.5,
            steps_log_path=out / "steps.jsonl",
            compilation=compilation,
            epoch=epoch,
            initial_steps=steps,
            active_seconds=clock.seconds,
            max_steps=max_steps,
            should_stop=_check_stop,
            on_step_completed=_on_step,
            check_counter_steps=check_steps,
        )
        if work.epoch_labels < corpus_labels:
            completed = False
            if stop_hit is not None:
                reason = stop_hit
            elif max_steps is not None and steps >= max_steps:
                reason = "max_adam_steps"
        if work.records:
            epoch_entry = _epoch_result(epoch, work.records, completed)
            if kind == C2_KIND:
                monitor.mark("offline_metrics", epoch=epoch, Adam_steps=steps)
                with clock.pause(), isolated_rng(shuffle):
                    epoch_entry.update(_offline_epoch_metrics(policy, fixed, holdout, selected))
                fixed.clear()
                if holdout is not None:
                    holdout.clear()
            history.append(epoch_entry)
            last_path = (
                f"checkpoints/epoch-{epoch:02d}.pt"
                if completed
                else f"checkpoints/partial-epoch-{epoch:02d}-step-{steps:05d}.pt"
            )
            final_reason = _checkpoint_stop_reason(
                config, budget, epoch, completed, reason, max_steps is not None and steps >= max_steps
            )
            save_checkpoint(last_path, epoch, work, completed, stop_reason=final_reason)
        if reason == "paused" or (completed and work.records and pause_due(out, "epoch", epoch)):
            if evaluate_milestones and completed and work.records and epoch in config["evaluate_epochs"]:
                evaluate_milestone(epoch)
                fixed.clear()
            write_json(out / "training.json", training_progress("paused"))
            write_json(
                out / "status.json",
                {"status": "paused", "checkpoint": last_path},
            )
            cache.clear()
            fixed.clear()
            if holdout is not None:
                holdout.clear()
            return False
        terminal = not completed or budget.expired() or (max_steps is not None and steps >= max_steps)
        if terminal and reason == "epoch_limit":
            reason = (
                "max_adam_steps" if (max_steps is not None and steps >= max_steps) else "active_time_limit"
            )
        if evaluate_milestones and work.records and (epoch in config["evaluate_epochs"] or terminal):
            evaluate_milestone(epoch)
            fixed.clear()
            monitor.mark("iterative_training", epoch=epoch)
        write_json(
            out / "training.json",
            {
                **training_progress("running"),
                "cache": cache.statistics(),
                "microbatch": selected,
            },
        )
        if terminal:
            break
    # Ensure state.json records the terminal stop_reason if we broke between epochs.
    if (out / "state.json").exists():
        final_state = read_json(out / "state.json")
        if final_state.get("stop_reason") != reason:
            final_state["stop_reason"] = reason
            final_state["active_seconds"] = clock.seconds()
            write_json(out / "state.json", final_state)
    # A budget boundary between epochs must also evaluate the last changed policy.
    if evaluate_milestones and steps:
        evaluate_milestone(history[-1]["epoch"])
    frozen.verify()
    elapsed = clock.seconds()
    if elapsed >= 600 and (out / "training-telemetry/resources.jsonl").exists():
        utilization = _write_utilization_review(out, cache)
    write_json(
        out / "training.json",
        {
            "status": "complete",
            "stop_reason": reason,
            "epochs": history,
            "Adam_steps": steps,
            "free_labels_processed": labels,
            "active_seconds": elapsed,
            "time_overshoot": max(0, elapsed - budget.limit()),
            "latest": last_path,
            "microbatch": selected,
            "cache": cache.statistics(),
            "value_head_unchanged": True,
            "GPU_utilization_review": utilization,
        },
    )
    cache.clear()
    fixed.clear()
    return True
