"""Restore and advance a PPO training session through its phases."""

from __future__ import annotations

import gc
import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from civ5rl.engine.collection.collector import Collector
from civ5rl.engine.collection.opponents import OpponentPool
from civ5rl.engine.ppo.adam import (
    check_counters,
    grouped_optimizer,
    optimizer_for,
    uses_grouped_adam,
)
from civ5rl.engine.ppo.storage import load_rollout
from civ5rl.engine.runtime.budget import (
    DeadlineBudget,
    deadline_reached,
)
from civ5rl.engine.runtime.control import (
    TrainingCounters,
    pause_due,
    target_bc_evaluation_due,
)
from civ5rl.engine.runtime.sequence import DEFAULT_PINNED_REPLAY_BYTES
from civ5rl.engine.runtime.telemetry import ResourceMonitor
from civ5rl.experiments.ppo.after_target_bc.boundary import (
    ActiveClock,
    RolloutBoundary,
    heal_interrupted_update,
)
from civ5rl.experiments.ppo.after_target_bc.evaluation import development
from civ5rl.experiments.ppo.after_target_bc.initialization import make_policy, u0_checkpoint
from civ5rl.experiments.ppo.after_target_bc.phases import (
    _commit_ppo_rollout,
    _PpoTrainingSession,
    _step_collect_phase,
    _step_critic_phase,
    _step_ppo_phase,
    skipped_metrics,
)
from civ5rl.experiments.ppo.checkpoints import prune_checkpoints
from civ5rl.protocols.ppo_after_target_bc import KIND
from civ5rl.protocols.ppo_target_scale import TargetFactory
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import (
    ROOT,
    configure,
    file_sha256,
    read_json,
    write_json,
)
from civ5rl.provenance.identity import policy_hash
from civ5rl.provenance.jsonl import read_jsonl


@dataclass(frozen=True)
class _CollectionRuntime:
    workers: int
    reply_size: int
    num_envs: int
    scheduler: str
    inference_min_batch: int
    inference_wait_seconds: float


def _recover_due_evaluation(session: _PpoTrainingSession) -> None:
    """Recover an evaluation only at its original committed policy boundary."""
    config, plan, out, state = session.config, session.plan, session.out, session.state
    policy, shuffle, clock = session.policy, session.shuffle, session.clock
    due = target_bc_evaluation_due(config, state, current_policy_hash=policy_hash(policy.state_dict()))
    if state["phase"] != "collect":
        ledger = out / "updates.jsonl"
        committed = read_jsonl(ledger) if ledger.exists() else []
        previous = {
            **state,
            "decisions": committed[-1]["total_decisions"] if committed else 0,
        }
        if target_bc_evaluation_due(config, previous, current_policy_hash=policy_hash(policy.state_dict())):
            raise ValueError("Cannot recover earlier development evaluation from a pending PPO boundary")
        # A milestone crossed by the pending rollout is evaluated only after PPO.
        due = False
    if due:
        if state["rollouts"]:
            rows = read_jsonl(out / "updates.jsonl")
            if (
                not rows
                or rows[-1].get("rollout") != state["rollouts"]
                or rows[-1].get("total_decisions") != state["decisions"]
            ):
                raise ValueError(
                    "Cannot recover development evaluation: update ledger and checkpoint disagree"
                )
            if not state["evaluated"]:
                raise ValueError("Cannot recover initial development evaluation from a later checkpoint")
            pending = [
                point
                for point in config["experiment"]["milestones"]
                if point <= state["decisions"] and str(point) not in state["development_points"]
            ]
            if any(row.get("total_decisions", 0) >= point for point in pending for row in rows[:-1]):
                raise ValueError(
                    "Cannot recover development evaluation: milestone was crossed before the latest checkpoint"
                )
        development(config, plan, out, state, shuffle, clock)


def _load_validated_collection_runtime(
    config: dict[str, Any], out: Path, runtime_path: Path | str | None
) -> _CollectionRuntime:
    # Runtime capacity can change rollout boundaries; record it separately.
    runtime_path = Path(runtime_path or ROOT / "configs/rl-ppo-after-target-bc-runtime.json")
    runtime = read_json(runtime_path)
    if not {"cpu_workers", "worker_reply_size"} <= set(runtime) or set(runtime) - {
        "cpu_workers",
        "worker_reply_size",
        "num_envs",
        "collection_scheduler",
        "inference_min_batch",
        "inference_wait_seconds",
    }:
        raise ValueError("Unsupported Collection runtime option")
    collection_workers = runtime["cpu_workers"]
    reply_size = runtime["worker_reply_size"]
    num_envs = runtime.get("num_envs", config["collection"]["num_envs"])
    scheduler = runtime.get("collection_scheduler", "ready")
    if (
        type(collection_workers) is not int
        or collection_workers < 1
        or type(num_envs) is not int
        or num_envs < 1
        or collection_workers > num_envs
        or (scheduler != "streaming" and num_envs % collection_workers)
        or type(reply_size) is not int
        or not 1 <= reply_size <= (num_envs + collection_workers - 1) // collection_workers
    ):
        raise ValueError("Runtime workers/reply size must fit the environment count")
    write_json(
        out / "runtime.json",
        {
            "collection_workers": collection_workers,
            "num_envs": num_envs,
            "envs_per_worker": (num_envs + collection_workers - 1) // collection_workers,
            "worker_reply_size": reply_size,
            "runtime_config": str(runtime_path),
            "runtime_config_sha256": file_sha256(runtime_path),
            "ppo_pinned_replay": True,
            "pinned_replay_limit_bytes": DEFAULT_PINNED_REPLAY_BYTES,
            "collection_scheduler": scheduler,
            "inference_min_batch": runtime.get("inference_min_batch", 0),
            "inference_wait_seconds": runtime.get("inference_wait_seconds", 0),
            "inference_coalescing": runtime.get("inference_wait_seconds", 0) > 0,
        },
    )
    return _CollectionRuntime(
        workers=collection_workers,
        reply_size=reply_size,
        num_envs=num_envs,
        scheduler=scheduler,
        inference_min_batch=runtime.get("inference_min_batch", 0),
        inference_wait_seconds=runtime.get("inference_wait_seconds", 0),
    )


def train(
    config: dict[str, Any],
    plan: dict[str, Any],
    out: Path,
    device: str | torch.device,
    runtime_path: Path | str | None = None,
    max_updates: int | None = None,
) -> None:
    state = read_json(out / "state.json")
    heal_interrupted_update(out, state)
    counters = TrainingCounters.from_dict(state, protocol=KIND)
    state.update(counters.to_state_dict(protocol=KIND))
    state.setdefault("updates", counters.rollouts_with_accepted_steps)
    boundary = RolloutBoundary.from_state(state)
    boundary.validate_sufficiency(out)
    configure(config.get("seed", 17), device)
    record = read_json(out / "run.json")
    identity = record.get("checkpoint_identity", {"run_id": record["run_id"]})
    if "checkpoint_identity" in record:
        checkpoint.verify_execution_passport(out, config, plan)
    shuffle = random.Random(record["shuffle_seed"])
    clock = ActiveClock(state, out)
    try:
        policy = make_policy(config, device)
        optimizer = (
            grouped_optimizer(policy, config, device)
            if uses_grouped_adam(config)
            else optimizer_for(policy, config, device)
        )
        restored = checkpoint.restore(out / state["latest"], policy, optimizer, shuffle, config, identity)
        if uses_grouped_adam(config):
            check_counters(optimizer, restored)
            # A value-only capsule, if present, is restored by adapt_critic below.
            state["critic_progress"] = restored.get("critic_progress", {})
        if restored.get("pending_update"):
            heal_interrupted_update(out, restored)
        for key in (
            "phase",
            "latest",
            "rollouts",
            "updates",
            "decisions",
            "optimizer_steps",
            "attempted_steps",
            "rejected_steps",
            "rollouts_collected",
            "rollouts_with_accepted_steps",
            "adam_steps_attempted",
            "adam_steps_accepted",
            "self_play_archive",
            "self_play_last_archived_optimizer_steps",
        ):
            if key in restored and key in state and restored[key] != state[key]:
                raise ValueError(f"Saved checkpoint and state disagree on {key}")
        session = _PpoTrainingSession(
            config,
            plan,
            out,
            device,
            state,
            policy,
            optimizer,
            shuffle,
            identity,
            clock,
        )
        del optimizer
        _recover_due_evaluation(session)
        if pause_due(out, "update", state["updates"], at_boundary=state["phase"] == "collect"):
            write_json(out / "status.json", {"status": "paused", "checkpoint": state["latest"]})
            return
        if state["stop_reason"]:
            return
        factory = TargetFactory(
            config,
            plan["base"],
            archive_provider=(
                (lambda: state.get("self_play_archive", []))
                if config.get("self_play", {}).get("enabled")
                else None
            ),
        )
        opponent_pool = None
        if config.get("self_play", {}).get("enabled"):
            opponent_pool = OpponentPool(config, device, out, parent_checkpoint=out / u0_checkpoint(config))
            opponent_pool.sync_archive(state.get("self_play_archive", []))
        runtime = _load_validated_collection_runtime(config, out, runtime_path)
        with (
            ResourceMonitor(out / "telemetry") as monitor,
            Collector(
                config,
                factory,
                device,
                out / "errors",
                cpu_workers=runtime.workers,
                reply_size=runtime.reply_size,
                num_envs=runtime.num_envs,
                collection_scheduler=runtime.scheduler,
                inference_min_batch=runtime.inference_min_batch,
                inference_wait_seconds=runtime.inference_wait_seconds,
                opponent_pool=opponent_pool,
                record_hidden=False,
            ) as collector,
            monitor.capture_cuda_oom(policy),
        ):
            checked_final_evaluation = False
            while not state["stop_reason"]:
                critic_started = perf_counter() if state["phase"] == "critic" else None
                if max_updates is not None and state["updates"] >= max_updates:
                    write_json(
                        out / "status.json",
                        {"status": "paused", "reason": "max_updates"},
                    )
                    return
                deadline = DeadlineBudget(lambda: clock.deadline(config))
                if deadline_reached(deadline):
                    state["stop_reason"] = "active_time_limit"
                    break
                if pause_due(
                    out,
                    "update",
                    state["updates"],
                    at_boundary=state["phase"] == "collect",
                ):
                    write_json(out / "status.json", {"status": "paused"})
                    return
                if state["phase"] == "collect":
                    episodes, critic_started = _step_collect_phase(session, collector, monitor, deadline)
                    if episodes is None:
                        break
                else:
                    episodes = load_rollout(out / state["pending_rollout"])

                metrics = skipped_metrics(state["stop_reason"] or "active_time_limit")
                monitor.record_cuda_memory(policy, "before_optimization", rollout=state["rollouts"])
                if not state["stop_reason"] and state["phase"] == "critic":
                    _step_critic_phase(session, episodes, monitor, deadline, critic_started)
                metrics = _step_ppo_phase(session, episodes, monitor, deadline, metrics)
                del episodes
                gc.collect()
                checked_final_evaluation = _commit_ppo_rollout(
                    session, metrics, monitor, factory, opponent_pool
                )
                if pause_due(out, "update", state["updates"]):
                    write_json(
                        out / "status.json",
                        {"status": "paused", "checkpoint": state["latest"]},
                    )
                    return
                prune_checkpoints(out, state)
                clock.progress()
                if max_updates is not None and state["updates"] >= max_updates and not state["stop_reason"]:
                    write_json(
                        out / "status.json",
                        {"status": "paused", "reason": "max_updates"},
                    )
                    return
            if not checked_final_evaluation and policy_hash(policy.state_dict()) not in state["evaluated"]:
                development(config, plan, out, state, shuffle, clock)
                monitor.record_cuda_memory(policy, "after_evaluation")
            clock.progress()
    finally:
        clock.progress()
