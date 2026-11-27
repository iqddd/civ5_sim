"""Execute rollout collection, critic fitting, and PPO at committed boundaries."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

import torch

from civ5rl.core.model import Policy
from civ5rl.engine.collection.collector import Collector
from civ5rl.engine.collection.opponents import OpponentPool, maybe_archive_snapshot
from civ5rl.engine.ppo.adam import (
    adapts_value,
    check_counters,
    optimizer_for,
    uses_grouped_adam,
)
from civ5rl.engine.ppo.backend import BatchedBackend
from civ5rl.engine.ppo.core import update
from civ5rl.engine.ppo.storage import save_rollout
from civ5rl.engine.runtime.budget import (
    DeadlineBudget,
    deadline_reached,
)
from civ5rl.engine.runtime.telemetry import (
    CompilationClock,
    ResourceMonitor,
)
from civ5rl.experiments.imitation.helpers import isolated_rng
from civ5rl.experiments.ppo.after_target_bc.boundary import ActiveClock, Phase, RolloutBoundary, save_boundary
from civ5rl.experiments.ppo.after_target_bc.evaluation import development
from civ5rl.experiments.ppo.critic import adapt_critic, prepare_critic
from civ5rl.experiments.ppo.rollout_summary import summarize_rollout
from civ5rl.protocols.ppo_after_target_bc import (
    record_collection,
    record_optimization,
)
from civ5rl.protocols.ppo_target_scale import TargetFactory
from civ5rl.provenance.common import write_json
from civ5rl.provenance.identity import policy_hash
from civ5rl.provenance.jsonl import append_jsonl

if TYPE_CHECKING:
    from civ5rl.core.contracts import Episode


def skipped_metrics(reason):
    return {
        "PPO_skipped": True,
        "stop_reason": reason,
        "optimizer_steps": 0,
        "attempted_optimizer_steps": 0,
        "rejected_optimizer_steps": 0,
    }


@dataclass
class _PpoTrainingSession:
    config: dict[str, Any]
    plan: dict[str, Any]
    out: Path
    device: str | torch.device
    state: dict[str, Any]
    policy: Policy
    optimizer: torch.optim.Optimizer
    shuffle: random.Random
    identity: dict[str, Any]
    clock: ActiveClock


def _step_collect_phase(
    session: _PpoTrainingSession,
    collector: Collector,
    monitor: ResourceMonitor,
    deadline: DeadlineBudget,
) -> tuple[list[dict[str, Any]] | None, float | None]:
    """Collect and publish a resumable pending rollout."""
    config, out, state = session.config, session.out, session.state
    policy, optimizer = session.policy, session.optimizer
    shuffle, identity, clock = session.shuffle, session.identity, session.clock
    critic_started = None
    if state["decisions"] >= config["experiment"]["decisions"]:
        state["stop_reason"] = "decision_limit"
        return None, critic_started
    monitor.mark("collection", rollout=state["rollouts"] + 1)
    monitor.record_cuda_memory(policy, "before_collection")
    episodes, following, timing = collector.collect(
        policy,
        None,
        state["next_episode"],
        monitor,
        deadline=deadline,
        max_decisions=config["experiment"]["decisions"] - state["decisions"],
        episode_length_hint=(
            state["rollout_summary"]["decisions"] / max(1, state["rollout_summary"]["episodes"])
            if state.get("rollout_summary")
            else None
        ),
    )
    if not state["critic_complete"]:
        critic_started = perf_counter()
    if not episodes:
        state["stop_reason"] = "active_time_limit"
        return None, critic_started
    summary = summarize_rollout(episodes)
    record_collection(state, summary, config)
    state["next_episode"] = following
    state["collection"] = timing
    state["rollout_summary"] = summary
    artifact = save_rollout(episodes, out / "rollouts", state["rollouts"])
    state["pending_rollout"] = "rollouts/" + artifact["path"]
    state["phase"] = Phase.PPO.value if state["critic_complete"] else Phase.CRITIC.value
    state["latest"] = "checkpoints/pending.pt"
    save_boundary(out, policy, optimizer, state, shuffle, config, identity, clock)
    RolloutBoundary.from_state(state).validate_sufficiency(out)
    print(
        f"Sampling rollout {state['rollouts']}: {summary}; total={state['decisions']}",
        flush=True,
    )
    return episodes, critic_started


def _step_critic_phase(
    session: _PpoTrainingSession,
    episodes: list[dict[str, Any]],
    monitor: ResourceMonitor,
    deadline: DeadlineBudget,
    critic_started: float | None,
) -> None:
    """Refresh critic values and checkpoint the resulting PPO input boundary."""
    config, out, state, device = (
        session.config,
        session.out,
        session.state,
        session.device,
    )
    policy = session.policy
    shuffle, identity, clock = session.shuffle, session.identity, session.clock
    monitor.mark("critic_preparation")
    with CompilationClock() as compiler:
        compiler.monitor = monitor
        state["critic"] = (
            adapt_critic(
                policy,
                session.optimizer,
                episodes,
                config,
                device,
                out,
                state,
                shuffle,
                identity,
                clock,
                deadline,
            )
            if adapts_value(config)
            else prepare_critic(
                policy,
                episodes,
                config["critic"],
                device,
                gamma=float(config["ppo"].get("gamma", 1.0)),
                gae_lambda=float(config["ppo"].get("gae_lambda", 1.0)),
                deadline=deadline,
                progress=clock.progress,
                preparation_started=critic_started,
            )
        )
    state["critic"]["runtime_compilation_seconds"] = sum(b - a for a, b in compiler.intervals)
    state["critic_complete"] = True
    state["phase"] = Phase.PPO.value
    artifact = save_rollout(episodes, out / "critic-rollout", 1)
    state["pending_rollout"] = "critic-rollout/" + artifact["path"]
    if not adapts_value(config):
        session.optimizer = optimizer_for(policy, config, device)
        assert not session.optimizer.state
    state["latest"] = "checkpoints/critic-ready.pt"
    write_json(out / "critic-preparation.json", state["critic"])
    save_boundary(out, policy, session.optimizer, state, shuffle, config, identity, clock)
    RolloutBoundary.from_state(state).validate_sufficiency(out)
    print("Critic preparation: " + json.dumps(state["critic"]), flush=True)


def _step_ppo_phase(
    session: _PpoTrainingSession,
    episodes: list[dict[str, Any]],
    monitor: ResourceMonitor,
    deadline: DeadlineBudget,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Run optimization when time remains and update its signed counters."""
    config, out, state, device = (
        session.config,
        session.out,
        session.state,
        session.device,
    )
    policy, optimizer, shuffle = session.policy, session.optimizer, session.shuffle
    if not state["stop_reason"] and not deadline_reached(deadline):
        if config["experiment"].get("diagnostic_training_only") and state["rollouts"] in (2, 3):
            from civ5rl.experiments.ppo.after_target_bc.diagnostic import record_pre_update

            with isolated_rng(shuffle):
                monitor.mark("diagnostic", rollout=state["rollouts"])
                record_pre_update(policy, episodes, config, out, device, state["rollouts"])
        monitor.mark("ppo", rollout=state["rollouts"])

        def inspect_step(step: dict[str, Any], optimizer: torch.optim.Optimizer = optimizer) -> None:
            if (
                step["accepted"]
                and step.get("full_kl_checked", True)
                and not (out / "first-accepted-step.json").exists()
            ):
                counters = {int(value["step"]) for value in optimizer.state.values()}
                expected_counter = (
                    int(state.get("origin_Adam_counter", 0))
                    + int(state["optimizer_steps"])
                    + int(step["committed_accepted_steps"])
                )
                if uses_grouped_adam(config):
                    grouped = check_counters(
                        optimizer,
                        state,
                        int(step["committed_accepted_steps"]),
                    )
                elif counters != {expected_counter} or int(step["accepted_step"]) != int(
                    step["committed_accepted_steps"]
                ):
                    raise RuntimeError(
                        f"First PPO Adam counter differs from committed steps: {counters} != {{{expected_counter}}}"
                    )
                write_json(
                    out / "first-accepted-step.json",
                    {
                        **(
                            {"Adam_counters": grouped}
                            if uses_grouped_adam(config)
                            else {"Adam_counter": expected_counter}
                        ),
                        "inherited_Adam_counter": state.get(
                            "origin_Adam_counters",
                            int(state.get("origin_Adam_counter", 0)),
                        ),
                        "committed_accepted_steps": state["optimizer_steps"]
                        + int(step["committed_accepted_steps"]),
                        "step": step,
                    },
                )

        with CompilationClock() as compiler:
            compiler.monitor = monitor
            operation_started = perf_counter()
            metrics = update(
                policy,
                optimizer,
                cast("list[Episode]", episodes),
                config,
                state["decisions"],
                device,
                shuffle,
                monitor=monitor,
                deadline=deadline,
                on_step=inspect_step,
                _backend=partial(BatchedBackend, pinned_replay=True, consume_encoded=True),
            )
            operation_ended = perf_counter()
        metrics["runtime_compilation_seconds"] = sum(b - a for a, b in compiler.intervals)
        metrics["operation_seconds"] = operation_ended - operation_started
        metrics["operation_without_compilation_seconds"] = metrics["operation_seconds"] - compiler.seconds(
            operation_started, operation_ended
        )
        metrics["PPO_skipped"] = False
        record_optimization(state, metrics, config)
    elif not state["stop_reason"]:
        state["stop_reason"] = "active_time_limit"
    return metrics


def _commit_ppo_rollout(
    session: _PpoTrainingSession,
    metrics: dict[str, Any],
    monitor: ResourceMonitor,
    factory: TargetFactory,
    opponent_pool: OpponentPool | None,
) -> bool:
    """Commit a rollout and return whether its terminal evaluation was considered."""
    config, plan, out, state = session.config, session.plan, session.out, session.state
    policy, optimizer = session.policy, session.optimizer
    shuffle, identity, clock = session.shuffle, session.identity, session.clock
    monitor.record_cuda_memory(policy, "after_update_release")
    clock.sync()
    if state["active_seconds"] >= clock.limit(config):
        state["stop_reason"] = state["stop_reason"] or "active_time_limit"
    elif state["decisions"] >= config["experiment"]["decisions"]:
        state["stop_reason"] = state["stop_reason"] or "decision_limit"
    row = {
        "rollout": state["rollouts"],
        **state["rollout_summary"],
        "total_decisions": state["decisions"],
        "active_seconds": state["active_seconds"],
        "winless_streak": state["winless_streak"],
        "zero_accepted_streak": state["zero_accepted_streak"],
        "decision_overshoot": max(0, state["decisions"] - config["experiment"]["decisions"]),
        "active_time_overshoot_seconds": max(
            0,
            state["active_seconds"] - clock.limit(config),
        ),
        "collection": state["collection"],
        "ppo": metrics,
    }
    # The checkpoint carries the row until the ledger append has completed.
    state["pending_update"] = row
    state["phase"] = Phase.COLLECT.value
    state["latest"] = f"checkpoints/update-{state['rollouts']:04}.pt"
    fingerprint_value: str | None = None

    def fingerprint() -> str:
        # Archive selection and final evaluation share only this rollout's policy hash.
        nonlocal fingerprint_value
        if fingerprint_value is None:
            fingerprint_value = policy_hash(policy.state_dict())
        return fingerprint_value

    archived = None
    if opponent_pool is not None:
        archived = maybe_archive_snapshot(
            state,
            config,
            checkpoint_relpath=state["latest"],
            policy_hash_value=fingerprint,
        )
    monitor.mark("checkpoint")
    save_boundary(out, policy, optimizer, state, shuffle, config, identity, clock)
    if opponent_pool is not None and archived is not None:
        opponent_pool.add_archive_snapshot(archived, policy)
    append_jsonl(out / "updates.jsonl", row)
    state.pop("pending_update", None)
    state.pop("pending_rollout", None)
    write_json(out / "state.json", state)
    RolloutBoundary.from_state(state).validate_sufficiency(out)
    write_json(out / "coverage.json", factory.summary())
    print(
        f"Rollout {state['rollouts']}: W/D/L={row['W']}/{row['D']}/{row['L']}; decisions={state['decisions']}; Adam={state['optimizer_steps']}; active={state['active_seconds']:.1f}s; {metrics['stop_reason']}",
        flush=True,
    )
    due = any(
        str(point) not in state["development_points"] and point <= state["decisions"]
        for point in config["experiment"]["milestones"]
    )
    if not config["experiment"].get("diagnostic_training_only") and (
        due or (state["stop_reason"] and fingerprint() not in state["evaluated"])
    ):
        monitor.mark("evaluation")
        development(config, plan, out, state, shuffle, clock)
        monitor.record_cuda_memory(policy, "after_evaluation")
    return bool(state["stop_reason"])
