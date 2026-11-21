"""Bounded PPO from BC or a preserved PPO boundary, with recoverable evaluation."""

from __future__ import annotations

import gc
import json
import random
import subprocess
from collections.abc import Iterator
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch.nn import functional as F

from civ5rl.core.model import Policy, batch_inputs
from civ5rl.core.preview import extends_parent, inherited_model
from civ5rl.engine.collection.collector import Collector
from civ5rl.engine.collection.opponents import OpponentPool, maybe_archive_snapshot
from civ5rl.engine.evaluation.executor import evaluation_session, runtime_for
from civ5rl.engine.ppo.adam import (
    adapts_value,
    check_counters,
    expected_counters,
    grouped_optimizer,
    parameter_group,
    uses_grouped_adam,
)
from civ5rl.engine.ppo.backend import BatchedBackend
from civ5rl.engine.ppo.core import (
    encode_episode,
    episode_returns,
    raw_gae_advantages,
    targets,
    update,
)
from civ5rl.engine.runtime.budget import (
    DeadlineBudget,
    deadline_reached,
    resolve_deadline,
)
from civ5rl.engine.runtime.control import (
    TrainingCounters,
    assert_not_closed,
    pause_due,
    target_bc_evaluation_due,
)
from civ5rl.engine.runtime.sequence import (
    DEFAULT_PINNED_REPLAY_BYTES,
    DEFAULT_PREFIX_BLOCK_SIZE,
)
from civ5rl.engine.runtime.telemetry import (
    ResourceMonitor,
    performance_markdown,
    summarize_performance,
)
from civ5rl.experiments._shared.critic_ops import (
    behavior_digest,
    critic_statistics,
    refresh_episode_old_values,
    value_only_grad_mode,
    verify_critic_invariance,
)
from civ5rl.experiments._shared.io import append_jsonl, read_jsonl
from civ5rl.experiments._shared.lifecycle import (
    PersistedActiveSessionClock,
    experiment_run_guard,
)
from civ5rl.experiments._shared.markdown import MarkdownReport, format_wdl
from civ5rl.experiments._shared.outcomes import collect_paired_outcomes
from civ5rl.experiments._shared.provenance_checks import verify_manifest_digest
from civ5rl.experiments.evaluation import paired_world_bootstrap
from civ5rl.experiments.h2h_evaluation import (
    dual_promotion_decision,
    evaluate_h2h,
    select_dual_milestone,
)
from civ5rl.experiments.imitation_helpers import (
    CompilationClock,
    assert_equal,
    isolated_rng,
    load_policy,
)
from civ5rl.experiments.ppo_after_imitation import (
    optimizer_for,
    policy_hash,
    summarize_rollout,
)
from civ5rl.experiments.ppo_artifacts import (
    load_rollout,
    prune_checkpoints,
    save_rollout,
)
from civ5rl.experiments.ppo_critic_adaptation import adapt_critic
from civ5rl.experiments.ppo_target_scale import evaluate
from civ5rl.protocols.ppo_after_target_bc import (
    KIND,
    check_development_gain,
    compute_time_budget_check,
    record_collection,
    record_optimization,
    validate_profile,
)
from civ5rl.protocols.ppo_target_scale import TargetFactory
from civ5rl.protocols.turn_reset import evaluation_plan, validate_case
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import (
    ROOT,
    configure,
    file_sha256,
    read_json,
    seed_for,
    write_json,
)
from civ5sim.engine import digest

if TYPE_CHECKING:
    from civ5rl.core.contracts import Episode


class Phase(str, Enum):
    COLLECT = "collect"
    CRITIC = "critic"
    PPO = "ppo"


def heal_interrupted_update(out: Path, state: dict) -> bool:
    """If pending_update is present and not flushed to updates.jsonl, flush it."""
    out = Path(out)
    pending = state.get("pending_update")
    if not pending:
        return False
    ledger = out / "updates.jsonl"
    committed = read_jsonl(ledger) if ledger.is_file() else []
    healed = False
    if not committed or committed[-1].get("rollout") != pending.get("rollout"):
        append_jsonl(ledger, pending)
        healed = True
    state.pop("pending_update", None)
    write_json(out / "state.json", state)
    return healed


@dataclass
class RolloutBoundary:
    checkpoint: str
    phase: Phase
    rollout_number: int
    pending_rollout: str | None = None
    decisions: int = 0
    optimizer_steps: int = 0

    def __post_init__(self):
        if isinstance(self.phase, str) and not isinstance(self.phase, Phase):
            self.phase = Phase(self.phase)

    @classmethod
    def from_state(cls, state: dict) -> RolloutBoundary:
        raw_phase = state.get("phase", Phase.COLLECT)
        phase = raw_phase if isinstance(raw_phase, Phase) else Phase(raw_phase)
        counters = TrainingCounters.from_dict(state, protocol=KIND)
        return cls(
            checkpoint=state.get("checkpoint", state.get("latest", "")),
            phase=phase,
            rollout_number=state.get("rollout_number", counters.rollouts_collected),
            pending_rollout=state.get("pending_rollout"),
            decisions=state.get("decisions", 0),
            optimizer_steps=state.get("optimizer_steps", counters.adam_steps_accepted),
        )

    def apply_to_state(self, state: dict) -> dict:
        state["phase"] = self.phase.value if isinstance(self.phase, Phase) else str(self.phase)
        state["latest"] = self.checkpoint
        state["rollouts"] = self.rollout_number
        state["rollouts_collected"] = self.rollout_number
        state["pending_rollout"] = self.pending_rollout
        state["decisions"] = self.decisions
        state["optimizer_steps"] = self.optimizer_steps
        state["adam_steps_accepted"] = self.optimizer_steps
        if "adam_steps_attempted" not in state or state["adam_steps_attempted"] < self.optimizer_steps:
            state["adam_steps_attempted"] = self.optimizer_steps
            state["attempted_steps"] = self.optimizer_steps
        if "rollouts_with_accepted_steps" not in state:
            state["rollouts_with_accepted_steps"] = state.get(
                "accepted_updates",
                0 if self.optimizer_steps == 0 else state.get("updates", 0),
            )
        if "accepted_updates" not in state:
            state["accepted_updates"] = state["rollouts_with_accepted_steps"]
        return state

    def to_state(self) -> dict:
        return self.apply_to_state({})

    def validate_sufficiency(self, out: Path) -> bool:
        """Validate that necessary files for the current phase exist on disk before proceeding."""
        out = Path(out)
        ckpt_path = out / self.checkpoint
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint file does not exist: {ckpt_path}")
        if self.phase in (Phase.CRITIC, Phase.PPO):
            if not self.pending_rollout:
                raise ValueError(f"Pending rollout path is required for phase '{self.phase.value}'")
            rollout_path = out / self.pending_rollout
            if not rollout_path.exists():
                raise FileNotFoundError(f"Pending rollout artifact does not exist: {rollout_path}")
        if self.phase == Phase.COLLECT and self.rollout_number > 0:
            ledger = out / "updates.jsonl"
            if not ledger.is_file():
                raise FileNotFoundError(f"Update ledger does not exist: {ledger}")
            rows = read_jsonl(ledger)
            if not rows:
                raise ValueError(f"Update ledger is empty: {ledger}")
            last = rows[-1]
            if last.get("rollout") != self.rollout_number:
                raise ValueError(
                    f"Update ledger rollout ({last.get('rollout')}) disagrees with boundary ({self.rollout_number})"
                )
            if "total_decisions" in last and last["total_decisions"] != self.decisions:
                raise ValueError(
                    f"Update ledger decisions ({last['total_decisions']}) disagrees with boundary ({self.decisions})"
                )
        return True

    @classmethod
    def heal_interrupted_update(cls, out: Path, state: dict) -> bool:
        return heal_interrupted_update(out, state)


@torch.no_grad()
def recurrent_features(policy, episodes, device):
    """Yield bounded FP32 GRU features, preserving complete episode prefixes."""
    for episode_index, episode in enumerate(episodes):
        encoded = encode_episode(episode, policy.variant)
        hidden = None
        for start in range(0, len(encoded), 128):
            if policy.spatial_cuda_graphs:
                torch.compiler.cudagraph_mark_step_begin()
            part = encoded[start : start + 128]
            batch = batch_inputs(part, device, non_blocking=True, spatial_only=True)
            _, context = policy.encode_spatial(batch, return_cells=False)
            features, hidden = policy.sequence_memory(context, [len(part)], hidden)
            # Own the host data: compiled/CUDA runners may reuse their buffers.
            yield episode_index, start, features.detach().cpu().clone()


@torch.no_grad()
def batched_recurrent_features(policy, episodes, device):
    """Stream bounded, episode-indexed FP32 feature blocks on the target device."""
    block = DEFAULT_PREFIX_BLOCK_SIZE
    group_size = 8
    for first in range(0, len(episodes), group_size):
        encoded = [encode_episode(ep, policy.variant) for ep in episodes[first : first + group_size]]
        hidden = (
            torch.zeros((1, len(encoded), policy.hidden_size), device=device) if policy.is_recurrent else None
        )
        for start in range(0, max(map(len, encoded)), block):
            parts = [(i, rows[start : start + block]) for i, rows in enumerate(encoded) if start < len(rows)]
            lengths = [len(rows) for _, rows in parts]
            slots = None
            if hidden is not None and len(parts) != len(encoded):
                slots = torch.tensor([i for i, _ in parts], device=device)
            if policy.spatial_cuda_graphs:
                torch.compiler.cudagraph_mark_step_begin()
            batch = batch_inputs(
                [row for _, rows in parts for row in rows],
                device,
                non_blocking=True,
                spatial_only=True,
            )
            _, context = policy.encode_spatial(batch, return_cells=False)
            features, last = policy.sequence_memory(
                context,
                lengths,
                hidden if slots is None else hidden.index_select(1, slots),
            )
            if slots is None:
                hidden = last
            else:
                hidden.index_copy_(1, slots, last)
            for (i, _), values in zip(parts, features.split(lengths)):
                yield first + i, start, values


@dataclass(frozen=True)
class _CriticFit:
    cache: torch.Tensor
    cached: int
    processed: int
    steps: int
    completed_epochs: int
    fit_seconds: float


def _fit_cached_critic_epochs(fit_batch, cached, batch_size, epochs, fit_deadline):
    completed_epochs = 1
    for _ in range(1, epochs):
        for first in range(0, cached, batch_size):
            if deadline_reached(fit_deadline):
                return completed_epochs
            fit_batch(first, min(first + batch_size, cached))
        completed_epochs += 1
    return completed_epochs


def _fit_initial_critic(
    policy, episodes, settings, device, returns, fit_deadline, started, progress
) -> _CriticFit:
    with value_only_grad_mode(policy):
        temporary = torch.optim.Adam(
            policy.value.parameters(),
            lr=settings["lr"],
            betas=tuple(settings["betas"]),
            eps=settings["eps"],
            weight_decay=settings["weight_decay"],
            fused=str(device).startswith("cuda"),
        )
        cache = torch.empty((len(returns), policy.hidden_size), dtype=torch.float32)
        cached = processed = steps = completed_epochs = 0
        batch_size = settings["batch_size"]

        def fit_batch(first, stop):
            nonlocal processed, steps
            temporary.zero_grad(set_to_none=True)
            features = cache[first:stop].to(device)
            labels = torch.from_numpy(returns[first:stop]).to(device)
            with torch.autocast(torch.device(device).type, enabled=False):
                prediction = policy.value(features).squeeze(-1)
                loss = F.mse_loss(prediction, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite critic preparation loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy.value.parameters(),
                settings["max_gradient_norm"],
                error_if_nonfinite=True,
            )
            temporary.step()
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            processed += stop - first
            steps += 1
            if progress:
                progress()

        portions = iter(recurrent_features(policy, episodes, device))
        trained = 0
        while cached < len(returns) and not deadline_reached(fit_deadline):
            _, _, features = next(portions)
            cache[cached : cached + len(features)] = features
            cached += len(features)
            if progress:
                progress()
            while cached - trained >= batch_size and not deadline_reached(fit_deadline):
                fit_batch(trained, trained + batch_size)
                trained += batch_size
        if cached == len(returns):
            if trained < cached and not deadline_reached(fit_deadline):
                fit_batch(trained, cached)
                trained = cached
            if trained == cached:
                completed_epochs = _fit_cached_critic_epochs(
                    fit_batch, cached, batch_size, settings["epochs"], fit_deadline
                )
        fit_seconds = perf_counter() - started
    return _CriticFit(cache, cached, processed, steps, completed_epochs, fit_seconds)


def prepare_critic(
    policy,
    episodes,
    settings,
    device,
    *,
    gamma=1.0,
    gae_lambda=1.0,
    deadline=None,
    progress=None,
    preparation_started=None,
):
    """Fit only value parameters, then refresh old_value without rewriting behavior."""
    started = preparation_started if preparation_started is not None else perf_counter()
    fit_deadline = DeadlineBudget(
        lambda: min(started + settings["active_seconds"], resolve_deadline(deadline))
    )
    original_policy_hash = policy_hash(policy.state_dict())
    original_behavior = behavior_digest(episodes)
    returns = np.concatenate(episode_returns(episodes, gamma=gamma))
    before = critic_statistics(np.concatenate([ep["old_value"] for ep in episodes]), returns)
    fit = _fit_initial_critic(policy, episodes, settings, device, returns, fit_deadline, started, progress)
    # The full refresh follows the fitting time limit.
    refresh_started = perf_counter()
    features = fit.cache if fit.cached == len(returns) else recurrent_features(policy, episodes, device)
    after_values = refresh_episode_old_values(
        policy,
        episodes,
        features,
        device,
        progress=progress if progress else None,
        expected_positions=len(returns),
    )
    verify_critic_invariance(
        policy,
        episodes,
        original_policy_hash,
        original_behavior,
        policy_hash_fn=policy_hash,
        behavior_hash_fn=behavior_digest,
    )
    raw_advantages = np.concatenate(raw_gae_advantages(episodes, gamma=gamma, gae_lambda=gae_lambda))
    advantages = targets(episodes, gamma=gamma, gae_lambda=gae_lambda)
    return {
        "steps": fit.steps,
        "positions_processed": fit.processed,
        "positions_cached": fit.cached,
        "epochs_completed": fit.completed_epochs,
        "fit_seconds": fit.fit_seconds,
        "fit_time_overshoot_seconds": max(0, fit.fit_seconds - settings["active_seconds"]),
        "refresh_seconds": perf_counter() - refresh_started,
        "before": before,
        "after": critic_statistics(after_values, returns),
        "raw_advantage_std": float(np.std(raw_advantages)),
        "normalized_advantage_std": float(np.concatenate(advantages).std()),
        "behavior_parameters_unchanged": True,
        "recorded_behavior_unchanged": True,
    }


class ActiveClock(PersistedActiveSessionClock):
    """One elapsed-time counter; evaluations explicitly pause it."""

    def __init__(self, state: dict[str, Any], out: Path) -> None:
        super().__init__(state, out, timer=lambda: perf_counter())


def make_policy(config, device):
    policy = load_policy(config, ROOT / config["experiment"]["parent"], device)
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    return policy


def save_boundary(out, policy, optimizer, state, shuffle, config, identity, clock=None):
    if clock:
        clock.sync()
    if isinstance(state, RolloutBoundary):
        state_dict = state.to_state()
    else:
        state_dict = state
    counters = TrainingCounters.from_dict(state_dict, protocol=KIND)
    state_dict.update(counters.to_state_dict(protocol=KIND))
    state_dict.setdefault("updates", 0)
    if uses_grouped_adam(config):
        state_dict.pop("Adam_counter_total", None)
        state_dict["Adam_counters"] = expected_counters(state_dict)
    else:
        state_dict["Adam_counter_total"] = (
            int(state_dict.get("origin_Adam_counter", 0)) + counters.adam_steps_accepted
        )
    if isinstance(state_dict.get("phase"), Phase):
        state_dict["phase"] = state_dict["phase"].value
    checkpoint.save(
        out / state_dict["latest"],
        policy,
        optimizer,
        state_dict,
        shuffle,
        config,
        identity,
    )
    if clock:
        clock.sync()
    write_json(out / "state.json", state_dict)


def _validate_inherited_groups(
    optimizer: torch.optim.Adam,
    source: dict[str, Any],
    origins: dict[str, int],
    saved_state: dict[str, Any],
    *,
    extension: bool,
) -> None:
    expected_names = {group["name"] for group in optimizer.param_groups}
    if extension:
        expected_names.remove("attack_preview")
    if set(origins) != expected_names or any(
        type(counter) is not int or counter < 0 for counter in origins.values()
    ):
        raise ValueError("Invalid inherited Adam group counters")
    if saved_state.get("Adam_counters") != origins:
        raise ValueError("Parent ledger and inherited Adam group counters differ")
    expected_groups = optimizer.state_dict()["param_groups"]
    if extension:
        expected_groups = expected_groups[:2]
    if len(source["param_groups"]) != len(expected_groups):
        raise ValueError("Parent Adam group count differs")
    for actual, expected in zip(source["param_groups"], expected_groups, strict=True):
        if any(actual.get(key) != expected[key] for key in ("name", "parameter_names", "params")):
            raise ValueError("Parent Adam parameter layout differs")


def _validate_adam_state_tensors(
    policy: Policy,
    optimizer: torch.optim.Adam,
    origins: dict[str, int] | None,
    expected: int,
    *,
    extension: bool,
) -> None:
    if len(optimizer.state) != len(list(policy.parameters())) - int(extension):
        raise ValueError("Parent Adam must contain all model and critic parameters")
    expected_by_parameter = {
        parameter: origins.get(group["name"], 0) if origins else expected
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for parameter, moments in optimizer.state.items():
        if float(moments["step"]) != expected_by_parameter[parameter]:
            raise ValueError("Parent Adam counters differ")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = moments[name]
            if (
                tensor.shape != parameter.shape
                or tensor.dtype != parameter.dtype
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"Invalid parent Adam {name}")


def _reset_value_optimizer(
    policy: Policy, config: dict[str, Any], inherited: torch.optim.Adam
) -> torch.optim.Adam:
    if len(inherited.param_groups) > 1:
        value_group = next(group for group in inherited.param_groups if group["name"] == "value")
        for parameter in value_group["params"]:
            inherited.state.pop(parameter, None)
        return inherited

    optimizer: torch.optim.Adam = grouped_optimizer(policy, config, "cpu")
    options = {key: value for key, value in inherited.param_groups[0].items() if key != "params"}
    for group in optimizer.param_groups:
        group.update(options)
    # Transfer inherited moments in model order and leave the value group empty.
    for name, parameter in policy.named_parameters():
        if parameter_group(name) != "value":
            optimizer.state[parameter] = inherited.state[parameter]
    return optimizer


def load_initial_optimizer(policy: Policy, saved: dict[str, Any], config: dict[str, Any]) -> torch.optim.Adam:
    """Preserve a PPO parent's complete Adam, or start the original BC handoff."""
    experiment = config["experiment"]
    origins = experiment.get("parent_optimizer_counters")
    extension = extends_parent(config)
    optimizer: torch.optim.Adam = (
        grouped_optimizer(policy, config, "cpu") if origins else optimizer_for(policy, config, "cpu")
    )
    if experiment.get("parent_optimizer") != "preserve":
        if optimizer.state:
            raise ValueError("New PPO Adam must be empty")
        return optimizer
    if (
        (
            saved.get("variant") != policy.variant
            and not (extension and saved.get("variant") == "C2" and policy.variant == "C2P")
        )
        or policy.variant != config["model"]["variant"]
        or not saved.get("state", {}).get("critic_complete")
    ):
        raise ValueError("PPO parent architecture/critic boundary is incompatible")
    if saved["state"].get("phase") != "collect":
        raise ValueError("PPO parent must be a completed update boundary")
    source = saved["optimizer"]
    if origins:
        _validate_inherited_groups(optimizer, source, origins, saved["state"], extension=extension)
    if extension:
        source = deepcopy(source)
        source["param_groups"].append(optimizer.state_dict()["param_groups"][2])
    optimizer.load_state_dict(source)
    assert_equal(optimizer.state_dict(), source)
    expected = int(experiment["parent_optimizer_steps"])
    if not origins and saved["state"]["optimizer_steps"] != expected:
        raise ValueError("Parent ledger and configured Adam counter differ")
    _validate_adam_state_tensors(policy, optimizer, origins, expected, extension=extension)
    parent_lr = experiment.get("parent_optimizer_lr", config["optimizer"]["lr"])
    for group in optimizer.param_groups:
        if group["lr"] != parent_lr:
            raise ValueError("Parent Adam lr differs from experiment")
        for name in ("eps", "weight_decay"):
            if group[name] != config["optimizer"][name]:
                raise ValueError(f"Parent Adam {name} differs from experiment")
        if tuple(group["betas"]) != tuple(config["optimizer"]["betas"]):
            raise ValueError("Parent Adam betas differ from experiment")
        group["lr"] = config["optimizer"]["lr"]
    if adapts_value(config):
        return _reset_value_optimizer(policy, config, optimizer)
    return optimizer


def validate_episode_origin(experiment, parent_final):
    """Weights may branch from an older best checkpoint than the data frontier."""
    frontier = parent_final
    source = experiment.get("episode_id_source")
    if source:
        for key in ("state", "run"):
            if file_sha256(ROOT / source[key]) != source[key + "_sha256"]:
                raise ValueError(f"Episode ID source {key} hash differs")
        record = read_json(ROOT / source["run"])
        frontier = read_json(ROOT / source["state"])
        origin = record.get("config", {}).get("experiment", {})
        if (
            origin.get("parent_sha256") != experiment["parent_sha256"]
            or origin.get("start_episode", -1) < parent_final["next_episode"]
        ):
            raise ValueError("Episode ID source does not extend the same parent lineage")
    if (
        parent_final.get("phase") != "collect"
        or not parent_final.get("stop_reason")
        or frontier.get("phase") != "collect"
        or not frontier.get("stop_reason")
        or frontier["next_episode"] != experiment["start_episode"]
        or experiment["start_episode"] < parent_final["next_episode"]
    ):
        raise ValueError("Continuation must start after all collected lineage episodes")


def _verify_parent_lineage(experiment: dict[str, Any], parent: Path, *, inherited: bool) -> dict[str, Any]:
    if file_sha256(parent) != experiment["parent_sha256"]:
        raise ValueError("Parent checkpoint SHA256 differs")
    if not inherited:
        return {}
    for key in ("parent_manifest", "parent_final_state"):
        if file_sha256(ROOT / experiment[key]) != experiment[key + "_sha256"]:
            raise ValueError(f"Parent lineage input differs: {key}")
    final = read_json(ROOT / experiment["parent_final_state"])
    parent_record_path = ROOT / experiment["parent_run"] / "run.json"
    if file_sha256(parent_record_path) != experiment["parent_run_sha256"]:
        raise ValueError("Parent run record differs")
    parent_record: dict[str, Any] = read_json(parent_record_path)
    validate_episode_origin(experiment, final)
    historical = experiment.get("historical_active_time_estimate")
    if historical and file_sha256(ROOT / historical["source"]) != historical["source_sha256"]:
        raise ValueError("Historical budget evidence differs")
    return parent_record


def _initial_evaluation_plan(config: dict[str, Any]) -> dict[str, Any]:
    experiment = config["experiment"]
    plan = read_json(ROOT / experiment["parent_manifest"])
    verify_manifest_digest(plan, error_msg="Target manifest hash differs")
    eval_purpose = config.get("ppo", {}).get("evaluation_purpose", "development")
    if (
        len(plan["development"]) != 128
        or len(plan["confirmation"]) != 256
        or (eval_purpose == "development_512" and len(plan.get("development_512", [])) != 512)
    ):
        raise ValueError("Unexpected evaluation cases")
    expected_rounds = int(config.get("environment", {}).get("max_rounds", 30))
    if any(int(c.get("max_rounds", 30)) != expected_rounds for c in plan[eval_purpose]) or any(
        int(c.get("max_rounds", 30)) != expected_rounds for c in plan["confirmation"]
    ):
        raise ValueError("Evaluation case max_rounds differs from environment config")
    derived_plan: dict[str, Any] = evaluation_plan(plan, config)
    return derived_plan


def initialize(config: dict[str, Any], out: Path) -> dict[str, Any]:
    experiment = config["experiment"]
    parent = ROOT / experiment["parent"]
    inherited = experiment.get("parent_optimizer") == "preserve"
    parent_record = _verify_parent_lineage(experiment, parent, inherited=inherited)
    plan = _initial_evaluation_plan(config)
    write_json(out / "manifest.json", plan)
    torch_seed = seed_for(17, 0, "torch", namespace=experiment["torch_namespace"]) if inherited else 17
    configure(torch_seed, "cpu")
    policy = make_policy(config, "cpu")
    saved = torch.load(parent, map_location="cpu", weights_only=False)
    if inherited and saved["identity"].get("run_id") != parent_record["run_id"]:
        raise ValueError("Parent checkpoint and final run lineage differ")
    extension = extends_parent(config)
    if extension:
        inherited_model(policy.state_dict(), saved["model"])
    elif not all(torch.equal(value, saved["model"][name]) for name, value in policy.state_dict().items()):
        raise ValueError("Initial model differs from parent")
    if policy_hash(saved["model"]) != experiment["parent_policy_hash"]:
        raise ValueError("Initial behavior hash differs")
    optimizer = load_initial_optimizer(policy, saved, config)
    shuffle_seed = seed_for(17, 0, "shuffle", namespace=experiment["shuffle_namespace"])
    record, identity = _initial_run_record(
        config,
        plan,
        out,
        policy,
        optimizer,
        inherited=inherited,
        shuffle_seed=shuffle_seed,
        torch_seed=torch_seed,
    )
    state = _initial_training_state(config, inherited=inherited, extension=extension)
    write_json(out / "run.json", record)
    save_boundary(
        out,
        policy,
        optimizer,
        state,
        random.Random(shuffle_seed),
        config,
        identity,
    )
    RolloutBoundary.from_state(state).validate_sufficiency(out)
    return plan


def _initial_run_record(
    config: dict[str, Any],
    plan: dict[str, Any],
    out: Path,
    policy: Policy,
    optimizer: torch.optim.Adam,
    *,
    inherited: bool,
    shuffle_seed: int,
    torch_seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    experiment = config["experiment"]
    budget_check = compute_time_budget_check(config)
    if budget_check is not None:
        print(
            "Active-time budget check: " + json.dumps(budget_check, ensure_ascii=False),
            flush=True,
        )
    record: dict[str, Any] = {
        "evaluation_runtime_default": "auto",
        "kind": KIND,
        "config": config,
        **({"time_budget_check": budget_check} if budget_check is not None else {}),
        "parent_sha256": experiment["parent_sha256"],
        "parent_model_equal": True,
        "new_Adam_empty": not inherited,
        **(
            {"parent_Adam_counters": experiment["parent_optimizer_counters"]}
            if experiment.get("parent_optimizer_counters")
            else {"parent_Adam_counter": int(experiment["parent_optimizer_steps"])}
        ),
        "critic_preserved": inherited,
        **(
            {
                "value_optimizer": "reset_then_preserve",
                "Adam_counter_groups": [g["name"] for g in optimizer.param_groups],
            }
            if adapts_value(config)
            else {}
        ),
        "behavior_hash": policy_hash(policy.state_dict()),
        "manifest_hash": plan["hash"],
        "seed": 17,
        "shuffle_seed": shuffle_seed,
        "torch_seed": torch_seed,
        "GPU": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "implementation_sha256": {
            name: file_sha256(ROOT / name)
            for name in (
                "civ5rl/experiments/ppo_after_target_bc.py",
                "civ5rl/protocols/ppo_after_target_bc.py",
                "tools/run_ppo.py",
            )
        },
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    record["run_id"] = digest(record)
    identity = {"run_id": record["run_id"]}
    if inherited:
        identity.update(checkpoint.passport(out, config, plan))
        record["checkpoint_identity"] = identity
        record["lineage"] = {
            key: experiment[key]
            for key in (
                "parent",
                "parent_sha256",
                "parent_manifest",
                "parent_manifest_sha256",
                "parent_final_state",
                "parent_final_state_sha256",
                "parent_run_sha256",
                "start_episode",
            )
        }
        if experiment.get("episode_id_source"):
            record["lineage"]["episode_id_source"] = experiment["episode_id_source"]
        runtime_for(out, config)
    return record, identity


def _initial_training_state(config: dict[str, Any], *, inherited: bool, extension: bool) -> dict[str, Any]:
    experiment = config["experiment"]
    counters = TrainingCounters()
    state: dict[str, Any] = {
        **counters.to_state_dict(protocol=KIND),
        "updates": 0,
        "decisions": 0,
        "next_episode": experiment["start_episode"],
        **(
            {"origin_Adam_counters": dict(experiment["parent_optimizer_counters"])}
            if experiment.get("parent_optimizer_counters")
            else {"origin_Adam_counter": int(experiment["parent_optimizer_steps"])}
        ),
        "winless_streak": 0,
        "zero_accepted_streak": 0,
        "active_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "critic_complete": inherited and not adapts_value(config),
        "phase": "collect",
        "latest": "checkpoints/initial.pt",
        "stop_reason": None,
        "evaluated": {},
        "development_points": {},
        "baseline_wins": None,
        "best": {"path": "checkpoints/initial.pt", "wins": -1, "update": 0},
    }
    if extension:
        state["origin_Adam_counters"]["attack_preview"] = 0
    if adapts_value(config) and "origin_Adam_counters" in state:
        state["value_optimizer"] = "reset_then_preserve"
        state["origin_Adam_counters"]["value"] = 0
    if config.get("self_play", {}).get("enabled"):
        state["self_play_archive"] = []
        state["self_play_last_archived_optimizer_steps"] = 0
    return state


def development(config, plan, out, state, shuffle, clock=None):
    if clock:
        clock.pause()
    started = perf_counter()
    purpose = config.get("ppo", {}).get("evaluation_purpose", "development")
    dual = bool(config.get("self_play", {}).get("enabled"))
    try:
        with isolated_rng(shuffle):
            result = evaluate(config, plan, out, state["latest"], purpose)
            h2h = None
            if dual:
                h2h = evaluate_h2h(
                    config,
                    plan,
                    out,
                    state["latest"],
                    "checkpoints/initial.pt",
                    purpose,
                )
        state["evaluation_seconds"] += perf_counter() - started
        entry = {
            **result,
            "decisions": state["decisions"],
            "update": state["updates"],
        }
        if h2h is not None:
            entry["h2h"] = h2h
        state["evaluated"][result["policy_hash"]] = entry
        wins = result["runs"]["argmax"]["cp"]["summary"]["W"]
        if state["baseline_wins"] is None:
            state["baseline_wins"] = wins
            if config["experiment"].get("early_gain_min_wins") is not None:
                state["early_gain_required_wins"] = wins + config["experiment"]["early_gain_min_wins"]
        if dual:
            candidates = []
            for p_hash, item in state["evaluated"].items():
                cp_sum = item["runs"]["argmax"]["cp"]["summary"]
                h2h_sum = item["h2h"]["summary"]
                candidates.append(
                    {
                        "path": item["checkpoint"],
                        "update": int(item.get("update", 0)),
                        "decisions": int(item.get("decisions", 0)),
                        "policy_hash": p_hash,
                        "cp_wins": int(cp_sum["W"]),
                        "h2h_wins": int(h2h_sum["W"]),
                        "h2h_losses": int(h2h_sum["L"]),
                        "cp_summary": cp_sum,
                        "h2h_summary": h2h_sum,
                    }
                )
            state["best"] = select_dual_milestone(candidates, state["baseline_wins"])
        elif wins > state["best"]["wins"]:
            state["best"] = {
                "path": result["checkpoint"],
                "wins": wins,
                "update": state["updates"],
                "policy_hash": result["policy_hash"],
            }
        for point in config["experiment"]["milestones"]:
            if point <= state["decisions"]:
                state["development_points"].setdefault(str(point), result["policy_hash"])
        check_development_gain(state, config)
        write_json(out / "state.json", state)
        report(config, out)
    finally:
        if clock:
            clock.resume()


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
                features_for=batched_recurrent_features,
                behavior_hash=behavior_digest,
                statistics=critic_statistics,
                policy_hash=policy_hash,
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
    save_boundary(out, policy, session.optimizer, state, shuffle, config, identity, clock)
    RolloutBoundary.from_state(state).validate_sufficiency(out)
    write_json(out / "critic-preparation.json", state["critic"])
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
    if due or (state["stop_reason"] and fingerprint() not in state["evaluated"]):
        monitor.mark("evaluation")
        development(config, plan, out, state, shuffle, clock)
        monitor.record_cuda_memory(policy, "after_evaluation")
    return bool(state["stop_reason"])


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
    configure(17, device)
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
            opponent_pool = OpponentPool(config, device, out)
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


def _development_gate_rejected(comparison, min_delta, min_ci_low):
    return comparison["delta"] < float(min_delta) or comparison["CI95"][0] <= min_ci_low


def finish(config, plan, out):
    if (out / "confirmation.json").exists():
        return
    state = read_json(out / "state.json")
    if target_bc_evaluation_due(config, state):
        raise ValueError("Cannot select a candidate while Development is incomplete")
    write_json(out / "selected.json", state["best"])
    if state["best"]["path"] == "checkpoints/initial.pt":
        write_json(out / "confirmation.json", {"status": "not_run_parent_selected"})
        return

    def outcomes(point):
        directory = out / point["runs"]["argmax"]["cp"]["directory"]
        results, cases, _ = collect_paired_outcomes(directory)
        return results, cases

    dev_comparison = None
    dev_purpose = config.get("ppo", {}).get("evaluation_purpose", "development")
    min_delta = config["experiment"].get("confirmation_min_win_rate_delta")
    if min_delta is not None:
        dev_baseline = evaluate(config, plan, out, "checkpoints/initial.pt", dev_purpose)
        dev_candidate = evaluate(config, plan, out, state["best"]["path"], dev_purpose)
        dev_before, dev_cases = outcomes(dev_baseline)
        dev_after, dev_paired = outcomes(dev_candidate)
        if dev_cases != dev_paired:
            raise ValueError(f"{dev_purpose} cases are not exactly paired")
        dev_comparison = paired_world_bootstrap(
            dev_before,
            dev_after,
            samples=10000,
            seed=20261103,
            cases=dev_cases,
            cases_per_world=len(dev_before) // 16,
        )
        min_ci_low = float(config["experiment"].get("confirmation_min_ci95_low", 0.0))
        if _development_gate_rejected(dev_comparison, min_delta, min_ci_low):
            write_json(
                out / "confirmation.json",
                {
                    "status": f"not_run_{dev_purpose}_gate_not_met",
                    f"{dev_purpose}_comparison": dev_comparison,
                },
            )
            return

    baseline = evaluate(config, plan, out, "checkpoints/initial.pt", "confirmation")
    candidate = evaluate(config, plan, out, state["best"]["path"], "confirmation")
    before, cases = outcomes(baseline)
    after, paired = outcomes(candidate)
    if cases != paired:
        raise ValueError("Additional cases are not exactly paired")
    comparison = paired_world_bootstrap(
        before, after, samples=10000, seed=20261103, cases=cases, cases_per_world=16
    )
    payload = {
        "baseline": baseline,
        "candidate": candidate,
        "comparison": comparison,
    }
    if config.get("self_play", {}).get("enabled"):
        h2h_confirmation = evaluate_h2h(
            config,
            plan,
            out,
            state["best"]["path"],
            "checkpoints/initial.pt",
            "confirmation",
        )
        payload["h2h_confirmation"] = h2h_confirmation
        payload["promotion"] = dual_promotion_decision(comparison, h2h_confirmation["bootstrap"])
    if dev_comparison is not None:
        payload[f"{dev_purpose}_comparison"] = dev_comparison
    write_json(out / "confirmation.json", payload)


def _development_rows(state: dict[str, Any], *, dual: bool) -> Iterator[tuple[object, ...]]:
    baseline_wins = state.get("baseline_wins") if dual else None
    for point in state["evaluated"].values():
        summary = point["runs"]["argmax"]["cp"]["summary"]
        if not dual:
            yield point["decisions"], point["checkpoint"], format_wdl(summary)
            continue
        h2h_summary = point.get("h2h", {}).get("summary")
        cp_delta = int(summary["W"]) - int(baseline_wins) if baseline_wins is not None else None
        if h2h_summary is not None:
            h2h_delta = int(h2h_summary["W"]) - int(h2h_summary["L"])
            h2h_wdl = format_wdl(h2h_summary)
            joint_score = cp_delta + h2h_delta if cp_delta is not None else None
        else:
            h2h_delta = None
            h2h_wdl = "—"
            joint_score = None
        yield (
            point["decisions"],
            point["checkpoint"],
            format_wdl(summary),
            f"{cp_delta:+d}" if cp_delta is not None else "—",
            h2h_wdl,
            f"{h2h_delta:+d}" if h2h_delta is not None else "—",
            f"{joint_score:+d}" if joint_score is not None else "—",
        )


def _rollout_report_rows(rows: list[dict[str, Any]]) -> Iterator[tuple[object, ...]]:
    for row in rows:
        metrics = row["ppo"]
        yield (
            row["rollout"],
            format_wdl(row),
            metrics.get("raw_advantage_std"),
            f"{metrics.get('critic_mse')}/{metrics.get('critic_explained_variance')}/{metrics.get('critic_mean')}",
            metrics.get("entropy"),
            metrics.get("final_accepted_kl"),
            f"{metrics['optimizer_steps']}/{metrics['rejected_optimizer_steps']}",
            metrics["PPO_skipped"],
        )


def _write_performance_reports(out: Path) -> None:
    for name, last in (("performance", None), ("performance-last10", 10)):
        performance = summarize_performance(out, last=last)
        write_json(out / f"{name}.json", performance)
        (out / f"{name}.md").write_text(performance_markdown(performance), encoding="utf-8")


def report(config: dict[str, Any], out: Path) -> None:
    state = read_json(out / "state.json")
    parent_label = config["experiment"].get("parent_label", "target BC epoch 10")
    eval_purpose = config.get("ppo", {}).get("evaluation_purpose", "development")
    max_rounds = config.get("environment", {}).get("max_rounds", 30)
    gamma = config["ppo"].get("gamma", 1)
    gae_lambda = config["ppo"].get("gae_lambda", 1)
    sp = config.get("self_play", {})
    dual = bool(sp.get("enabled"))
    opponent_desc = (
        f"смешанный пул (CP {sp['cp_share']:.0%} / current {sp['current_share']:.0%} / "
        f"u0 {sp['parent_share']:.0%} / archive {sp['archive_share']:.0%}, "
        f"T_opp={sp.get('opponent_temperature', 0.5)}, архив={len(state.get('self_play_archive', []))})"
        if dual
        else "обычный CP"
    )
    lines = MarkdownReport(
        [
            f"# PPO после {parent_label}",
            "",
            f"Статус: {state['stop_reason'] or state['phase']}; решения {state['decisions']}; active {state['active_seconds']:.2f} с.",
            f"Лимит active: {state.get('active_seconds_limit', config['experiment']['active_seconds']):.0f} с; исходный бюджет {config['experiment']['active_seconds']} с; продления в active-budget.json.",
            f"Sampling rollouts {state['rollouts']}; PPO updates {state['updates']}; accepted/rejected Adam {state['optimizer_steps']}/{state['rejected_steps']}.",
            (
                f"Adam groups: {state.get('Adam_counters', expected_counters(state))}; critic preparation {state.get('critic_progress', {}).get('steps', 0)}; new PPO accepted {state['optimizer_steps']}."
                if uses_grouped_adam(config)
                else f"Adam: inherited {state.get('origin_Adam_counter', 0)}; new accepted {state['optimizer_steps']}; total {state.get('origin_Adam_counter', 0) + state['optimizer_steps']}."
            ),
            f"GPU {read_json(out / 'run.json')['GPU']}; T={config.get('policy_temperature', 1)}; max_rounds={max_rounds}; gamma={gamma}; gae_lambda={gae_lambda}; microbatch {config['ppo']['sequences_per_microbatch']}; {opponent_desc}; reward 1/0/0.",
            "",
        ]
    )
    if dual:
        lines.table(
            (
                "Решения",
                "Checkpoint",
                f"W/D/L {eval_purpose} (CP)",
                "ΔW_cp",
                "W/D/L H2H vs u0",
                "ΔW_h2h",
                "J",
            ),
            ("right", "left", "left", "right", "left", "right", "right"),
            _development_rows(state, dual=True),
        )
    else:
        lines.table(
            ("Решения", "Checkpoint", f"W/D/L {eval_purpose}"),
            ("right", "left", "left"),
            _development_rows(state, dual=False),
        )
    lines += ["", f"Выбор: {state['best']}."]
    if (out / "runtime.json").exists():
        lines += [
            "",
            "Runtime: " + json.dumps(read_json(out / "runtime.json"), ensure_ascii=False),
        ]
    if (out / "updates.jsonl").exists():
        _write_performance_reports(out)
        lines += [
            "",
            "Performance: [all cycles](performance.md), [last 10](performance-last10.md).",
        ]
        rows = read_jsonl(out / "updates.jsonl", skip_blank=False, encoding=None)
        lines.append("")
        lines.table(
            (
                "Rollout",
                "Sampling W/D/L",
                "Raw advantage std",
                "Critic MSE / EV / mean",
                "Entropy",
                "Full KL",
                "Adam accept/reject",
                "PPO skipped",
            ),
            ("right", "left", "right", "left", "right", "right", "left", "left"),
            _rollout_report_rows(rows),
        )
        lines += [
            "",
            f"Rollout без побед: {sum(row['W'] == 0 for row in rows)}/{len(rows)}.",
        ]
    for point in state["evaluated"].values():
        lines += [
            "",
            f"Срезы development {point['checkpoint']}",
            "",
            json.dumps(point.get("slices", {}), ensure_ascii=False),
        ]
    if state["stop_reason"] == "no_development_gain":
        lines += ["", "Прироста в установленном раннем бюджете не получено."]
    lines += [
        "",
        (
            f"Ранний порог: {state.get('early_gain_required_wins', 'по конфигу')} побед после {config['experiment']['early_gain_decisions']} новых решений; полезность оценивается по парному additional-приросту и CI."
            if config["experiment"].get("early_gain_enabled", True)
            else (
                "Ранний порог продолжения отключён; двухлинейный выбор по Development-512 (CP) и H2H-512 (ΔW_cp ≥ 0, ΔW_h2h ≥ 0, max J), проверка нового кандидата на Additional-256 (CP + H2H)."
                if dual
                else "Ранний порог продолжения отключён; выбор по Development, проверка нового кандидата на парном Additional."
            )
        ),
    ]
    for name in ("critic-preparation.json", "confirmation.json"):
        if (out / name).exists():
            lines += [
                "",
                name,
                "",
                json.dumps(read_json(out / name), ensure_ascii=False),
            ]
    lines.write(out / "report.md")


def run(
    phase: str,
    config: dict[str, Any],
    out: Path | str,
    device: str | torch.device,
    runtime_path: Path | str | None = None,
    max_updates: int | None = None,
) -> None:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    validate_profile(config)
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be a positive operational limit")
    if phase == "pause":
        (out / "pause.request").write_text("pause requested\n", encoding="utf-8")
        return
    trains = phase in ("train", "run", "resume")
    with experiment_run_guard(
        out,
        phase=phase,
        session=evaluation_session,
        preflight=partial(assert_not_closed, out) if trains else None,
    ):
        if phase == "resume":
            (out / "pause.request").unlink(missing_ok=True)
        if not (out / "run.json").exists():
            initialize(config, out)
        if read_json(out / "run.json")["config"] != config:
            raise ValueError("Run configuration changed")
        if config.get("evaluation_runtime"):
            runtime_for(out, config)
        plan = read_json(out / "manifest.json")
        for purpose in ("development", "development_512", "confirmation"):
            for case in plan.get(purpose, []):
                validate_case(case, config)
        if trains:
            write_json(out / "status.json", {"status": "training"})
            train(
                config,
                plan,
                out,
                device,
                runtime_path=runtime_path,
                max_updates=max_updates,
            )
            if read_json(out / "status.json").get("status") == "paused":
                report(config, out)
                return
        if phase in ("evaluate", "run", "resume"):
            if not read_json(out / "state.json")["stop_reason"]:
                raise ValueError("Training has not stopped")
            finish(config, plan, out)
            write_json(out / "status.json", {"status": "complete"})
        report(config, out)
