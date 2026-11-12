"""Bounded PPO from BC or a preserved PPO boundary, with recoverable evaluation."""

import gc
import json
import random
import traceback
from dataclasses import dataclass
from enum import Enum
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from civ5rl.engine.budget import ActiveBudget, deadline_reached, resolve_deadline
from civ5rl.core.model import batch_inputs
from civ5rl.engine.collector import Collector
from civ5rl.engine.constants import DEFAULT_PINNED_REPLAY_BYTES
from civ5rl.engine.control import (
    RunLock,
    assert_not_closed,
    pause_due,
    target_bc_evaluation_due,
)
from civ5rl.engine.counters import TrainingCounters
from civ5rl.engine.evaluation import runtime_for
from civ5rl.engine.performance import performance_markdown, summarize_performance
from civ5rl.engine.ppo import (
    encode_episode,
    episode_returns,
    raw_gae_advantages,
    targets,
    update,
)
from civ5rl.engine.ppo_fast import BatchedBackend
from civ5rl.engine.resources import ResourceMonitor
from civ5rl.experiments.evaluation import paired_world_bootstrap
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
    committed = (
        [
            json.loads(line)
            for line in ledger.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if ledger.is_file()
        else []
    )
    healed = False
    if not committed or committed[-1].get("rollout") != pending.get("rollout"):
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(pending, allow_nan=False) + "\n")
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
    def from_state(cls, state: dict) -> "RolloutBoundary":
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
            rows = [
                json.loads(line)
                for line in ledger.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
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


def critic_statistics(predictions, returns):
    predictions = np.asarray(predictions, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    residual = returns - predictions
    variance = returns.var()
    return {
        "prediction_mean": float(predictions.mean()),
        "return_mean": float(returns.mean()),
        "mse": float(np.mean(residual**2)),
        "constant_return_mean_mse": float(np.mean((returns - returns.mean())**2)),
        "explained_variance": float(1 - residual.var() / variance)
        if variance > 0
        else None,
    }


def behavior_digest(episodes):
    """The actual sampled choices and their probabilities are immutable."""
    import hashlib

    result = hashlib.sha256()
    for episode in episodes:
        result.update(np.asarray(episode["chosen"], dtype=np.int64).tobytes())
        result.update(np.asarray(episode["old_logp"], dtype=np.float64).tobytes())
        for row in episode["old_log_probs"]:
            result.update(row.tobytes())
    return result.hexdigest()


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
            batch = batch_inputs(part, device, non_blocking=True)
            _, context = policy.encode_spatial(batch)
            features, hidden = policy.sequence_memory(context, [len(part)], hidden)
            # Own the host data: compiled/CUDA runners may reuse their buffers.
            yield episode_index, start, features.detach().cpu().clone()


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
    fit_deadline = lambda: min(started + settings["active_seconds"], resolve_deadline(deadline))
    original_policy_hash = policy_hash(policy.state_dict())
    original_behavior = behavior_digest(episodes)
    returns = np.concatenate(episode_returns(episodes, gamma=gamma))
    before = critic_statistics(
        np.concatenate([ep["old_value"] for ep in episodes]), returns
    )
    flags = [parameter.requires_grad for parameter in policy.parameters()]
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.value.parameters():
        parameter.requires_grad_(True)
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

    try:
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
                completed_epochs = 1
                for _ in range(1, settings["epochs"]):
                    finished = True
                    for first in range(0, cached, batch_size):
                        if deadline_reached(fit_deadline):
                            finished = False
                            break
                        fit_batch(first, min(first + batch_size, cached))
                    if not finished:
                        break
                    completed_epochs += 1
        fit_seconds = perf_counter() - started
    finally:
        for parameter, enabled in zip(policy.parameters(), flags):
            parameter.requires_grad_(enabled)
        temporary = None

    # Required full refresh is outside the 120-second fitting limit.
    refresh_started = perf_counter()
    flat_offset = 0
    after_values = []
    with torch.no_grad():
        if cached == len(returns):
            for episode in episodes:
                ep_len = len(episode["old_value"])
                for start in range(0, ep_len, 128):
                    chunk_len = min(128, ep_len - start)
                    features = cache[flat_offset : flat_offset + chunk_len].to(device)
                    values = policy.value(features).squeeze(-1).cpu().tolist()
                    episode["old_value"][start : start + chunk_len] = values
                    after_values.extend(values)
                    flat_offset += chunk_len
                    if progress:
                        progress()
        else:
            for episode_index, start, features in recurrent_features(
                policy, episodes, device
            ):
                values = policy.value(features.to(device)).squeeze(-1).cpu().tolist()
                episodes[episode_index]["old_value"][start : start + len(values)] = values
                after_values.extend(values)
                flat_offset += len(values)
                if progress:
                    progress()
    if flat_offset != len(returns):
        raise ValueError("Incomplete first-rollout old_value refresh")
    if policy_hash(policy.state_dict()) != original_policy_hash:
        raise RuntimeError("Critic preparation changed encoder/GRU/policy parameters")
    if behavior_digest(episodes) != original_behavior:
        raise RuntimeError("Critic preparation changed recorded behavior")
    raw_advantages = np.concatenate(
        raw_gae_advantages(episodes, gamma=gamma, gae_lambda=gae_lambda)
    )
    advantages = targets(episodes, gamma=gamma, gae_lambda=gae_lambda)
    return {
        "steps": steps,
        "positions_processed": processed,
        "positions_cached": cached,
        "epochs_completed": completed_epochs,
        "fit_seconds": fit_seconds,
        "fit_time_overshoot_seconds": max(0, fit_seconds - settings["active_seconds"]),
        "refresh_seconds": perf_counter() - refresh_started,
        "before": before,
        "after": critic_statistics(after_values, returns),
        "raw_advantage_std": float(np.std(raw_advantages)),
        "normalized_advantage_std": float(np.concatenate(advantages).std()),
        "behavior_parameters_unchanged": True,
        "recorded_behavior_unchanged": True,
    }


class ActiveClock:
    """One elapsed-time counter; evaluations explicitly pause it."""

    def __init__(self, state, out):
        self.state = state
        self.out = out
        self.started = perf_counter()
        self.budget = ActiveBudget(out, state)

    def sync(self):
        if self.started is not None:
            now = perf_counter()
            self.state["active_seconds"] += now - self.started
            self.started = now

    def progress(self):
        self.sync()
        write_json(self.out / "state.json", self.state)

    def pause(self):
        self.sync()
        self.started = None

    def resume(self):
        self.started = perf_counter()

    def limit(self, config):
        return self.budget.limit()

    def deadline(self, config):
        self.sync()
        return perf_counter() + max(
            0, self.limit(config) - self.state["active_seconds"]
        )


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
    state_dict["Adam_counter_total"] = int(state_dict.get("origin_Adam_counter", 0)) + counters.adam_steps_accepted
    if isinstance(state_dict.get("phase"), Phase):
        state_dict["phase"] = state_dict["phase"].value
    checkpoint.save(
        out / state_dict["latest"], policy, optimizer, state_dict, shuffle, config, identity
    )
    if clock:
        clock.sync()
    write_json(out / "state.json", state_dict)


def load_initial_optimizer(policy, saved, config):
    """Preserve a PPO parent's complete Adam, or start the original BC handoff."""
    optimizer = optimizer_for(policy, config, "cpu")
    experiment = config["experiment"]
    if experiment.get("parent_optimizer") != "preserve":
        if optimizer.state:
            raise ValueError("New PPO Adam must be empty")
        return optimizer
    if (saved.get("variant") != policy.variant
            or saved.get("variant") != config["model"]["variant"]
            or not saved.get("state", {}).get("critic_complete")):
        raise ValueError("PPO parent architecture/critic boundary is incompatible")
    if saved["state"].get("phase") != "collect":
        raise ValueError("PPO parent must be a completed update boundary")
    source = saved["optimizer"]
    optimizer.load_state_dict(source)
    assert_equal(optimizer.state_dict(), source)
    expected = int(experiment["parent_optimizer_steps"])
    if saved["state"]["optimizer_steps"] != expected:
        raise ValueError("Parent ledger and configured Adam counter differ")
    if len(optimizer.state) != len(list(policy.parameters())):
        raise ValueError("Parent Adam must contain all model and critic parameters")
    for parameter, item in optimizer.state.items():
        if float(item["step"]) != expected:
            raise ValueError("Parent Adam counters differ")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = item[name]
            if tensor.shape != parameter.shape or tensor.dtype != parameter.dtype or not torch.isfinite(tensor).all():
                raise ValueError(f"Invalid parent Adam {name}")
    for group in optimizer.param_groups:
        for name in ("lr", "eps", "weight_decay"):
            if group[name] != config["optimizer"][name]:
                raise ValueError(f"Parent Adam {name} differs from experiment")
        if tuple(group["betas"]) != tuple(config["optimizer"]["betas"]):
            raise ValueError("Parent Adam betas differ from experiment")
    return optimizer


def initialize(config, out):
    experiment = config["experiment"]
    parent = ROOT / experiment["parent"]
    if file_sha256(parent) != experiment["parent_sha256"]:
        raise ValueError("Parent checkpoint SHA256 differs")
    inherited = experiment.get("parent_optimizer") == "preserve"
    if inherited:
        for key in ("parent_manifest", "parent_final_state"):
            if file_sha256(ROOT / experiment[key]) != experiment[key + "_sha256"]:
                raise ValueError(f"Parent lineage input differs: {key}")
        final = read_json(ROOT / experiment["parent_final_state"])
        parent_record_path = ROOT / experiment["parent_run"] / "run.json"
        if file_sha256(parent_record_path) != experiment["parent_run_sha256"]:
            raise ValueError("Parent run record differs")
        parent_record = read_json(parent_record_path)
        if final["next_episode"] != experiment["start_episode"] or final.get("phase") != "collect" or not final.get("stop_reason"):
            raise ValueError("Continuation must start after all collected parent episodes")
        historical = experiment.get("historical_active_time_estimate")
        if historical and file_sha256(ROOT / historical["source"]) != historical["source_sha256"]:
            raise ValueError("Historical budget evidence differs")
    plan = read_json(ROOT / experiment["parent_manifest"])
    if digest({k: v for k, v in plan.items() if k != "hash"}) != plan["hash"]:
        raise ValueError("Target manifest hash differs")
    eval_purpose = config.get("ppo", {}).get("evaluation_purpose", "development")
    if (
        len(plan["development"]) != 128
        or len(plan["confirmation"]) != 256
        or (
            eval_purpose == "development_512"
            and len(plan.get("development_512", [])) != 512
        )
    ):
        raise ValueError("Unexpected evaluation cases")
    expected_rounds = int(config.get("environment", {}).get("max_rounds", 30))
    if any(
        int(c.get("max_rounds", 30)) != expected_rounds for c in plan[eval_purpose]
    ) or any(
        int(c.get("max_rounds", 30)) != expected_rounds for c in plan["confirmation"]
    ):
        raise ValueError("Evaluation case max_rounds differs from environment config")
    write_json(out / "manifest.json", plan)
    torch_seed = seed_for(17, 0, "torch", namespace=experiment["torch_namespace"]) if inherited else 17
    configure(torch_seed, "cpu")
    policy = make_policy(config, "cpu")
    saved = torch.load(parent, map_location="cpu", weights_only=False)
    if inherited and saved["identity"].get("run_id") != parent_record["run_id"]:
        raise ValueError("Parent checkpoint and final run lineage differ")
    if not all(
        torch.equal(value, saved["model"][name])
        for name, value in policy.state_dict().items()
    ):
        raise ValueError("Initial model differs from parent")
    if policy_hash(policy.state_dict()) != experiment["parent_policy_hash"]:
        raise ValueError("Initial behavior hash differs")
    optimizer = load_initial_optimizer(policy, saved, config)
    shuffle_seed = seed_for(17, 0, "shuffle", namespace=experiment["shuffle_namespace"])
    budget_check = compute_time_budget_check(config)
    if budget_check is not None:
        print("Active-time budget check: " + json.dumps(budget_check, ensure_ascii=False), flush=True)
    record = {
        "evaluation_runtime_default": "auto",
        "kind": KIND,
        "config": config,
        **({"time_budget_check": budget_check} if budget_check is not None else {}),
        "parent_sha256": experiment["parent_sha256"],
        "parent_model_equal": True,
        "new_Adam_empty": not inherited,
        "parent_Adam_counter": int(experiment["parent_optimizer_steps"]),
        "critic_preserved": inherited,
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
        "source_commit": __import__("subprocess")
        .check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
        .strip(),
    }
    record["run_id"] = digest(record)
    identity = {"run_id": record["run_id"]}
    if inherited:
        identity.update(checkpoint.passport(out, config, plan))
        record["checkpoint_identity"] = identity
        record["lineage"] = {key: experiment[key] for key in (
            "parent", "parent_sha256", "parent_manifest", "parent_manifest_sha256",
            "parent_final_state", "parent_final_state_sha256", "parent_run_sha256", "start_episode")}
        runtime_for(out, config)
    counters = TrainingCounters()
    state = {
        **counters.to_state_dict(protocol=KIND),
        "updates": 0,
        "decisions": 0,
        "next_episode": experiment["start_episode"],
        "origin_Adam_counter": int(experiment["parent_optimizer_steps"]),
        "winless_streak": 0,
        "zero_accepted_streak": 0,
        "active_seconds": 0.0,
        "evaluation_seconds": 0.0,
        "critic_complete": inherited,
        "phase": "collect",
        "latest": "checkpoints/initial.pt",
        "stop_reason": None,
        "evaluated": {},
        "development_points": {},
        "baseline_wins": None,
        "best": {"path": "checkpoints/initial.pt", "wins": -1, "update": 0},
    }
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


def development(config, plan, out, state, shuffle, clock=None):
    if clock:
        clock.pause()
    started = perf_counter()
    purpose = config.get("ppo", {}).get("evaluation_purpose", "development")
    try:
        with isolated_rng(shuffle):
            result = evaluate(config, plan, out, state["latest"], purpose)
        state["evaluation_seconds"] += perf_counter() - started
        state["evaluated"][result["policy_hash"]] = {
            **result,
            "decisions": state["decisions"],
        }
        wins = result["runs"]["argmax"]["cp"]["summary"]["W"]
        if state["baseline_wins"] is None:
            state["baseline_wins"] = wins
            if config["experiment"].get("early_gain_min_wins") is not None:
                state["early_gain_required_wins"] = wins + config["experiment"]["early_gain_min_wins"]
        if wins > state["best"]["wins"]:
            state["best"] = {
                "path": result["checkpoint"],
                "wins": wins,
                "update": state["updates"],
                "policy_hash": result["policy_hash"],
            }
        for point in config["experiment"]["milestones"]:
            if point <= state["decisions"]:
                state["development_points"].setdefault(
                    str(point), result["policy_hash"]
                )
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


def train(config, plan, out, device, runtime_path=None, max_updates=None):
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
        optimizer = optimizer_for(policy, config, device)
        restored = checkpoint.restore(
            out / state["latest"], policy, optimizer, shuffle, config, identity
        )
        if restored.get("pending_update"):
            heal_interrupted_update(out, restored)
        for key in (
            "phase", "latest", "rollouts", "updates", "decisions",
            "optimizer_steps", "attempted_steps", "rejected_steps",
            "rollouts_collected", "rollouts_with_accepted_steps",
            "adam_steps_attempted", "adam_steps_accepted",
        ):
            if key in restored and key in state and restored[key] != state[key]:
                raise ValueError(f"Saved checkpoint and state disagree on {key}")
        due = target_bc_evaluation_due(config, state, current_policy_hash=policy_hash(policy.state_dict()))
        if state["phase"] != "collect":
            ledger = out / "updates.jsonl"
            committed = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()] if ledger.exists() else []
            previous = {**state, "decisions": committed[-1]["total_decisions"] if committed else 0}
            if target_bc_evaluation_due(config, previous, current_policy_hash=policy_hash(policy.state_dict())):
                raise ValueError("Cannot recover earlier development evaluation from a pending PPO boundary")
            # A milestone crossed by the pending rollout is evaluated only after PPO.
            due = False
        if due:
            if state["rollouts"]:
                rows = [json.loads(line) for line in (out / "updates.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
                if (not rows or rows[-1].get("rollout") != state["rollouts"]
                        or rows[-1].get("total_decisions") != state["decisions"]):
                    raise ValueError("Cannot recover development evaluation: update ledger and checkpoint disagree")
                if not state["evaluated"]:
                    raise ValueError("Cannot recover initial development evaluation from a later checkpoint")
                pending = [point for point in config["experiment"]["milestones"]
                           if point <= state["decisions"] and str(point) not in state["development_points"]]
                if any(row.get("total_decisions", 0) >= point
                       for point in pending for row in rows[:-1]):
                    raise ValueError("Cannot recover development evaluation: milestone was crossed before the latest checkpoint")
            development(config, plan, out, state, shuffle, clock)
        if pause_due(out, "update", state["updates"], at_boundary=state["phase"] == "collect"):
            write_json(out / "status.json", {"status": "paused", "checkpoint": state["latest"]})
            return
        if state["stop_reason"]:
            return
        factory = TargetFactory(config, plan["base"])
        # Runtime capacity can change rollout boundaries; record it separately.
        runtime_path = Path(
            runtime_path or ROOT / "configs/rl-ppo-after-target-bc-runtime.json"
        )
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
            or not 1
            <= reply_size
            <= (num_envs + collection_workers - 1) // collection_workers
        ):
            raise ValueError(
                "Runtime workers/reply size must fit the environment count"
            )
        write_json(
            out / "runtime.json",
            {
                "collection_workers": collection_workers,
                "num_envs": num_envs,
                "envs_per_worker": (num_envs + collection_workers - 1)
                // collection_workers,
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
        with (
            ResourceMonitor(out / "telemetry") as monitor,
            Collector(
                config,
                factory,
                device,
                out / "errors",
                cpu_workers=collection_workers,
                reply_size=reply_size,
                num_envs=num_envs,
                collection_scheduler=scheduler,
                inference_min_batch=runtime.get("inference_min_batch", 0),
                inference_wait_seconds=runtime.get("inference_wait_seconds", 0),
            ) as collector,
        ):
            while not state["stop_reason"]:
                critic_started = perf_counter() if state["phase"] == "critic" else None
                if max_updates is not None and state["updates"] >= max_updates:
                    write_json(out / "status.json", {"status": "paused", "reason": "max_updates"})
                    return
                deadline = lambda: clock.deadline(config)
                if deadline_reached(deadline):
                    state["stop_reason"] = "active_time_limit"
                    break
                if pause_due(out, "update", state["updates"], at_boundary=state["phase"] == "collect"):
                    write_json(out / "status.json", {"status": "paused"})
                    return
                if state["phase"] == "collect":
                    if state["decisions"] >= config["experiment"]["decisions"]:
                        state["stop_reason"] = "decision_limit"
                        break
                    monitor.mark("collection", rollout=state["rollouts"] + 1)
                    episodes, following, timing = collector.collect(
                        policy,
                        None,
                        state["next_episode"],
                        monitor,
                        deadline=deadline,
                        max_decisions=config["experiment"]["decisions"]
                        - state["decisions"],
                        episode_length_hint=(
                            state["rollout_summary"]["decisions"]
                            / max(1, state["rollout_summary"]["episodes"])
                            if state.get("rollout_summary")
                            else None
                        ),
                    )
                    if not state["critic_complete"]:
                        critic_started = perf_counter()
                    if not episodes:
                        state["stop_reason"] = "active_time_limit"
                        break
                    summary = summarize_rollout(episodes)
                    record_collection(state, summary, config)
                    state["next_episode"] = following
                    state["collection"] = timing
                    state["rollout_summary"] = summary
                    artifact = save_rollout(
                        episodes, out / "rollouts", state["rollouts"]
                    )
                    state["pending_rollout"] = "rollouts/" + artifact["path"]
                    state["phase"] = (
                        Phase.PPO.value if state["critic_complete"] else Phase.CRITIC.value
                    )
                    state["latest"] = "checkpoints/pending.pt"
                    save_boundary(
                        out, policy, optimizer, state, shuffle, config, identity, clock
                    )
                    RolloutBoundary.from_state(state).validate_sufficiency(out)
                    print(
                        f"Sampling rollout {state['rollouts']}: {summary}; total={state['decisions']}",
                        flush=True,
                    )
                else:
                    episodes = load_rollout(out / state["pending_rollout"])

                metrics = skipped_metrics(state["stop_reason"] or "active_time_limit")
                if not state["stop_reason"] and state["phase"] == "critic":
                    monitor.mark("critic_preparation")
                    with CompilationClock() as compiler:
                        compiler.monitor = monitor
                        state["critic"] = prepare_critic(
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
                    state["critic"]["runtime_compilation_seconds"] = sum(
                        b - a for a, b in compiler.intervals
                    )
                    state["critic_complete"] = True
                    state["phase"] = Phase.PPO.value
                    artifact = save_rollout(episodes, out / "critic-rollout", 1)
                    state["pending_rollout"] = "critic-rollout/" + artifact["path"]
                    optimizer = optimizer_for(policy, config, device)
                    assert not optimizer.state
                    state["latest"] = "checkpoints/critic-ready.pt"
                    save_boundary(
                        out, policy, optimizer, state, shuffle, config, identity, clock
                    )
                    RolloutBoundary.from_state(state).validate_sufficiency(out)
                    write_json(out / "critic-preparation.json", state["critic"])
                    print(
                        "Critic preparation: " + json.dumps(state["critic"]), flush=True
                    )

                if not state["stop_reason"] and not deadline_reached(deadline):
                    monitor.mark("ppo", rollout=state["rollouts"])

                    def inspect_step(step, optimizer=optimizer):
                        if (
                            step["accepted"]
                            and step.get("full_kl_checked", True)
                            and not (out / "first-accepted-step.json").exists()
                        ):
                            counters = {
                                int(value["step"]) for value in optimizer.state.values()
                            }
                            expected_counter = (int(state.get("origin_Adam_counter", 0))
                                                + int(state["optimizer_steps"])
                                                + int(step["committed_accepted_steps"]))
                            if (
                                counters != {expected_counter}
                                or int(step["accepted_step"]) != int(step["committed_accepted_steps"])
                            ):
                                raise RuntimeError(
                                    f"First PPO Adam counter differs from committed steps: {counters} != {{{expected_counter}}}"
                                )
                            write_json(
                                out / "first-accepted-step.json",
                                {
                                    "Adam_counter": expected_counter,
                                    "inherited_Adam_counter": int(state.get("origin_Adam_counter", 0)),
                                    "committed_accepted_steps": state["optimizer_steps"] + int(step["committed_accepted_steps"]),
                                    "step": step,
                                },
                            )

                    with CompilationClock() as compiler:
                        compiler.monitor = monitor
                        operation_started = perf_counter()
                        metrics = update(
                            policy,
                            optimizer,
                            episodes,
                            config,
                            state["decisions"],
                            device,
                            shuffle,
                            monitor=monitor,
                            deadline=deadline,
                            on_step=inspect_step,
                            _backend=partial(BatchedBackend, pinned_replay=True),
                        )
                        operation_ended = perf_counter()
                    metrics["runtime_compilation_seconds"] = sum(
                        b - a for a, b in compiler.intervals
                    )
                    metrics["operation_seconds"] = operation_ended - operation_started
                    metrics["operation_without_compilation_seconds"] = (
                        metrics["operation_seconds"]
                        - compiler.seconds(operation_started, operation_ended)
                    )
                    metrics["PPO_skipped"] = False
                    record_optimization(state, metrics, config)
                elif not state["stop_reason"]:
                    state["stop_reason"] = "active_time_limit"
                del episodes
                gc.collect()
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
                    "decision_overshoot": max(
                        0, state["decisions"] - config["experiment"]["decisions"]
                    ),
                    "active_time_overshoot_seconds": max(
                        0,
                        state["active_seconds"]
                        - clock.limit(config),
                    ),
                    "collection": state["collection"],
                    "ppo": metrics,
                }
                state["pending_update"] = row
                state["phase"] = Phase.COLLECT.value
                state["latest"] = f"checkpoints/update-{state['rollouts']:04}.pt"
                monitor.mark("checkpoint")
                save_boundary(
                    out, policy, optimizer, state, shuffle, config, identity, clock
                )
                with (out / "updates.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
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
                    str(point) not in state["development_points"]
                    and point <= state["decisions"]
                    for point in config["experiment"]["milestones"]
                )
                changed = policy_hash(policy.state_dict()) not in state["evaluated"]
                if due or (state["stop_reason"] and changed):
                    monitor.mark("evaluation")
                    development(config, plan, out, state, shuffle, clock)
                if pause_due(out, "update", state["updates"]):
                    write_json(out / "status.json", {"status": "paused", "checkpoint": state["latest"]})
                    return
                prune_checkpoints(out, state)
                clock.progress()
                if max_updates is not None and state["updates"] >= max_updates and not state["stop_reason"]:
                    write_json(out / "status.json", {"status": "paused", "reason": "max_updates"})
                    return
            if policy_hash(policy.state_dict()) not in state["evaluated"]:
                development(config, plan, out, state, shuffle, clock)
            clock.progress()
    finally:
        clock.progress()


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
        results, cases = {}, {}
        for path in sorted(directory.glob("*/metrics.json")):
            row = read_json(path)
            case = read_json(path.parent / "case.json")
            key = (case["parent_world"], case["case_id"], case["learner_side"])
            results[key] = int(row["winner"] == row["side"])
            cases[key] = case
        return results, cases

    dev_comparison = None
    dev_purpose = config.get("ppo", {}).get("evaluation_purpose", "development")
    min_delta = config["experiment"].get("confirmation_min_win_rate_delta")
    if min_delta is not None:
        dev_baseline = evaluate(
            config, plan, out, "checkpoints/initial.pt", dev_purpose
        )
        dev_candidate = evaluate(
            config, plan, out, state["best"]["path"], dev_purpose
        )
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
        if (
            dev_comparison["delta"] < float(min_delta)
            or dev_comparison["CI95"][0] <= min_ci_low
        ):
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
    if dev_comparison is not None:
        payload[f"{dev_purpose}_comparison"] = dev_comparison
    write_json(out / "confirmation.json", payload)


def report(config, out):
    state = read_json(out / "state.json")
    parent_label = config["experiment"].get("parent_label", "target BC epoch 10")
    eval_purpose = config.get("ppo", {}).get("evaluation_purpose", "development")
    max_rounds = config.get("environment", {}).get("max_rounds", 30)
    gamma = config["ppo"].get("gamma", 1)
    gae_lambda = config["ppo"].get("gae_lambda", 1)
    lines = [
        f"# PPO после {parent_label}",
        "",
        f"Статус: {state['stop_reason'] or state['phase']}; решения {state['decisions']}; active {state['active_seconds']:.2f} с.",
        f"Лимит active: {state.get('active_seconds_limit', config['experiment']['active_seconds']):.0f} с; исходный бюджет {config['experiment']['active_seconds']} с; продления в active-budget.json.",
        f"Sampling rollouts {state['rollouts']}; PPO updates {state['updates']}; accepted/rejected Adam {state['optimizer_steps']}/{state['rejected_steps']}.",
        f"Adam: inherited {state.get('origin_Adam_counter', 0)}; new accepted {state['optimizer_steps']}; total {state.get('origin_Adam_counter', 0) + state['optimizer_steps']}.",
        f"GPU {read_json(out / 'run.json')['GPU']}; T={config.get('policy_temperature', 1)}; max_rounds={max_rounds}; gamma={gamma}; gae_lambda={gae_lambda}; microbatch {config['ppo']['sequences_per_microbatch']}; обычный CP; reward 1/0/0.",
        "",
        f"| Решения | Checkpoint | W/D/L {eval_purpose} |",
        "|---:|---|---|",
    ]
    for point in state["evaluated"].values():
        summary = point["runs"]["argmax"]["cp"]["summary"]
        lines.append(
            f"| {point['decisions']} | {point['checkpoint']} | {summary['W']}/{summary['D']}/{summary['L']} |"
        )
    lines += ["", f"Выбор: {state['best']}."]
    if (out / "runtime.json").exists():
        lines += ["", "Runtime: " + json.dumps(read_json(out / "runtime.json"), ensure_ascii=False)]
    if (out / "updates.jsonl").exists():
        for name, last in (("performance", None), ("performance-last10", 10)):
            performance = summarize_performance(out, last=last)
            write_json(out / f"{name}.json", performance)
            (out / f"{name}.md").write_text(
                performance_markdown(performance), encoding="utf-8"
            )
        lines += ["", "Performance: [all cycles](performance.md), [last 10](performance-last10.md)."]
        rows = [
            json.loads(line)
            for line in (out / "updates.jsonl").read_text().splitlines()
        ]
        lines += [
            "",
            "| Rollout | Sampling W/D/L | Raw advantage std | Critic MSE / EV / mean | Entropy | Full KL | Adam accept/reject | PPO skipped |",
            "|---:|---|---:|---|---:|---:|---|---|",
        ]
        for row in rows:
            metrics = row["ppo"]
            lines.append(
                f"| {row['rollout']} | {row['W']}/{row['D']}/{row['L']} | {metrics.get('raw_advantage_std')} | "
                f"{metrics.get('critic_mse')}/{metrics.get('critic_explained_variance')}/{metrics.get('critic_mean')} | "
                f"{metrics.get('entropy')} | {metrics.get('final_accepted_kl')} | "
                f"{metrics['optimizer_steps']}/{metrics['rejected_optimizer_steps']} | {metrics['PPO_skipped']} |"
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
            json.dumps(point["slices"], ensure_ascii=False),
        ]
    if state["stop_reason"] == "no_development_gain":
        lines += ["", "Прироста в установленном раннем бюджете не получено."]
    lines += [
        "",
        f"Ранний порог: {state.get('early_gain_required_wins', 'по конфигу')} побед после {config['experiment']['early_gain_decisions']} новых решений; полезность оценивается по парному additional-приросту и CI.",
    ]
    for name in ("critic-preparation.json", "confirmation.json"):
        if (out / name).exists():
            lines += [
                "",
                name,
                "",
                json.dumps(read_json(out / name), ensure_ascii=False),
            ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(phase, config, out, device, runtime_path=None, max_updates=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    validate_profile(config)
    if max_updates is not None and (type(max_updates) is not int or max_updates < 1):
        raise ValueError("max_updates must be a positive operational limit")
    if phase == "pause":
        (out / "pause.request").write_text("pause requested\n", encoding="utf-8")
        return
    with RunLock(out):
        if phase in ("train", "run", "resume"):
            assert_not_closed(out)
        try:
            if phase == "resume":
                (out / "pause.request").unlink(missing_ok=True)
            if not (out / "run.json").exists():
                initialize(config, out)
            if read_json(out / "run.json")["config"] != config:
                raise ValueError("Run configuration changed")
            if config.get("evaluation_runtime"):
                runtime_for(out, config)
            plan = read_json(out / "manifest.json")
            if phase in ("train", "run", "resume"):
                write_json(out / "status.json", {"status": "training"})
                train(config, plan, out, device, runtime_path=runtime_path, max_updates=max_updates)
                if read_json(out / "status.json").get("status") == "paused":
                    report(config, out)
                    return
            if phase in ("evaluate", "run", "resume"):
                if not read_json(out / "state.json")["stop_reason"]:
                    raise ValueError("Training has not stopped")
                finish(config, plan, out)
                write_json(out / "status.json", {"status": "complete"})
            report(config, out)
        except BaseException:
            (out / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            write_json(
                out / "status.json", {"status": "technical_error", "phase": phase}
            )
            raise
