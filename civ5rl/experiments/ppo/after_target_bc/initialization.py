"""Validate parent lineage and publish the initial PPO boundary."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from civ5rl.core.model import Policy
from civ5rl.core.preview import ModelExtension, inherited_model, model_extension
from civ5rl.engine.collection.opponents import U0_CHECKPOINT, u0_checkpoint
from civ5rl.engine.evaluation.executor import runtime_for
from civ5rl.engine.ppo.adam import adapts_value
from civ5rl.engine.runtime.control import TrainingCounters
from civ5rl.experiments._shared.provenance_checks import verify_manifest_digest
from civ5rl.experiments.imitation.helpers import load_policy
from civ5rl.experiments.ppo.after_target_bc.boundary import RolloutBoundary, save_boundary
from civ5rl.experiments.ppo.after_target_bc.optimizer import load_initial_optimizer
from civ5rl.protocols.ppo_after_target_bc import (
    KIND,
    compute_time_budget_check,
)
from civ5rl.protocols.turn_reset import evaluation_plan
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import (
    ROOT,
    atomic_output,
    configure,
    file_sha256,
    read_json,
    seed_for,
    write_json,
)
from civ5rl.provenance.identity import policy_hash
from civ5sim.engine import digest

def make_policy(config, device):
    policy = load_policy(config, ROOT / config["experiment"]["parent"], device)
    for parameter in policy.parameters():
        parameter.requires_grad_(True)
    return policy


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
        same_parent_run = (
            source["state_sha256"] == experiment.get("parent_final_state_sha256")
            and source["run_sha256"] == experiment.get("parent_run_sha256")
        )
        if not same_parent_run and (
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
    path = ROOT / experiment.get("evaluation_manifest", experiment["parent_manifest"])
    if "evaluation_manifest" in experiment and file_sha256(path) != experiment["evaluation_manifest_sha256"]:
        raise ValueError("Frozen evaluation manifest SHA-256 differs")
    plan = read_json(path)
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
    if experiment.get("three_opponent_evaluation"):
        from civ5rl.protocols.ppo_target_scale import paired_validation_cases
        from civ5rl.protocols.common import own_state_physical_hash
        from civ5sim.engine import digest

        existing = []
        seen = set()
        for key, count in (("development", 128), ("confirmation", 256), ("development_512", 512)):
            cases = derived_plan[key]
            hashes = {own_state_physical_hash(c["scenario"]) for c in cases}
            if len(cases) != count or len(hashes) != count // 2 or seen & hashes or sum(
                c["placement"] == "contact" for c in cases
            ) != count // 2:
                raise ValueError(f"Unbalanced or overlapping evaluation layouts: {key}")
            seen.update(hashes)
            existing.extend(cases)
        derived_plan["reserve_256"] = paired_validation_cases(
            config, derived_plan["base"], 8, 28, "ppo_c2pa_cpv2_reserve_256_v1", existing
        )
        if len(derived_plan["reserve_256"]) != 256:
            raise ValueError("Reserve-256 requires sixteen validation worlds")
        derived_plan["hash"] = digest({k: v for k, v in derived_plan.items() if k != "hash"})
    return derived_plan


def initialize(config: dict[str, Any], out: Path) -> dict[str, Any]:
    experiment = config["experiment"]
    parent = ROOT / experiment["parent"]
    inherited = experiment.get("parent_optimizer") == "preserve"
    parent_record = _verify_parent_lineage(experiment, parent, inherited=inherited)
    plan = _initial_evaluation_plan(config)
    write_json(out / "manifest.json", plan)
    seed = config.get("seed", 17)
    torch_seed = seed_for(seed, 0, "torch", namespace=experiment["torch_namespace"]) if inherited else seed
    configure(torch_seed, "cpu")
    policy = make_policy(config, "cpu")
    saved = torch.load(parent, map_location="cpu", weights_only=False)
    if inherited and saved["identity"].get("run_id") != parent_record["run_id"]:
        raise ValueError("Parent checkpoint and final run lineage differ")
    extension = model_extension(config)
    if extension:
        if (saved["variant"], policy.variant) != (extension.parent, extension.child):
            raise ValueError("Model extension requires its declared parent architecture")
        inherited_model(policy.state_dict(), saved["model"], extension)
    elif not all(torch.equal(value, saved["model"][name]) for name, value in policy.state_dict().items()):
        raise ValueError("Initial model differs from parent")
    if policy_hash(saved["model"]) != experiment["parent_policy_hash"]:
        raise ValueError("Initial behavior hash differs")
    optimizer = load_initial_optimizer(policy, saved, config)
    shuffle_seed = seed_for(seed, 0, "shuffle", namespace=experiment["shuffle_namespace"])
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
    if u0_checkpoint(config) == U0_CHECKPOINT:
        anchor = out / U0_CHECKPOINT
        anchor.parent.mkdir(parents=True, exist_ok=True)
        with atomic_output(anchor) as temporary:
            shutil.copyfile(parent, temporary)
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
        "seed": config.get("seed", 17),
        "shuffle_seed": shuffle_seed,
        "torch_seed": torch_seed,
        "GPU": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    if experiment.get("three_opponent_evaluation"):
        from civ5sim.combat_ai import LIBRARY_PATH

        record["algorithmic_opponents"] = {
            "cp_search": config["teacher_search"],
            "cpv2_agent_sha256": file_sha256(ROOT / "civ5sim/cpv2_agent.py"),
            "combat_ai_sha256": file_sha256(LIBRARY_PATH),
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


def _initial_training_state(config: dict[str, Any], *, inherited: bool, extension: ModelExtension | None) -> dict[str, Any]:
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
        state["origin_Adam_counters"][extension.adam_group] = 0
    if "value_optimizer" in experiment:
        state["value_optimizer"] = experiment["value_optimizer"]
    if adapts_value(config) and "origin_Adam_counters" in state:
        state["value_optimizer"] = "reset_then_preserve"
        state["origin_Adam_counters"]["value"] = 0
        if extension and extension.adds_action_context:
            state["origin_Adam_counters"]["value_actions"] = 0
    if config.get("self_play", {}).get("enabled"):
        state["self_play_archive"] = []
        state["self_play_last_archived_optimizer_steps"] = 0
    return state
