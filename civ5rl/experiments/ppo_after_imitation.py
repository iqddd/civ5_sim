"""Bounded on-policy P(win) fine-tuning from the frozen own-state BC policy."""

from __future__ import annotations

import ast
import gc
import hashlib
import random
import tempfile
import zipfile
from collections import Counter
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import torch

from civ5rl.engine.collection.collector import Collector
from civ5rl.engine.ppo.core import update
from civ5rl.engine.runtime.control import TrainingCounters
from civ5rl.engine.runtime.telemetry import ResourceMonitor, analyze_telemetry
from civ5rl.experiments._shared.evaluation_runner import run_chunked_pool
from civ5rl.experiments._shared.io import append_jsonl, read_jsonl
from civ5rl.experiments._shared.lifecycle import experiment_run_guard
from civ5rl.experiments._shared.markdown import MarkdownReport, format_wdl
from civ5rl.experiments._shared.outcomes import (
    WDLSummary,
    collect_paired_outcomes,
    select_outcome_representatives,
    summarize_wdl,
    verify_match_artifacts,
)
from civ5rl.experiments._shared.provenance_checks import verify_manifest_digest
from civ5rl.experiments.evaluation import (
    paired_world_bootstrap,
    play_case,
    verify_replay,
)
from civ5rl.experiments.imitation_helpers import (
    CompilationClock,
    isolated_rng,
    load_policy,
    pool,
)
from civ5rl.experiments.ppo_artifacts import prune_checkpoints, save_rollout
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
from civ5rl.provenance.validation import validate_config
from civ5sim.engine import digest

if TYPE_CHECKING:
    from civ5rl.core.model import Policy


def policy_hash(model):
    hasher = hashlib.sha256()
    for key, value in sorted(model.items()):
        if not key.startswith("value."):
            hasher.update(key.encode())
            hasher.update(value.detach().cpu().contiguous().numpy().tobytes())
    return hasher.hexdigest()


def optimizer_for(policy, config, device):
    optimizer_config = config["optimizer"]
    return torch.optim.Adam(
        policy.parameters(),
        lr=optimizer_config["lr"],
        betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["eps"],
        weight_decay=optimizer_config["weight_decay"],
        fused=str(device).startswith("cuda"),
    )


def create_policy(config, device):
    policy = load_policy(config, ROOT / config["experiment"]["parent"], device)
    for parameter in policy.value.parameters():
        parameter.requires_grad_(True)
    with torch.no_grad():
        policy.value[-1].weight.zero_()
        policy.value[-1].bias.zero_()
    return policy


def execution_identity(config, plan):
    with tempfile.TemporaryDirectory() as tmp:
        return checkpoint.passport(tmp, config, plan)


def check_identity(config, out, *, for_execution=False):
    """Validate frozen artifacts for analysis; opt in to live-code checks to run."""
    out = Path(out)
    plan = read_json(out / "manifest.json")
    verify_manifest_digest(plan)
    if not for_execution:
        checkpoint.verify_frozen_passport(out, config=config, manifest=plan)
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
    configure(17, "cuda" if torch.cuda.is_available() else "cpu")
    out.mkdir(parents=True, exist_ok=True)
    if file_hash(ROOT / config["experiment"]["parent"]) != config["experiment"]["parent_sha256"]:
        raise ValueError("Parent checkpoint differs")
    parent = ROOT / config["experiment"]["parent_run"]
    previous = checkpoint.verify_frozen_passport(parent, sources=False)
    plan = prepare_cases(config)
    current = execution_identity(config, plan)
    write_json(out / "manifest.json", plan)
    checkpoint.passport(out, config, plan)
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
    rng = random.Random(17)
    checkpoint.save(out / state["latest"], policy, optimizer, state, rng, config, current)
    write_json(out / "state.json", state)
    changes = {
        name: {
            "parent": previous["source_hashes"].get(name),
            "current": current["source_hashes"].get(name),
        }
        for name in previous["source_hashes"].keys() | current["source_hashes"].keys()
        if previous["source_hashes"].get(name) != current["source_hashes"].get(name)
    }
    write_json(
        out / "lineage.json",
        {
            "parent": config["experiment"]["parent"],
            "parent_sha256": config["experiment"]["parent_sha256"],
            "parent_Adam_counter": 0 if controlled(config) else 2480,
            "BC_origin_Adam_counter": 2480,
            "reused_training_streams": controlled(config),
            "initial_model_bitwise_equal": controlled(config),
            "PPO_Adam_counter": 0,
            "initial_policy_equal": True,
            "value_last_layer_zero": True,
            "changed_sources": changes,
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
    return play_case(config, case, checkpoint_path, mode, enemy, directory)


def match_summary(directory: str | Path, expected: int) -> WDLSummary:
    rows = [read_json(p) for p in sorted(Path(directory).glob("*/metrics.json"))]
    if len(rows) != expected or any(r["status"] != "completed" for r in rows):
        raise ValueError("Incomplete evaluation")
    return summarize_wdl(rows)


_BASELINE_EVALUATOR_DEPENDENCIES = (
    "civ5rl/core/agent.py",
    "civ5rl/core/model.py",
    "civ5rl/core/observation.py",
    "civ5rl/core/schema.py",
    "civ5rl/protocols/imitation.py",
    "civ5sim/match.py",
    "civ5sim/engine.py",
    "civ5sim/cp_agent.py",
    "civ5sim/greedy_agent.py",
    "civ5sim/combat_ai.py",
    "civ5sim/_native.py",
    "build/combat.dll",
)


def _verify_parent_evaluator_closure(
    previous_hashes: Mapping[str, str],
    current_hashes: Iterable[tuple[str, str]],
) -> Iterable[str]:
    for dependency, current_sha256 in current_hashes:
        if previous_hashes.get(dependency) != current_sha256:
            yield dependency


def _verify_saved_evaluation_cases(
    directory: Path,
    cases: Iterable[Mapping[str, Any]],
    *,
    case_error: str,
    replay_error: str,
) -> None:
    for case in cases:
        case_dir = directory / f"case-{case['case_id']:03}-side-{case['learner_side']}"
        if read_json(case_dir / "case.json") != case:
            raise ValueError(case_error)
        if not verify_match_artifacts(case_dir, ("replay.json", "final.json", "result.json")):
            raise ValueError(replay_error)


def _parent_baseline_directories(parent, cases, mismatched):
    dirs = {}
    try:
        for mode in ("argmax", "sample"):
            dirs[mode] = {}
            for enemy in ("cp", "greedy"):
                directory = parent / f"cycle-02/evaluation/{mode}/{enemy}"
                _verify_saved_evaluation_cases(
                    directory,
                    cases,
                    case_error="Baseline case differs",
                    replay_error="Missing baseline replay",
                )
                dirs[mode][enemy] = {
                    "directory": str(directory),
                    "summary": match_summary(directory, 64),
                }
    except (OSError, ValueError) as exc:
        mismatched.append(str(exc))
    return dirs


def reusable_baseline(config, plan, out):
    parent = ROOT / config["experiment"]["parent_run"]
    old = read_json(parent / "passport.json")
    previous_hashes = old["source_hashes"]
    closure = _BASELINE_EVALUATOR_DEPENDENCIES
    mismatched = list(
        _verify_parent_evaluator_closure(
            previous_hashes, ((name, file_hash(ROOT / name)) for name in closure)
        )
    )
    if previous_hashes.get("build/combat_ai.dll") != file_hash(ROOT / "build/combat_ai.dll"):
        mismatched.append("unverified_parent_planner_DLL")
    old_config = read_json(parent / "resolved-config.json")
    if old_config["teacher_search"] != config["teacher_search"]:
        mismatched.append("teacher_search")
    dirs = _parent_baseline_directories(parent, plan["development"], mismatched)
    write_json(
        out / "baseline-reuse-check.json",
        {
            "dependency_hashes": {name: file_hash(ROOT / name) for name in closure},
            "mismatched": mismatched,
            "reused": not mismatched,
        },
    )
    return dirs if not mismatched else None


EVALUATOR_VERSION = "1.0.0"

EVALUATION_DEPENDENCIES = [
    "civ5rl/core/agent.py",
    "civ5rl/core/model.py",
    "civ5rl/core/observation.py",
    "civ5rl/core/schema.py",
    "civ5rl/core/context.py",
    "civ5rl/protocols/imitation.py",
    "civ5rl/protocols/imitation_helpers.py",
    "civ5sim/match.py",
    "civ5sim/engine.py",
    "civ5sim/cp_agent.py",
    "civ5sim/greedy_agent.py",
    "civ5sim/combat_ai.py",
    "civ5sim/_native.py",
    "build/combat.dll",
    "build/combat_ai.dll",
]


def evaluator_dependency_hashes():
    existing = [p for p in EVALUATION_DEPENDENCIES if (ROOT / p).exists()]
    return {p: file_hash(ROOT / p) for p in existing}


def match_execution(source: str) -> str:
    try:
        tree = ast.parse(source)
    except Exception as exc:
        raise ValueError(f"Failed to parse source AST: {exc}") from exc

    def find_directory_start(body):
        for i, n in enumerate(body):
            if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "directory" for t in n.targets
            ):
                return i
            if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "directory":
                return i
        for i, n in enumerate(body):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if any(isinstance(node, ast.Name) and node.id == "directory" for node in ast.walk(t)):
                        return i
            if isinstance(n, ast.AnnAssign) and any(
                isinstance(node, ast.Name) and node.id == "directory" for node in ast.walk(n.target)
            ):
                return i
        return None

    play_case = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "play_case"),
        None,
    )
    if play_case is not None:
        start = find_directory_start(play_case.body)
        body = play_case.body[start:] if start is not None else play_case.body
        return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)

    eval_task = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluation_task"),
        None,
    )
    if eval_task is not None:
        start = find_directory_start(eval_task.body)
        body = eval_task.body[start:] if start is not None else eval_task.body
        return ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False)

    raise ValueError("Neither play_case nor evaluation_task found in evaluator source")


def _verify_archived_match_execution(parent: Path, mismatched: list[str]) -> None:
    # The controlled kind adds a manifest guard before the ordinary match code.
    try:
        archive_path = parent / "sources.zip"
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing archive: {archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            source_path = "civ5rl/experiments/ppo_after_imitation.py"
            if source_path not in archive.namelist():
                raise KeyError(f"{source_path} not found in archive")
            old_source = archive.read(source_path).decode("utf-8")
        current_source = Path(__file__).read_text(encoding="utf-8")
        if match_execution(old_source) != match_execution(current_source):
            mismatched.append("evaluator_match_execution")
    except Exception as exc:  # noqa: BLE001 -- record unreadable or altered archive format as mismatch
        mismatched.append(f"unrecognized_archive_format: {exc}")


def _controlled_reuse_mismatches(
    parent: Path,
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    purpose: str,
    dependency_hashes: Mapping[str, str],
) -> list[str]:
    mismatched: list[str] = []
    try:
        recorded = read_json(parent / "passport.json")
        mismatched.extend(
            _verify_parent_evaluator_closure(recorded.get("source_hashes", {}), dependency_hashes.items())
        )
    except Exception as exc:  # noqa: BLE001 -- record unreadable passport as mismatch
        mismatched.append(f"unrecognized_parent_passport: {exc}")
    _verify_archived_match_execution(parent, mismatched)

    try:
        resolved = read_json(parent / "resolved-config.json")
        if resolved.get("teacher_search") != config.get("teacher_search"):
            mismatched.append("teacher_search")
    except Exception as exc:  # noqa: BLE001 -- record unreadable config as mismatch
        mismatched.append(f"unrecognized_config: {exc}")

    try:
        previous_plan = read_json(parent / "manifest.json")
        if plan.get(purpose) != previous_plan.get(purpose):
            mismatched.append("cases")
    except Exception as exc:  # noqa: BLE001 -- record unreadable manifest as mismatch
        mismatched.append(f"unrecognized_manifest: {exc}")
    return mismatched


def _verify_saved_controlled_evaluation(result, cases, fingerprint):
    model = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)["model"]
    if policy_hash(model) != fingerprint:
        raise ValueError("Evaluation policy weights changed")
    for enemies in result["runs"].values():
        for record in enemies.values():
            directory = Path(record["directory"])
            if match_summary(directory, len(cases)) != record["summary"]:
                raise ValueError("Evaluation results changed")
            _verify_saved_evaluation_cases(
                directory,
                cases,
                case_error="Evaluation provenance/replay missing",
                replay_error="Evaluation provenance/replay missing",
            )


def reusable_controlled_evaluation(config, plan, out, fingerprint, purpose):
    parent = ROOT / config["experiment"]["parent_run"]
    dep_hashes = evaluator_dependency_hashes()
    mismatched = _controlled_reuse_mismatches(parent, config, plan, purpose, dep_hashes)
    result = None
    try:
        old_cache = read_json(parent / "evaluations.json")
        result = old_cache.get(purpose + ":" + fingerprint)
        if result is None:
            mismatched.append("policy_not_evaluated")
    except Exception as exc:  # noqa: BLE001 -- record unreadable evaluations as mismatch
        mismatched.append(f"unrecognized_evaluations: {exc}")

    if not mismatched and result is not None:
        try:
            _verify_saved_controlled_evaluation(result, plan[purpose], fingerprint)
        except (OSError, ValueError) as exc:
            mismatched.append(str(exc))

    write_json(
        out / f"reuse-{purpose}-{fingerprint[:16]}.json",
        {
            "source": str(parent),
            "mismatched": mismatched,
            "reused": not mismatched and result is not None,
            "dependency_hashes": dep_hashes,
        },
    )
    if mismatched or result is None:
        return None

    return {
        **result,
        "reused_parent": True,
        "seconds": 0.0,
        "source_result": result,
        "evaluator_version": result.get("evaluator_version", EVALUATOR_VERSION),
        "dependency_hashes": result.get("dependency_hashes", dep_hashes),
    }


def evaluate_checkpoint(config, plan, out, path, purpose="development"):
    model = torch.load(path, map_location="cpu", weights_only=False)["model"]
    fingerprint = policy_hash(model)
    cache_path = out / "evaluations.json"
    cache = read_json(cache_path) if cache_path.exists() else {}
    key = purpose + ":" + fingerprint
    if key in cache:
        return cache[key]
    if controlled(config):
        reused_result = reusable_controlled_evaluation(config, plan, out, fingerprint, purpose)
        if reused_result is not None:
            cache[key] = reused_result
            write_json(cache_path, cache)
            return reused_result
    if (
        purpose == "development"
        and not controlled(config)
        and (out / "baseline-policy.json").is_file()
        and fingerprint == read_json(out / "baseline-policy.json")["hash"]
    ):
        reused = reusable_baseline(config, plan, out)
    else:
        reused = None
    started = perf_counter()
    runs = {}
    if reused:
        runs = reused
    else:
        modes = ("argmax", "sample") if purpose == "development" else ("argmax",)
        enemies = ("cp", "greedy") if purpose == "development" else ("cp",)
        with pool(config["workers"]) as executor:
            for mode in modes:
                runs[mode] = {}
                for enemy in enemies:
                    directory = out / f"evaluation/{purpose}/{fingerprint[:16]}/{mode}/{enemy}"
                    tasks = []
                    for i, case in enumerate(plan[purpose]):
                        d = directory / f"case-{case['case_id']:03}-side-{case['learner_side']}"
                        if not (d / "metrics.json").exists():
                            tasks.append(
                                (
                                    config,
                                    str(out / "manifest.json"),
                                    purpose,
                                    i,
                                    str(path),
                                    mode,
                                    enemy,
                                    str(d),
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
    dep_hashes = evaluator_dependency_hashes()
    result = {
        "policy_hash": fingerprint,
        "checkpoint": str(path),
        "purpose": purpose,
        "evaluator_version": EVALUATOR_VERSION,
        "dependency_hashes": dep_hashes,
        "runs": runs,
        "reused_parent": bool(reused),
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


def summarize_rollout(episodes):
    count = Counter(
        "D"
        if e["result"]["winner"] is None
        else "W"
        if e["result"]["winner"] == e["spec"]["learner_side"]
        else "L"
        for e in episodes
    )
    free = sum(sum(len(a) >= 2 for a in e["actions"]) for e in episodes)
    total = sum(len(e["chosen"]) for e in episodes)
    summary = {
        "episodes": len(episodes),
        "W": count["W"],
        "D": count["D"],
        "L": count["L"],
        "win_rate": count["W"] / len(episodes),
        "decisions": total,
        "free_decisions": free,
        "forced_decisions": total - free,
        "positive_reward_episodes": count["W"],
    }
    if any(e["spec"].get("opponent") == "policy" for e in episodes):
        buckets = {}
        for e in episodes:
            spec = e["spec"]
            if spec.get("opponent") != "policy":
                bucket = "cp"
            else:
                opp_id = spec.get("opponent_id") or "current"
                bucket = "archive" if opp_id.startswith("archive:") else opp_id
            row = buckets.setdefault(bucket, Counter())
            outcome = (
                "D"
                if e["result"]["winner"] is None
                else "W"
                if e["result"]["winner"] == spec["learner_side"]
                else "L"
            )
            row[outcome] += 1
            row["episodes"] += 1
            row["decisions"] += len(e["chosen"])
        summary["by_opponent"] = {
            name: {
                "episodes": int(b["episodes"]),
                "W": int(b["W"]),
                "D": int(b["D"]),
                "L": int(b["L"]),
                "win_rate": b["W"] / max(1, b["episodes"]),
                "decisions": int(b["decisions"]),
            }
            for name, b in sorted(buckets.items())
        }
    return summary


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
            from civ5rl.experiments.ppo_controlled_checks import preflight
        else:
            from civ5rl.experiments.ppo_after_imitation_checks import preflight
        preflight(config, plan, out)
    if not read_json(out / "preflight.json")["passed"]:
        raise RuntimeError("Acceptance checks failed; training is disabled")
    configure(17, device)
    policy = create_policy(config, device)
    optimizer = optimizer_for(policy, config, device)
    rng = random.Random(17)
    identity = read_json(out / "passport.json")
    restored = checkpoint.restore(out / state["latest"], policy, optimizer, rng, config, identity)
    # state.json can include a completed evaluation after the model checkpoint was saved.
    assert (
        restored["decisions"] == state["decisions"]
        and restored["optimizer_steps"] == state["optimizer_steps"]
    )
    if not (out / "baseline-policy.json").exists():
        write_json(out / "baseline-policy.json", {"hash": policy_hash(policy.state_dict())})
    if state["pending_evaluation"]:
        development(config, plan, out, state)
    return _PpoTraining(state, protocol, policy, optimizer, rng, identity)


def _warmup_ppo_training(
    training: _PpoTraining, monitor: ResourceMonitor, device: str | torch.device
) -> None:
    # One no-update warmup uses saved public inputs; RNG is restored before collection.
    monitor.mark("initial_warmup")
    prep = perf_counter()
    # The warmup module also imports this runner's factories.
    from civ5rl.experiments.ppo_after_imitation_checks import warmup

    with isolated_rng(training.rng):
        warmup(training.policy, device)
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
        evaluated_hashes = {e["policy_hash"] for e in state["evaluated"].values()}
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


def _report_gpu_metrics(out: Path) -> dict[str, Any] | None:
    telemetry = out / "telemetry/resources.jsonl"
    if not telemetry.exists():
        return None
    gpu: dict[str, Any] = analyze_telemetry(telemetry, active_phase="ppo")
    write_json(out / "gpu-analysis.json", {k: v for k, v in gpu.items() if k != "active"})
    return gpu


def _render_report(
    config: dict[str, Any],
    out: Path,
    state: dict[str, Any],
    rows: list[dict[str, Any]],
    final: dict[str, Any] | None,
) -> MarkdownReport:
    lines = MarkdownReport(
        [
            "# PPO с контролем принятого изменения политики"
            if controlled(config)
            else "# PPO после имитации CP",
            "",
            f"Статус: {read_json(out / 'status.json')['status']}; причина остановки: {state['stop_reason']}.",
            f"Решения: {state['decisions']}; rollouts: {state['updates']}; Adam steps: {state['optimizer_steps']}; активное время: {state['active_seconds']:.2f} с.",
            f"Evaluation: {state['evaluation_seconds']:.2f} с; подготовка: {state['preparation_seconds']:.2f} с; сохранение checkpoints: {state['save_seconds']:.2f} с.",
            "",
            "Основной режим — argmax; sampling и greedy — отдельные контрольные показатели.",
            "",
        ]
    )
    lines.table(
        (
            "Update",
            "Решения",
            "Adam",
            "Active, с",
            "Train W/D/L",
            "Win rate",
            "Free/forced",
            "KL free / all",
            "Stop",
        ),
        ("left", "right", "right", "right", "right", "right", "right", "left", "left"),
        (
            (
                r["update"],
                r["total_decisions"],
                r["total_Adam_steps"],
                f"{r['active_seconds']:.1f}",
                format_wdl(r),
                f"{r['win_rate']:.1%}",
                f"{r['free_decisions']}/{r['forced_decisions']}",
                f"{r['ppo'].get('sampled_kl_free')} / {r['ppo'].get('sampled_kl')}",
                r["ppo"]["stop_reason"],
            )
            for r in rows
        ),
    )
    lines += ["", "## Development", ""]
    lines.table(
        ("Update", "Режим", "Соперник", "W/D/L", "Win rate"),
        ("left", "left", "left", "right", "right"),
        (
            (
                number,
                mode,
                enemy,
                format_wdl(record["summary"]),
                f"{record['summary']['win_rate']:.1%}",
            )
            for number, evaluation in state["evaluated"].items()
            for mode, enemies in evaluation["runs"].items()
            for enemy, record in enemies.items()
        ),
    )
    lines += ["", f"Выбранный кандидат: {state['best']}."]
    if final:
        lines += [
            "",
            "## Дополнительные 256 матчей argmax против CP",
            "",
            f"Baseline: {final['baseline']['runs']['argmax']['cp']['summary']}",
            f"Кандидат: {final['candidate']['runs']['argmax']['cp']['summary']}",
            f"Парный прирост и 95% CI: {final['comparison']}",
            f"Подтверждённый прирост: {final['confirmed_improvement']}; ориентир +10 п.п.: {final['practical_10pp']}.",
        ]
    if rows:
        r = rows[-1]
        lines += [
            "",
            f"Последний rollout: {r['decisions']} решений; превышение общего лимита: {r['decision_overshoot']}; превышение времени: {r['active_time_overshoot_seconds']:.3f} с.",
        ]
    gpu = _report_gpu_metrics(out)
    if gpu is not None:
        lines += [
            "",
            "## NVML",
            "",
            f"PPO samples: {gpu['metrics']}; непрерывных 60-секундных окон: {len(gpu['windows'])}.",
            "CPU collection и evaluation исключены. Компиляция внутри update учитывается в активном бюджете; начальная подготовка/прогрев отделены.",
            "Предыдущие аппаратные измерения выявили CPU/autograd/launch overhead и низкую occupancy spatial GEMM без устойчивого насыщения DRAM. Здесь условия обучения не менялись ради загрузки; длительности непрерывных GPU-фаз и окна публикуются отдельно.",
        ]
    lines += [
        "",
        "## Границы вывода",
        "",
        "Дополнительная проверка повторно использует 256 validation-матчей прежнего PPO; это открытый validation, не test."
        if controlled(config)
        else "Дополнительная проверка использует новые расстановки знакомых validation-worlds. Test закрыт; 75% — цель проекта, не автоматическое подтверждение обобщения.",
        "Отсутствие побед в rollout явно означает отсутствие положительной терминальной награды. Награда, модель, температура и бюджет автоматически не менялись.",
    ]
    if controlled(config):
        # Controlled diagnostics use the policy and optimizer factories above.
        from civ5rl.experiments.ppo_controlled_checks import extend_report

        lines += extend_report(config, out, state, rows, final)
    return lines


def _plot_learning_curves(out: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(
        [r["total_decisions"] for r in rows],
        [r["win_rate"] for r in rows],
        label="Train P(win)",
    )
    axes[1].plot(
        [r["total_decisions"] for r in rows],
        [r["ppo"].get("sampled_kl_free") or 0 for r in rows],
        label="Post-update sampled KL (free)",
    )
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.25)
        ax.set_xlabel("Learner decisions")
    fig.savefig(out / "learning-curves.png", dpi=150)
    plt.close(fig)


def report(config: dict[str, Any], out: Path) -> dict[str, Any]:
    state: dict[str, Any] = read_json(out / "state.json")
    rows = read_jsonl(out / "updates.jsonl", missing_ok=True, skip_blank=False, encoding=None)
    final_path = out / "confirmation.json"
    final = read_json(final_path) if final_path.exists() else None
    _render_report(config, out, state, rows, final).write(out / "report.md")
    if rows:
        _plot_learning_curves(out, rows)
    return state


def _audit_replay_selection(cache: dict[str, Any]) -> list[Path]:
    selected: dict[Path, None] = {}
    for evaluation in cache.values():
        if evaluation["reused_parent"]:
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
    checks = [verify_replay(p) for p in selected]
    write_json(out / "replay-verification.json", checks)
    if controlled(config):
        descriptions = []
        for directory, check in zip(selected, checks):
            replay = read_json(directory / "replay.json")
            descriptions.append(
                {
                    "directory": str(directory),
                    "metrics": read_json(directory / "metrics.json"),
                    "commands": dict(Counter(s["action"]["kind"] for s in replay["steps"])),
                    "verification": check,
                }
            )
        write_json(out / "representative-battles.json", descriptions)
    hashes = {
        p.relative_to(out).as_posix(): file_hash(p)
        for p in out.rglob("*")
        if p.is_file()
        and p.name not in ("artifact-hashes.json",)
        and p.name != "run.lock"
        and "__pycache__" not in p.parts
        and p.suffix not in (".log",)
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
