"""Bounded on-policy P(win) fine-tuning from the frozen own-state BC policy."""

import ast
import gc
import hashlib
import json
import random
import tempfile
import traceback
import zipfile
from collections import Counter
from pathlib import Path
from time import perf_counter

import torch

from civ5rl.core.agent import RLAgent
from civ5rl.engine.collector import Collector
from civ5rl.engine.control import RunLock
from civ5rl.engine.counters import TrainingCounters
from civ5rl.engine.ppo import update
from civ5rl.engine.resources import ResourceMonitor, analyze_telemetry
from civ5rl.experiments.evaluation import paired_world_bootstrap, verify_replay
from civ5rl.experiments.imitation_helpers import (
    CompilationClock,
    isolated_rng,
    load_policy,
    pool,
)
from civ5rl.experiments.ppo_artifacts import prune_checkpoints, save_rollout
from civ5rl.protocols.imitation import search_config
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
from civ5sim import MatchConfig
from civ5sim.cp_agent import CPAgent
from civ5sim.engine import digest
from civ5sim.match import run_match


def policy_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.items()):
        if not key.startswith("value."):
            h.update(key.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def optimizer_for(policy, config, device):
    o = config["optimizer"]
    return torch.optim.Adam(
        policy.parameters(),
        lr=o["lr"],
        betas=tuple(o["betas"]),
        eps=o["eps"],
        weight_decay=o["weight_decay"],
        fused=str(device).startswith("cuda"),
    )


def create_policy(config, device):
    policy = load_policy(config, ROOT / config["experiment"]["parent"], device)
    for p in policy.value.parameters():
        p.requires_grad_(True)
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
    if digest({k: v for k, v in plan.items() if k != "hash"}) != plan["hash"]:
        raise ValueError("Manifest changed")
    if not for_execution:
        checkpoint.verify_frozen_passport(out, config=config, manifest=plan)
        return plan
    validate_config(config)
    checkpoint.verify_execution_passport(out, config, plan)
    if (
        file_hash(ROOT / config["experiment"]["parent"])
        != config["experiment"]["parent_sha256"]
    ):
        raise ValueError("Parent changed")
    # Validate only open terrain named by the frozen plan, without loading closed maps.
    from civ5rl.protocols.corpus import validate_corpus

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
    if (
        file_hash(ROOT / config["experiment"]["parent"])
        != config["experiment"]["parent_sha256"]
    ):
        raise ValueError("Parent checkpoint differs")
    parent = ROOT / config["experiment"]["parent_run"]
    previous = read_json(parent / "passport.json")
    with zipfile.ZipFile(parent / "sources.zip") as archive:
        for name, h in previous["source_hashes"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != h:
                raise ValueError(f"Parent archive corrupted: {name}")
    plan = prepare_cases(config)
    current = execution_identity(config, plan)
    for key in ("python", "platform", "packages", "cuda", "gpu"):
        if current[key] != previous[key]:
            raise ValueError(f"Parent environment differs: {key}")
    write_json(out / "manifest.json", plan)
    checkpoint.passport(out, config, plan)
    saved = torch.load(
        ROOT / config["experiment"]["parent"], map_location="cpu", weights_only=False
    )
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
    checkpoint.save(
        out / state["latest"], policy, optimizer, state, rng, config, current
    )
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
    config, manifest_path, purpose, index, checkpoint_path, mode, enemy, directory = (
        task
    )
    plan = read_json(manifest_path)
    if plan["kind"] != config["kind"] or plan["kind"] not in (KIND, CONTROLLED_KIND) or purpose not in ("development", "confirmation"):
        raise ValueError("Closed or untrusted evaluation")
    if digest({k: v for k, v in plan.items() if k != "hash"}) != plan["hash"]:
        raise ValueError("Evaluation manifest hash changed")
    case = plan[purpose][index]
    if (
        case["split"] != "validation"
        or case["parent_world"] not in plan["base"]["worlds"]["validation"]
    ):
        raise ValueError("Closed evaluation case")
    if digest({k: v for k, v in case.items() if k != "case_hash"}) != case["case_hash"]:
        raise ValueError("Evaluation case changed")
    return play_case(config, case, checkpoint_path, mode, enemy, directory)


def play_case(
    config, case, checkpoint_path, mode, enemy, directory, *,
    temperature=1.0, collect_selection_metrics=False,
):
    """Shared ordinary-match execution; protocol callers supply validated cases."""
    directory = Path(directory)
    configure(17, "cpu")
    if checkpoint_path == "cp":
        agent = CPAgent(search_config(config))
    else:
        agent = RLAgent.from_checkpoint(
            checkpoint_path, "cpu", mode, temperature=temperature,
            collect_selection_metrics=collect_selection_metrics,
        )
    opponent = CPAgent(search_config(config)) if enemy == "cp" else "greedy"
    side = case["learner_side"]
    agents = [None, None]
    agents[side], agents[1 - side] = agent, opponent
    write_json(directory / "case.json", case)
    started = perf_counter()
    try:
        outcome = run_match(
            MatchConfig(
                case["scenario"]["map"],
                case["armies"],
                int(case["max_rounds"]),
                max_calls_per_turn=config["environment"]["max_calls_per_turn"],
            ),
            agents,
            scenario=case["scenario"],
            agent_seeds=case["agent_seeds"],
        )
        outcome.save(directory)
        if outcome.result["status"] != "completed":
            raise RuntimeError(str(outcome.result["error"]))
    except BaseException:
        (directory / "traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise
    finally:
        if isinstance(agent, CPAgent):
            agent.close()
        if isinstance(opponent, CPAgent):
            opponent.close()
    result = {
        "status": "completed",
        "case_id": case["case_id"],
        "side": side,
        "parent_world": case["parent_world"],
        "winner": outcome.result["winner"],
        "reason": outcome.result["reason"],
        "rounds": outcome.result["completed_rounds"],
        "seconds": perf_counter() - started,
        "remaining_hp": outcome.result["remaining_hp"],
        "case_hash": case["case_hash"],
    }
    if collect_selection_metrics:
        write_json(directory / "selection-metrics.json", agent.selection_metrics)
    write_json(directory / "metrics.json", result)
    return result


def match_summary(directory, expected):
    rows = [read_json(p) for p in sorted(Path(directory).glob("*/metrics.json"))]
    if len(rows) != expected or any(r["status"] != "completed" for r in rows):
        raise ValueError("Incomplete evaluation")
    counts = Counter(
        "D" if r["winner"] is None else "W" if r["winner"] == r["side"] else "L"
        for r in rows
    )
    return {
        "matches": len(rows),
        **{k: counts[k] for k in ("W", "D", "L")},
        "win_rate": counts["W"] / len(rows),
    }


def reusable_baseline(config, plan, out):
    parent = ROOT / config["experiment"]["parent_run"]
    old = read_json(parent / "passport.json")
    closure = [
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
    ]
    mismatched = [
        p for p in closure if old["source_hashes"].get(p) != file_hash(ROOT / p)
    ]
    # Old passports do not pin the planner DLL; do not claim reusable identity without that proof.
    if old["source_hashes"].get("build/combat_ai.dll") != file_hash(
        ROOT / "build/combat_ai.dll"
    ):
        mismatched.append("unverified_parent_planner_DLL")
    old_config = read_json(parent / "resolved-config.json")
    if old_config["teacher_search"] != config["teacher_search"]:
        mismatched.append("teacher_search")
    dirs = {}
    try:
        for mode in ("argmax", "sample"):
            dirs[mode] = {}
            for enemy in ("cp", "greedy"):
                directory = parent / f"cycle-02/evaluation/{mode}/{enemy}"
                for case in plan["development"]:
                    d = (
                        directory
                        / f"case-{case['case_id']:03}-side-{case['learner_side']}"
                    )
                    if read_json(d / "case.json") != case:
                        raise ValueError("Baseline case differs")
                    if any(
                        not (d / name).is_file()
                        for name in ("replay.json", "final.json", "result.json")
                    ):
                        raise ValueError("Missing baseline replay")
                dirs[mode][enemy] = {
                    "directory": str(directory),
                    "summary": match_summary(directory, 64),
                }
    except (OSError, ValueError) as exc:
        mismatched.append(str(exc))
    write_json(
        out / "baseline-reuse-check.json",
        {
            "dependency_hashes": {p: file_hash(ROOT / p) for p in closure},
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
            if (
                isinstance(n, ast.AnnAssign)
                and isinstance(n.target, ast.Name)
                and n.target.id == "directory"
            ):
                return i
        for i, n in enumerate(body):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if any(
                        isinstance(node, ast.Name) and node.id == "directory"
                        for node in ast.walk(t)
                    ):
                        return i
            if isinstance(n, ast.AnnAssign) and any(
                isinstance(node, ast.Name) and node.id == "directory"
                for node in ast.walk(n.target)
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
        return ast.dump(
            ast.Module(body=body, type_ignores=[]), include_attributes=False
        )

    eval_task = next(
        (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "evaluation_task"),
        None,
    )
    if eval_task is not None:
        start = find_directory_start(eval_task.body)
        body = eval_task.body[start:] if start is not None else eval_task.body
        return ast.dump(
            ast.Module(body=body, type_ignores=[]), include_attributes=False
        )

    raise ValueError("Neither play_case nor evaluation_task found in evaluator source")



def reusable_controlled_evaluation(config, plan, out, fingerprint, purpose):
    parent = ROOT / config["experiment"]["parent_run"]
    dep_hashes = evaluator_dependency_hashes()
    dependencies = list(dep_hashes.keys())
    mismatched = []

    try:
        recorded = read_json(parent / "passport.json")
        for p in dependencies:
            if recorded.get("source_hashes", {}).get(p) != dep_hashes[p]:
                mismatched.append(p)
    except Exception as exc:  # noqa: BLE001 -- record unreadable passport as mismatch
        mismatched.append(f"unrecognized_parent_passport: {exc}")

    # The new kind adds a manifest guard; actual match execution must be unchanged.
    try:
        archive_path = parent / "sources.zip"
        if not archive_path.exists():
            raise FileNotFoundError(f"Missing archive: {archive_path}")
        with zipfile.ZipFile(archive_path) as archive:
            namelist = archive.namelist()
            if "civ5rl/experiments/ppo_after_imitation.py" not in namelist:
                raise KeyError("civ5rl/experiments/ppo_after_imitation.py not found in archive")
            old_source = archive.read("civ5rl/experiments/ppo_after_imitation.py").decode("utf-8")
        current_source = Path(__file__).read_text(encoding="utf-8")
        if match_execution(old_source) != match_execution(current_source):
            mismatched.append("evaluator_match_execution")
    except Exception as exc:  # noqa: BLE001 -- record unreadable or altered archive format as mismatch
        mismatched.append(f"unrecognized_archive_format: {exc}")

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
            model = torch.load(result["checkpoint"], map_location="cpu", weights_only=False)["model"]
            if policy_hash(model) != fingerprint:
                raise ValueError("Evaluation policy weights changed")
            for enemies in result["runs"].values():
                for record in enemies.values():
                    directory = Path(record["directory"])
                    if match_summary(directory, len(plan[purpose])) != record["summary"]:
                        raise ValueError("Evaluation results changed")
                    for case in plan[purpose]:
                        d = directory / f"case-{case['case_id']:03}-side-{case['learner_side']}"
                        if read_json(d / "case.json") != case or any(
                            not (d / n).is_file() for n in ("replay.json", "final.json", "result.json")
                        ):
                            raise ValueError("Evaluation provenance/replay missing")
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
                    directory = (
                        out / f"evaluation/{purpose}/{fingerprint[:16]}/{mode}/{enemy}"
                    )
                    tasks = []
                    for i, case in enumerate(plan[purpose]):
                        d = (
                            directory
                            / f"case-{case['case_id']:03}-side-{case['learner_side']}"
                        )
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
                    for first in range(0, len(tasks), 4):
                        list(executor.map(evaluation_task, tasks[first : first + 4]))
                        print(
                            f"evaluate {purpose}/{mode}/{enemy}: {min(first + 4, len(tasks))}/{len(tasks)}",
                            flush=True,
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
    checkpoint.save(
        out / state["latest"], policy, optimizer, state, rng, config, identity
    )
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
    return {
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


def train(config, plan, out, device):
    state = read_json(out / "state.json")
    if state["stop_reason"]:
        return state
    protocol = config["kind"]
    has_updates = "updates" in state
    saved_updates = state.get("updates")
    counters = TrainingCounters.from_dict(state, protocol=protocol)
    state.update(counters.to_state_dict(protocol=protocol))
    if not controlled(config) and has_updates:
        state["updates"] = saved_updates
    if not (out / "preflight.json").exists():
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
    restored = checkpoint.restore(
        out / state["latest"], policy, optimizer, rng, config, identity
    )
    # state.json can include a completed evaluation after the model checkpoint was saved.
    assert (
        restored["decisions"] == state["decisions"]
        and restored["optimizer_steps"] == state["optimizer_steps"]
    )
    if not (out / "baseline-policy.json").exists():
        write_json(
            out / "baseline-policy.json", {"hash": policy_hash(policy.state_dict())}
        )
    if state["pending_evaluation"]:
        development(config, plan, out, state)
    with (
        ResourceMonitor(out / "telemetry") as monitor,
        Collector(
            config, TrainingFactory(config, plan["base"]), device, out / "errors"
        ) as collector,
    ):
        # One no-update warmup uses saved public inputs; RNG is restored before collection.
        monitor.mark("initial_warmup")
        prep = perf_counter()
        from civ5rl.experiments.ppo_after_imitation_checks import warmup

        with isolated_rng(rng):
            warmup(policy, device)
        state["preparation_seconds"] += perf_counter() - prep
        records_path = out / "updates.jsonl"
        while not state["stop_reason"]:
            if state["active_seconds"] >= config["experiment"]["active_seconds"]:
                state["stop_reason"] = "active_time_limit"
                break
            if state["decisions"] >= config["experiment"]["decisions"]:
                state["stop_reason"] = "decision_limit"
                break
            tick = perf_counter()
            deadline = (
                tick + config["experiment"]["active_seconds"] - state["active_seconds"]
            )
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
                break
            summary = summarize_rollout(episodes)
            before_decisions = state["decisions"]
            state["decisions"] += summary["decisions"]
            state["next_episode"] = next_episode
            monitor.mark("rollout_save")
            rollout_record = save_rollout(episodes, out / "rollouts", number)
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
                metrics["runtime_compilation_seconds"] = sum(
                    max(0, b - a) for a, b in compiler.intervals
                )
                delta = float(
                    torch.sqrt(
                        sum(
                            (v - before[k]).float().square().sum()
                            for k, v in policy.state_dict().items()
                        )
                    )
                )
                del before
            rollout_hash = rollout_record["sha256"]
            if controlled(config):
                # Buffer cleanup services this update, so charge it before evaluation.
                del episodes
                gc.collect()
            state["active_seconds"] += perf_counter() - tick
            counters = TrainingCounters.from_dict(state, protocol=protocol)
            accepted_steps = int(metrics["optimizer_steps"])
            attempted_steps = int(
                metrics.get("attempted_optimizer_steps", accepted_steps)
            )
            counters = counters.replace(
                rollouts_collected=number,
                rollouts_with_accepted_steps=counters.rollouts_with_accepted_steps
                + (1 if accepted_steps > 0 else 0),
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
            timed = (
                state["active_seconds"] >= config["experiment"]["active_seconds"]
                or metrics["stop_reason"] == "active_time_limit"
            )
            if timed:
                state["stop_reason"] = "active_time_limit"
            elif state["decisions"] >= config["experiment"]["decisions"]:
                state["stop_reason"] = "decision_limit"
            elif controlled(config) and state["zero_accepted_streak"] >= config["experiment"]["max_zero_accepted_rollouts"]:
                state["stop_reason"] = "no_accepted_updates"
            due = any(
                before_decisions < m <= state["decisions"]
                for m in config["experiment"]["milestones"]
            )
            state["pending_evaluation"] = bool(
                (due or first_accepted or state["stop_reason"]) if controlled(config)
                else metrics["optimizer_steps"] and (due or state["stop_reason"])
            )
            row = {
                "update": number,
                **({"total_attempted_steps": state["attempted_steps"],
                    "total_rejected_steps": state["rejected_steps"],
                    "accepted_updates": state["accepted_updates"],
                    "zero_accepted_streak": state["zero_accepted_streak"]} if controlled(config) else {}),
                "trained": bool(metrics["optimizer_steps"]),
                **summary,
                "total_decisions": state["decisions"],
                "total_Adam_steps": state["optimizer_steps"],
                "active_seconds": state["active_seconds"],
                "active_time_overshoot_seconds": max(
                    0, state["active_seconds"] - config["experiment"]["active_seconds"]
                ),
                "decision_overshoot": max(
                    0, state["decisions"] - config["experiment"]["decisions"]
                ),
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
                if state["active_seconds"] >= config["experiment"]["active_seconds"]:
                    state["stop_reason"] = "active_time_limit"
                    state["pending_evaluation"] = True
                write_json(out / "state.json", state)
            with records_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
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
                        "estimated_total_active_seconds": config["experiment"][
                            "decisions"
                        ]
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
        monitor.mark("stopped")
        evaluated_hashes = {e["policy_hash"] for e in state["evaluated"].values()}
        if policy_hash(policy.state_dict()) not in evaluated_hashes:
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
        directory = Path(result["runs"]["argmax"]["cp"]["directory"])
        values = {}
        cases = {}
        for p in directory.glob("*/metrics.json"):
            metric = read_json(p)
            case = read_json(p.parent / "case.json")
            key = (case["parent_world"], case["case_id"], metric["side"])
            if key in values:
                raise ValueError("Duplicate confirmation")
            values[key] = int(metric["winner"] == metric["side"])
            cases[key] = case
        return values, cases

    b, bc = rows(before)
    a, ac = rows(after)
    if bc != ac:
        raise ValueError("Confirmation pairing differs")
    baseline_selected = controlled(config) and before["policy_hash"] == after["policy_hash"]
    ci = None if baseline_selected else paired_world_bootstrap(
        b,
        a,
        samples=config["experiment"]["bootstrap_samples"],
        seed=config["experiment"]["bootstrap_seed"],
        cases=ac,
        cases_per_world=16,
    )
    result = {
        "baseline": before,
        "candidate": after,
        "comparison": ci,
        "baseline_selected": baseline_selected,
        "selection_result": "PPO-кандидат не прошёл development-отбор" if baseline_selected else "new_candidate",
        "confirmed_improvement": False if ci is None else ci["CI95"][0] > 0,
        "practical_10pp": False if ci is None else ci["delta"] >= 0.1,
        "selected_before_confirmation": True,
    }
    write_json(out / "confirmation.json", result)
    return result


def report(config, out):
    state = read_json(out / "state.json")
    rows = (
        [json.loads(l) for l in (out / "updates.jsonl").read_text().splitlines()]
        if (out / "updates.jsonl").exists()
        else []
    )
    final = (
        read_json(out / "confirmation.json")
        if (out / "confirmation.json").exists()
        else None
    )
    lines = [
        "# PPO с контролем принятого изменения политики" if controlled(config) else "# PPO после имитации CP",
        "",
        f"Статус: {read_json(out / 'status.json')['status']}; причина остановки: {state['stop_reason']}.",
        f"Решения: {state['decisions']}; rollouts: {state['updates']}; Adam steps: {state['optimizer_steps']}; активное время: {state['active_seconds']:.2f} с.",
        f"Evaluation: {state['evaluation_seconds']:.2f} с; подготовка: {state['preparation_seconds']:.2f} с; сохранение checkpoints: {state['save_seconds']:.2f} с.",
        "",
        "Основной режим — argmax; sampling и greedy — отдельные контрольные показатели.",
        "",
        "| Update | Решения | Adam | Active, с | Train W/D/L | Win rate | Free/forced | KL free / all | Stop |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows:
        p = r["ppo"]
        lines.append(
            f"| {r['update']} | {r['total_decisions']} | {r['total_Adam_steps']} | {r['active_seconds']:.1f} | {r['W']}/{r['D']}/{r['L']} | {r['win_rate']:.1%} | {r['free_decisions']}/{r['forced_decisions']} | {p.get('sampled_kl_free')} / {p.get('sampled_kl')} | {p['stop_reason']} |"
        )
    lines += [
        "",
        "## Development",
        "",
        "| Update | Режим | Соперник | W/D/L | Win rate |",
        "|---|---|---|---:|---:|",
    ]
    for number, evaluation in state["evaluated"].items():
        for mode, enemies in evaluation["runs"].items():
            for enemy, record in enemies.items():
                s = record["summary"]
                lines.append(
                    f"| {number} | {mode} | {enemy} | {s['W']}/{s['D']}/{s['L']} | {s['win_rate']:.1%} |"
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
    telemetry = out / "telemetry/resources.jsonl"
    if telemetry.exists():
        gpu = analyze_telemetry(telemetry, active_phase="ppo")
        write_json(
            out / "gpu-analysis.json", {k: v for k, v in gpu.items() if k != "active"}
        )
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
        "Дополнительная проверка повторно использует 256 validation-матчей прежнего PPO; это открытый validation, не test." if controlled(config) else "Дополнительная проверка использует новые расстановки знакомых validation-worlds. Test закрыт; 75% — цель проекта, не автоматическое подтверждение обобщения.",
        "Отсутствие побед в rollout явно означает отсутствие положительной терминальной награды. Награда, модель, температура и бюджет автоматически не менялись.",
    ]
    if controlled(config):
        from civ5rl.experiments.ppo_controlled_checks import extend_report
        lines += extend_report(config, out, state, rows, final)
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if rows:
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
    return state


def audit_replays(config, out):
    cache = read_json(out / "evaluations.json")
    selected = []
    for evaluation in cache.values():
        if evaluation["reused_parent"]:
            continue
        for enemies in evaluation["runs"].values():
            for value in enemies.values():
                paths = sorted(Path(value["directory"]).glob("*/metrics.json"))
                for outcome in ("W", "D", "L"):
                    for p in paths:
                        r = read_json(p)
                        o = (
                            "D"
                            if r["winner"] is None
                            else "W"
                            if r["winner"] == r["side"]
                            else "L"
                        )
                        if o == outcome:
                            selected.append(p.parent)
                            break
    selected = list(dict.fromkeys(selected))
    if controlled(config):
        # One representative of each observed outcome is sufficient here.
        representatives = {}
        for path in selected:
            result = read_json(path / "metrics.json")
            outcome = "D" if result["winner"] is None else "W" if result["winner"] == result["side"] else "L"
            representatives.setdefault(outcome, path)
        selected = list(representatives.values())
    checks = [verify_replay(p) for p in selected]
    write_json(out / "replay-verification.json", checks)
    if controlled(config):
        descriptions = []
        for directory, check in zip(selected, checks):
            replay = read_json(directory / "replay.json")
            descriptions.append({
                "directory": str(directory), "metrics": read_json(directory / "metrics.json"),
                "commands": dict(Counter(s["action"]["kind"] for s in replay["steps"])),
                "verification": check,
            })
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


def run(phase, config, out, device):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    with RunLock(out):
        try:
            if phase == "prepare":
                return prepare(config, out)
            plan = check_identity(
                config, out, for_execution=phase in ("train", "evaluate", "run")
            )
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
        except BaseException:
            (out / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
            write_json(
                out / "status.json", {"status": "technical_error", "phase": phase}
            )
            if (out / "state.json").exists():
                try:
                    report(config, out)
                except Exception:  # noqa: BLE001, S110 -- best-effort report before re-raising
                    pass
            raise
