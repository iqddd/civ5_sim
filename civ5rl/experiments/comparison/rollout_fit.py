"""One bounded rollout-comparison cycle, with separate collection/train budgets."""

from __future__ import annotations

import os

# This must run before importing the CUDA-dependent modules below.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import shutil
from collections import Counter
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any, TypedDict

from civ5rl.engine.comparison.continuation import (
    continuation_seconds,
    reuse_parent_development,
    validate_continuation,
)
from civ5rl.engine.comparison.storage import archive_tree, restore_tree
from civ5rl.engine.comparison.training import epoch_fit_options, fit, training_runtime
from civ5rl.engine.evaluation.executor import (
    EvaluationExecutor,
    EvaluationRuntime,
    evaluation_options,
    evaluation_precision,
)
from civ5rl.engine.evaluation.storage import archive_evaluations, readable_evaluations
from civ5rl.engine.runtime import is_int_at_least
from civ5rl.engine.runtime.budget import PhaseBudget
from civ5rl.engine.runtime.control import RunLock
from civ5rl.experiments._shared.lifecycle import record_experiment_errors
from civ5rl.experiments._shared.provenance_checks import verify_file_hashes
from civ5rl.experiments.comparison.data import (
    ForkLabel,
    prepare_comparison_dataset,
    validate_fit_options,
    write_labels_and_collect_pairs,
)
from civ5rl.experiments.comparison.executor import (
    ProposalExecutor,
    preference,
    select_forks,
)
from civ5rl.experiments.comparison.match import match_task
from civ5rl.experiments.evaluation.paired import (
    CANDIDATE_CHECKPOINT,
    PARENT_CHECKPOINT,
    evaluate_paired_dev512,
)
from civ5rl.experiments.ppo.target_scale import evaluate as ordinary_evaluate
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
from civ5rl.reporting.comparison_report import report
from civ5rl.reporting.evaluation_audit import clustered_paired_bootstrap, load_eval_rows, outcome

KIND = "rl01.rollout_comparison"


def load_config(path):
    cfg = read_json(path)
    if cfg["kind"] != KIND or cfg["seed"] != 17:
        raise ValueError("Unknown comparison profile")
    for key in ("collection_seconds", "training_seconds", "start_episode"):
        if not is_int_at_least(cfg.get(key)):
            raise ValueError(f"{key} must be a positive integer")
    if not cfg.get("collection_input") and not is_int_at_least(cfg.get("max_forks")):
        raise ValueError("max_forks must be a positive integer")
    if not is_int_at_least(cfg.get("workers", 8)):
        raise ValueError("workers must be a positive integer")
    if len(cfg["parents"]) != 2 or len(cfg["parent_sha256"]) != 2:
        raise ValueError("Comparison requires two parents and their hashes")
    for parent, checksum in zip(cfg["parents"], cfg["parent_sha256"], strict=True):
        if file_sha256(ROOT / parent) != checksum:
            raise ValueError("Parent SHA mismatch")
    training = cfg.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("training must be an object")
    mode = training.get("mode", "epoch_kl")
    if mode == "epoch_kl":
        epoch_fit_options(cfg)
    else:
        if mode not in ("full_corpus_step_kl", "frozen_heads_step_kl"):
            raise ValueError("Unknown fitting mode")
        continuation = bool(cfg.get("fit_continuation"))
        validate_fit_options(training)
        if type(training.get("cycle_anchors", False)) is not bool:
            raise ValueError("training.cycle_anchors must be a boolean")
        if not cfg.get("collection_input"):
            raise ValueError("Full-corpus fitting requires a frozen collection")
        if cfg.get("evaluation_protocol") != "paired_development_only":
            raise ValueError("Full-corpus fitting requires paired Development only")
        if continuation:
            validate_continuation(cfg)
    return cfg


def prepare(cfg, out):
    if (out / "run.json").exists():
        return
    if cfg.get("collection_input"):
        return prepare_collection_input(cfg, out)
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / cfg["source_run"]
    saved = read_json(source / "run.json")
    source_state = read_json(source / "state.json")
    if cfg["start_episode"] < source_state["next_episode"]:
        raise ValueError("Episode IDs overlap")
    plan = read_json(source / "manifest.json")
    identity = checkpoint.passport(out, cfg, plan)
    write_json(
        out / "run.json",
        {
            "kind": KIND,
            "config": cfg,
            "identity": identity,
            "training_runtime": training_runtime(cfg),
            "source_manifest_sha256": file_sha256(source / "manifest.json"),
            "parent_sha256": cfg["parent_sha256"],
        },
    )
    write_json(out / "match-config.json", saved["config"])
    write_json(out / "manifest.json", plan)
    write_json(
        out / "collection.json",
        {
            "seconds": 0.0,
            "next_episode": cfg["start_episode"],
            "forks": [],
            "sources": [],
            "complete": False,
            "samples": [],
        },
    )
    write_json(out / "status.json", {"status": "prepared", "phase": "collect"})


def prepare_collection_input(cfg, out):
    """Import immutable collection evidence into a separately passported fit.

    The collector's source snapshot is verified as frozen evidence, never against
    the current training executable. All consumed files are pinned by the fit
    config. Only the fit's copy of public histories may be unpacked or cached.
    Publishing run.json last makes interrupted preparation safely repeatable.
    """
    source = ROOT / cfg["collection_input"]["run"]
    expected = cfg["collection_input"]["sha256"]
    required = {
        "dataset.json",
        "anchors.json",
        "collection.json",
        "match-config.json",
        "manifest.json",
        "passport.json",
        "resolved-config.json",
        "corpus-manifest.json",
        "sources.storage/manifest.json",
        "sources.storage/json.7z",
    }
    if not required.issubset(expected):
        raise ValueError("Incomplete frozen collection identity")
    with RunLock(source):
        verify_file_hashes(source, expected, label="Collection input changed")
        checkpoint.verify_frozen_passport(source, sources=False)
        collection = read_json(source / "collection.json")
        original = read_json(source / "resolved-config.json")
        if not collection.get("finalized"):
            raise ValueError("Collection is not finalized")
        if original["parent_sha256"] != cfg["parent_sha256"]:
            raise ValueError("Collection parents differ")
        dataset = read_json(source / "dataset.json")
        if dataset["completed_forks"] != sum(fork["status"] == "completed" for fork in collection["forks"]):
            raise ValueError("Collection dataset count differs")
        out.mkdir(parents=True, exist_ok=True)
        for name in (
            "dataset.json",
            "anchors.json",
            "collection.json",
            "match-config.json",
            "manifest.json",
            "sources.storage/manifest.json",
            "sources.storage/json.7z",
        ):
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target)
        write_json(out / "collection-lineage.json", cfg["collection_input"])
    plan = read_json(out / "manifest.json")
    identity = checkpoint.passport(out, cfg, plan)
    write_json(out / "status.json", {"status": "prepared", "phase": "train"})
    write_json(
        out / "run.json",
        {
            "kind": KIND,
            "config": cfg,
            "identity": identity,
            "training_runtime": training_runtime(cfg),
            "collection_input": cfg["collection_input"],
            "source_manifest_sha256": file_sha256(source / "manifest.json"),
            "parent_sha256": cfg["parent_sha256"],
        },
    )


def collect(cfg, out, *, limit=None, budget=None):
    if budget is not None:
        return _collect(cfg, out, limit=limit, budget=budget)
    with PhaseBudget(out, "collection", cfg["collection_seconds"]) as collection_budget:
        return _collect(cfg, out, limit=limit, budget=collection_budget)


class _CollectionFork(TypedDict):
    id: int
    episode: int
    index: int
    command: int
    actions: list[dict[str, Any]]
    status: str


class _CollectionSample(TypedDict):
    work: int
    seconds: float


class _CollectionWaveState(TypedDict, total=False):
    seconds: float
    next_episode: int
    forks: list[_CollectionFork]
    sources: list[int]
    complete: bool
    samples: list[_CollectionSample]
    pending_wave: list[int]
    finalized: bool
    stage: str
    overshoot_seconds: float


class _ContinuationBranch(TypedDict):
    source: str
    command: int
    seed: int
    action: dict[str, Any]


# Keep the worker wire format as a plain tuple, including for spawned processes.
_MatchTask = tuple[dict[str, Any], dict[str, Any], str, str, _ContinuationBranch | None]


def _stage_source_wave(
    out: Path,
    state: _CollectionWaveState,
    factory: TargetFactory,
    save: Callable[[], None],
) -> list[int]:
    # Persist the wave before launch. A failed/resumed wave keeps its exact cases.
    if not state.get("pending_wave"):
        state["pending_wave"] = []
        for _ in range(8):
            episode = state["next_episode"]
            state["next_episode"] += 1
            case = factory.make(episode)
            directory = out / "sources" / str(episode)
            write_json(directory / "case.json", case)
            state["pending_wave"].append(episode)
        save()
    return state["pending_wave"]


def _source_wave_tasks(
    cfg: dict[str, Any], config: dict[str, Any], out: Path, wave: list[int]
) -> list[_MatchTask]:
    tasks: list[_MatchTask] = []
    for episode in wave:
        directory = out / "sources" / str(episode)
        if not (directory / "completed.json").exists():
            tasks.append(
                (
                    config,
                    read_json(directory / "case.json"),
                    str(directory),
                    str(ROOT / cfg["parents"][0]),
                    None,
                )
            )
    return tasks


def _commit_source_wave(
    cfg: dict[str, Any],
    out: Path,
    state: _CollectionWaveState,
    wave: list[int],
    maximum: int,
    save: Callable[[], None],
) -> None:
    """Admit each source history once and preserve global fork numbering."""
    for episode in wave:
        if episode in state["sources"]:
            continue
        state["sources"].append(episode)
        history = read_json(out / "sources" / str(episode) / "history.json")
        for index in select_forks(
            history,
            episode,
            cfg["namespace"],
            maximum=min(4, maximum - len(state["forks"])),
        ):
            if len(state["forks"]) >= maximum:
                break
            row = history[index]
            state["forks"].append(
                {
                    "id": len(state["forks"]),
                    "episode": episode,
                    "index": index,
                    "command": row["command"],
                    "actions": [
                        row["actions"][row["chosen"]],
                        row["actions"][row["alternative"]],
                    ],
                    "status": "not_started",
                }
            )
    state["pending_wave"] = []
    save()


def _collect(cfg: dict[str, Any], out: Path, *, limit: int | None = None, budget: PhaseBudget) -> None:
    state: _CollectionWaveState = read_json(out / "collection.json")
    if state.get("finalized") and (out / "dataset.json").exists():
        return
    if state["complete"] or state.get("stage") == "finalizing":
        return finalize_collection(cfg, out, state, budget, limit)
    config = read_json(out / "match-config.json")
    config = deepcopy(config)
    config["experiment"].update(
        start_episode=cfg["start_episode"],
        training_namespace=cfg["namespace"] + "_training",
    )
    factory = TargetFactory(config, read_json(out / "manifest.json")["base"])
    runtime = EvaluationRuntime("cuda", cfg.get("workers", 8), 32, 0.001)
    maximum = limit or cfg["max_forks"]

    def save() -> None:
        state["seconds"] = budget.elapsed()
        write_json(out / "collection.json", state)

    def expired() -> bool:
        return budget.expired()

    with evaluation_precision(), ExitStack() as sessions:
        proposal: ProposalExecutor | None = None
        continuation: EvaluationExecutor | None = None

        def get_proposal() -> ProposalExecutor:
            nonlocal proposal
            if proposal is None:
                proposal = sessions.enter_context(
                    ProposalExecutor([ROOT / p for p in cfg["parents"]], runtime)
                )
            return proposal

        def get_continuation() -> EvaluationExecutor:
            nonlocal continuation
            if continuation is None:
                continuation = sessions.enter_context(
                    EvaluationExecutor(str(ROOT / cfg["parents"][0]), runtime)
                )
            return continuation

        if any(f["status"] == "running" for f in state["forks"]):
            process_forks(cfg, out, state, config, get_continuation, expired, save)
        while (len(state["forks"]) < maximum and not expired()) or state.get("pending_wave"):
            wave = _stage_source_wave(out, state, factory, save)
            tasks = _source_wave_tasks(cfg, config, out, wave)
            if tasks:
                executor = get_proposal()
                list(executor.run(match_task, tasks))
            _commit_source_wave(cfg, out, state, wave, maximum, save)
            if expired():
                break
            process_forks(cfg, out, state, config, get_continuation, expired, save)
        # Selected forks are never replaced when a label is ambiguous.
        if not expired():
            process_forks(cfg, out, state, config, get_continuation, expired, save)
    for fork in state["forks"]:
        if fork["status"] == "not_started":
            fork["status"] = "not_started_budget"
    state["stage"] = "finalizing"
    save()
    finalize_collection(cfg, out, state, budget, limit)


def finalize_collection(
    cfg: dict[str, Any],
    out: Path,
    state: _CollectionWaveState,
    budget: PhaseBudget,
    limit: int | None = None,
) -> None:
    if not (out / "dataset.json").exists():
        restore_tree(out, "sources")
        restore_tree(out, "continuations")
        write_json(out / "dataset.json", build_dataset(cfg, out, state))
    if limit is None:
        archive_tree(out, "continuations")
    state.update(
        {
            "complete": True,
            "finalized": True,
            "stage": "complete",
            "seconds": budget.elapsed(),
            "overshoot_seconds": max(0, budget.elapsed() - cfg["collection_seconds"]),
        }
    )
    write_json(out / "collection.json", state)


def _continuation_tasks(
    cfg: dict[str, Any], out: Path, config: dict[str, Any], fork: _CollectionFork
) -> list[_MatchTask]:
    case = read_json(out / "sources" / str(fork["episode"]) / "case.json")
    tasks: list[_MatchTask] = []
    for j in range(8):
        seed = (
            seed_for(
                cfg["seed"],
                fork["episode"],
                fork["id"],
                j,
                namespace=cfg["namespace"] + "_futures",
            )
            or 1
        )
        for alternative in range(2):
            folder = out / "continuations" / str(fork["id"]) / f"{j}-{alternative}"
            if (folder / "completed.json").exists():
                continue
            branch: _ContinuationBranch = {
                "source": str(out / "sources" / str(fork["episode"])),
                "command": fork["command"],
                "seed": seed,
                "action": fork["actions"][alternative],
            }
            write_json(folder / "branch.json", branch)
            tasks.append((config, case, str(folder), str(ROOT / cfg["parents"][0]), branch))
    return tasks


def process_forks(
    cfg: dict[str, Any],
    out: Path,
    state: _CollectionWaveState,
    config: dict[str, Any],
    get_executor: Callable[[], EvaluationExecutor],
    expired: Callable[[], bool],
    save: Callable[[], None],
) -> None:
    for fork in state["forks"]:
        if fork["status"] == "completed":
            continue
        if fork["status"] not in ("running",) and expired():
            break
        start = perf_counter()
        fork["status"] = "running"
        save()
        tasks = _continuation_tasks(cfg, out, config, fork)
        if tasks:
            executor = get_executor()
            list(executor.run(match_task, tasks))
        fork["status"] = "completed"
        state["samples"].append({"work": 1, "seconds": perf_counter() - start})
        save()
        print(
            f"Completed fork {fork['id'] + 1}/{len(state['forks'])}; collection={state['seconds']:.1f}s",
            flush=True,
        )


def build_dataset(cfg: dict[str, Any], out: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze anchors in collection order before reading continuation winners."""
    inputs = prepare_comparison_dataset(
        out,
        state,
        seed=17,
        namespace=cfg["namespace"],
        sort_sources=False,
        load_learner_sides=True,
    )
    reasons: Counter[str] = Counter()

    def label_fork(fork: dict[str, Any]) -> ForkLabel | None:
        if fork["status"] != "completed":
            return None
        winners = [
            [
                read_json(out / "continuations" / str(fork["id"]) / f"{j}-{a}" / "completed.json")["winner"]
                for j in range(8)
            ]
            for a in range(2)
        ]
        label = preference(winners, inputs.learner_sides[fork["episode"]])
        reasons[label["reason"]] += 1
        return f"{fork['id']}.json", {"fork": fork, **label}

    pairs = write_labels_and_collect_pairs(out, state["forks"], inputs.fork_actions, label_fork)
    return {
        "pairs": pairs,
        "anchors": inputs.anchors,
        "source_hashes": inputs.source_hashes,
        "selected_forks": len(state["forks"]),
        "completed_forks": sum(f["status"] == "completed" for f in state["forks"]),
        "accepted_pairs": len(pairs),
        "rejections": dict(reasons),
    }


def evaluate(cfg, out):
    training = read_json(out / "training.json")
    if training.get("candidate_sha256") and file_sha256(out / "candidate.pt") != training["candidate_sha256"]:
        raise ValueError("Candidate checkpoint changed")
    if (out / "result.json").exists() and read_json(out / "result.json").get("status") == "complete":
        return
    if not training.get("changed"):
        write_json(
            out / "result.json",
            {
                "status": training.get("status"),
                "training_steps": training.get("training_steps", 0),
                "selected": "parent",
                "evaluations_skipped": True,
                **dataset_counts(out),
            },
        )
        return
    config = read_json(out / "match-config.json")
    plan = read_json(out / "manifest.json")
    if cfg.get("evaluation_protocol") == "paired_development_only":
        return evaluate_paired_development(cfg, out, config, plan, ordinary_evaluate)
    with (
        readable_evaluations(out),
        evaluation_options(
            device="cuda",
            workers=cfg.get("workers", 8),
            max_batch=32,
            wait_seconds=0.001,
        ),
    ):
        candidate = ordinary_evaluate(config, plan, out, "candidate.pt", "development_512")
        before, after = _historical_development_rows(cfg, out, plan, candidate)
        parent_wins = sum(outcome(metric) == "W" for metric, _, _ in before.values())
        candidate_wins = sum(outcome(metric) == "W" for metric, _, _ in after.values())
        improved = candidate_wins > parent_wins
        result = {
            "status": "complete",
            "development": candidate,
            "comparison": clustered_paired_bootstrap(before, after),
            "selected": "candidate" if improved else "parent",
            "historical_gpu_comparison": True,
            **dataset_counts(out),
        }
        result["additional"] = (
            _evaluate_confirmation(cfg, config, plan, out)
            if improved
            else {"status": "not_run_no_development_gain"}
        )
        write_json(out / "result.json", result)


def _argmax_rows(run, record, plan):
    return load_eval_rows(run / record["runs"]["argmax"]["cp"]["directory"], plan=plan)


def _historical_development_rows(cfg, out, plan, candidate):
    source = ROOT / cfg["source_run"]
    with RunLock(source), readable_evaluations(source):
        baseline = next(
            record
            for record in read_json(source / "evaluations.json").values()
            if record["checkpoint"] == "checkpoints/initial.pt" and record["purpose"] == "development_512"
        )
        parent = _argmax_rows(source, baseline, plan["development_512"])
    return parent, _argmax_rows(out, candidate, plan["development_512"])


def _evaluate_confirmation(cfg, config, plan, out):
    parent_path = out / "checkpoints/parent.pt"
    parent_path.parent.mkdir(exist_ok=True)
    if not parent_path.exists():
        shutil.copy2(ROOT / cfg["parents"][0], parent_path)
    parent = ordinary_evaluate(config, plan, out, "checkpoints/parent.pt", "confirmation")
    candidate = ordinary_evaluate(config, plan, out, "candidate.pt", "confirmation")
    return clustered_paired_bootstrap(
        _argmax_rows(out, parent, plan["confirmation"]),
        _argmax_rows(out, candidate, plan["confirmation"]),
    )


def evaluate_paired_development(cfg, out, config, plan, evaluator):
    """Compare the last verified candidate and parent through one local runtime."""
    reuse_parent = bool(cfg.get("fit_continuation"))
    paired = evaluate_paired_dev512(
        cfg,
        out,
        config,
        plan,
        evaluator,
        reuse_parent_fn=reuse_parent_development if reuse_parent else None,
    )
    write_json(
        out / "result.json",
        {
            "status": paired["status"],
            "development": paired["development"],
            "parent_development": paired["parent_development"],
            "comparison": paired["comparison"],
            "candidate_selection": paired["candidate_selection"],
            "candidate_checkpoint": CANDIDATE_CHECKPOINT,
            "parent_checkpoint": PARENT_CHECKPOINT,
            "parent_evaluation_reused": reuse_parent,
            "additional": paired["additional"],
            **dataset_counts(out),
        },
    )


def dataset_counts(out):
    data = read_json(out / "dataset.json")
    return {
        **{k: data[k] for k in ("selected_forks", "completed_forks", "accepted_pairs")},
        "rejections": data.get("rejections", data.get("rejection_reasons", {})),
    }


def _comparison_work_remaining(cfg, out, phase):
    collection = read_json(out / "collection.json") if (out / "collection.json").exists() else {}
    training = read_json(out / "training.json") if (out / "training.json").exists() else {}
    collect_needed = not cfg.get("collection_input") and (
        not collection.get("finalized") or not (out / "dataset.json").exists()
    )
    if cfg.get("collection_input") and phase == "collect":
        raise ValueError("Frozen collection inputs cannot collect again")
    train_needed = training.get("status") not in ("complete", "no_accepted_pairs")
    return collect_needed, train_needed


def _start_training_budget(cfg, out, timing):
    return timing.enter_context(
        PhaseBudget(
            out,
            "training",
            cfg["training_seconds"],
            initial_seconds=continuation_seconds(cfg),
        )
    )


def _start_comparison_budget(cfg, out, phase, timing, *, collect_needed, train_needed):
    if phase in ("prepare", "collect", "resume") and collect_needed:
        return timing.enter_context(PhaseBudget(out, "collection", cfg["collection_seconds"]))
    prepares_full_corpus = phase == "prepare" and cfg.get("training", {}).get("mode") in (
        "full_corpus_step_kl",
        "frozen_heads_step_kl",
    )
    if (phase in ("train", "resume") or prepares_full_corpus) and train_needed:
        return _start_training_budget(cfg, out, timing)
    return None


def run(phase, cfg, out, *, preparation_seconds=0.0):
    out = Path(out).resolve()
    with RunLock(out), ExitStack() as timing:
        collect_needed, train_needed = _comparison_work_remaining(cfg, out, phase)
        budget = _start_comparison_budget(
            cfg,
            out,
            phase,
            timing,
            collect_needed=collect_needed,
            train_needed=train_needed,
        )
        if budget is not None:
            budget.base += preparation_seconds
        configure(cfg["seed"], "cuda")
        prepare(cfg, out)
        checkpoint.verify_execution_passport(out, cfg, read_json(out / "manifest.json"))
        write_json(out / "status.json", {"status": "running", "phase": phase})
        with record_experiment_errors(out, phase=phase, error_message_only=True):
            if phase in ("collect", "resume") and collect_needed:
                collect(cfg, out, budget=budget)
                timing.close()
                budget = None
            if phase in ("train", "resume"):
                if not (out / "dataset.json").exists():
                    raise ValueError("Collection incomplete")
                if budget is None and train_needed:
                    budget = _start_training_budget(cfg, out, timing)
                fit(
                    cfg,
                    out,
                    read_json(out / "dataset.json"),
                    read_json(out / "passport.json"),
                    budget=budget,
                )
                timing.close()
            if phase in ("evaluate", "resume"):
                evaluate(cfg, out)
            write_json(
                out / "status.json",
                {
                    "status": "complete" if phase in ("evaluate", "resume") else "prepared",
                    "phase": phase,
                },
            )
    report(out)
    if phase in ("evaluate", "resume") and (out / "evaluation").exists():
        archive_evaluations(out)
