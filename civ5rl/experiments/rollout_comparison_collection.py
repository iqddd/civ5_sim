"""Bounded rollout-comparison collection: shared queue and two-stage continuations.

A separate run kind from ``rl01.rollout_comparison``. It produces frozen
sources, labels and dataset.json for a later, separately passported training
run. The scientific config fixes parents, namespace, first episode ID, case
generator, fork selection, seed scheme and the preference filter. Worker count,
source share, stock and buffer are execution settings recorded per session in
collection-runtime.json; under a time limit they affect corpus volume.

Episode IDs are assigned when a source match is queued. A fork is keyed by
(episode_id, decision_index); a continuation additionally by (repetition,
alternative). Continuation seeds are ``seed_for(seed, episode_id,
decision_index, repetition, namespace + '_futures')``; both alternatives of a
repetition share it. Repetitions 0-3 form the first group and 4-7 the second.
"""

from __future__ import annotations

import os

# This must run before importing the CUDA-dependent modules below.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, NotRequired, Protocol, TypedDict

from civ5rl.engine.evaluation.executor import EvaluationRuntime, evaluation_precision
from civ5rl.engine.runtime.budget import PhaseBudget
from civ5rl.engine.runtime.control import RunLock
from civ5rl.experiments._shared.comparison_dataset import (
    ForkLabel,
    prepare_comparison_dataset,
    write_labels_and_collect_pairs,
)
from civ5rl.experiments._shared.lifecycle import record_experiment_errors
from civ5rl.experiments.comparison_collection import select_forks
from civ5rl.experiments.comparison_match import match_task
from civ5rl.experiments.comparison_queue import (
    CollectionExecutor,
    Scheduler,
    two_stage_label,
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
from civ5rl.reporting.comparison_report import report
from civ5rl.reporting.comparison_storage import archive_tree

KIND = "rl01.rollout_comparison_collection"

MatchConfig = dict[str, Any]
MatchCase = dict[str, Any]
ActionRecord = dict[str, Any]
Winner = int | None


class Branch(TypedDict):
    source: str
    command: int
    seed: int
    action: ActionRecord


MatchTask = tuple[MatchConfig, MatchCase, str, str, Branch | None]


class ForkRecord(TypedDict):
    id: int
    episode: int
    index: int
    command: int
    side: int
    actions: list[ActionRecord]
    status: str
    winners: NotRequired[dict[str, Winner]]
    label: NotRequired[str]
    reason: NotRequired[str | None]
    deltas: NotRequired[list[int | None]]
    accepted: NotRequired[bool]
    better: NotRequired[int | None]
    weight: NotRequired[float]
    executed_repetitions: NotRequired[int]
    wdl: NotRequired[list[dict[str, int]]]


class CollectionSample(TypedDict):
    work: int
    seconds: float


class CollectionState(TypedDict):
    seconds: float
    next_episode: int
    started_sources: list[int]
    sources: list[int]
    forks: list[ForkRecord]
    samples: list[CollectionSample]
    complete: bool
    stage: NotRequired[str]
    finalized: NotRequired[bool]
    output_bytes: NotRequired[int]
    factory_seconds: NotRequired[float]
    scheduler: NotRequired[dict[str, Any]]
    overshoot_seconds: NotRequired[float]


class _ElapsedBudget(Protocol):
    def elapsed(self) -> float: ...


def load_config(path: str | Path) -> dict[str, Any]:
    cfg: dict[str, Any] = read_json(path)
    if cfg.get("kind") != KIND or cfg.get("seed") != 17:
        raise ValueError("Unknown comparison collection profile")
    for key in ("max_forks", "collection_seconds", "start_episode"):
        if type(cfg.get(key)) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    for parent, expected_hash in zip(cfg["parents"], cfg["parent_sha256"], strict=True):
        if file_sha256(ROOT / parent) != expected_hash:
            raise ValueError("Parent SHA mismatch")
    return cfg


@dataclass(frozen=True)
class CollectionRuntime:
    """Execution settings; not part of the scientific config."""

    workers: int = 8
    source_share: float = 0.25
    stock_factor: float = 2.0
    max_active_forks: int = 0  # 0: equal to workers
    buffer_limit: int = 32
    max_batch: int = 0  # 0: equal to workers
    wait_seconds: float = 0.001
    pack_mib_per_second: float = 20.0
    prefix_states: bool = False

    def resolved(self) -> CollectionRuntime:
        if (
            self.workers < 1
            or not 0 <= self.source_share < 1
            or self.stock_factor <= 0
            or self.pack_mib_per_second <= 0
            or type(self.prefix_states) is not bool
        ):
            raise ValueError("Invalid collection runtime")
        return CollectionRuntime(
            self.workers,
            self.source_share,
            self.stock_factor,
            self.max_active_forks or self.workers,
            max(1, self.buffer_limit),
            self.max_batch or self.workers,
            self.wait_seconds,
            self.pack_mib_per_second,
            self.prefix_states,
        )


def prepare(cfg: dict[str, Any], out: Path) -> None:
    if (out / "run.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / cfg["source_run"]
    saved = read_json(source / "run.json")
    if cfg["start_episode"] < read_json(source / "state.json")["next_episode"]:
        raise ValueError("Episode IDs overlap the source run")
    plan = read_json(source / "manifest.json")
    identity = checkpoint.passport(out, cfg, plan)
    write_json(
        out / "run.json",
        {
            "kind": KIND,
            "config": cfg,
            "identity": identity,
            "source_manifest_sha256": file_sha256(source / "manifest.json"),
            "parent_sha256": cfg["parent_sha256"],
        },
    )
    match_config = deepcopy(saved["config"])
    match_config["experiment"].update(
        start_episode=cfg["start_episode"],
        training_namespace=cfg["namespace"] + "_training",
    )
    write_json(out / "match-config.json", match_config)
    write_json(out / "manifest.json", plan)
    write_json(
        out / "collection.json",
        {
            "seconds": 0.0,
            "next_episode": cfg["start_episode"],
            "started_sources": [],
            "sources": [],
            "forks": [],
            "samples": [],
            "complete": False,
        },
    )
    write_json(out / "status.json", {"status": "prepared", "phase": "collect"})


def fork_id(episode: int, index: int) -> str:
    return f"{episode}-{index}"


def directory_bytes(path: Path) -> int:
    total = 0
    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file():
                total += entry.stat().st_size
    return total


class CollectionProtocol:
    """Files, persisted state and fixed scientific rules for the Scheduler."""

    def __init__(
        self,
        cfg: dict[str, Any],
        out: Path,
        state: CollectionState,
        runtime: CollectionRuntime,
        *,
        save_interval: float = 2.0,
    ) -> None:
        self.cfg, self.out, self.state, self.runtime = cfg, out, state, runtime
        self.config: MatchConfig = read_json(out / "match-config.json")
        self.factory = TargetFactory(self.config, read_json(out / "manifest.json")["base"])
        self.parent = str(ROOT / cfg["parents"][0])
        self.records = {fork_id(record["episode"], record["index"]): record for record in state["forks"]}
        self.cases: dict[int, MatchCase] = {}
        self.save_interval = save_interval
        self.last_save = 0.0
        self.last_sample = (self.completed(), state["seconds"])
        self.bytes = state.get("output_bytes", 0)
        finished_sources = set(state["sources"])
        self.pending_sources = [
            episode for episode in state["started_sources"] if episode not in finished_sources
        ]
        self.budget: _ElapsedBudget | None = None
        self.factory_seconds = state.get("factory_seconds", 0.0)

    def completed(self) -> int:
        return sum(record["status"] == "completed" for record in self.state["forks"])

    def case(self, episode: int) -> MatchCase:
        if episode not in self.cases:
            self.cases[episode] = read_json(self.out / "sources" / str(episode) / "case.json")
        return self.cases[episode]

    def has_sources(self, running_sources: int) -> bool:
        if self.pending_sources:
            return True
        return len(self.state["forks"]) + 4 * running_sources < self.cfg["max_forks"]

    def next_source(self) -> tuple[int, MatchTask]:
        if self.pending_sources:
            episode = self.pending_sources.pop(0)
        else:
            started = perf_counter()
            episode = self.state["next_episode"]
            self.state["next_episode"] += 1
            directory = self.out / "sources" / str(episode)
            write_json(directory / "case.json", self.factory.make(episode))
            self.state["started_sources"].append(episode)
            self.factory_seconds += perf_counter() - started
            self.save(force=True)
        directory = self.out / "sources" / str(episode)
        return episode, (
            self.config,
            self.case(episode),
            str(directory),
            self.parent,
            None,
        )

    def source_done(self, episode: int, seconds: float) -> list[ForkRecord]:
        directory = self.out / "sources" / str(episode)
        self.bytes += directory_bytes(directory)
        self.state["sources"].append(episode)
        history = read_json(directory / "history.json")
        side = self.case(episode)["learner_side"]
        remaining = self.cfg["max_forks"] - len(self.state["forks"])
        selected: list[ForkRecord] = []
        for decision_index in select_forks(
            history, episode, self.cfg["namespace"], maximum=max(0, min(4, remaining))
        ):
            row = history[decision_index]
            record: ForkRecord = {
                "id": len(self.state["forks"]),
                "episode": episode,
                "index": decision_index,
                "command": row["command"],
                "side": side,
                "actions": [
                    row["actions"][row["chosen"]],
                    row["actions"][row["alternative"]],
                ],
                "status": "pending",
                "winners": {},
            }
            self.state["forks"].append(record)
            self.records[fork_id(episode, decision_index)] = record
            selected.append(record)
        self.save(force=True)
        return selected

    def seed(self, record: ForkRecord, repetition: int) -> int:
        return (
            seed_for(
                self.cfg["seed"],
                record["episode"],
                record["index"],
                repetition,
                namespace=self.cfg["namespace"] + "_futures",
            )
            or 1
        )

    def folder(self, record: ForkRecord, repetition: int, alternative: int) -> Path:
        return (
            self.out
            / "continuations"
            / fork_id(record["episode"], record["index"])
            / f"{repetition}-{alternative}"
        )

    def continuation_task(self, record: ForkRecord, repetition: int, alternative: int) -> MatchTask:
        folder = self.folder(record, repetition, alternative)
        branch: Branch = {
            "source": str(self.out / "sources" / str(record["episode"])),
            "command": record["command"],
            "seed": self.seed(record, repetition),
            "action": record["actions"][alternative],
        }
        write_json(folder / "branch.json", branch)
        return (
            self.config,
            self.case(record["episode"]),
            str(folder),
            self.parent,
            branch,
        )

    def continuation_done(
        self, record: ForkRecord, repetition: int, alternative: int, seconds: float
    ) -> Winner:
        folder = self.folder(record, repetition, alternative)
        self.bytes += directory_bytes(folder)
        return read_json(folder / "completed.json")["winner"]

    def decide(self, record: ForkRecord, winners: dict[tuple[int, int], Winner]) -> None:
        record["status"] = "completed"
        record.update(two_stage_label(winners, record["side"]))

    def reserve_seconds(self) -> float:
        # Packing of sources and continuations plus a 25% margin.
        return 1.25 * self.bytes / (self.runtime.pack_mib_per_second * 2**20)

    def save(self, *, force: bool = False, stats: dict[str, Any] | None = None) -> None:
        now = perf_counter()
        if not force and now - self.last_save < self.save_interval:
            return
        self.last_save = now
        if self.budget is not None:
            self.state["seconds"] = self.budget.elapsed()
        completed = self.completed()
        work, seconds = (
            completed - self.last_sample[0],
            self.state["seconds"] - self.last_sample[1],
        )
        if work > 0 and seconds > 0:
            self.state["samples"].append({"work": work, "seconds": seconds})
            self.last_sample = (completed, self.state["seconds"])
        self.state.update({"output_bytes": self.bytes, "factory_seconds": self.factory_seconds})
        if stats is not None:
            self.state["scheduler"] = stats
        write_json(self.out / "collection.json", self.state)

    def tick(self, stats: dict[str, Any]) -> None:
        self.save(stats=stats)


def restore_winners(protocol: CollectionProtocol, record: ForkRecord) -> None:
    """Fill winners from completed continuation folders after an interruption."""
    for repetition in range(8):
        for alternative in (0, 1):
            key = f"{repetition}-{alternative}"
            path = protocol.folder(record, repetition, alternative) / "completed.json"
            if key not in record.setdefault("winners", {}) and path.exists():
                record["winners"][key] = read_json(path)["winner"]


def record_runtime(
    out: Path,
    runtime: CollectionRuntime,
    executor_runtime: EvaluationRuntime,
    session: dict[str, Any],
) -> None:
    path = out / "collection-runtime.json"
    value: dict[str, Any] = read_json(path) if path.exists() else {"schema": 1, "sessions": []}
    value["sessions"].append(
        dict(
            session,
            runtime=asdict(runtime),
            evaluation_runtime=executor_runtime.identity(),
        )
    )
    write_json(path, value)


def collect(cfg: dict[str, Any], out: Path, runtime: CollectionRuntime, *, budget: PhaseBudget) -> None:
    state: CollectionState = read_json(out / "collection.json")
    if state.get("finalized") and (out / "dataset.json").exists():
        return
    if not state.get("complete") and state.get("stage") != "finalizing":
        protocol = CollectionProtocol(cfg, out, state, runtime)
        protocol.budget = budget
        executor_runtime = EvaluationRuntime("cuda", runtime.workers, runtime.max_batch, runtime.wait_seconds)
        active_forks: list[ForkRecord] = []
        buffered_forks: list[ForkRecord] = []
        for record in state["forks"]:
            if record["status"] == "running":
                restore_winners(protocol, record)
                active_forks.append(record)
            elif record["status"] == "pending":
                buffered_forks.append(record)
        record_runtime(
            out,
            runtime,
            executor_runtime.resolve(),
            {
                "started_seconds": budget.elapsed(),
                "resumed_active_forks": len(active_forks),
            },
        )
        with (
            evaluation_precision(),
            CollectionExecutor(
                [ROOT / parent for parent in cfg["parents"]],
                executor_runtime,
                prefix_states=runtime.prefix_states,
            ) as executor,
        ):
            scheduler = Scheduler(
                executor,
                protocol,
                budget,
                workers=runtime.workers,
                source_share=runtime.source_share,
                stock_factor=runtime.stock_factor,
                max_active_forks=runtime.max_active_forks,
                buffer_limit=runtime.buffer_limit,
                function=match_task,
            )
            scheduler.restore(active_forks, buffered_forks)
            stats: dict[str, Any] = scheduler.run()
            stats["inference"] = executor.statistics()
            stats["proposal_items"] = getattr(executor.inference, "proposal_items", 0)
        for record in state["forks"]:
            if record["status"] == "pending":
                record["status"] = "not_started_budget"
        state["stage"] = "finalizing"
        protocol.save(force=True, stats=stats)
        print(
            json.dumps({name: value for name, value in stats.items() if name != "inference"}),
            flush=True,
        )
    finalize_collection(cfg, out, state, budget)


def finalize_collection(
    cfg: dict[str, Any], out: Path, state: CollectionState, budget: _ElapsedBudget
) -> None:
    """Publish public training histories after all admitted continuations finish.

    Binary prefix captures (and interrupted capture writes) are collection-only
    caches. They must remain available for collection resume, but must not enter
    the source packet imported by training as manifest.json plus json.7z.
    """
    if not (out / "dataset.json").exists():
        write_json(out / "dataset.json", build_dataset(cfg, out, state))
    archive_tree(out, "continuations")
    for pattern in ("*/prefix-states.npz", "*/prefix-states.tmp"):
        for cache in (out / "sources").glob(pattern):
            cache.unlink()
    archive_tree(out, "sources")
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


def build_dataset(cfg: dict[str, Any], out: Path, state: CollectionState) -> dict[str, Any]:
    """Freeze anchors in episode order before reading saved fork labels."""
    inputs = prepare_comparison_dataset(
        out,
        state,
        seed=cfg["seed"],
        namespace=cfg["namespace"],
        sort_sources=True,
        load_learner_sides=False,
    )
    labels: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    statuses: Counter[str] = Counter()

    def label_fork(fork: ForkRecord) -> ForkLabel | None:
        statuses[fork["status"]] += 1
        if fork["status"] != "completed":
            return None
        labels[fork["label"]] += 1
        if fork["reason"]:
            reasons[fork["reason"]] += 1
        return f"{fork_id(fork['episode'], fork['index'])}.json", fork

    pairs = write_labels_and_collect_pairs(out, state["forks"], inputs.fork_actions, label_fork)
    executed = sum(f.get("executed_repetitions", 0) * 2 for f in state["forks"] if f["status"] == "completed")
    return {
        "pairs": pairs,
        "anchors": inputs.anchors,
        "source_hashes": inputs.source_hashes,
        "source_matches": len(state["sources"]),
        "selected_forks": len(state["forks"]),
        "completed_forks": statuses["completed"],
        "fork_statuses": dict(statuses),
        "accepted_pairs": len(pairs),
        "labels": dict(labels),
        "rejection_reasons": dict(reasons),
        "continuations": executed,
    }


def run(
    phase: str,
    cfg: dict[str, Any],
    out: str | Path,
    runtime: CollectionRuntime,
    *,
    preparation_seconds: float = 0.0,
) -> None:
    out = Path(out).resolve()
    with RunLock(out):
        configure(cfg["seed"], "cuda")
        prepare(cfg, out)
        checkpoint.verify_execution_passport(out, cfg, read_json(out / "manifest.json"))
        if phase == "prepare":
            return
        write_json(out / "status.json", {"status": "running", "phase": "collect"})
        with record_experiment_errors(out, phase="collect", error_message_only=True):
            with PhaseBudget(out, "collection", cfg["collection_seconds"]) as budget:
                budget.base += preparation_seconds
                collect(cfg, out, runtime, budget=budget)
            write_json(out / "status.json", {"status": "complete", "phase": "collect"})
    report(out)
