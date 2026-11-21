"""Execution shared by experiment evaluation phases."""

from __future__ import annotations

import shutil
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Executor
from pathlib import Path
from typing import Any, Literal, TypedDict, TypeVar

from civ5rl.engine.evaluation.storage import readable_evaluations
from civ5rl.provenance.common import ROOT, file_sha256
from civ5rl.reporting.evaluation_audit import clustered_paired_bootstrap, load_eval_rows

Task = TypeVar("Task")

Config = dict[str, Any]
EvaluationRecord = Mapping[str, Any]
EvaluateCheckpoint = Callable[[Config, Config, Path, str, str], EvaluationRecord]
ReuseParentEvaluation = Callable[[Config, Path, Config, Config], object]

DEVELOPMENT_PURPOSE = "development_512"
PARENT_CHECKPOINT = "checkpoints/parent.pt"
CANDIDATE_CHECKPOINT = "candidate.pt"


class _AdditionalStatus(TypedDict):
    status: Literal["disabled_by_protocol"]


class PairedDevelopmentResult(TypedDict):
    status: Literal["complete"]
    development: EvaluationRecord
    parent_development: EvaluationRecord
    comparison: dict[str, Any]
    candidate_selection: Literal["last_accepted"]
    additional: _AdditionalStatus


def run_chunked_pool(
    executor: Executor,
    fn: Callable[[Task], object],
    tasks: Sequence[Task],
    chunk_size: int,
    on_chunk: Callable[[int], None] | None = None,
) -> None:
    """Drain each chunk before submitting the next and report cumulative progress."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")
    for first in range(0, len(tasks), chunk_size):
        chunk = tasks[first : first + chunk_size]
        list(executor.map(fn, chunk))
        if on_chunk is not None:
            on_chunk(first + len(chunk))


def prepare_paired_evaluation_parent(
    out: Path,
    original: Path,
    expected_sha256: str,
    *,
    error_message: str = "Evaluation parent changed",
) -> None:
    parent_path = out / PARENT_CHECKPOINT
    parent_path.parent.mkdir(exist_ok=True)
    if not parent_path.exists():
        shutil.copy2(original, parent_path)
    if file_sha256(parent_path) != expected_sha256:
        raise ValueError(error_message)


def evaluate_paired_dev512(
    cfg: Config,
    out: Path,
    match_cfg: Config,
    plan: Config,
    evaluator: EvaluateCheckpoint,
    *,
    reuse_parent_fn: ReuseParentEvaluation | None = None,
    parent_prepared: bool = False,
) -> PairedDevelopmentResult:
    """Evaluate and strictly pair the parent and last candidate on Development-512.

    The caller holds the run lock and checks candidate eligibility. A reuse hook
    runs before evaluation with (cfg, out, match_cfg, plan) and can populate the
    evaluator's cache. Its errors propagate. The caller publishes the result and
    manages phase status and archiving. Some protocols prepare the parent copy
    before opening the evaluation manifest and lock.
    """
    from civ5rl.engine.evaluation.executor import evaluation_options

    if not parent_prepared:
        prepare_paired_evaluation_parent(out, ROOT / cfg["parents"][0], cfg["parent_sha256"][0])

    with (
        readable_evaluations(out),
        evaluation_options(device="cuda", workers=8, max_batch=32, wait_seconds=0.001),
    ):
        if reuse_parent_fn is not None:
            reuse_parent_fn(cfg, out, match_cfg, plan)
        parent = evaluator(match_cfg, plan, out, PARENT_CHECKPOINT, DEVELOPMENT_PURPOSE)
        candidate = evaluator(match_cfg, plan, out, CANDIDATE_CHECKPOINT, DEVELOPMENT_PURPOSE)
        parent_rows = load_eval_rows(
            out / parent["runs"]["argmax"]["cp"]["directory"],
            512,
            plan=plan[DEVELOPMENT_PURPOSE],
        )
        candidate_rows = load_eval_rows(
            out / candidate["runs"]["argmax"]["cp"]["directory"],
            512,
            plan=plan[DEVELOPMENT_PURPOSE],
        )
        comparison = clustered_paired_bootstrap(parent_rows, candidate_rows)

    return {
        "status": "complete",
        "development": candidate,
        "parent_development": parent,
        "comparison": comparison,
        "candidate_selection": "last_accepted",
        "additional": {"status": "disabled_by_protocol"},
    }
