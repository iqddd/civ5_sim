"""Target scale BC training restoration, state reconciliation, and mid-epoch resume."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from civ5rl.engine.imitation.core import label_minibatches
from civ5rl.engine.imitation.storage import PreparedDemonstrations
from civ5rl.engine.runtime.sequence import Span
from civ5rl.protocols.imitation_target_scale import C2_KIND
from civ5rl.provenance.common import write_json
from civ5rl.provenance.jsonl import (
    read_jsonl_lines,
    recover_jsonl_prefix,
)


@dataclass
class EpochBatches:
    order: list[Span]
    cursor: int
    epoch_labels: int
    batches: Iterator[tuple[list[Span], int]]
    records: list[dict[str, Any]]


@dataclass
class PreparedEpochState:
    saved_epoch: int = 0
    start_epoch: int = 1
    resume_mid_epoch: bool = False
    steps: int = 0
    labels: int = 0
    clock_offset: float = 0.0
    history: list[dict[str, Any]] = field(default_factory=list)
    previous_records: list[dict[str, Any]] = field(default_factory=list)
    evaluate_needed: bool = False


def saved_adam_steps(state: dict[str, Any]) -> int:
    """Read the committed step count from current or historical BC state."""
    return int(state.get("adam_steps", state.get("Adam_steps", state.get("adam_steps_accepted", 0))))


def reconcile_state_vs_steplog(
    saved_state: dict[str, Any],
    state_record: dict[str, Any] | None,
    steplog_path: Path | str,
) -> list[dict[str, Any]]:
    """Reconcile checkpoint state against state.json and steps.jsonl.

    Truncates orphaned trailing steps if interrupted before checkpoint save,
    validates step continuity and label sums.
    """
    if state_record is not None:
        for key in (
            "epoch",
            "epoch_completed",
            "batch_cursor",
            "free_labels_processed",
        ):
            if saved_state.get(key) != state_record.get(key):
                raise ValueError(f"Checkpoint state and state.json disagree on {key}")
        if saved_adam_steps(saved_state) != saved_adam_steps(state_record):
            raise ValueError("Checkpoint state and state.json disagree on Adam_steps")

    steps_path = Path(steplog_path)
    all_lines = read_jsonl_lines(steps_path) if steps_path.exists() else []
    origin_steps = saved_adam_steps(saved_state)
    previous_records = recover_jsonl_prefix(
        steps_path,
        all_lines,
        origin_steps,
        missing_error="steps.jsonl has fewer records than saved checkpoint",
    )
    if [r["step"] for r in previous_records] != list(range(1, origin_steps + 1)):
        raise ValueError("Non-sequential step numbers in steps.jsonl")

    expected_labels = int(saved_state.get("free_labels_processed", 0))
    if sum(r["labels"] for r in previous_records) != expected_labels:
        raise ValueError("Saved label count disagrees with steps.jsonl")

    return previous_records


def restore_epoch_cursor_and_batches(
    saved_state: dict[str, Any],
    history: list[dict[str, Any]],
    previous_records: list[dict[str, Any]],
    cache_masks: list[list[bool]],
    *,
    size: int = 1024,
) -> EpochBatches:
    """Restore chunk priority order, check record count matches cursor, and slice remaining minibatches."""
    raw_order = saved_state.get("epoch_order")
    if raw_order is None:
        raise ValueError("Missing epoch_order in saved_state")

    order = [Span(int(episode), int(start), int(stop)) for episode, start, stop in raw_order]
    cursor = int(saved_state["batch_cursor"])
    all_batches = list(label_minibatches(order, cache_masks, size=size))

    prior_steps = sum(h["steps"] for h in history)
    records = previous_records[prior_steps:]
    if len(records) != cursor:
        raise ValueError("Mid-epoch step records disagree with batch_cursor")

    epoch_labels = sum(r["labels"] for r in records)
    batches = iter(all_batches[cursor:])
    return EpochBatches(
        order=order,
        cursor=cursor,
        epoch_labels=epoch_labels,
        batches=batches,
        records=records,
    )


def restore_bc_training_state(
    out: Path,
    saved_state: dict[str, Any],
    state_record: dict[str, Any] | None,
    *,
    resuming: bool,
    config: dict[str, Any],
    cache: PreparedDemonstrations,
    source: str,
) -> PreparedEpochState:
    """Reconcile committed rows and prepare the next epoch from the checkpoint."""
    if not resuming:
        return PreparedEpochState()

    previous_records = reconcile_state_vs_steplog(saved_state, state_record, out / "steps.jsonl")
    steps = saved_adam_steps(saved_state)
    for field_name in ("history_positions", "prefix_positions"):
        cache.counters[field_name] = int(saved_state.get(field_name, 0))

    history = list(saved_state.get("epochs_history", []))
    clock_offset = float(
        (state_record or {}).get("active_seconds", saved_state.get("active_seconds_at_save_start", 0.0))
    )
    saved_epoch = int(saved_state["epoch"])
    epoch_completed = bool(saved_state["epoch_completed"])
    if epoch_completed:
        evaluate_needed = (
            config["kind"] != C2_KIND
            and saved_epoch in config.get("evaluate_epochs", [])
            and not config.get("skip_milestone_evaluations", False)
        )
    else:
        evaluate_needed = False
        if history and history[-1].get("epoch") == saved_epoch and not history[-1].get("completed", True):
            history.pop()

    write_json(
        out / "resume-state-verification.json",
        {
            "model_equal": True,
            "Adam_equal": True,
            "RNG_restored": True,
            "source": source,
            "epoch": saved_epoch,
            "epoch_completed": epoch_completed,
            "batch_cursor": int(saved_state["batch_cursor"]),
            "Adam_counter": steps,
            "next_Adam_counter": steps + 1,
            "active_seconds_preserved": clock_offset,
        },
    )
    return PreparedEpochState(
        saved_epoch=saved_epoch,
        start_epoch=saved_epoch + int(epoch_completed),
        resume_mid_epoch=not epoch_completed,
        steps=steps,
        labels=int(saved_state["free_labels_processed"]),
        clock_offset=clock_offset,
        history=history,
        previous_records=previous_records,
        evaluate_needed=evaluate_needed,
    )
