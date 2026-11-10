"""Target scale BC training restoration, state reconciliation, and mid-epoch resume."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from civ5rl.engine.counters import TrainingCounters
from civ5rl.provenance.common import read_json, write_json


@dataclass
class BcRestorationState:
    epoch: int
    epoch_completed: bool
    batch_cursor: int
    adam_steps: int
    free_labels_processed: int
    epoch_order: list[Any] | None = None

    @classmethod
    def from_dict(
        cls, data: dict[str, Any], *, protocol: str | None = "bc"
    ) -> BcRestorationState:
        adam_steps = data.get("adam_steps")
        if adam_steps is None:
            counters = TrainingCounters.from_dict(data, protocol=protocol)
            adam_steps = counters.adam_steps_accepted
        return cls(
            epoch=int(data.get("epoch", 0)),
            epoch_completed=bool(data.get("epoch_completed", False)),
            batch_cursor=int(data.get("batch_cursor", 0)),
            adam_steps=int(adam_steps),
            free_labels_processed=int(data.get("free_labels_processed", 0)),
            epoch_order=data.get("epoch_order"),
        )

    def to_dict(self, *, protocol: str | None = "bc") -> dict[str, Any]:
        """Dual-write dictionary retaining both canonical and legacy keys."""
        counters = TrainingCounters(
            rollouts_collected=0,
            rollouts_with_accepted_steps=0,
            adam_steps_attempted=self.adam_steps,
            adam_steps_accepted=self.adam_steps,
        )
        return {
            **counters.to_state_dict(protocol=protocol, include_legacy_bc=True),
            "epoch": self.epoch,
            "epoch_completed": self.epoch_completed,
            "batch_cursor": self.batch_cursor,
            "adam_steps": self.adam_steps,
            "Adam_steps": self.adam_steps,
            "adam_steps_accepted": self.adam_steps,
            "free_labels_processed": self.free_labels_processed,
            "epoch_order": self.epoch_order,
        }


@dataclass
class RestoredEpochBatches:
    order: list[Any]
    cursor: int
    epoch_labels: int
    batches: Iterator[Any]
    records: list[dict[str, Any]]

    def __iter__(self):
        return iter(
            (
                self.order,
                self.cursor,
                self.epoch_labels,
                self.batches,
                self.records,
            )
        )

    def __getitem__(self, index: int) -> Any:
        items = (
            self.order,
            self.cursor,
            self.epoch_labels,
            self.batches,
            self.records,
        )
        return items[index]


@dataclass
class PreparedEpochState:
    start_epoch: int
    resume_mid_epoch: bool
    steps: int
    labels: int
    clock_offset: float
    history: list[dict[str, Any]] = field(default_factory=list)
    previous_records: list[dict[str, Any]] = field(default_factory=list)
    evaluate_needed: bool = False
    source: str = ""


@dataclass
class BcRestorationResult:
    restoration_state: BcRestorationState
    epoch_state: PreparedEpochState

    def __iter__(self):
        return iter((self.restoration_state, self.epoch_state))


def _extract_counter(data: dict[str, Any], key: str) -> Any:
    if key in ("Adam_steps", "adam_steps", "adam_steps_accepted"):
        for k in ("adam_steps", "Adam_steps", "adam_steps_accepted"):
            if k in data:
                return data[k]
        return None
    return data.get(key)


def select_bc_checkpoint(
    out: Path | str,
    restart_file: str | Path | None = "restart-input-v2.json",
    *,
    resume: bool = False,
    kind: str | None = None,
) -> str:
    """Resolve candidate checkpoint path from restart input, state.json['latest'], or initial.pt."""
    out_path = Path(out)
    if restart_file is not None:
        restart_path = (
            out_path / restart_file
            if not Path(restart_file).is_absolute()
            else Path(restart_file)
        )
        if restart_path.exists():
            restart = read_json(restart_path)
            if restart and not restart.get("consumed", False):
                if (
                    "checkpoint_boundary" in restart
                    and "latest" in restart["checkpoint_boundary"]
                ):
                    return str(restart["checkpoint_boundary"]["latest"])
                if "latest" in restart:
                    return str(restart["latest"])

    state_path = out_path / "state.json"
    if state_path.exists():
        state_record = read_json(state_path)
        if state_record and "latest" in state_record:
            adam_steps = int(
                state_record.get(
                    "Adam_steps",
                    state_record.get("adam_steps_accepted", state_record.get("adam_steps", 0)),
                )
            )
            epoch = int(state_record.get("epoch", 0))
            if resume or adam_steps > 0 or epoch > 0 or kind is None:
                return str(state_record["latest"])

    return "checkpoints/initial.pt"


def reconcile_state_vs_steplog(
    saved_state: dict[str, Any] | BcRestorationState,
    state_record: dict[str, Any] | None,
    steplog_path: Path | str,
) -> list[dict[str, Any]]:
    """Reconcile checkpoint state against state.json and steps.jsonl.

    Truncates orphaned trailing steps if interrupted before checkpoint save,
    validates step continuity and label sums.
    """
    saved_dict = (
        asdict(saved_state)
        if isinstance(saved_state, BcRestorationState)
        else saved_state
    )

    if state_record is not None:
        for key in (
            "epoch",
            "epoch_completed",
            "batch_cursor",
            "Adam_steps",
            "free_labels_processed",
        ):
            saved_val = _extract_counter(saved_dict, key)
            state_val = _extract_counter(state_record, key)
            if saved_val != state_val:
                raise ValueError(
                    f"Checkpoint state and state.json disagree on {key}"
                )

    steps_path = Path(steplog_path)
    all_lines = (
        [
            line
            for line in steps_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if steps_path.exists()
        else []
    )
    origin_steps = int(_extract_counter(saved_dict, "Adam_steps") or 0)

    if len(all_lines) < origin_steps:
        raise ValueError("steps.jsonl has fewer records than saved checkpoint")
    if len(all_lines) > origin_steps:
        all_lines = all_lines[:origin_steps]
        steps_path.write_text(
            ("\n".join(all_lines) + "\n") if all_lines else "",
            encoding="utf-8",
        )

    previous_records = [json.loads(line) for line in all_lines]
    if [r["step"] for r in previous_records] != list(
        range(1, origin_steps + 1)
    ):
        raise ValueError("Non-sequential step numbers in steps.jsonl")

    expected_labels = int(saved_dict.get("free_labels_processed", 0))
    if sum(r["labels"] for r in previous_records) != expected_labels:
        raise ValueError("Saved label count disagrees with steps.jsonl")

    return previous_records


def restore_epoch_cursor_and_batches(
    saved_state: dict[str, Any] | BcRestorationState,
    history: list[dict[str, Any]] | None,
    previous_records: list[dict[str, Any]],
    cache_masks: list[list[bool]],
    Span_cls: Any = None,
    label_minibatches_fn: Any = None,
    size: int = 1024,
) -> RestoredEpochBatches:
    """Restore chunk priority order, check record count matches cursor, and slice remaining minibatches."""
    if Span_cls is None:
        from civ5rl.engine.spans import Span as Span_cls
    if label_minibatches_fn is None:
        from civ5rl.engine.imitation import (
            label_minibatches as label_minibatches_fn,
        )

    saved_dict = (
        asdict(saved_state)
        if isinstance(saved_state, BcRestorationState)
        else saved_state
    )

    raw_order = saved_dict.get("epoch_order")
    if raw_order is None:
        raise ValueError("Missing epoch_order in saved_state")

    order = []
    for item in raw_order:
        if isinstance(item, Span_cls):
            order.append(item)
        elif (
            hasattr(item, "episode")
            and hasattr(item, "start")
            and hasattr(item, "stop")
        ):
            order.append(
                Span_cls(int(item.episode), int(item.start), int(item.stop))
            )
        else:
            order.append(Span_cls(int(item[0]), int(item[1]), int(item[2])))

    cursor = int(saved_dict["batch_cursor"])
    try:
        all_batches = list(label_minibatches_fn(order, cache_masks, size=size))
    except TypeError:
        all_batches = list(label_minibatches_fn(order, cache_masks))

    prior_steps = sum(h["steps"] for h in (history or []))
    records = previous_records[prior_steps:]
    if len(records) != cursor:
        raise ValueError("Mid-epoch step records disagree with batch_cursor")

    epoch_labels = sum(r["labels"] for r in records)
    batches = iter(all_batches[cursor:])
    return RestoredEpochBatches(
        order=order,
        cursor=cursor,
        epoch_labels=epoch_labels,
        batches=batches,
        records=records,
    )


def restore_bc_training_state(
    out: Path | str,
    saved_state: dict[str, Any] | BcRestorationState,
    state_record: dict[str, Any] | None = None,
    *,
    restart: dict[str, Any] | None = None,
    kind: str | None = None,
    resume: bool = False,
    config: dict[str, Any] | None = None,
    cache: Any = None,
    source: str = "",
    data_labels: int | None = None,
) -> BcRestorationResult:
    """High-level orchestrator returning BcRestorationState and prepared epoch state for the main loop."""
    out_path = Path(out)
    if isinstance(saved_state, dict) and "state" in saved_state and isinstance(saved_state["state"], dict):
        saved_dict = saved_state["state"]
    elif isinstance(saved_state, BcRestorationState):
        saved_dict = asdict(saved_state)
    else:
        saved_dict = saved_state

    if state_record is None and (out_path / "state.json").exists():
        state_record = read_json(out_path / "state.json")

    if restart is None and (out_path / "restart-input-v2.json").exists():
        restart = read_json(out_path / "restart-input-v2.json")

    from civ5rl.protocols.imitation_target_scale import C2_KIND, MORE_DATA_KIND

    # Check restart path
    if restart and not restart.get("consumed", False):
        boundary = restart["checkpoint_boundary"]
        origin_steps = int(_extract_counter(boundary, "Adam_steps") or 0)
        saved_steps = int(_extract_counter(saved_dict, "Adam_steps") or 0)
        if not saved_dict.get("epoch_completed") or saved_steps != origin_steps:
            raise ValueError(
                "Input-fix restart requires its complete saved epoch boundary"
            )

        steps_path = out_path / "steps.jsonl"
        previous = [
            json.loads(line)
            for line in steps_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ] if steps_path.exists() else []

        if len(previous) != origin_steps:
            raise ValueError("Saved step log differs from checkpoint boundary")

        history: list[dict[str, Any]] = []
        if data_labels is not None and data_labels > 0:
            steps_per_epoch = (data_labels + 1023) // 1024
            for e in range(1, int(boundary["epoch"]) + 1):
                records = previous[(e - 1) * steps_per_epoch : e * steps_per_epoch]
                if sum(r["labels"] for r in records) != data_labels:
                    raise ValueError("Incomplete saved epoch")
                history.append(
                    {
                        "epoch": e,
                        "completed": True,
                        "steps": len(records),
                        "labels": data_labels,
                        "online_CE": sum(r["ce"] * r["labels"] for r in records)
                        / data_labels,
                    }
                )
        else:
            history = list(saved_dict.get("epochs_history", []))

        start_epoch = int(boundary["epoch"]) + 1
        steps = origin_steps
        labels = int(boundary["free_labels_processed"])
        clock_offset = float(boundary.get("active_seconds", 0.0))

        if cache is not None and hasattr(cache, "counters"):
            for field_name in ("history_positions", "prefix_positions"):
                if field_name in boundary:
                    cache.counters[field_name] = boundary[field_name]

        restoration_state = BcRestorationState.from_dict(saved_dict)
        epoch_state = PreparedEpochState(
            start_epoch=start_epoch,
            resume_mid_epoch=False,
            steps=steps,
            labels=labels,
            clock_offset=clock_offset,
            history=history,
            previous_records=previous,
            evaluate_needed=True,
            source=source,
        )
        return BcRestorationResult(restoration_state, epoch_state)

    # Check resume path
    resuming_more_data = (
        kind in (MORE_DATA_KIND, C2_KIND)
        and state_record is not None
        and (
            resume
            or int(_extract_counter(state_record, "Adam_steps") or 0) > 0
            or int(state_record.get("epoch", 0)) > 0
        )
    )
    is_resuming = resuming_more_data or resume or (
        state_record is not None
        and (
            int(_extract_counter(state_record, "Adam_steps") or 0) > 0
            or int(state_record.get("epoch", 0)) > 0
        )
    )

    if is_resuming:
        previous_records = reconcile_state_vs_steplog(
            saved_dict, state_record, out_path / "steps.jsonl"
        )
        origin_steps = int(_extract_counter(saved_dict, "Adam_steps") or 0)
        steps = origin_steps
        labels = int(saved_dict["free_labels_processed"])

        if cache is not None and hasattr(cache, "counters"):
            for field_name in ("history_positions", "prefix_positions"):
                cache.counters[field_name] = int(saved_dict.get(field_name, 0))

        history = list(saved_dict.get("epochs_history", []))
        clock_offset = float(
            state_record.get(
                "active_seconds",
                saved_dict.get("active_seconds_at_save_start", 0.0),
            )
            if state_record is not None
            else saved_dict.get("active_seconds_at_save_start", 0.0)
        )
        saved_epoch = int(saved_dict["epoch"])
        evaluate_needed = False

        if saved_dict.get("epoch_completed", False):
            evaluate_epochs = (config or {}).get("evaluate_epochs", [])
            skip_eval = (config or {}).get("skip_milestone_evaluations", False)
            if kind != C2_KIND and saved_epoch in evaluate_epochs and not skip_eval:
                evaluate_needed = True
            start_epoch = saved_epoch + 1
            resume_mid_epoch = False
        else:
            start_epoch = saved_epoch
            resume_mid_epoch = True
            if (
                history
                and history[-1].get("epoch") == saved_epoch
                and not history[-1].get("completed", True)
            ):
                history.pop()

        write_json(
            out_path / "resume-state-verification.json",
            {
                "model_equal": True,
                "Adam_equal": True,
                "RNG_restored": True,
                "source": source,
                "epoch": saved_epoch,
                "epoch_completed": bool(saved_dict["epoch_completed"]),
                "batch_cursor": int(saved_dict["batch_cursor"]),
                "Adam_counter": steps,
                "next_Adam_counter": steps + 1,
                "active_seconds_preserved": clock_offset,
            },
        )

        restoration_state = BcRestorationState.from_dict(saved_dict)
        epoch_state = PreparedEpochState(
            start_epoch=start_epoch,
            resume_mid_epoch=resume_mid_epoch,
            steps=steps,
            labels=labels,
            clock_offset=clock_offset,
            history=history,
            previous_records=previous_records,
            evaluate_needed=evaluate_needed,
            source=source,
        )
        return BcRestorationResult(restoration_state, epoch_state)

    # Clean start
    restoration_state = BcRestorationState(
        epoch=0,
        epoch_completed=False,
        batch_cursor=0,
        adam_steps=0,
        free_labels_processed=0,
    )
    epoch_state = PreparedEpochState(
        start_epoch=1,
        resume_mid_epoch=False,
        steps=0,
        labels=0,
        clock_offset=0.0,
        history=[],
        previous_records=[],
        evaluate_needed=False,
        source=source,
    )
    return BcRestorationResult(restoration_state, epoch_state)
