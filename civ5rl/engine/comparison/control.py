"""Transactional full-corpus updates with a fixed parent-distribution bound.

The checkpoint contains only a verified model and Adam, including while an
attempt is pending. Pending work owns a stable ID, LR and anchor slice. Resume
replays that ID from its saved RNG boundary rather than applying tentative
weights. Attempt history in the checkpoint is authoritative; JSONL views can
be rebuilt after a publication failure without duplicating accepted steps.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, fields
from time import perf_counter

from civ5rl.engine.runtime.transactions import AdamSnapshot

MODE = "full_corpus_step_kl"


@dataclass
class ComparisonTrainingState:
    """Progress shared by epoch and per-step fitting; checkpoint payloads are dictionaries."""

    steps: int = 0
    epoch: int = 0
    cursor: int = 0
    seconds: float = 0.0
    anchor_order: list[int] = field(default_factory=list)
    anchor_cursor: int = 0
    order: list[int] | None = None
    last_kl: float | None = None
    stop_reason: str | None = None
    attempted_final_kl: float | None = None
    mode: str | None = None
    attempts: int = 0
    rejected_attempts: int = 0
    lr: float | None = None
    pending: dict | None = None
    history: list[dict] = field(default_factory=list)
    last_full_kl: float = 0.0
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_checkpoint_dict(cls, data):
        payload = dict(data)
        progress = {
            item.name: payload.pop(item.name)
            for item in fields(cls)
            if item.name != "metadata" and item.name in payload
        }
        return cls(**progress, metadata=payload)

    def to_checkpoint_dict(self):
        # Epoch and full-corpus checkpoints keep their respective key sets.
        progress = {
            "steps": self.steps,
            "epoch": self.epoch,
            "cursor": self.cursor,
            "seconds": self.seconds,
            "anchor_order": self.anchor_order,
            "anchor_cursor": self.anchor_cursor,
        }
        if self.mode is not None:
            progress.update(
                mode=self.mode,
                attempts=self.attempts,
                rejected_attempts=self.rejected_attempts,
                lr=self.lr,
                pending=self.pending,
                history=self.history,
                last_full_kl=self.last_full_kl,
            )
        if self.order is not None:
            progress["order"] = self.order
        if self.last_kl is not None:
            progress["last_kl"] = self.last_kl
        if self.mode is not None or self.stop_reason is not None:
            progress.update(stop_reason=self.stop_reason, attempted_final_kl=self.attempted_final_kl)
        return {**progress, **self.metadata}

    def begin_attempt(self):
        if self.pending is None:
            self.pending = {
                "attempt": self.attempts + 1,
                "step": self.steps + 1,
                "retry": 0,
                "lr": self.lr,
                "anchor_cursor": self.anchor_cursor,
                "entered": False,
            }
        if not self.pending["entered"]:
            self.attempts += 1
            self.pending["entered"] = True
        return self.pending

    def record_accepted(self, pending, count, kl):
        self.steps += 1
        self.anchor_cursor += count
        self.lr = pending["lr"]
        self.last_full_kl = kl

    def record_rejected(self):
        self.rejected_attempts += 1

    def record_epoch_step(self, count):
        self.cursor += count
        self.steps += 1


def initial_state(anchors, options, seed):
    order = list(range(len(anchors)))
    random.Random(seed).shuffle(order)
    return ComparisonTrainingState(mode=MODE, anchor_order=order, lr=options["initial_lr"])


def run_updates(policy, optimizer, state, anchors, options, budget, save, update, measure):
    """Apply bounded updates; callbacks execute the existing loss and full KL.

    save() snapshots the current verified weights, optimizer and RNG. It is
    called before touching Adam and after the accept/restore decision. A
    deadline never interrupts an entered update or its required KL check.
    """
    if state.stop_reason:
        return
    while state.steps < options["max_accepted_steps"]:
        if budget.expired():
            if state.pending is not None and state.pending["entered"]:
                state.history.append(
                    dict(
                        **state.pending,
                        accepted=False,
                        outcome="interrupted_time_limit",
                        seconds=0.0,
                    )
                )
            state.pending = None
            state.stop_reason = "time_limit"
            break
        pending = state.begin_attempt()
        count = options["anchors_per_step"]
        start = pending["anchor_cursor"]
        if options.get("cycle_anchors"):
            if count > len(anchors):
                raise ValueError("A step cannot repeat anchors")
            selected = [anchors[state.anchor_order[(start + i) % len(anchors)]] for i in range(count)]
        else:
            selected = [anchors[i] for i in state.anchor_order[start : start + count]]
        if len(selected) != count:
            raise ValueError("Not enough distinct anchors for the accepted-step limit")
        # A restored pending attempt already has its ID and RNG in the checkpoint.
        snapshot = AdamSnapshot(policy, optimizer)
        started = perf_counter()
        save()
        for group in optimizer.param_groups:
            group["lr"] = pending["lr"]
        try:
            step_started = perf_counter()
            metrics = update(selected)
            step_seconds = perf_counter() - step_started
            kl_started = perf_counter()
            measured = measure()
            kl, learning = measured if isinstance(measured, tuple) else (measured, None)
            kl_seconds = perf_counter() - kl_started
            if not math.isfinite(kl):
                raise FloatingPointError("Non-finite full KL after Adam attempt")
        except BaseException:
            snapshot.restore()
            save()
            raise
        accepted = kl <= options["kl_limit"]
        state.attempted_final_kl = kl
        rollback_started = perf_counter()
        if accepted:
            state.record_accepted(pending, count, kl)
        else:
            snapshot.restore()
            state.record_rejected()
        record = dict(
            **pending,
            accepted=accepted,
            outcome="accepted" if accepted else "kl_rejected",
            full_kl=kl,
            step_seconds=step_seconds,
            kl_seconds=kl_seconds,
            rollback_seconds=0.0 if accepted else perf_counter() - rollback_started,
            seconds=perf_counter() - started,
            **metrics,
        )
        if accepted and learning is not None:
            record["learning"] = learning
            state.metadata["learning"] = learning
        state.history.append(record)
        state.pending = None
        if not accepted:
            if pending["retry"] >= options["max_retries"]:
                state.stop_reason = "retry_limit"
            elif budget.expired():
                state.stop_reason = "time_limit"
            else:
                # Reserve the retry ID only when entering it, after checking time.
                state.pending = dict(
                    pending,
                    attempt=state.attempts + 1,
                    retry=pending["retry"] + 1,
                    lr=pending["lr"] / 2,
                    entered=False,
                )
        save()
        if state.stop_reason:
            break
    if state.steps >= options["max_accepted_steps"]:
        state.stop_reason = "accepted_step_limit"
    save()
