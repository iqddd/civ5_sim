"""Transactional full-corpus updates with a fixed parent-distribution bound.

The checkpoint contains only a verified model and Adam, including while an
attempt is pending. Pending work owns a stable ID, LR and anchor slice. Resume
replays that ID from its saved RNG boundary rather than applying tentative
weights. Attempt history in the checkpoint is authoritative; JSONL views can
be rebuilt after a publication failure without duplicating accepted steps.
"""

import math
import random
from dataclasses import dataclass, field, fields
from time import perf_counter

from civ5rl.engine.runtime.transactions import AdamSnapshot, atomic_torch_save
from civ5rl.provenance.common import atomic_write_jsonl
from civ5rl.provenance.rng import capture_rng, restore_rng

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
        progress = dict(
            steps=self.steps,
            epoch=self.epoch,
            cursor=self.cursor,
            seconds=self.seconds,
            anchor_order=self.anchor_order,
            anchor_cursor=self.anchor_cursor,
        )
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
            progress.update(
                stop_reason=self.stop_reason, attempted_final_kl=self.attempted_final_kl
            )
        return {**progress, **self.metadata}

    def begin_attempt(self):
        if self.pending is None:
            self.pending = dict(
                attempt=self.attempts + 1,
                step=self.steps + 1,
                retry=0,
                lr=self.lr,
                anchor_cursor=self.anchor_cursor,
                entered=False,
            )
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
    return ComparisonTrainingState(
        mode=MODE, anchor_order=order, lr=options["initial_lr"]
    )


def run_updates(
    policy, optimizer, state, anchors, options, budget, save, update, measure
):
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
            selected = [
                anchors[state.anchor_order[(start + i) % len(anchors)]]
                for i in range(count)
            ]
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


def fit_full_corpus(cfg, out, dataset, identity, budget):
    from pathlib import Path

    import torch

    from civ5rl.engine.comparison.training import (
        Histories,
        configure_policy,
        fit_identity,
        full_kl,
        logical_step,
        prepare_fit,
        prepare_histories,
        prepare_references,
        publish_candidate,
    )
    from civ5rl.provenance.common import seed_for, write_json

    out = Path(out)
    options = cfg["training"]
    anchors, pairs = dataset["anchors"], dataset["pairs"]
    if len({tuple(key) for key in anchors}) != len(anchors):
        raise ValueError("Full-corpus control requires distinct anchors")
    required = options["anchors_per_step"] * (
        1 if options.get("cycle_anchors") else options["max_accepted_steps"]
    )
    if len(anchors) < required:
        raise ValueError("Anchor pool does not cover the accepted-step limit")
    identity = fit_identity(identity, dataset)
    path = out / "training.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
    elif cfg.get("fit_continuation"):
        from civ5rl.engine.comparison.continuation import inherited_state

        state = inherited_state(cfg, identity)
        atomic_torch_save(state, path)
        write_json(out / "fit-lineage.json", cfg["fit_continuation"])
    else:
        state = initial_state(
            anchors,
            options,
            seed_for(cfg["seed"], 0, "anchors", namespace=cfg["namespace"]),
        )
    state = (
        state
        if isinstance(state, ComparisonTrainingState)
        else ComparisonTrainingState.from_checkpoint_dict(state)
    )
    done, result = prepare_fit(cfg, out, state.to_checkpoint_dict(), identity)
    if done:
        return result
    head_only = options["mode"] == "frozen_heads_step_kl"
    if head_only:
        from civ5rl.engine.comparison.heads import HeadFeatures, configure_heads

        policy = configure_heads(cfg["parents"][0])
    else:
        policy = configure_policy(cfg["parents"][0])
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad],
        lr=options["initial_lr"],
        betas=(0.9, 0.999),
        eps=1e-5,
        weight_decay=0,
    )
    cache_dir = None
    if cfg.get("fit_continuation"):
        from civ5rl.provenance.common import ROOT

        previous = Histories(ROOT / cfg["fit_continuation"]["run"])
        if previous.directory.exists():
            cache_dir = previous.directory
    histories = prepare_histories(out, dataset, policy, budget, cache_dir=cache_dir)
    features = None
    if head_only:
        features = HeadFeatures(policy, histories, dataset)
        try:
            features.prepare(cfg["parents"][0], budget=budget)
            _, baseline = features.measure()
            state.metadata.setdefault("baseline_learning", baseline)
            state.metadata.setdefault("learning", baseline)
            state.metadata["feature_parity"] = features.parity
        except TimeoutError:
            if not budget.expired():
                raise
        state.mode = options["mode"]
    else:
        references = prepare_references(
            policy, histories, anchors, out / "references.pt", budget=budget
        )
    if "model" in state.metadata:
        policy.load_state_dict(state.metadata["model"])
        optimizer.load_state_dict(state.metadata["optimizer"])
        restore_rng(state.metadata["rng"])

    def save():
        state.metadata.update(
            model=policy.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=capture_rng(),
            identity=identity,
            input_cache=histories.statistics(),
        )
        if features is not None:
            state.metadata["input_cache"]["head_features"] = features.statistics
        state.seconds = budget.elapsed()
        data = state.to_checkpoint_dict()
        atomic_torch_save(data, path)
        cycle = 0.0
        accepted = []
        for row in state.history:
            cycle += row["seconds"]
            if row["accepted"]:
                accepted.append(dict(row, cycle_seconds=cycle))
                cycle = 0.0
        atomic_write_jsonl(out / "attempts.jsonl", state.history)
        atomic_write_jsonl(out / "steps.jsonl", accepted)
        if features is not None:
            learning_rows = (
                [dict(step=0, learning=state.metadata["baseline_learning"])]
                if "baseline_learning" in state.metadata
                else []
            )
            learning_rows += [
                dict(step=row["step"], learning=row["learning"])
                for row in state.history
                if row["accepted"]
            ]
            atomic_write_jsonl(out / "learning.jsonl", learning_rows)
        write_json(
            out / "training.json",
            {
                k: data[k]
                for k in (
                    "mode",
                    "steps",
                    "attempts",
                    "rejected_attempts",
                    "lr",
                    "anchor_cursor",
                    "pending",
                    "last_full_kl",
                    "seconds",
                    "stop_reason",
                    "input_cache",
                )
            }
            | {
                "learning": state.metadata.get("learning"),
                "inherited_steps": state.metadata.get("inherited_steps", 0),
                "inherited_seconds": state.metadata.get("inherited_seconds", 0.0),
            },
        )

    def update(selected):
        if features is not None:
            return features.update(optimizer, pairs, selected)
        return logical_step(
            policy, optimizer, histories, pairs, selected, references["mixed"]
        )

    run_updates(
        policy,
        optimizer,
        state,
        anchors,
        options,
        budget,
        save,
        update,
        features.measure
        if features is not None
        else lambda: full_kl(policy, histories, anchors, references["fp32"]),
    )
    frozen = torch.load(path, map_location="cpu", weights_only=False)
    return publish_candidate(cfg, out, frozen, identity)
