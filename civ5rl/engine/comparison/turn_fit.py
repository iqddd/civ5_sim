"""Whole-turn fit orchestration using the shared transactional update controller."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import torch

from civ5rl.core.agent import RLAgent
from civ5rl.engine.comparison import PREFERENCE_TEMPERATURE
from civ5rl.engine.comparison import training as shared
from civ5rl.engine.comparison.control import ComparisonTrainingState, run_updates
from civ5rl.engine.comparison.turn_inputs import TurnHistories
from civ5rl.engine.comparison.turn_training import (
    BPTT_WINDOW_LENGTH,
    AnchorSubset,
    HistoryKey,
    PreparedPreferences,
    TurnBatch,
    control_kl,
    current_chosen,
    hidden_states_at,
    logical_step,
    pack_batch,
    prepare_anchor_subset,
    prepare_preferences,
    reference_distributions,
)
from civ5rl.engine.runtime.transactions import atomic_torch_save, cpu_snapshot
from civ5rl.provenance.common import ROOT, atomic_write_jsonl, read_json, seed_for, write_json
from civ5rl.provenance.rng import capture_rng, restore_rng
from civ5sim.engine import digest


class TurnTrainingState(ComparisonTrainingState):
    def record_accepted(self, pending, count, kl):
        super().record_accepted(pending, count, kl)
        per_epoch = self.metadata["steps_per_epoch"]
        self.epoch, step = divmod(self.steps, per_epoch)
        self.cursor = step * self.metadata["positions_per_step"]


def configure_policy(parent, *, compile_model=True):
    # Keep the inherited FP32 operators in both execution modes.
    if compile_model:
        return shared.configure_policy(parent, mixed=False)
    policy = RLAgent.from_checkpoint(parent, "cpu", "argmax").policy.to("cuda").train()
    policy.enable_amp(False)
    policy.enable_inverse_routing()
    policy.enable_action_routing()
    policy.enable_maximum_routing()
    policy.configure_execution(activation_checkpointing=False, attention_backend=shared.ATTENTION_BACKEND)
    for parameter in policy.value.parameters():
        parameter.requires_grad_(False)
    return policy


def parent_references(policy, histories, dataset, keys, out, cfg, *, branch_steps=None):
    identity = {
        "version": 2,
        "dataset": digest(dataset),
        "parent": cfg["parent_sha256"][0],
        "keys": keys,
        "precision": "deterministic_cuda_fp32_no_amp_no_tf32",
        "temperature": PREFERENCE_TEMPERATURE,
        "attention_backend": policy.attention_backend,
        "torch": str(torch.__version__),
        "gpu": torch.cuda.get_device_name(),
    }
    path = Path(out) / "parent-references.pt"
    saved = torch.load(path, map_location="cpu", weights_only=False) if path.exists() else None
    if saved is None or saved["identity"] != identity:
        with torch.no_grad(), shared.evaluation_precision():
            chosen = current_chosen(policy, histories, keys, branch_steps=branch_steps)
            anchors = reference_distributions(policy, histories, dataset["anchors"])
        saved = cpu_snapshot({"identity": identity, "chosen": chosen, "anchors": anchors})
        atomic_torch_save(saved, path)
    device = next(policy.parameters()).device
    chosen = saved["chosen"].to(device)
    return (
        {key: chosen[i] for i, key in enumerate(keys)},
        shared.DeviceReferences(saved["anchors"], device),
        shared.DeviceReferences(saved["anchors"], device, control=True),
    )


def _load_training_state(path, identity, cfg, dataset):
    if path.exists():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload["identity"] != identity:
            raise ValueError("Whole-turn fit input changed")
        return TurnTrainingState.from_checkpoint_dict(payload)

    options = cfg["training"]
    position_orders = []
    for epoch in range(options["max_epochs"]):
        order = list(range(len(dataset["positions"])))
        random.Random(seed_for(cfg["seed"], epoch, "positions", namespace=cfg["namespace"])).shuffle(order)
        position_orders.append(order)
    anchor_order = list(range(len(dataset["anchors"])))
    random.Random(seed_for(cfg["seed"], 0, "anchors", namespace=cfg["namespace"])).shuffle(anchor_order)
    return TurnTrainingState(
        mode="turn_preferences_step_kl",
        lr=options["initial_lr"],
        anchor_order=anchor_order,
        metadata={
            "position_orders": position_orders,
            "positions_per_step": options["positions_per_step"],
            "steps_per_epoch": len(dataset["positions"]) // options["positions_per_step"],
        },
    )


def _save_training_state(out, path, state, policy, optimizer, budget, identity):
    state.seconds = budget.elapsed()
    state.metadata.update(
        identity=identity, model=policy.state_dict(), optimizer=optimizer.state_dict(), rng=capture_rng()
    )
    payload = cpu_snapshot(state.to_checkpoint_dict())
    atomic_torch_save(payload, path)
    write_json(
        out / "training.json",
        {
            "status": "training",
            "training_steps": state.steps,
            "attempts": state.attempts,
            "rejected_attempts": state.rejected_attempts,
            "epoch": state.epoch,
            "cursor": state.cursor,
            "seconds": state.seconds,
            "lr": state.lr,
            "last_full_kl": state.last_full_kl,
            "stop_reason": state.stop_reason,
            "learning": state.metadata.get("learning"),
        },
    )
    atomic_write_jsonl(out / "attempts.jsonl", state.history)
    atomic_write_jsonl(out / "steps.jsonl", [record for record in state.history if record["accepted"]])


@dataclass
class _TurnPositions:
    accepted_steps: int
    ids: list[int]
    batch: TurnBatch
    reference: torch.Tensor
    branch_prefixes: dict[HistoryKey, torch.Tensor]


@dataclass
class _AnchorPrefixes:
    accepted_steps: int
    keys: tuple[HistoryKey, ...]
    hidden: dict[HistoryKey, torch.Tensor]

    def matches(self, accepted_steps, keys):
        return self.accepted_steps == accepted_steps and self.keys == keys


@dataclass
class _TurnFitPhases:
    """Keep transient inputs and detached prefixes outside the checkpoint state."""

    policy: torch.nn.Module
    optimizer: torch.optim.Optimizer
    histories: TurnHistories
    state: TurnTrainingState
    preferences: PreparedPreferences
    parent: dict[HistoryKey, torch.Tensor]
    training_refs: shared.DeviceReferences
    control_refs: shared.DeviceReferences
    anchor_subset: AnchorSubset
    all_anchors: Sequence[Sequence[int]]
    options: dict
    device: str
    microbatch: int
    positions: _TurnPositions | None = None
    accepted_prefixes: _AnchorPrefixes | None = None
    pending_prefixes: _AnchorPrefixes | None = None

    def _prepare_positions(self) -> _TurnPositions:
        epoch, step = divmod(self.state.steps, self.state.metadata["steps_per_epoch"])
        size = self.options["positions_per_step"]
        ids = self.state.metadata["position_orders"][epoch][step * size : (step + 1) * size]
        batch = pack_batch(
            [self.preferences.rows[index] for index in ids],
            self.device,
            histories=self.histories,
            branch_steps=self.preferences.branch_steps,
        )
        reference = (
            torch.stack([self.parent[key] for key in batch.keys])
            if batch.keys
            else torch.empty(0, device=self.device)
        )
        needed = [key for position in batch.positions for key in position.boundary_keys if key is not None]
        prefixes = hidden_states_at(
            self.policy, self.histories, needed, branch_steps=self.preferences.branch_steps
        )
        return _TurnPositions(self.state.steps, ids, batch, reference, prefixes)

    def _prefixes_for(self, keys) -> _AnchorPrefixes:
        if self.accepted_prefixes is not None and self.accepted_prefixes.matches(self.state.steps, keys):
            return self.accepted_prefixes
        if self.pending_prefixes is not None and self.pending_prefixes.matches(self.state.steps, keys):
            return self.pending_prefixes
        boundaries = {
            (episode, index): (episode, index - BPTT_WINDOW_LENGTH)
            for episode, index in set(keys)
            if index >= BPTT_WINDOW_LENGTH
        }
        hidden = hidden_states_at(self.policy, self.histories, boundaries.values())
        return _AnchorPrefixes(
            self.state.steps, keys, {key: hidden[boundary] for key, boundary in boundaries.items()}
        )

    def update(self, anchors):
        # A retry retains the accepted-step index and starts from restored weights.
        positions = self.positions
        if positions is None or positions.accepted_steps != self.state.steps:
            self.positions = positions = self._prepare_positions()
        self.accepted_prefixes = self._prefixes_for(tuple(map(tuple, anchors)))
        self.pending_prefixes = None

        metrics = logical_step(
            self.policy,
            self.optimizer,
            self.histories,
            positions.batch,
            anchors,
            positions.reference,
            self.training_refs,
            microbatch=self.microbatch,
            anchor_prefixes=self.accepted_prefixes.hidden,
            branch_prefixes=positions.branch_prefixes,
        )
        return {**metrics, "positions": positions.ids}

    def _next_anchors(self):
        if self.state.steps + 1 >= self.options["max_accepted_steps"]:
            return None
        count = self.options["anchors_per_step"]
        next_start = self.state.anchor_cursor + count
        anchor_count = len(self.all_anchors)
        return tuple(
            tuple(self.all_anchors[self.state.anchor_order[(next_start + index) % anchor_count]])
            for index in range(count)
        )

    def _add_preference_metrics(self, metrics):
        positions = self.positions
        if positions is None or not positions.batch.keys:
            return
        batch = positions.batch
        values = current_chosen(
            self.policy, self.histories, batch.keys, branch_steps=self.preferences.branch_steps
        )
        scores = ((values - positions.reference)[batch.indices] * batch.mask).sum(-1) / batch.mask.sum(
            -1
        ).clamp_min(1)
        margins = batch.signs * (scores[:, 0] - scores[:, 1])
        metrics.update(
            preference_margin_mean=float(margins.mean()),
            winning_execution_preferred=int((margins > 0).sum()),
            informative_comparisons=len(margins),
        )

    def measure(self):
        next_keys = self._next_anchors()
        collected = {} if next_keys is not None else None
        kl, metrics = control_kl(
            self.policy,
            self.histories,
            self.all_anchors,
            self.control_refs,
            anchor_subset=self.anchor_subset,
            collect_prefixes=next_keys,
            collected_prefixes=collected,
        )
        if kl <= self.options["kl_limit"]:
            if next_keys is not None:
                # These prefixes were measured on the candidate now being accepted.
                self.pending_prefixes = _AnchorPrefixes(self.state.steps + 1, next_keys, collected)
            self._add_preference_metrics(metrics)
        else:
            self.pending_prefixes = None
        return kl, metrics


def train(cfg, out, dataset, histories, budget, *, microbatch=2, compile_model=True, monitor=None):
    out = Path(out)
    device = "cuda"
    options = cfg["training"]
    identity = {"dataset": digest(dataset), "config": digest(cfg)}
    path = out / "training.pt"
    state = _load_training_state(path, identity, cfg, dataset)
    if state.stop_reason:
        return publish(cfg, out, state.to_checkpoint_dict(), identity)

    policy = configure_policy(str(ROOT / cfg["parents"][0]), compile_model=compile_model)
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=options["initial_lr"],
        betas=(0.9, 0.999),
        eps=1e-5,
        weight_decay=0,
    )
    preferences = prepare_preferences(histories, dataset["positions"], budget=budget)
    chosen_keys = sorted(histories.targets)
    anchor_subset = prepare_anchor_subset(
        histories, dataset["anchors"], preferences.training_observations, budget=budget
    )
    keys = list(dict.fromkeys([*chosen_keys, *map(tuple, dataset["anchors"])]))
    histories.prepare(keys, device=device, budget=budget)
    parent, training_refs, control_refs = parent_references(
        policy, histories, dataset, chosen_keys, out, cfg, branch_steps=preferences.branch_steps
    )
    if "model" in state.metadata:
        policy.load_state_dict(state.metadata["model"])
        optimizer.load_state_dict(state.metadata["optimizer"])
        restore_rng(state.metadata["rng"])

    controller = {
        **options,
        "cycle_anchors": True,
        "max_accepted_steps": min(
            options["max_accepted_steps"], options["max_epochs"] * state.metadata["steps_per_epoch"]
        ),
    }
    phases = _TurnFitPhases(
        policy,
        optimizer,
        histories,
        state,
        preferences,
        parent,
        training_refs,
        control_refs,
        anchor_subset,
        dataset["anchors"],
        controller,
        device,
        microbatch,
    )
    save = partial(_save_training_state, out, path, state, policy, optimizer, budget, identity)
    try:
        if monitor is not None:
            monitor.mark("training")
        run_updates(
            policy,
            optimizer,
            state,
            dataset["anchors"],
            controller,
            budget,
            save,
            phases.update,
            phases.measure,
        )
    finally:
        save()
    return publish(cfg, out, torch.load(path, map_location="cpu", weights_only=False), identity)


def publish(cfg, out, payload, identity):
    if payload.get("identity", identity) != identity:
        raise ValueError("Candidate identity differs")
    if "model" not in payload:
        parent = torch.load(ROOT / cfg["parents"][0], map_location="cpu", weights_only=False)
        payload.update(model=parent["model"], optimizer={}, identity=identity)
    candidate = shared.publish_candidate(cfg, out, payload, identity)
    report = read_json(Path(out) / "training.json")
    report.update(
        attempts=payload.get("attempts", 0),
        lr=payload.get("lr"),
        rejected_attempts=payload.get("rejected_attempts", 0),
        last_full_kl=payload.get("last_full_kl", 0),
        learning=payload.get("learning"),
        automatic_development=False,
    )
    write_json(Path(out) / "training.json", report)
    return candidate
