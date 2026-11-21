"""Behavior cloning of public command labels; independent from PPO rollout math."""

from __future__ import annotations

from collections import Counter, defaultdict

import torch
from torch.nn import functional as F

from civ5rl.core.observation import ObservationEncoder
from civ5rl.core.schema import MIN_FREE_ACTIONS
from civ5rl.core.variants import PolicyVariant
from civ5rl.engine.runtime import is_int_at_least
from civ5rl.engine.runtime.sequence import (
    DEFAULT_BC_MICROBATCH_SIZE,
    SequenceBuffer,
    pack_minibatches,
    prefixes,
)
from civ5rl.engine.runtime.transactions import GradientAccumulator
from civ5sim import Action


def decision_indices(row):
    """CE follows advice; recurrent history follows the actual command."""
    if "teacher_action" in row or "executed_action" in row:
        teacher, executed = row["teacher_action"], row["executed_action"]
    else:
        teacher = executed = row["chosen"]
    if any(not is_int_at_least(i, 0) or i >= len(row["actions"]) for i in (teacher, executed)) or row[
        "supervised"
    ] != (len(row["actions"]) >= MIN_FREE_ACTIONS):
        raise ValueError("Invalid demonstration indices or CE mask")
    return teacher, executed


def encode_demonstration(episode, variant=PolicyVariant.CONTACTS_GRU):
    encoder = ObservationEncoder(episode["context"], variant)
    encoded = []
    for row in episode["decisions"]:
        legal = tuple(Action(**a) for a in row["actions"])
        _, executed = decision_indices(row)
        encoded.append(encoder.encode(row["observation"], legal))
        encoder.remember_action(legal[executed], row["observation"])
    return encoded


class DemonstrationCache(SequenceBuffer):
    def __init__(self, episodes, device):
        self.episodes, self.device = episodes, device
        encoded = [encode_demonstration(episode) for episode in episodes]
        super().__init__(device=device, **self._pack_encoded(encoded, device))
        self.masks = [[r["supervised"] for r in ep["decisions"]] for ep in episodes]

    def batch(self, spans):
        rows = [r for ep, a, b in spans for r in self.episodes[ep]["decisions"][a:b]]
        labels = {
            "indices": torch.tensor([decision_indices(r)[0] for r in rows], device=self.device),
            "mask": torch.tensor([r["supervised"] for r in rows], device=self.device, dtype=torch.bool),
        }
        return self.gather_inputs(spans), labels, [b - a for _, a, b in spans]


def label_minibatches(order, masks, size=1024):
    """Pack labelled decisions; forced states remain inside recurrent fragments."""
    yield from pack_minibatches(order, masks, size)


def ce_sum(logits, labels):
    mask = labels["mask"]
    if not mask.any():
        raise ValueError("BC loss requires at least one supervised decision")
    return F.cross_entropy(logits[mask], labels["indices"][mask], reduction="sum")


def masked_ce_sum(logits, labels):
    """Apply the supervision mask without transferring it to the host."""
    losses = F.cross_entropy(logits, labels["indices"], reduction="none")
    return (losses * labels["mask"]).sum()


def _hidden_slice(hidden, first, size):
    return None if hidden is None else hidden[:, first : first + size].contiguous()


def backward_minibatch_async(policy, cache, spans, count, micro_size=DEFAULT_BC_MICROBATCH_SIZE):
    """Same accumulation; return GPU loss and check finiteness without host waiting."""
    if policy.spatial_cuda_graphs:
        torch.compiler.cudagraph_mark_step_begin()
    hidden = prefixes(policy, cache, spans)
    total = torch.zeros((), device=cache.device)
    accumulator = GradientAccumulator(policy.parameters(), coalesced=policy.coalesced_accumulation)
    for first in range(0, len(spans), micro_size):
        part = spans[first : first + micro_size]
        if not any(any(cache.masks[ep][a:b]) for ep, a, b in part):
            continue
        batch, labels, lengths = cache.batch(part)
        if policy.spatial_cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        logits, _, _ = policy.forward_sequences(batch, lengths, _hidden_slice(hidden, first, len(part)))
        loss = masked_ce_sum(logits, labels)
        accumulator.backward(loss / count)
        total.add_(loss.detach())
    result = total / count
    torch._assert_async(torch.isfinite(result), "Non-finite BC loss")
    return result


def backward_minibatch(policy, cache, spans, count, micro_size=DEFAULT_BC_MICROBATCH_SIZE):
    if policy.coalesced_accumulation:
        return float(backward_minibatch_async(policy, cache, spans, count, micro_size))
    if policy.spatial_cuda_graphs:
        torch.compiler.cudagraph_mark_step_begin()
    hidden = prefixes(policy, cache, spans)
    total = 0.0
    accumulator = GradientAccumulator(policy.parameters(), coalesced=False)
    for first in range(0, len(spans), micro_size):
        part = spans[first : first + micro_size]
        batch, labels, lengths = cache.batch(part)
        if not labels["mask"].any():
            continue
        logits, _, _ = policy.forward_sequences(batch, lengths, _hidden_slice(hidden, first, len(part)))
        loss = ce_sum(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite BC loss")
        accumulator.backward(loss / count)
        total += loss.detach().item()
    return total / count


@torch.no_grad()
def teacher_metrics(
    policy, cache, *, micro_size=DEFAULT_BC_MICROBATCH_SIZE, by_unit_type=False, move_details=False
):
    """Score the teacher's labels over the executed public command history."""
    losses, correct, counts = defaultdict(float), Counter(), Counter()
    units, units_correct = Counter(), Counter()
    movement = Counter()
    for spans, _ in label_minibatches(cache.chunks(), cache.masks):
        if policy.spatial_cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        hidden = prefixes(policy, cache, spans)
        for first in range(0, len(spans), micro_size):
            part = spans[first : first + micro_size]
            batch, labels, lengths = cache.batch(part)
            logits, _, _ = policy.forward_sequences(batch, lengths, _hidden_slice(hidden, first, len(part)))
            losses_per_row = F.cross_entropy(logits, labels["indices"], reduction="none").cpu().tolist()
            predictions = logits.argmax(-1).cpu().tolist()
            rows = (
                row
                for episode, start, stop in part
                for row in cache.episodes[episode]["decisions"][start:stop]
            )
            for row, loss, predicted in zip(rows, losses_per_row, predictions, strict=True):
                if not row["supervised"]:
                    continue
                target, _ = decision_indices(row)
                action = row["actions"][target]
                kind = action["kind"]
                matched = predicted == target
                counts[kind] += 1
                losses[kind] += loss
                correct[kind] += matched
                if by_unit_type:
                    unit_types = {unit["id"]: unit["type"] for unit in row["observation"]["units"]}
                    unit_type = unit_types.get(action.get("unit"), "no_unit")
                    units[unit_type] += 1
                    units_correct[unit_type] += matched
                if move_details and kind == "move":
                    selected = row["actions"][predicted]
                    movement["count"] += 1
                    movement["type"] += selected["kind"] == "move"
                    movement["type_unit"] += selected["kind"] == "move" and selected["unit"] == action["unit"]
                    movement["exact"] += selected == action
    count = sum(counts.values())
    result = {
        "labels": count,
        "ce": sum(losses.values()) / count if count else None,
        "accuracy": sum(correct.values()) / count if count else None,
        "by_kind": {
            kind: {
                "labels": counts[kind],
                "ce": losses[kind] / counts[kind],
                "accuracy": correct[kind] / counts[kind],
            }
            for kind in counts
        },
    }
    if by_unit_type:
        result["by_unit_type"] = {
            unit_type: {
                "labels": unit_count,
                "accuracy": units_correct[unit_type] / unit_count,
            }
            for unit_type, unit_count in units.items()
        }
    if move_details:
        result["move"] = {
            "denominator": movement["count"],
            **{
                name: movement[name] / movement["count"] if movement["count"] else None
                for name in ("type", "type_unit", "exact")
            },
        }
    return result
