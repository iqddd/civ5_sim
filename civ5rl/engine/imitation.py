"""Behavior cloning of public command labels; independent from PPO rollout math."""

from collections import Counter, defaultdict

import torch
from torch.nn import functional as F

from civ5rl.core.observation import ObservationEncoder
from civ5rl.core.schema import MIN_FREE_ACTIONS
from civ5rl.core.variants import PolicyVariant
from civ5rl.engine.constants import DEFAULT_BC_MICROBATCH_SIZE
from civ5rl.engine.ppo_fast import prefixes
from civ5rl.engine.sequence import SequenceBuffer, pack_minibatches
from civ5rl.engine.training import GradientAccumulator
from civ5sim import Action


def decision_indices(row):
    """CE follows advice; recurrent history follows the actual command."""
    if "teacher_action" in row or "executed_action" in row:
        teacher, executed = row["teacher_action"], row["executed_action"]
    else:
        teacher = executed = row["chosen"]
    if any(
        type(i) is not int or not 0 <= i < len(row["actions"])
        for i in (teacher, executed)
    ) or row["supervised"] != (len(row["actions"]) >= MIN_FREE_ACTIONS):
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
        super().__init__([encode_demonstration(ep) for ep in episodes], device)
        self.masks = [[r["supervised"] for r in ep["decisions"]] for ep in episodes]

    def batch(self, spans):
        rows = [r for ep, a, b in spans for r in self.episodes[ep]["decisions"][a:b]]
        labels = {
            "indices": torch.tensor(
                [decision_indices(r)[0] for r in rows], device=self.device
            ),
            "mask": torch.tensor(
                [r["supervised"] for r in rows], device=self.device, dtype=torch.bool
            ),
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


def backward_minibatch(
    policy, cache, spans, count, micro_size=DEFAULT_BC_MICROBATCH_SIZE
):
    if policy.coalesced_accumulation:
        from civ5rl.engine.imitation_performance import backward_minibatch_async

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
        logits, _, _ = policy.forward_sequences(
            batch, lengths, hidden[:, first : first + len(part)].contiguous()
        )
        loss = ce_sum(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite BC loss")
        accumulator.backward(loss / count)
        total += loss.detach().item()
    return total / count


@torch.no_grad()
def teacher_metrics(policy, cache, *, move_details=False):
    losses, correct, counts = defaultdict(float), Counter(), Counter()
    movement = Counter()
    for spans, _ in label_minibatches(cache.chunks(), cache.masks):
        if policy.spatial_cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        hidden = prefixes(policy, cache, spans)
        for first in range(0, len(spans), DEFAULT_BC_MICROBATCH_SIZE):
            part = spans[first : first + DEFAULT_BC_MICROBATCH_SIZE]
            batch, labels, lengths = cache.batch(part)
            logits, _, _ = policy.forward_sequences(
                batch, lengths, hidden[:, first : first + len(part)].contiguous()
            )
            loss = F.cross_entropy(logits, labels["indices"], reduction="none")
            predicted = logits.argmax(-1)
            rows = [
                r for ep, a, b in part for r in cache.episodes[ep]["decisions"][a:b]
            ]
            for j, row in enumerate(rows):
                if row["supervised"]:
                    target, _ = decision_indices(row)
                    kind = row["actions"][target]["kind"]
                    counts[kind] += 1
                    losses[kind] += loss[j].item()
                    correct[kind] += int(predicted[j].item() == target)
                    if move_details and kind == "move":
                        actual = row["actions"][target]
                        selected = row["actions"][int(predicted[j].item())]
                        movement["count"] += 1
                        movement["type"] += selected["kind"] == "move"
                        movement["type_unit"] += (
                            selected["kind"] == "move"
                            and selected["unit"] == actual["unit"]
                        )
                        movement["exact"] += selected == actual
    n = sum(counts.values())
    result = {
        "labels": n,
        "ce": sum(losses.values()) / n if n else None,
        "accuracy": sum(correct.values()) / n if n else None,
        "by_kind": {
            kind: {
                "labels": counts[kind],
                "ce": losses[kind] / counts[kind],
                "accuracy": correct[kind] / counts[kind],
            }
            for kind in counts
        },
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
