"""Behavior cloning of public command labels; independent from PPO rollout math."""

from collections import Counter, defaultdict

import torch
from torch.nn import functional as F

from civ5rl.core.model import batch_inputs
from civ5rl.core.observation import ObservationEncoder
from civ5rl.engine.ppo_fast import prefixes
from civ5sim import Action


def decision_indices(row):
    """CE follows advice; recurrent history follows the actual command."""
    if "teacher_action" in row or "executed_action" in row:
        teacher, executed = row["teacher_action"], row["executed_action"]
    else:
        teacher = executed = row["chosen"]
    if (
        any(type(i) is not int or not 0 <= i < len(row["actions"]) for i in (teacher, executed))
        or row["supervised"] != (len(row["actions"]) >= 2)
    ):
        raise ValueError("Invalid demonstration indices or CE mask")
    return teacher, executed


def encode_demonstration(episode):
    encoder = ObservationEncoder(episode["context"], "C")
    encoded = []
    for row in episode["decisions"]:
        legal = tuple(Action(**a) for a in row["actions"])
        _, executed = decision_indices(row)
        encoded.append(encoder.encode(row["observation"], legal))
        encoder.remember_action(legal[executed], row["observation"])
    return encoded


class DemonstrationCache:
    def __init__(self, episodes, device):
        self.episodes, self.device = episodes, device
        self.encoded = [encode_demonstration(ep) for ep in episodes]
        self.masks = [[r["supervised"] for r in ep["decisions"]] for ep in episodes]

    def batch(self, spans):
        encoded = [x for ep, a, b in spans for x in self.encoded[ep][a:b]]
        rows = [r for ep, a, b in spans for r in self.episodes[ep]["decisions"][a:b]]
        labels = {
            "indices": torch.tensor([decision_indices(r)[0] for r in rows], device=self.device),
            "mask": torch.tensor(
                [r["supervised"] for r in rows], device=self.device, dtype=torch.bool
            ),
        }
        return batch_inputs(encoded, self.device), labels, [b - a for _, a, b in spans]

    def chunks(self):
        return [
            (i, a, min(a + 128, len(encoded)))
            for i, encoded in enumerate(self.encoded)
            for a in range(0, len(encoded), 128)
        ]


def label_minibatches(order, masks, size=1024):
    """Pack labelled decisions; forced states remain inside recurrent fragments."""
    batch, count = [], 0
    for episode, start, stop in order:
        piece = start
        for j in range(start, stop):
            count += int(masks[episode][j])
            if count == size:
                batch.append((episode, piece, j + 1))
                yield batch, count
                batch, count, piece = [], 0, j + 1
        if piece < stop and any(masks[episode][piece:stop]):
            batch.append((episode, piece, stop))
    if count:
        yield batch, count


def ce_sum(logits, labels):
    mask = labels["mask"]
    if mask.any():
        return F.cross_entropy(logits[mask], labels["indices"][mask], reduction="sum")
    return (
        logits[:, 0].sum() * 0
    )  # caller skips backward for entirely forced microbatches


def backward_minibatch(policy, cache, spans, count, micro_size=8):
    if policy.coalesced_accumulation:
        from civ5rl.engine.imitation_performance import backward_minibatch_async

        return float(backward_minibatch_async(policy, cache, spans, count, micro_size))
    if policy.spatial_cuda_graphs:
        torch.compiler.cudagraph_mark_step_begin()
    hidden = prefixes(policy, cache, spans)
    total = 0.0
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
        (loss / count).backward()
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
        for first in range(0, len(spans), 8):
            part = spans[first : first + 8]
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
