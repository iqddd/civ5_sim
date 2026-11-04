"""Behavior cloning of public command labels; independent from PPO rollout math."""

from collections import Counter, defaultdict

import torch
from torch.nn import functional as F

from civ5rl.core.model import batch_inputs
from civ5rl.core.observation import ObservationEncoder
from civ5rl.engine.ppo_fast import prefixes
from civ5sim import Action


def encode_demonstration(episode):
    encoder = ObservationEncoder(episode["context"], "C")
    encoded = []
    for row in episode["decisions"]:
        legal = tuple(Action(**a) for a in row["actions"])
        index = row["chosen"]
        if not 0 <= index < len(legal) or row["supervised"] != (len(legal) >= 2):
            raise ValueError("Invalid demonstration label or CE mask")
        encoded.append(encoder.encode(row["observation"], legal))
        encoder.remember_action(legal[index], row["observation"])
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
            "indices": torch.tensor([r["chosen"] for r in rows], device=self.device),
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
def teacher_metrics(policy, cache):
    losses, correct, counts = defaultdict(float), Counter(), Counter()
    for spans, _ in label_minibatches(cache.chunks(), cache.masks):
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
                    kind = row["actions"][row["chosen"]]["kind"]
                    counts[kind] += 1
                    losses[kind] += loss[j].item()
                    correct[kind] += int(predicted[j].item() == row["chosen"])
    n = sum(counts.values())
    return {
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
