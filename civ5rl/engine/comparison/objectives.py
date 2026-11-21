"""Batched policy objectives and full-history KL for rollout comparisons.

Targets share spatial encodings within each map-sized microbatch. Each target
keeps its own truncated GRU graph, initialized from a no-grad prefix computed
with the current weights. No activations survive an Adam step.
"""

from __future__ import annotations

from functools import lru_cache

import torch
from torch.nn import functional as F

from civ5rl.core.distributions import masked_kl_rows
from civ5rl.core.model import batch_inputs
from civ5rl.engine.comparison import PREFERENCE_TEMPERATURE
from civ5rl.engine.comparison.policy import step_comparison_policy
from civ5rl.engine.comparison.reference import (
    HISTORY_BATCH,
    DeviceReferences,
    encode_histories,
    history_groups,
    reference_logits_batches,
)
from civ5rl.engine.evaluation.runtime import evaluation_precision


def pair_loss(logits, better, worse, weight):
    return weight * F.softplus((logits[worse] - logits[better]) / PREFERENCE_TEMPERATURE)


def categorical_kl(reference, logits, *, double=False):
    # Both inputs contain only the target's real legal actions (no padding).
    current = (logits / PREFERENCE_TEMPERATURE).log_softmax(-1)
    if double:
        reference, current = reference.double(), current.double()
    return masked_kl_rows(reference, current)


@torch.no_grad()
def prefix_states(policy, entries):
    """One current-weight no-grad GRU prefix per episode, scoped to one Adam step."""
    episodes = {}
    result = {}
    device = next(policy.parameters()).device
    zero = torch.zeros(1, 1, policy.hidden_size, device=device)
    for key, history in entries:
        boundary = max(0, key[1] + 1 - 128)
        if not boundary:
            result[key] = zero
            continue
        item = episodes.setdefault(key[0], {"history": history, "targets": []})
        item["targets"].append((key, boundary))
    groups = {}
    for episode, item in episodes.items():
        item["episode"] = episode
        groups.setdefault(len(item["history"][0].cells), []).append(item)
    for group in groups.values():
        for first in range(0, len(group), HISTORY_BATCH):
            part = group[first : first + HISTORY_BATCH]
            sequences = [p["history"][: max(b for _, b in p["targets"])] for p in part]
            _, context = encode_histories(policy, sequences)
            lengths = [len(seq) for seq in sequences]
            h, _ = policy.sequence_memory(context, lengths)
            keys = []
            indices = []
            offset = 0
            for item, length in zip(part, lengths):
                for key, boundary in item["targets"]:
                    keys.append(key)
                    indices.append(offset + boundary - 1)
                offset += length
            selected = h.index_select(0, torch.tensor(indices, device=device))
            result.update((key, selected[i : i + 1].unsqueeze(0)) for i, key in enumerate(keys))
    return result


def _window_features(policy, entries, boundaries):
    """Encode the union of overlapping spatial windows once, without sharing GRU graphs."""
    grouped = {}
    for (key, history), start in zip(entries, boundaries):
        item = grouped.setdefault(key[0], {"history": history, "intervals": []})
        item["intervals"].append((start, key[1] + 1))
    segments = []
    lookup = {}
    for episode, item in grouped.items():
        merged = []
        for start, stop in sorted(item["intervals"]):
            if merged and start <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
            else:
                merged.append((start, stop))
        lookup[episode] = []
        for start, stop in merged:
            lookup[episode].append((start, stop, len(segments)))
            segments.append(item["history"][start:stop])
    x, context = encode_histories(policy, segments)
    lengths = [len(seq) for seq in segments]
    xs = x.split(lengths)
    contexts = context.split(lengths)
    windows = []
    targets = []
    for (key, _), boundary in zip(entries, boundaries):
        start, _, slot = next(row for row in lookup[key[0]] if row[0] <= boundary and key[1] < row[1])
        windows.append(contexts[slot][boundary - start : key[1] + 1 - start])
        targets.append(xs[slot][key[1] - start])
    return torch.stack(targets), torch.cat(windows, 0)


def batched_target_logits(policy, entries, *, gradient=True, padded=False, prefixes=None):
    """Independent current-weight histories; detach each one's own BPTT boundary."""
    keys, histories = zip(*entries)
    indices = [key[1] for key in keys]
    boundaries = [max(0, i + 1 - 128) if gradient else 0 for i in indices]
    hidden = None
    if gradient:
        if prefixes is None:
            prefixes = prefix_states(policy, entries)
        hidden = torch.cat([prefixes[key] for key in keys], 1)
    with torch.set_grad_enabled(gradient):
        x, context = _window_features(policy, entries, boundaries)
        lengths = [i + 1 - b for i, b in zip(indices, boundaries)]
        h, _ = policy.sequence_memory(context, lengths, hidden)
        ends = torch.tensor([sum(lengths[: i + 1]) - 1 for i in range(len(lengths))], device=x.device)
        targets = [history[index] for history, index in zip(histories, indices)]
        batch = batch_inputs(targets, x.device, actions_only=True)
        logits, _ = policy.score_heads(batch, x, h[ends])
        if padded:
            return logits
        return [row[: len(target.sources)] for row, target in zip(logits, targets)]


def full_kl(policy, histories, anchors, references):
    """Reduce FP32 log-softmax differences in FP64 on device; read one scalar.

    Episode histories are encoded once per pass, with bounded spatial blocks.
    Duplicate anchors retain their multiplicity in both numerator and denominator.
    """
    device = next(policy.parameters()).device
    if (
        not isinstance(references, DeviceReferences)
        or references.probabilities is None
        or references.flat.device != device
    ):
        references = DeviceReferences(references, device, control=True)
    multiplicity = {}
    for key in map(tuple, anchors):
        multiplicity[key] = multiplicity.get(key, 0) + 1
    previous_amp, previous_batch = policy.amp_enabled, policy.spatial_batch_size
    policy.enable_amp(False)
    try:
        total = torch.zeros((), device=device, dtype=torch.float64)
        with evaluation_precision():
            for part, logits in reference_logits_batches(policy, histories, anchors):
                keys = [key for key, _ in part]
                parent, probabilities = references.gather(keys, logits.shape[1])
                values = control_runner(logits.is_cuda)(logits, parent, probabilities)
                counts = torch.tensor(
                    [multiplicity[key] for key in keys],
                    device=device,
                    dtype=torch.float64,
                )
                total += (values * counts).sum()
        return float(total / len(anchors))
    finally:
        policy.enable_amp(previous_amp, spatial_batch_size=previous_batch)


def logical_step(policy, optimizer, histories, pairs, anchor_keys, references):
    """Accumulate batched preference/KL losses, then clip and apply one Adam step.

    CUDA gradient finiteness is asserted before Adam. Model/Adam finiteness and
    loss metrics share the post-step host readback; no finite check is skipped.
    """
    optimizer.zero_grad(set_to_none=True)
    pair_total = torch.zeros((), device=next(policy.parameters()).device)
    kl_total = torch.zeros_like(pair_total)
    pair_rows = {}
    anchor_counts = {}
    for row in pairs:
        pair_rows.setdefault((row["episode"], row["index"]), []).append(row)
    for key in map(tuple, anchor_keys):
        anchor_counts[key] = anchor_counts.get(key, 0) + 1
    keys = list(dict.fromkeys([*pair_rows, *anchor_counts]))
    prefixes = prefix_states(policy, [(key, histories.get(key[0])[0]) for key in keys])
    if not isinstance(references, DeviceReferences):
        references = DeviceReferences(references, pair_total.device)
    if any(key not in references for key in anchor_counts):
        raise ValueError("Missing anchor reference")
    for entries in history_groups(histories, keys):
        logits = batched_target_logits(policy, entries, padded=True, prefixes=prefixes)
        pair_indices = []
        pair_weights = []
        weights = []
        for index, (key, _) in enumerate(entries):
            for row in pair_rows.get(key, ()):
                pair_indices.append((index, row["better"], row["worse"]))
                pair_weights.append(row["weight"] / len(pairs))
            weights.append(anchor_counts.get(key, 0) / len(anchor_keys) if anchor_keys else 0.0)
        parent, _ = references.gather([key for key, _ in entries], logits.shape[1], allow_missing=True)
        indices = torch.tensor(pair_indices, device=logits.device, dtype=torch.long).reshape(-1, 3)
        factors = torch.tensor(pair_weights, device=logits.device, dtype=logits.dtype)
        anchor_weights = torch.tensor(weights, device=logits.device, dtype=logits.dtype)
        pair, kl = loss_runner(logits.is_cuda)(logits, indices, factors, parent, anchor_weights)
        pair_total += pair.detach()
        kl_total += kl.detach()
        (pair + kl).backward()
    return step_comparison_policy(
        policy, optimizer, pair_total, kl_total, gradient_error="Non-finite gradient before Adam"
    )


def preference_and_kl(logits, pair_indices, pair_weights, reference, anchor_weights):
    # Rows with no pair are omitted; duplicates retain every original weight.
    pair_logits = logits.index_select(0, pair_indices[:, 0])
    better = pair_logits.gather(1, pair_indices[:, 1:2]).squeeze(1)
    worse = pair_logits.gather(1, pair_indices[:, 2:3]).squeeze(1)
    pair = (pair_weights * F.softplus((worse - better) / PREFERENCE_TEMPERATURE)).sum()
    logq = (logits / PREFERENCE_TEMPERATURE).log_softmax(-1)
    kl = (masked_kl_rows(reference, logq, ~torch.isneginf(logits)) * anchor_weights).sum()
    return pair, kl


def full_kl_rows(logits, reference, probabilities):
    # Retain FP32 log-softmax, followed by the established FP64 subtraction/exp.
    # Padding is -inf in logits, but must contribute zero without 0*inf NaNs.
    logq = (logits / PREFERENCE_TEMPERATURE).log_softmax(-1).double()
    return masked_kl_rows(reference, logq, ~torch.isneginf(logits), probabilities=probabilities)


@lru_cache(maxsize=2)
def loss_runner(cuda):
    return (
        torch.compile(preference_and_kl, dynamic=True, fullgraph=True, mode="default")
        if cuda
        else preference_and_kl
    )


@lru_cache(maxsize=2)
def control_runner(cuda):
    return torch.compile(full_kl_rows, dynamic=True, fullgraph=True, mode="default") if cuda else full_kl_rows
