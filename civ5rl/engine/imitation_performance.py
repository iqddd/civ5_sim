"""Optional resident BC inputs and fewer host synchronizations; same CE objective."""

import math
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F

from civ5rl.core.model import batch_inputs
from civ5rl.core.schema import (
    ACTION_DIM,
    GLOBAL_DIM,
    HEX_DIRECTIONS,
    PREV_ACTION_DIM,
    TILE_DIM,
)
from civ5rl.engine.ppo_fast import prefixes
from civ5rl.engine.imitation import decision_indices


class ResidentDemonstrations:
    def __init__(self, base, *, deadline=None, reserve_bytes=2 * 1024**3):
        self.episodes, self.device = base.episodes, base.device
        self.encoded, self.masks = base.encoded, base.masks
        self.offsets = np.cumsum([0, *map(len, self.encoded)])
        flat = [item for episode in self.encoded for item in episode]
        self.action_counts = np.asarray([len(item.sources) for item in flat])
        n, a, rows = (
            max(len(x.cells) for x in flat),
            int(self.action_counts.max()),
            len(flat),
        )
        template = batch_inputs(flat[:1], self.device)
        shapes = {
            "cells": (rows, n, TILE_DIM),
            "cell_mask": (rows, n),
            "neighbors": (rows, n, HEX_DIRECTIONS),
            "globals": (rows, GLOBAL_DIM),
            "previous": (rows, PREV_ACTION_DIM),
            "sources": (rows, a),
            "targets": (rows, a),
            "action_features": (rows, a, ACTION_DIM),
            "action_mask": (rows, a),
        }
        self.required_bytes = sum(
            math.prod(shape) * template[key].element_size()
            for key, shape in shapes.items()
        )
        if str(self.device).startswith("cuda"):
            free, _ = torch.cuda.mem_get_info()
            # Keep space for gradients, spatial activations and allocator fluctuations.
            if self.required_bytes + reserve_bytes > free:
                raise MemoryError(
                    f"Resident input cache requires {self.required_bytes} bytes plus {reserve_bytes} reserve; free={free}"
                )
        self.data = {
            key: torch.empty(shape, dtype=template[key].dtype, device=self.device)
            for key, shape in shapes.items()
        }
        for first in range(0, rows, 256):
            if deadline is not None and perf_counter() >= deadline:
                raise TimeoutError("Resident preparation exceeded optimization budget")
            part = batch_inputs(flat[first : first + 256], self.device)
            count = len(part["cells"])
            for key, target in self.data.items():
                destination = target[first : first + count]
                destination.fill_(
                    -1 if key in ("neighbors", "sources", "targets") else 0
                )
                destination[tuple(slice(0, size) for size in part[key].shape)] = part[
                    key
                ]
        self.labels = {
            "indices": torch.tensor(
                [decision_indices(r)[0] for ep in self.episodes for r in ep["decisions"]],
                device=self.device,
            ),
            "mask": torch.tensor(
                [r["supervised"] for ep in self.episodes for r in ep["decisions"]],
                dtype=torch.bool,
                device=self.device,
            ),
        }
        self._gather_runner = None
        self.async_transfers = False

    def enable_compiled_gather(self):
        self._gather_runner = torch.compile(self._gather, dynamic=True, fullgraph=True)

    def enable_async_transfers(self):
        self.async_transfers = True

    def _gather(self, indices, width):
        data = {key: value.index_select(0, indices) for key, value in self.data.items()}
        for key in ("sources", "targets", "action_features", "action_mask"):
            data[key] = data[key][:, :width].contiguous()
        return data, {
            key: value.index_select(0, indices) for key, value in self.labels.items()
        }

    def chunks(self):
        return [
            (i, a, min(a + 128, len(ep)))
            for i, ep in enumerate(self.encoded)
            for a in range(0, len(ep), 128)
        ]

    def batch(self, spans):
        indices = torch.tensor(
            [
                i
                for ep, a, b in spans
                for i in range(int(self.offsets[ep]) + a, int(self.offsets[ep]) + b)
            ],
            device="cpu" if self.async_transfers else self.device,
        )
        if self.async_transfers:
            indices = indices.pin_memory().to(self.device, non_blocking=True)
        width = max(
            int(
                self.action_counts[
                    int(self.offsets[ep]) + a : int(self.offsets[ep]) + b
                ].max()
            )
            for ep, a, b in spans
        )
        data, labels = (self._gather_runner or self._gather)(indices, width)
        return (
            data,
            labels,
            [b - a for _, a, b in spans],
        )


def backward_minibatch_async(policy, cache, spans, count, micro_size=8):
    """Same accumulation; return GPU loss and check finiteness without host waiting."""
    if policy.spatial_cuda_graphs:
        torch.compiler.cudagraph_mark_step_begin()
    hidden = prefixes(policy, cache, spans)
    total = torch.zeros((), device=cache.device)
    parameters = (
        tuple(p for p in policy.parameters() if p.requires_grad)
        if policy.coalesced_accumulation else None
    )
    accumulated = None
    for first in range(0, len(spans), micro_size):
        part = spans[first : first + micro_size]
        if not any(any(cache.masks[ep][a:b]) for ep, a, b in part):
            continue
        batch, labels, lengths = cache.batch(part)
        logits, _, _ = policy.forward_sequences(
            batch, lengths, hidden[:, first : first + len(part)].contiguous()
        )
        losses = F.cross_entropy(logits, labels["indices"], reduction="none")
        loss = (losses * labels["mask"]).sum()
        if parameters is None:
            (loss / count).backward()
        else:
            gradients = torch.autograd.grad(loss / count, parameters)
            if accumulated is None:
                accumulated = list(gradients)
                for parameter, gradient in zip(parameters, accumulated):
                    parameter.grad = gradient
            else:
                torch._foreach_add_(accumulated, gradients)
        total.add_(loss.detach())
    result = total / count
    torch._assert_async(torch.isfinite(result), "Non-finite BC loss")
    return result


def backward_minibatch_fast(policy, cache, spans, count, micro_size=8):
    result = float(backward_minibatch_async(policy, cache, spans, count, micro_size))
    if not math.isfinite(result):
        raise FloatingPointError("Non-finite BC loss")
    return result
