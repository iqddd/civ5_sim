"""Shared storage and packing of recurrent decision sequences."""

from time import perf_counter

import numpy as np
import torch

from civ5rl.core.encoded_layout import ENCODED_FIELDS
from civ5rl.core.model import batch_inputs
from civ5rl.core.schema import INVALID_INDEX
from civ5rl.engine.constants import DEFAULT_GPU_RESERVE_BYTES, DEFAULT_PREFIX_BLOCK_SIZE
from civ5rl.engine.spans import Span


class SequenceBuffer:
    def __init__(
        self,
        encoded,
        device,
        *,
        resident=False,
        deadline=None,
        reserve_bytes=DEFAULT_GPU_RESERVE_BYTES,
    ):
        self.encoded, self.device = encoded, device
        self.offsets = np.cumsum([0, *map(len, encoded)])
        flat = [item for episode in encoded for item in episode]
        self.action_counts = np.asarray([len(item.sources) for item in flat])
        self.data = None
        self.required_bytes = 0
        if resident:
            # Derive shapes and dtypes from the canonical batching implementation.
            template = batch_inputs(flat[:1], "cpu")
            rows = len(flat)
            cells = max(len(x.cells) for x in flat)
            actions = int(self.action_counts.max())
            shapes = {}
            axes = {field.name: field.axis for field in ENCODED_FIELDS}
            axes.update(cell_mask="cells", action_mask="actions")
            for key, value in template.items():
                shape = list(value.shape)
                shape[0] = rows
                if axes[key] == "cells":
                    shape[1] = cells
                elif axes[key] == "actions":
                    shape[1] = actions
                shapes[key] = shape
            self.required_bytes = sum(
                int(np.prod(shapes[k])) * v.element_size() for k, v in template.items()
            )
            if str(device).startswith("cuda"):
                free, _ = torch.cuda.mem_get_info(device)
                if self.required_bytes + reserve_bytes > free:
                    raise MemoryError(
                        f"Resident input cache requires {self.required_bytes} bytes plus {reserve_bytes} reserve; free={free}"
                    )
            if deadline is not None and perf_counter() >= deadline:
                raise TimeoutError("Resident preparation exceeded optimization budget")
            self.data = {
                k: torch.empty(shapes[k], dtype=v.dtype, device=device)
                for k, v in template.items()
            }
            for first in range(0, rows, 256):
                if deadline is not None and perf_counter() >= deadline:
                    raise TimeoutError(
                        "Resident preparation exceeded optimization budget"
                    )
                part = batch_inputs(
                    flat[first : first + 256], device, non_blocking=True
                )
                for key, target in self.data.items():
                    destination = target[first : first + len(part["cells"])]
                    destination.fill_(
                        INVALID_INDEX
                        if key in ("neighbors", "sources", "targets")
                        else 0
                    )
                    destination[tuple(slice(0, size) for size in part[key].shape)] = (
                        part[key]
                    )
        self.async_transfers = False
        self._gather_runner = None

    def chunks(self, size=DEFAULT_PREFIX_BLOCK_SIZE):
        return [
            Span(episode_index, start, min(start + size, len(episode)))
            for episode_index, episode in enumerate(self.encoded)
            for start in range(0, len(episode), size)
        ]

    def indices(self, spans):
        indices = torch.tensor(
            [
                i
                for episode_index, start, stop in spans
                for i in range(
                    int(self.offsets[episode_index]) + start,
                    int(self.offsets[episode_index]) + stop,
                )
            ],
            device="cpu" if self.async_transfers else self.device,
        )
        if self.async_transfers:
            indices = indices.pin_memory().to(self.device, non_blocking=True)
        return indices

    def _gather(self, indices, width):
        data = {key: value.index_select(0, indices) for key, value in self.data.items()}
        for key in ("sources", "targets", "action_features", "action_mask"):
            data[key] = data[key][:, :width].contiguous()
        return data

    def gather_inputs(self, spans, indices=None):
        if self.data is None:
            return batch_inputs(
                [
                    item
                    for episode_index, start, stop in spans
                    for item in self.encoded[episode_index][start:stop]
                ],
                self.device,
                non_blocking=True,
            )
        if indices is None:
            indices = self.indices(spans)
        width = max(
            int(
                self.action_counts[
                    int(self.offsets[episode_index]) + start : int(
                        self.offsets[episode_index]
                    )
                    + stop
                ].max()
            )
            for episode_index, start, stop in spans
        )
        return (self._gather_runner or self._gather)(indices, width)

    def enable_compiled_gather(self):
        self._gather_runner = torch.compile(self._gather, dynamic=True, fullgraph=True)

    def enable_async_transfers(self):
        self.async_transfers = True


def take_counted_span(episode, start, stop, masks, limit):
    """Return the end offset and count, stopping immediately at the limit."""
    if masks is None:
        end = min(stop, start + limit)
        return end, end - start
    count = 0
    end = start
    while end < stop and count < limit:
        count += int(masks[episode][end])
        end += 1
    return end, count


def pack_minibatches(order, masks, size):
    """Pack counted decisions while keeping unlabelled recurrent context in spans."""
    batch, count = [], 0
    for episode, start, stop in order:
        while start < stop:
            end, used = take_counted_span(episode, start, stop, masks, size - count)
            if used:
                batch.append(Span(episode, start, end))
            count += used
            start = end
            if count == size:
                yield batch, count
                batch, count = [], 0
    if count:
        yield batch, count
