"""Shared storage and packing of recurrent decision sequences."""

from contextlib import contextmanager
from time import perf_counter

import numpy as np
import torch

from civ5rl.engine.budget import deadline_reached
from civ5rl.core.encoded_layout import ENCODED_FIELDS
from civ5rl.core.model import batch_inputs
from civ5rl.core.schema import INVALID_INDEX
from civ5rl.core.spatial import SPATIAL_KEYS
from civ5rl.engine.constants import DEFAULT_GPU_RESERVE_BYTES, DEFAULT_PREFIX_BLOCK_SIZE
from civ5rl.engine.spans import Span


def cuda_memory_snapshot(device):
    """Distinguish driver-free memory from live tensors and allocator caches."""
    free, total = torch.cuda.mem_get_info(device)
    stats = torch.cuda.memory_stats(device)
    return {
        "driver_free_bytes": free,
        "device_total_bytes": total,
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "active_bytes": stats.get("active_bytes.all.current"),
        "inactive_split_bytes": stats.get("inactive_split_bytes.all.current"),
    }


@contextmanager
def allocation_with_reserve(device, reserve_bytes):
    """Allocate real tensors in the body, then probe allocator headroom.

    The caching allocator, not driver-free telemetry, decides availability.
    A single contiguous reserve probe is conservative admission headroom, not a
    guarantee for future training. No cache flush or allocator setting changes.
    Callers must discard partially built containers on a genuine CUDA OOM.
    """
    if not str(device).startswith("cuda"):
        yield None
        return
    result = {
        "before": cuda_memory_snapshot(device),
        "reserve_bytes": reserve_bytes,
        "reserve_policy": "contiguous_allocator_probe_after_data",
    }
    stage = "data_allocation"
    try:
        yield result
        result["after_data"] = cuda_memory_snapshot(device)
        stage = "reserve_probe"
        if reserve_bytes:
            probe = torch.empty(reserve_bytes, dtype=torch.uint8, device=device)
            del probe
        result["after_probe"] = cuda_memory_snapshot(device)
        result["status"] = "admitted"
    except torch.cuda.OutOfMemoryError as error:
        result.update(status="out_of_memory", stage=stage)
        result["at_failure"] = cuda_memory_snapshot(device)
        error.add_note(f"CUDA allocation admission: {result}")
        raise


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
        self.memory_admission = None
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
            if deadline_reached(deadline):
                raise TimeoutError("Resident preparation exceeded optimization budget")
            data = {}
            try:
                with allocation_with_reserve(device, reserve_bytes) as admission:
                    for key, value in template.items():
                        data[key] = torch.empty(shapes[key], dtype=value.dtype, device=device)
            except torch.cuda.OutOfMemoryError:
                data.clear()
                raise
            self.data = data
            self.memory_admission = admission
            for first in range(0, rows, 256):
                if deadline_reached(deadline):
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
        for key in ("sources", "targets", "action_features", "action_mask", "attack_preview"):
            data[key] = data[key][:, :width].contiguous()
        return data

    def gather_inputs(self, spans, indices=None, *, device=None, min_cells=0):
        if self.data is None:
            return batch_inputs(
                [
                    item
                    for episode_index, start, stop in spans
                    for item in self.encoded[episode_index][start:stop]
                ],
                self.device if device is None else device,
                non_blocking=True,
                min_cells=min_cells,
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

    def gather_spatial_inputs(self, spans):
        """Recurrent prefixes need neither action tensors nor PPO labels."""
        if self.data is None:
            return batch_inputs(
                [
                    item
                    for episode, start, stop in spans
                    for item in self.encoded[episode][start:stop]
                ],
                self.device,
                non_blocking=True,
                spatial_only=True,
            )
        indices = self.indices(spans)
        return {key: self.data[key].index_select(0, indices) for key in SPATIAL_KEYS}

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
