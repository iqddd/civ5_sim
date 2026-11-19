"""Shared storage and packing of recurrent decision sequences."""

from contextlib import contextmanager
from typing import NamedTuple, Protocol

import numpy as np
import torch

from civ5rl.core.encoded_layout import ENCODED_FIELDS
from civ5rl.core.model import batch_inputs
from civ5rl.core.schema import INVALID_INDEX
from civ5rl.core.spatial import SPATIAL_KEYS
from civ5rl.engine.runtime.budget import deadline_reached


class Span(NamedTuple):
    episode: int
    start: int
    stop: int

    @property
    def length(self) -> int:
        return self.stop - self.start


DEFAULT_PREFIX_BLOCK_SIZE = 128
DEFAULT_BC_MICROBATCH_SIZE = 8
DEFAULT_GPU_RESERVE_BYTES = 2 * 1024**3
DEFAULT_CPU_CACHE_BYTES = 4 * 1024**3
DEFAULT_PINNED_REPLAY_BYTES = 1024**3
PREPARED_VIEW_BYTES_PER_DECISION = 2048
PREPARED_FIXED_OVERHEAD_BYTES = 4096


class SequenceCache(Protocol):
    """Ordered episode spans, shared by replay and demonstration caches."""

    device: str | torch.device

    def chunks(self, size: int = DEFAULT_PREFIX_BLOCK_SIZE) -> list[Span]: ...

    def batch(self, spans: list[Span]) -> tuple[dict, dict, list[int]]: ...

    def gather_spatial_inputs(self, spans: list[Span]) -> dict: ...


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


_ACTION_TRIM_KEYS = (
    "sources",
    "targets",
    "action_features",
    "action_mask",
    "attack_preview",
)
_COMPILED_GATHER = None


def _gather_resident(data, indices, width):
    gathered = {key: value.index_select(0, indices) for key, value in data.items()}
    for key in _ACTION_TRIM_KEYS:
        if key in gathered:
            gathered[key] = gathered[key][:, :width].contiguous()
    return gathered


def _compiled_gather():
    global _COMPILED_GATHER
    if _COMPILED_GATHER is None:
        _COMPILED_GATHER = torch.compile(
            _gather_resident, dynamic=True, fullgraph=True
        )
    return _COMPILED_GATHER


class SequenceBuffer:
    """Pack episode offsets, action counts and inputs for BC/PPO caches.

    Resident inputs require successful real allocation plus a temporary 2-GiB
    allocator reserve probe. Optional whole-replay caches use a 4-GiB probe and
    can fall back to host/on-demand data without changing KL; mandatory resident
    allocation failures propagate. Admission snapshots are telemetry, not a
    promise of enough memory for all subsequent optimizer work. Ragged gathers
    pad legal actions only to the current batch width.
    """

    def __init__(
        self,
        data,
        offsets,
        action_counts,
        *,
        device,
        encoded=None,
        required_bytes=None,
        memory_admission=None,
    ):
        self.data = data
        self.offsets = offsets
        self.action_counts = action_counts
        self.device = device
        self.encoded = encoded
        self.memory_admission = memory_admission
        self.required_bytes = (
            sum(
                tensor.numel() * tensor.element_size()
                for tensor in (data or {}).values()
            )
            if required_bytes is None
            else required_bytes
        )
        self.async_transfers = False
        self._gather_runner = None

    @classmethod
    def from_spatial_tensors(cls, data, device):
        """Reuse gathering for immutable prepacked spatial inputs (no labels/actions)."""
        return cls(data, np.asarray([0, len(data["cells"])]), None, device=device)

    @classmethod
    def from_encoded(cls, encoded, device, **options):
        """Construct a sequence buffer from complete encoded episodes."""
        return cls(device=device, **cls._pack_encoded(encoded, device, **options))

    @staticmethod
    def _pack_encoded(
        encoded,
        device,
        *,
        resident=False,
        deadline=None,
        reserve_bytes=DEFAULT_GPU_RESERVE_BYTES,
    ):
        flat = [item for episode in encoded for item in episode]
        action_counts = np.asarray([len(item.sources) for item in flat])
        packed = {
            "data": None,
            "encoded": encoded,
            "offsets": np.cumsum([0, *map(len, encoded)]),
            "action_counts": action_counts,
        }
        if not resident:
            return packed

        # Derive shapes and dtypes from the canonical batching implementation.
        template = batch_inputs(flat[:1], "cpu")
        rows = len(flat)
        cells = max(len(item.cells) for item in flat)
        actions = int(action_counts.max())
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
        required_bytes = sum(
            int(np.prod(shapes[key])) * value.element_size()
            for key, value in template.items()
        )
        if deadline_reached(deadline):
            raise TimeoutError("Resident preparation exceeded optimization budget")
        data = {}
        try:
            with allocation_with_reserve(device, reserve_bytes) as admission:
                for key, value in template.items():
                    data[key] = torch.empty(
                        shapes[key], dtype=value.dtype, device=device
                    )
        except torch.cuda.OutOfMemoryError:
            data.clear()
            raise
        for first in range(0, rows, 256):
            if deadline_reached(deadline):
                raise TimeoutError("Resident preparation exceeded optimization budget")
            part = batch_inputs(flat[first : first + 256], device, non_blocking=True)
            for key, target in data.items():
                destination = target[first : first + len(part["cells"])]
                destination.fill_(
                    INVALID_INDEX if key in ("neighbors", "sources", "targets") else 0
                )
                destination[tuple(slice(0, size) for size in part[key].shape)] = part[
                    key
                ]
        packed.update(
            data=data, required_bytes=required_bytes, memory_admission=admission
        )
        return packed

    def chunks(self, size=DEFAULT_PREFIX_BLOCK_SIZE):
        return [
            Span(episode, start, min(start + size, length))
            for episode, length in enumerate(np.diff(self.offsets).tolist())
            for start in range(0, length, size)
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
        return _gather_resident(self.data, indices, width)

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
        if self._gather_runner is not None:
            return self._gather_runner(self.data, indices, width)
        return _gather_resident(self.data, indices, width)

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
        keys = SPATIAL_KEYS + (("unit_indices",) if "unit_indices" in self.data else ())
        if self.action_counts is None and len(spans) == 1:
            episode, start, stop = spans[0]
            offset = int(self.offsets[episode])
            return {key: self.data[key][offset + start : offset + stop] for key in keys}
        indices = self.indices(spans)
        return {key: self.data[key].index_select(0, indices) for key in keys}

    def enable_compiled_gather(self):
        self._gather_runner = _compiled_gather()

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
