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
_COMPILED_RAGGED_GATHER = None
_PACKED_INDEX_KEYS = ("neighbors", "sources", "targets")


def _gather_resident(data, indices, width):
    gathered = {}
    for key, value in data.items():
        if key in _ACTION_TRIM_KEYS:
            value = value[:, :width]
        rows = value.index_select(0, indices)
        gathered[key] = rows.long() if key in _PACKED_INDEX_KEYS else rows
    return gathered


def _gather_ragged(data, indices, width, include_behavior):
    starts, counts = data["action_spans"].index_select(0, indices).unbind(-1)
    columns = torch.arange(width, device=starts.device)
    mask = columns < counts[:, None]
    # Every flat action tensor ends in one shared padding row.
    padding = data["sources"].shape[0] - 1
    actions = torch.where(mask, starts[:, None] + columns, padding)
    gathered = {"action_mask": mask}
    for key, value in data.items():
        if key == "action_spans" or (key == "old_log_probs" and not include_behavior):
            continue
        per_action = key in _ACTION_TRIM_KEYS or key == "old_log_probs"
        rows = value[actions] if per_action else value.index_select(0, indices)
        gathered[key] = rows.long() if key in _PACKED_INDEX_KEYS else rows
    return gathered


def _compiled_gather(*, ragged_actions=False):
    global _COMPILED_GATHER, _COMPILED_RAGGED_GATHER
    if ragged_actions:
        if _COMPILED_RAGGED_GATHER is None:
            _COMPILED_RAGGED_GATHER = torch.compile(_gather_ragged, dynamic=True, fullgraph=True)
        return _COMPILED_RAGGED_GATHER
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
        self._host_index_slots = []
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
        ragged_actions=False,
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
        # Encoded cell indices fit in int16, including the invalid sentinel.
        # Widen only the rows in use to satisfy the model's gather operations.
        template = batch_inputs(flat[:1], "cpu", index_dtype=np.int16)
        rows = len(flat)
        cells = max(len(item.cells) for item in flat)
        actions = int(action_counts.max())
        shapes = {}
        axes = {field.name: field.axis for field in ENCODED_FIELDS}
        axes.update(cell_mask="cells", action_mask="actions")
        if ragged_actions:
            action_offsets = np.cumsum(np.r_[0, action_counts])
            action_spans = np.column_stack((action_offsets[:-1], action_counts))
            template.pop("action_mask")
            template["action_spans"] = torch.from_numpy(action_spans[:1])
            axes["action_spans"] = "fixed"
        for key, value in template.items():
            shape = list(value.shape)
            shape[0] = rows
            if axes[key] == "cells":
                shape[1] = cells
            elif axes[key] == "actions":
                shape[1] = actions
                if ragged_actions:
                    shape = [int(action_offsets[-1]) + 1, *value.shape[2:]]
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
            items = flat[first : first + 256]
            part = batch_inputs(
                items, device, non_blocking=True, index_dtype=np.int16, spatial_only=ragged_actions
            )
            for key, target in data.items():
                if ragged_actions and key in _ACTION_TRIM_KEYS:
                    stop = first + len(items)
                    dtype = np.int16 if key in _PACKED_INDEX_KEYS else np.float32
                    values = np.concatenate([getattr(item, key) for item in items], dtype=dtype)
                    destination = target[action_offsets[first] : action_offsets[stop]]
                    destination.copy_(torch.from_numpy(values), non_blocking=True)
                    continue
                if key == "action_spans":
                    continue
                destination = target[first : first + len(part["cells"])]
                destination.fill_(
                    INVALID_INDEX if key in _PACKED_INDEX_KEYS else 0
                )
                destination[tuple(slice(0, size) for size in part[key].shape)] = part[
                    key
                ]
        if ragged_actions:
            data["action_spans"].copy_(torch.from_numpy(action_spans), non_blocking=True)
            for key in _ACTION_TRIM_KEYS:
                if key in data:
                    data[key][-1].fill_(INVALID_INDEX if key in _PACKED_INDEX_KEYS else 0)
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
        if self.async_transfers:
            size = sum(stop - start for _, start, stop in spans)
            indices, copied = self._host_index_slot(size)
            host = indices.numpy()
            cursor = 0
            for episode, start, stop in spans:
                end = cursor + stop - start
                offset = int(self.offsets[episode])
                host[cursor:end] = np.arange(offset + start, offset + stop, dtype=np.int64)
                cursor = end
            gpu = indices.to(self.device, non_blocking=True)
            if copied is not None:
                copied.record(torch.cuda.current_stream(self.device))
            return gpu
        return torch.tensor(
            [
                i
                for episode_index, start, stop in spans
                for i in range(
                    int(self.offsets[episode_index]) + start,
                    int(self.offsets[episode_index]) + stop,
                )
            ],
            device=self.device,
        )

    def _host_index_slot(self, size):
        # DMA can still be reading a live pinned slot after .to() returns.
        # Reuse only completed slots and fall back to allocator-owned storage
        # when several gathers are in flight, without waiting on the CPU.
        for row, (storage, copied) in enumerate(self._host_index_slots):
            if copied.query():
                if len(storage) < size:
                    storage = torch.empty(1 << (size - 1).bit_length(), dtype=torch.long, pin_memory=True)
                    self._host_index_slots[row] = storage, copied
                return storage[:size], copied
        if len(self._host_index_slots) == 4:
            return torch.empty(size, dtype=torch.long, pin_memory=True), None
        capacity = 1 << max(0, (size - 1).bit_length())
        storage = torch.empty(capacity, dtype=torch.long, pin_memory=True)
        copied = torch.cuda.Event()
        self._host_index_slots.append((storage, copied))
        return storage[:size], copied

    def _gather(self, indices, width, *, include_behavior=True):
        if "action_spans" in self.data:
            runner = self._gather_runner or _gather_ragged
            return runner(self.data, indices, width, include_behavior)
        runner = self._gather_runner or _gather_resident
        return runner(self.data, indices, width)

    def gather_inputs(self, spans, indices=None, *, device=None, min_cells=0, include_behavior=True):
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
        return self._gather(indices, width, include_behavior=include_behavior)

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
        return _gather_resident({key: self.data[key] for key in keys}, indices, None)

    def enable_compiled_gather(self):
        self._gather_runner = _compiled_gather(
            ragged_actions=self.data is not None and "action_spans" in self.data
        )

    def enable_async_transfers(self):
        self.async_transfers = str(self.device).startswith("cuda")


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
