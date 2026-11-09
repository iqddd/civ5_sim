"""Reusable page-locked (pinned) host tensor buffers for async CPU-to-GPU pipelines."""

from collections.abc import Callable
import math
import queue
import threading
from typing import Generic, TypeVar

import torch


class PinnedBufferSlot:
    """Reusable flat page-locked host buffers sliced into arbitrary tensor shapes via views.

    Background worker threads call ``reset()`` and ``alloc(...)`` without making any CUDA
    driver calls (safe during concurrent CUDA stream graph capture on the main thread).
    When a requested shape exceeds the current pinned capacity at a cursor position,
    ``alloc(...)`` returns an ordinary unpinned CPU tensor for the current step and
    records a growth request. Once the GPU consumer event for this slot completes, the
    main CUDA thread calls ``fulfill_growth()`` to allocate/expand pinned storage.
    """

    def __init__(
        self,
        *,
        growth_factor: float = 1.25,
        min_elements: int = 4096,
    ):
        if growth_factor < 1.0:
            raise ValueError("growth_factor must be >= 1.0")
        if min_elements < 1:
            raise ValueError("min_elements must be >= 1")
        self.growth_factor = float(growth_factor)
        self.min_elements = int(min_elements)
        self.buffers: dict[tuple[str, torch.dtype], list[torch.Tensor]] = {}
        self.cursors: dict[tuple[str, torch.dtype], int] = {}
        self.grow_requests: dict[tuple[str, torch.dtype, int], int] = {}

    def reset(self) -> None:
        """Reset per-channel allocation cursors before packing a new batch."""
        self.cursors.clear()

    def alloc(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        channel: str = "default",
    ) -> torch.Tensor:
        """Return a tensor of ``shape`` and ``dtype``, reusing pinned storage when sized."""
        numel = math.prod(shape)
        key = (channel, dtype)
        lst = self.buffers.get(key)
        idx = self.cursors.get(key, 0)
        self.cursors[key] = idx + 1
        if lst is not None and idx < len(lst) and lst[idx].numel() >= numel:
            return lst[idx][:numel].view(shape)
        req_key = (channel, dtype, idx)
        target = max(numel, self.min_elements)
        if target > self.grow_requests.get(req_key, 0):
            self.grow_requests[req_key] = target
        return torch.empty(shape, dtype=dtype, pin_memory=False)

    def fulfill_growth(self) -> None:
        """Allocate or expand ``pin_memory=True`` storage on the main CUDA thread."""
        if not self.grow_requests:
            return
        for (channel, dtype, idx), min_numel in sorted(
            self.grow_requests.items(), key=lambda kv: kv[0][2]
        ):
            lst = self.buffers.setdefault((channel, dtype), [])
            alloc_numel = max(min_numel, int(min_numel * self.growth_factor))
            while len(lst) <= idx:
                size = alloc_numel if len(lst) == idx else self.min_elements
                lst.append(torch.empty(size, dtype=dtype, pin_memory=True))
            if lst[idx].numel() < min_numel:
                lst[idx] = torch.empty(alloc_numel, dtype=dtype, pin_memory=True)
        self.grow_requests.clear()

    def pinned_bytes(self) -> int:
        """Total bytes currently held in page-locked host storage for this slot."""
        return sum(
            tensor.numel() * tensor.element_size()
            for lst in self.buffers.values()
            for tensor in lst
        )


SlotT = TypeVar("SlotT", bound=PinnedBufferSlot)


class PinnedSlotPool(Generic[SlotT]):
    """Bounded ring of ``PinnedBufferSlot`` instances shared between CPU producers and GPU."""

    def __init__(
        self,
        size: int,
        *,
        slot_factory: Callable[[], SlotT] = PinnedBufferSlot,  # type: ignore[assignment]
    ):
        if size < 1:
            raise ValueError("Pool size must be >= 1")
        self.size = int(size)
        self._free: queue.Queue[SlotT] = queue.Queue()
        for _ in range(self.size):
            self._free.put(slot_factory())

    def acquire(
        self,
        *,
        stop_event: threading.Event | None = None,
        timeout: float = 0.05,
    ) -> SlotT | None:
        """Acquire and reset a free slot on a producer thread, or return ``None`` if stopped."""
        if stop_event is None:
            slot = self._free.get()
            slot.reset()
            return slot
        while not stop_event.is_set():
            try:
                slot = self._free.get(timeout=timeout)
                slot.reset()
                return slot
            except queue.Empty:
                continue
        return None

    def release(self, slot: SlotT | None, *, fulfill_growth: bool = True) -> None:
        """Fulfill pending pinned allocations on the main CUDA thread and return ``slot``."""
        if slot is None:
            return
        if fulfill_growth:
            slot.fulfill_growth()
        self._free.put(slot)
