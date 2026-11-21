"""Optional resident BC inputs and overlapped pinned-memory BC training pipeline."""

from __future__ import annotations

import json
import math
import os
import queue
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from time import perf_counter
from typing import NamedTuple

import torch
from torch.profiler import record_function

from civ5rl.engine.imitation.core import backward_minibatch_async as backward_minibatch_async
from civ5rl.engine.imitation.core import decision_indices, masked_ce_sum
from civ5rl.engine.imitation.storage import PreparedDemonstrations
from civ5rl.engine.runtime.pinned import PinnedBufferSlot, PinnedSlotPool
from civ5rl.engine.runtime.sequence import (
    DEFAULT_BC_MICROBATCH_SIZE,
    DEFAULT_GPU_RESERVE_BYTES,
    DEFAULT_PREFIX_BLOCK_SIZE,
    SequenceBuffer,
    Span,
)
from civ5rl.engine.runtime.transactions import GradientAccumulator, bc_optimizer_step


class ResidentDemonstrations(SequenceBuffer):
    def __init__(self, base, *, deadline=None, reserve_bytes=DEFAULT_GPU_RESERVE_BYTES):
        self.episodes, self.masks = base.episodes, base.masks
        super().__init__(
            device=base.device,
            **self._pack_encoded(
                base.encoded,
                base.device,
                resident=True,
                deadline=deadline,
                reserve_bytes=reserve_bytes,
            ),
        )
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

    def batch(self, spans):
        indices = self.indices(spans)
        return (
            self.gather_inputs(spans, indices),
            {k: v.index_select(0, indices) for k, v in self.labels.items()},
            [b - a for _, a, b in spans],
        )


def backward_minibatch_fast(policy, cache, spans, count, micro_size=DEFAULT_BC_MICROBATCH_SIZE):
    result = float(backward_minibatch_async(policy, cache, spans, count, micro_size))
    if not math.isfinite(result):
        raise FloatingPointError("Non-finite BC loss")
    return result


class ImitationPinnedBatchSlot(PinnedBufferSlot):
    """Pinned buffer slot separating variable-count prefix blocks from microbatches."""

    def alloc_prefix(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return self.alloc(shape, dtype, channel="prefix")

    def alloc_micro(self, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return self.alloc(shape, dtype, channel="micro")


class _PreparedMinibatchCPU(NamedTuple):
    spans_len: int
    label_count: int
    prefix_blocks: list
    microbatches: list
    prefix_positions: int
    history_positions: int
    slot: ImitationPinnedBatchSlot | None


class _StagedMinibatchGPU(NamedTuple):
    spans_len: int
    label_count: int
    prefix_blocks: list
    microbatches: list
    h2d_ready: torch.cuda.Event


class PendingBCStep(NamedTuple):
    metrics_buffer_index: int
    pinned_slot: ImitationPinnedBatchSlot | None
    done_event: torch.cuda.Event
    label_count: int
    prefix_positions: int
    history_positions: int
    started_at: float


_STREAM_END = object()


class _ProducerError(NamedTuple):
    exception: BaseException


def _prepare_minibatch_cpu(
    cache,
    spans,
    count: int,
    *,
    is_recurrent: bool,
    micro_size: int,
    cell_padding: int,
    slot: ImitationPinnedBatchSlot | None = None,
    block_size: int = DEFAULT_PREFIX_BLOCK_SIZE,
) -> _PreparedMinibatchCPU:
    if slot is not None:
        slot.reset()
    alloc_prefix = slot.alloc_prefix if slot is not None else None
    alloc_micro = slot.alloc_micro if slot is not None else None
    prefix_blocks = []
    prefix_pos = 0
    if is_recurrent:
        max_start = max((int(start) for _, start, _ in spans), default=0)
        for offset in range(0, max_start, block_size):
            active = [
                (slot_idx, int(episode), int(start))
                for slot_idx, (episode, start, _) in enumerate(spans)
                if int(start) > offset
            ]
            subspans = [
                Span(episode, offset, min(start, offset + block_size)) for _, episode, start in active
            ]
            cpu_spatial, _, lengths = cache.pack_cpu_spans(
                subspans,
                spatial_only=True,
                pin=(slot is None),
                cell_padding=cell_padding,
                alloc=alloc_prefix,
            )
            if len(active) == len(spans):
                slots_cpu = None
            elif alloc_prefix is not None:
                slots_cpu = alloc_prefix((len(active),), torch.long)
                slots_cpu.numpy()[:] = [s for s, _, _ in active]
            else:
                slots_cpu = torch.tensor(
                    [s for s, _, _ in active],
                    dtype=torch.long,
                    pin_memory=False,
                )
            prefix_pos += sum(lengths)
            prefix_blocks.append((slots_cpu, cpu_spatial, lengths))

    microbatches = []
    history_pos = 0
    for first in range(0, len(spans), micro_size):
        part = spans[first : first + micro_size]
        if not any(any(cache.masks[ep][a:b]) for ep, a, b in part):
            continue
        cpu_batch, cpu_labels, lengths = cache.pack_cpu_spans(
            part,
            spatial_only=False,
            pin=(slot is None),
            cell_padding=cell_padding,
            alloc=alloc_micro,
        )
        history_pos += sum(lengths)
        microbatches.append((first, len(part), cpu_batch, cpu_labels, lengths))

    return _PreparedMinibatchCPU(
        spans_len=len(spans),
        label_count=int(count),
        prefix_blocks=prefix_blocks,
        microbatches=microbatches,
        prefix_positions=prefix_pos,
        history_positions=history_pos,
        slot=slot,
    )


def _stage_minibatch_to_gpu(
    item: _PreparedMinibatchCPU,
    device: torch.device,
    h2d_stream: torch.cuda.Stream,
    compute_stream: torch.cuda.Stream,
) -> _StagedMinibatchGPU:
    with torch.cuda.stream(h2d_stream):
        gpu_prefixes = []
        for slots_cpu, cpu_spatial, lengths in item.prefix_blocks:
            slots_gpu = None
            if slots_cpu is not None:
                slots_gpu = slots_cpu.to(device, non_blocking=True)
                slots_gpu.record_stream(compute_stream)
            gpu_spatial = {}
            for k, v in cpu_spatial.items():
                gv = v.to(device, non_blocking=True)
                gv.record_stream(compute_stream)
                gpu_spatial[k] = gv
            gpu_prefixes.append((slots_gpu, gpu_spatial, lengths))

        gpu_micros = []
        for first, part_len, cpu_batch, cpu_labels, lengths in item.microbatches:
            gpu_batch = {}
            for k, v in cpu_batch.items():
                gv = v.to(device, non_blocking=True)
                gv.record_stream(compute_stream)
                gpu_batch[k] = gv
            gpu_labels = {}
            for k, v in cpu_labels.items():
                gv = v.to(device, non_blocking=True)
                gv.record_stream(compute_stream)
                gpu_labels[k] = gv
            gpu_micros.append((first, part_len, gpu_batch, gpu_labels, lengths))

        h2d_ready = torch.cuda.Event()
        h2d_ready.record(h2d_stream)

    return _StagedMinibatchGPU(
        spans_len=item.spans_len,
        label_count=item.label_count,
        prefix_blocks=gpu_prefixes,
        microbatches=gpu_micros,
        h2d_ready=h2d_ready,
    )


def _enqueue_gpu_step(
    *,
    policy,
    optimizer,
    staged: _StagedMinibatchGPU,
    device: torch.device,
    max_gradient_norm: float,
    compute_stream: torch.cuda.Stream,
    metrics_cpu_buf: torch.Tensor,
    annotate: bool,
) -> torch.cuda.Event:
    compute_stream.wait_event(staged.h2d_ready)
    optimizer.zero_grad(set_to_none=True)

    if policy.spatial_cuda_graphs:
        torch.compiler.cudagraph_mark_step_begin()

    if policy.is_recurrent:
        with (
            record_function("bc::prefixes") if annotate else nullcontext(),
            torch.no_grad(),
        ):
            hidden = torch.zeros((1, staged.spans_len, policy.hidden_size), device=device)
            for slots_gpu, gpu_spatial, lengths in staged.prefix_blocks:
                if policy.spatial_cuda_graphs:
                    torch.compiler.cudagraph_mark_step_begin()
                if slots_gpu is None:
                    hidden = policy.forward_sequences(gpu_spatial, lengths, hidden, memory_only=True)
                else:
                    last = policy.forward_sequences(
                        gpu_spatial,
                        lengths,
                        hidden.index_select(1, slots_gpu),
                        memory_only=True,
                    )
                    hidden.index_copy_(1, slots_gpu, last)
    else:
        hidden = None

    total = torch.zeros((), device=device)
    accumulator = GradientAccumulator(policy.parameters(), coalesced=policy.coalesced_accumulation)
    for first, part_len, gpu_batch, gpu_labels, lengths in staged.microbatches:
        if policy.spatial_cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        h_slice = None if hidden is None else hidden[:, first : first + part_len].contiguous()
        logits, _, _ = policy.forward_sequences(gpu_batch, lengths, h_slice)
        loss = masked_ce_sum(logits, gpu_labels)
        accumulator.backward(loss / staged.label_count)
        total.add_(loss.detach())

    ce = total / staged.label_count
    torch._assert_async(torch.isfinite(ce), "Non-finite BC loss")
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_gradient_norm, error_if_nonfinite=False)
    torch._assert_async(torch.isfinite(norm), "Non-finite gradient")
    optimizer.step()

    metrics_gpu = torch.stack((ce.detach().double(), norm.detach().double()))
    metrics_cpu_buf.copy_(metrics_gpu, non_blocking=True)
    done_event = torch.cuda.Event()
    done_event.record(compute_stream)
    return done_event


class _BCStepJournal:
    def __init__(self, epoch, initial_steps, compilation, steps_log_path, active_seconds, on_step_completed):
        self.epoch = epoch
        self.steps = initial_steps
        self.compilation = compilation
        self.steps_log_path = steps_log_path
        self.active_seconds = active_seconds
        self.on_step_completed = on_step_completed
        self.records = []

    def publish(
        self,
        labels,
        ce,
        norm,
        started_at,
        completed_at,
        prefix_positions,
        history_positions,
        *,
        log_handle=None,
    ):
        self.steps += 1
        record = {
            "step": self.steps,
            "epoch": self.epoch,
            "labels": int(labels),
            "ce": ce,
            "gradient_norm": norm,
            "seconds": completed_at - started_at,
            "prefix_positions": prefix_positions,
            "history_positions": history_positions,
            "compile_seconds": (
                self.compilation.seconds(started_at, completed_at) if self.compilation is not None else 0.0
            ),
        }
        self.records.append(record)
        if self.active_seconds is not None:
            record["active_seconds"] = self.active_seconds()
        if log_handle is not None:
            log_handle.write(json.dumps(record) + "\n")
            log_handle.flush()
        elif self.steps_log_path is not None:
            with self.steps_log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")
        if self.on_step_completed is not None:
            self.on_step_completed(record, self.steps)


def _stop_before_step(steps, max_steps, should_stop):
    return (max_steps is not None and steps >= max_steps) or (
        should_stop is not None and should_stop() is not None
    )


def _require_finite_step(ce, norm):
    if not math.isfinite(ce) or not math.isfinite(norm):
        raise FloatingPointError("Non-finite BC loss or gradient norm")


def _run_synchronous_minibatches(
    policy, optimizer, cache, batches, journal, *, micro_size, max_gradient_norm, max_steps, should_stop
):
    def backward(policy, cache, spans, count):
        return backward_minibatch_async(policy, cache, spans, count, micro_size)

    for spans, count in batches:
        if _stop_before_step(journal.steps, max_steps, should_stop):
            break
        started_at = perf_counter()
        before = dict(cache.counters)
        ce, norm = bc_optimizer_step(
            policy,
            optimizer,
            backward,
            cache,
            spans,
            count,
            max_gradient_norm=max_gradient_norm,
            asynchronous=False,
        )
        ce, norm = float(ce), float(norm)
        _require_finite_step(ce, norm)
        journal.publish(
            count,
            ce,
            norm,
            started_at,
            perf_counter(),
            cache.counters["prefix_positions"] - before.get("prefix_positions", 0),
            cache.counters["history_positions"] - before.get("history_positions", 0),
        )


class _PreparedBatchPrefetch:
    """The producer owns each slot until it publishes the packed minibatch."""

    def __init__(self, cache, batches, *, is_recurrent, micro_size, cell_padding):
        self.cache = cache
        self.batches = batches
        self.is_recurrent = is_recurrent
        self.micro_size = micro_size
        self.cell_padding = cell_padding
        self.queue: queue.Queue = queue.Queue(maxsize=2)
        self.slot_pool = PinnedSlotPool(5, slot_factory=ImitationPinnedBatchSlot)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._produce, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join(timeout=5.0)

    def _publish(self, item):
        while not self.stop_event.is_set():
            try:
                self.queue.put(item, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def get(self):
        item = self.queue.get()
        if isinstance(item, _ProducerError):
            raise item.exception
        return None if item is _STREAM_END else item

    def _produce(self):
        upcoming = deque()
        io_workers = min(4, max(2, (os.cpu_count() or 4) // 2))
        try:
            batches = iter(self.batches)
            with ThreadPoolExecutor(max_workers=io_workers) as io_pool:
                while not self.stop_event.is_set():
                    while len(upcoming) < 4:
                        try:
                            upcoming.append(next(batches))
                        except StopIteration:
                            break
                    if not upcoming:
                        break
                    self.cache.preload_episodes(
                        (episode for spans, _ in upcoming for episode, _, _ in spans),
                        io_pool,
                    )
                    slot = self.slot_pool.acquire(stop_event=self.stop_event)
                    if slot is None:
                        return
                    spans, count = upcoming.popleft()
                    prepared = _prepare_minibatch_cpu(
                        self.cache,
                        spans,
                        count,
                        is_recurrent=self.is_recurrent,
                        micro_size=self.micro_size,
                        cell_padding=self.cell_padding,
                        slot=slot,
                    )
                    if not self._publish(prepared):
                        return
                self._publish(_STREAM_END)
        except BaseException as error:  # noqa: BLE001 -- propagate failure to the consumer thread
            self._publish(_ProducerError(error))
        finally:
            self.cache._inflight.clear()


class _CudaBCPipeline:
    """Enqueue transfers and optimizer work, then drain one pending metrics readback."""

    def __init__(
        self,
        policy,
        optimizer,
        cache,
        device,
        journal,
        prefetch,
        *,
        max_gradient_norm,
        check_counter_steps,
        annotate,
        log_handle,
    ):
        self.policy = policy
        self.optimizer = optimizer
        self.cache = cache
        self.device = device
        self.journal = journal
        self.prefetch = prefetch
        self.max_gradient_norm = max_gradient_norm
        self.check_steps = set(check_counter_steps)
        self.annotate = annotate
        self.log_handle = log_handle

        self.compute_stream = torch.cuda.current_stream(device)
        self.h2d_stream = torch.cuda.Stream(device=device)
        self.metric_buffers = [torch.empty(2, dtype=torch.float64, pin_memory=True) for _ in range(2)]
        self.metric_index = 0
        self.pending: PendingBCStep | None = None
        self.last_wall_anchor = perf_counter()
        self.queued_steps = journal.steps

    def _verify_adam_steps(self, expected_steps):
        counters = {int(state["step"]) for state in self.optimizer.state.values()}
        if counters != {expected_steps}:
            raise ValueError("BC Adam counters differ from completed steps")

    def _complete(self, step, *, verify_counter):
        if step is None:
            return
        with record_function("bc::sync_and_bookkeeping") if self.annotate else nullcontext():
            step.done_event.synchronize()
        self.prefetch.slot_pool.release(step.pinned_slot)
        completed_at = perf_counter()
        ce, norm = self.metric_buffers[step.metrics_buffer_index].tolist()
        _require_finite_step(ce, norm)
        if verify_counter:
            self._verify_adam_steps(self.journal.steps + 1)
        self.cache.counters["prefix_positions"] += step.prefix_positions
        self.cache.counters["history_positions"] += step.history_positions
        self.last_wall_anchor = completed_at
        self.journal.publish(
            step.label_count,
            ce,
            norm,
            step.started_at,
            completed_at,
            step.prefix_positions,
            step.history_positions,
            log_handle=self.log_handle,
        )

    def _enqueue(self, item):
        with record_function("bc::h2d_stage") if self.annotate else nullcontext():
            staged = _stage_minibatch_to_gpu(item, self.device, self.h2d_stream, self.compute_stream)

        # A boundary callback must read this Adam clock before the next update.
        previous = self.pending
        if previous is not None and self.journal.steps + 1 in self.check_steps:
            self.pending = None
            self._complete(previous, verify_counter=True)
            previous = None

        metric_index = self.metric_index
        self.metric_index = 1 - metric_index
        with (
            record_function("bc::step_total") if self.annotate else nullcontext(),
            record_function("bc::optimizer_step") if self.annotate else nullcontext(),
        ):
            done_event = _enqueue_gpu_step(
                policy=self.policy,
                optimizer=self.optimizer,
                staged=staged,
                device=self.device,
                max_gradient_norm=self.max_gradient_norm,
                compute_stream=self.compute_stream,
                metrics_cpu_buf=self.metric_buffers[metric_index],
                annotate=self.annotate,
            )
        del staged
        # Keep the new pinned slot alive if the previous step's callback fails.
        self.pending = PendingBCStep(
            metrics_buffer_index=metric_index,
            pinned_slot=item.slot,
            done_event=done_event,
            label_count=item.label_count,
            prefix_positions=item.prefix_positions,
            history_positions=item.history_positions,
            started_at=self.last_wall_anchor,
        )
        if previous is not None:
            self._complete(previous, verify_counter=False)
            self.pending = self.pending._replace(started_at=self.last_wall_anchor)

    def run(self, max_steps, should_stop):
        initial_steps = self.journal.steps
        while True:
            if _stop_before_step(self.queued_steps, max_steps, should_stop):
                self.prefetch.stop_event.set()
                break
            item = self.prefetch.get()
            if item is None:
                break
            self.queued_steps += 1
            self._enqueue(item)
            del item

        self._complete(self.pending, verify_counter=self.journal.steps + 1 in self.check_steps)
        self.pending = None
        torch.cuda.synchronize(self.device)
        if self.journal.steps > initial_steps:
            self._verify_adam_steps(self.journal.steps)


def run_bc_minibatch_stream(
    *,
    policy,
    optimizer,
    cache,
    batches,
    micro_size=DEFAULT_BC_MICROBATCH_SIZE,
    max_gradient_norm=0.5,
    steps_log_path=None,
    compilation=None,
    annotate=False,
    epoch=1,
    initial_steps=0,
    max_steps=None,
    should_stop=None,
    on_step_completed=None,
    check_counter_steps=(),
    active_seconds=None,
):
    """Execute minibatches, overlapping prepared CPU input work with CUDA updates.

    Step N+1 is enqueued before step N's callback, except at the synchronous
    boundaries in ``check_counter_steps``. Only boundary callbacks can safely
    inspect or save the model or optimizer state for a specific step.
    """
    journal = _BCStepJournal(
        epoch, initial_steps, compilation, steps_log_path, active_seconds, on_step_completed
    )
    device = torch.device(cache.device)
    if device.type != "cuda" or not isinstance(cache, PreparedDemonstrations):
        _run_synchronous_minibatches(
            policy,
            optimizer,
            cache,
            batches,
            journal,
            micro_size=micro_size,
            max_gradient_norm=max_gradient_norm,
            max_steps=max_steps,
            should_stop=should_stop,
        )
        return journal.records
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    prefetch = _PreparedBatchPrefetch(
        cache,
        batches,
        is_recurrent=bool(policy.is_recurrent),
        micro_size=micro_size,
        cell_padding=int(getattr(policy, "spatial_cell_padding", 0) or 0),
    )
    log = nullcontext() if steps_log_path is None else steps_log_path.open("a", encoding="utf-8")
    with log as log_handle:
        pipeline = _CudaBCPipeline(
            policy,
            optimizer,
            cache,
            device,
            journal,
            prefetch,
            max_gradient_norm=max_gradient_norm,
            check_counter_steps=check_counter_steps,
            annotate=annotate,
            log_handle=log_handle,
        )
        with prefetch:
            pipeline.run(max_steps, should_stop)
    return journal.records
