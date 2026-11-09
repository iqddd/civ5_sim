"""Optional resident BC inputs and overlapped pinned-memory BC training pipeline."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
import queue
import threading
from time import perf_counter
from typing import NamedTuple

import torch
from torch.nn import functional as F
from torch.profiler import record_function

from civ5rl.engine.constants import (
    DEFAULT_BC_MICROBATCH_SIZE,
    DEFAULT_GPU_RESERVE_BYTES,
    DEFAULT_PREFIX_BLOCK_SIZE,
)
from civ5rl.engine.imitation import decision_indices
from civ5rl.engine.pinned_pool import PinnedBufferSlot, PinnedSlotPool
from civ5rl.engine.ppo_fast import prefixes
from civ5rl.engine.sequence import SequenceBuffer
from civ5rl.engine.spans import Span
from civ5rl.engine.training import GradientAccumulator, bc_optimizer_step


class ResidentDemonstrations(SequenceBuffer):
    def __init__(self, base, *, deadline=None, reserve_bytes=DEFAULT_GPU_RESERVE_BYTES):
        self.episodes, self.masks = base.episodes, base.masks
        super().__init__(
            base.encoded,
            base.device,
            resident=True,
            deadline=deadline,
            reserve_bytes=reserve_bytes,
        )
        self.labels = {
            "indices": torch.tensor(
                [
                    decision_indices(r)[0]
                    for ep in self.episodes
                    for r in ep["decisions"]
                ],
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


def backward_minibatch_async(
    policy, cache, spans, count, micro_size=DEFAULT_BC_MICROBATCH_SIZE
):
    """Same accumulation; return GPU loss and check finiteness without host waiting."""
    if policy.spatial_cuda_graphs:
        torch.compiler.cudagraph_mark_step_begin()
    hidden = prefixes(policy, cache, spans)
    total = torch.zeros((), device=cache.device)
    accumulator = GradientAccumulator(
        policy.parameters(), coalesced=policy.coalesced_accumulation
    )
    for first in range(0, len(spans), micro_size):
        part = spans[first : first + micro_size]
        if not any(any(cache.masks[ep][a:b]) for ep, a, b in part):
            continue
        batch, labels, lengths = cache.batch(part)
        if policy.spatial_cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        logits, _, _ = policy.forward_sequences(
            batch, lengths, hidden[:, first : first + len(part)].contiguous()
        )
        losses = F.cross_entropy(logits, labels["indices"], reduction="none")
        loss = (losses * labels["mask"]).sum()
        accumulator.backward(loss / count)
        total.add_(loss.detach())
    result = total / count
    torch._assert_async(torch.isfinite(result), "Non-finite BC loss")
    return result


def backward_minibatch_fast(
    policy, cache, spans, count, micro_size=DEFAULT_BC_MICROBATCH_SIZE
):
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
                Span(episode, offset, min(start, offset + block_size))
                for _, episode, start in active
            ]
            cpu_spatial, _, lengths = cache._pack_cpu_spans(
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
        cpu_batch, cpu_labels, lengths = cache._pack_cpu_spans(
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
        if annotate:
            with record_function("bc::prefixes"):
                with torch.no_grad():
                    hidden = torch.zeros(
                        (1, staged.spans_len, policy.hidden_size), device=device
                    )
                    for slots_gpu, gpu_spatial, lengths in staged.prefix_blocks:
                        if policy.spatial_cuda_graphs:
                            torch.compiler.cudagraph_mark_step_begin()
                        if slots_gpu is None:
                            hidden = policy.forward_sequences(
                                gpu_spatial, lengths, hidden, memory_only=True
                            )
                        else:
                            last = policy.forward_sequences(
                                gpu_spatial,
                                lengths,
                                hidden.index_select(1, slots_gpu),
                                memory_only=True,
                            )
                            hidden.index_copy_(1, slots_gpu, last)
        else:
            with torch.no_grad():
                hidden = torch.zeros(
                    (1, staged.spans_len, policy.hidden_size), device=device
                )
                for slots_gpu, gpu_spatial, lengths in staged.prefix_blocks:
                    if policy.spatial_cuda_graphs:
                        torch.compiler.cudagraph_mark_step_begin()
                    if slots_gpu is None:
                        hidden = policy.forward_sequences(
                            gpu_spatial, lengths, hidden, memory_only=True
                        )
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
    accumulator = GradientAccumulator(
        policy.parameters(), coalesced=policy.coalesced_accumulation
    )
    for first, part_len, gpu_batch, gpu_labels, lengths in staged.microbatches:
        if policy.spatial_cuda_graphs:
            torch.compiler.cudagraph_mark_step_begin()
        h_slice = (
            None
            if hidden is None
            else hidden[:, first : first + part_len].contiguous()
        )
        logits, _, _ = policy.forward_sequences(gpu_batch, lengths, h_slice)
        losses = F.cross_entropy(logits, gpu_labels["indices"], reduction="none")
        loss = (losses * gpu_labels["mask"]).sum()
        accumulator.backward(loss / staged.label_count)
        total.add_(loss.detach())

    ce = total / staged.label_count
    torch._assert_async(torch.isfinite(ce), "Non-finite BC loss")
    norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), max_gradient_norm, error_if_nonfinite=False
    )
    torch._assert_async(torch.isfinite(norm), "Non-finite gradient")
    optimizer.step()

    metrics_gpu = torch.stack((ce.detach().double(), norm.detach().double()))
    metrics_cpu_buf.copy_(metrics_gpu, non_blocking=True)
    done_event = torch.cuda.Event()
    done_event.record(compute_stream)
    return done_event


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
):
    """Execute BC minibatches with background .npz decompression, pinned H2D stream, and CPU/GPU overlap."""
    device = torch.device(cache.device)
    if device.type != "cuda" or not hasattr(cache, "_pack_cpu_spans"):
        def backward(p, c, s, n):
            return backward_minibatch_async(p, c, s, n, micro_size)

        records = []
        steps = initial_steps
        for spans, n in batches:
            if max_steps is not None and steps >= max_steps:
                break
            if should_stop is not None and should_stop() is not None:
                break
            start = perf_counter()
            before = dict(cache.counters)
            ce, norm = bc_optimizer_step(
                policy,
                optimizer,
                backward,
                cache,
                spans,
                n,
                max_gradient_norm=max_gradient_norm,
                asynchronous=False,
            )
            ce = float(ce)
            norm = float(norm)
            if not math.isfinite(ce) or not math.isfinite(norm):
                raise FloatingPointError("Non-finite BC loss or gradient norm")
            steps += 1
            record = {
                "step": steps,
                "epoch": epoch,
                "labels": int(n),
                "ce": ce,
                "gradient_norm": norm,
                "seconds": perf_counter() - start,
                "prefix_positions": cache.counters["prefix_positions"]
                - before.get("prefix_positions", 0),
                "history_positions": cache.counters["history_positions"]
                - before.get("history_positions", 0),
                "compile_seconds": compilation.seconds(start, perf_counter())
                if compilation is not None
                else 0.0,
            }
            records.append(record)
            if steps_log_path is not None:
                with steps_log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
            if on_step_completed is not None:
                on_step_completed(record, steps)
        return records

    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    cell_padding = int(getattr(policy, "spatial_cell_padding", 0) or 0)
    is_recurrent = bool(policy.is_recurrent)
    check_set = set(check_counter_steps)

    prefetch_q: queue.Queue = queue.Queue(maxsize=2)
    slot_pool = PinnedSlotPool(5, slot_factory=ImitationPinnedBatchSlot)
    stop_event = threading.Event()
    io_workers = min(4, max(2, (os.cpu_count() or 4) // 2))

    def _put_safe(obj):
        while not stop_event.is_set():
            try:
                prefetch_q.put(obj, timeout=0.05)
                return True
            except queue.Full:
                continue
        return False

    def _producer():
        batch_iter = iter(batches)
        upcoming = deque()
        lookahead = 4
        try:
            with ThreadPoolExecutor(max_workers=io_workers) as io_pool:
                while not stop_event.is_set():
                    while len(upcoming) < lookahead:
                        try:
                            upcoming.append(next(batch_iter))
                        except StopIteration:
                            break
                    if not upcoming:
                        break
                    cache.preload_episodes(
                        (ep for s_list, _ in upcoming for ep, _, _ in s_list),
                        io_pool,
                    )
                    slot = slot_pool.acquire(stop_event=stop_event)
                    if slot is None:
                        return
                    spans, count = upcoming.popleft()
                    prepared = _prepare_minibatch_cpu(
                        cache,
                        spans,
                        count,
                        is_recurrent=is_recurrent,
                        micro_size=micro_size,
                        cell_padding=cell_padding,
                        slot=slot,
                    )
                    if not _put_safe(prepared):
                        return
                _put_safe(_STREAM_END)
        except BaseException as exc:
            _put_safe(_ProducerError(exc))
        finally:
            cache._inflight.clear()

    producer_thread = threading.Thread(target=_producer, daemon=True)
    producer_thread.start()

    compute_stream = torch.cuda.current_stream(device)
    h2d_stream = torch.cuda.Stream(device=device)
    metrics_bufs = [
        torch.empty(2, dtype=torch.float64, pin_memory=True) for _ in range(2)
    ]

    records = []
    completed_steps = initial_steps
    pending = None  # (slot_idx, item, done_event, wall_anchor)
    last_wall_anchor = perf_counter()
    log_handle = (
        steps_log_path.open("a", encoding="utf-8")
        if steps_log_path is not None
        else None
    )

    def _finalize_pending(verify_counter: bool = False):
        nonlocal pending, completed_steps, last_wall_anchor
        if pending is None:
            return
        slot_idx, p_meta, p_slot, done_event, wall_anchor = pending
        p_labels, p_prefix_pos, p_history_pos = p_meta
        pending = None
        if annotate:
            with record_function("bc::sync_and_bookkeeping"):
                done_event.synchronize()
        else:
            done_event.synchronize()
        slot_pool.release(p_slot)
        now = perf_counter()
        ce_val = float(metrics_bufs[slot_idx][0].item())
        norm_val = float(metrics_bufs[slot_idx][1].item())
        if not math.isfinite(ce_val) or not math.isfinite(norm_val):
            raise FloatingPointError("Non-finite BC loss or gradient norm")
        completed_steps += 1
        if verify_counter:
            counters = {int(s["step"]) for s in optimizer.state.values()}
            if counters != {completed_steps}:
                raise ValueError("BC Adam counters differ from completed steps")
        cache.counters["prefix_positions"] += p_prefix_pos
        cache.counters["history_positions"] += p_history_pos
        record = {
            "step": completed_steps,
            "epoch": epoch,
            "labels": p_labels,
            "ce": ce_val,
            "gradient_norm": norm_val,
            "seconds": now - wall_anchor,
            "prefix_positions": p_prefix_pos,
            "history_positions": p_history_pos,
            "compile_seconds": compilation.seconds(wall_anchor, now)
            if compilation is not None
            else 0.0,
        }
        last_wall_anchor = now
        records.append(record)
        if log_handle is not None:
            log_handle.write(json.dumps(record) + "\n")
            log_handle.flush()
        if on_step_completed is not None:
            on_step_completed(record, completed_steps)

    slot_cursor = 0
    queued_steps = initial_steps
    try:
        while True:
            if max_steps is not None and queued_steps >= max_steps:
                stop_event.set()
                break
            if should_stop is not None and should_stop() is not None:
                stop_event.set()
                break
            item = prefetch_q.get()
            if item is _STREAM_END:
                break
            if isinstance(item, _ProducerError):
                raise item.exception
            queued_steps += 1

            if annotate:
                with record_function("bc::h2d_stage"):
                    staged = _stage_minibatch_to_gpu(
                        item, device, h2d_stream, compute_stream
                    )
            else:
                staged = _stage_minibatch_to_gpu(
                    item, device, h2d_stream, compute_stream
                )
            item_meta = (
                item.label_count,
                item.prefix_positions,
                item.history_positions,
            )
            item_slot = item.slot
            del item

            # If the in-flight step requires an exact synchronous Adam counter check
            # (e.g. step 1), finalize it before mutating optimizer state with the next step.
            if pending is not None and (completed_steps + 1) in check_set:
                _finalize_pending(verify_counter=True)

            cur_slot = slot_cursor
            slot_cursor = 1 - slot_cursor
            anchor = last_wall_anchor if pending is None else perf_counter()

            if annotate:
                with record_function("bc::step_total"):
                    with record_function("bc::optimizer_step"):
                        done_ev = _enqueue_gpu_step(
                            policy=policy,
                            optimizer=optimizer,
                            staged=staged,
                            device=device,
                            max_gradient_norm=max_gradient_norm,
                            compute_stream=compute_stream,
                            metrics_cpu_buf=metrics_bufs[cur_slot],
                            annotate=True,
                        )
            else:
                done_ev = _enqueue_gpu_step(
                    policy=policy,
                    optimizer=optimizer,
                    staged=staged,
                    device=device,
                    max_gradient_norm=max_gradient_norm,
                    compute_stream=compute_stream,
                    metrics_cpu_buf=metrics_bufs[cur_slot],
                    annotate=False,
                )
            del staged

            if pending is not None:
                _finalize_pending(verify_counter=False)
                # Update anchor of the newly-running step to the completion time of the previous step.
                pending = (cur_slot, item_meta, item_slot, done_ev, last_wall_anchor)
            else:
                pending = (cur_slot, item_meta, item_slot, done_ev, anchor)

        _finalize_pending(
            verify_counter=(pending is not None and (completed_steps + 1) in check_set)
        )
        torch.cuda.synchronize(device)
        if completed_steps > initial_steps:
            counters = {int(s["step"]) for s in optimizer.state.values()}
            if counters != {completed_steps}:
                raise ValueError("BC Adam counters differ from completed steps")
    finally:
        stop_event.set()
        producer_thread.join(timeout=5.0)
        if log_handle is not None:
            log_handle.close()

    return records
