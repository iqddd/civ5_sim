"""Complete-episode waves with ready-worker pipelining; learner owns the GPU."""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import random
from dataclasses import dataclass
from multiprocessing.connection import wait
from pathlib import Path
from time import perf_counter

import numpy as np

from civ5rl.core.context import context_from_spec
from civ5rl.core.contracts import (
    WorkerCommand,
    WorkerPayload,
)
from civ5rl.core.distributions import policy_logits, validate_temperature
from civ5rl.core.observation import ObservationEncoder
from civ5rl.engine.collection.state import (
    WORKER_TIMEOUT_SECONDS,
    CollectionScheduler,
    CollectionSlot,
    CollectionWave,
    EpisodeBuilder,
)
from civ5rl.engine.collection.streaming import StreamingScheduler
from civ5rl.engine.collection.workers import ready_worker, sync_worker
from civ5rl.engine.runtime import is_int_at_least
from civ5rl.engine.runtime.budget import deadline_reached
from civ5rl.engine.runtime.shared import EncodedSlots
from civ5rl.provenance import validate_cases
from civ5rl.provenance.common import choose, write_json
from civ5rl.provenance.validation import validate_config


@dataclass
class _InferenceOutput:
    encoded: list
    log_probs: np.ndarray
    critic_values: np.ndarray
    old_hidden: np.ndarray | None


class WaveScheduler:
    supports_diagnostics = False

    def __init__(self):
        self._episode_length_hint = 128.0

    def _poll_workers(self, collector, wave, ready_mode, times):
        pending = wave.pending
        start = perf_counter()
        if ready_mode:
            connections = [collector.workers[w][0] for w in sorted(pending)]
            available = wait(connections, timeout=WORKER_TIMEOUT_SECONDS)
            if not available:
                raise TimeoutError("No collector worker replied within 300 seconds")
            # Include all replies ready now; never wait for a slow worker.
            ready = [w for w in sorted(pending) if collector.workers[w][0].poll()]
        else:
            ready = sorted(pending)
        refreshed = []
        for w in ready:
            payload = collector.receive_worker(w)
            times["worker_encoding_seconds"] += payload["worker_encoding_seconds"]
            offset = collector.worker_offsets[w]
            for j, state in enumerate(payload["states"]):
                wave.slots[offset + j].state = state
                refreshed.append(offset + j)
            pending.remove(w)
        times["worker_wait_seconds"] += perf_counter() - start
        return ready, refreshed

    def _partial_wave(self, collector, policy, wave, worker_encoding, times):
        for slot in wave.slots:
            slot.generation = 0
        active_workers = sorted(wave.pending)
        connections = [collector.workers[w][0] for w in active_workers]
        publish_delays, infer_delays = [], []
        completed = 0
        while completed < len(wave.slots):
            tick = perf_counter()
            if not wait(connections, timeout=WORKER_TIMEOUT_SECONDS):
                raise TimeoutError("No partial collector reply within 300 seconds")
            live, ready, ready_at = [], [], []
            # Briefly combine replies already in flight. Never wait for a slot
            # whose next action depends on this inference, including wave tails.
            coalesce_until = perf_counter() + collector.inference_wait_seconds
            while True:
                for w in active_workers:
                    conn = collector.workers[w][0]
                    while conn.poll():
                        payload = collector.receive_worker(w)
                        if payload["wave"] != wave.wave_id:
                            raise RuntimeError("Stale worker wave")
                        times["worker_encoding_seconds"] += payload["worker_encoding_seconds"]
                        times["worker_request_seconds"] += payload["worker_request_seconds"]
                        times["worker_replies"] += 1
                        for item in payload["items"]:
                            slot = item.slot
                            generation = item.generation
                            state = item.state
                            at = item.completed_at
                            i = collector.worker_offsets[w] + slot
                            if (
                                not 0 <= slot < collector.worker_capacities[w]
                                or i >= len(wave.slots)
                                or generation != wave.slots[i].generation + 1
                                or wave.slots[i].completed_at is not None
                            ):
                                raise RuntimeError("Stale or duplicate slot reply")
                            wave.slots[i].generation = generation
                            wave.slots[i].state = state
                            publish_delays.append(payload["published_at"] - at)
                            if state["done"]:
                                wave.slots[i].builder.finalize(state)
                                wave.slots[i].completed_at = perf_counter()
                                completed += 1
                            else:
                                live.append(i)
                                ready_at.append(at)
                        if w not in ready:
                            ready.append(w)
                remaining = coalesce_until - perf_counter()
                if (
                    len(live) >= collector.inference_min_batch
                    or len(live) + completed == len(wave.slots)
                    or remaining <= 0
                    or not wait(connections, timeout=remaining)
                ):
                    break
            times["worker_wait_seconds"] += perf_counter() - tick
            if live:
                now = perf_counter()
                infer_delays.extend(now - at for at in ready_at)
                wave.decisions += collector.infer_and_select(
                    policy, live, wave, ready, worker_encoding, times
                )
        times["worker_publish_delay_seconds"] += sum(publish_delays)
        times["ready_to_infer_dispatch_seconds"] += sum(infer_delays)
        times["readiness_samples"] += len(infer_delays)
        times["worker_publish_delay_max_seconds"] = max(
            times["worker_publish_delay_max_seconds"], max(publish_delays, default=0.0)
        )

    def _collect_waves(
        self,
        collector,
        policy,
        stage,
        next_episode,
        monitor,
        fixed_specs,
        deadline,
        max_decisions,
        episode_length_hint,
        *,
        ready_mode,
    ):
        policy.eval()
        episodes, decisions = [], 0
        times = {
            "encoding_seconds": 0.0,
            "inference_seconds": 0.0,
            "worker_wait_seconds": 0.0,
            "idle_slot_seconds": 0.0,
            "worker_encoding_seconds": 0.0,
            "inference_batches": 0,
            "inference_batch_items": 0,
            "worker_request_seconds": 0.0,
            "worker_replies": 0,
            "worker_publish_delay_seconds": 0.0,
            "worker_publish_delay_max_seconds": 0.0,
            "ready_to_infer_dispatch_seconds": 0.0,
            "readiness_samples": 0,
        }
        performance = collector.config.get("performance", {})
        if performance.get("collection_scheduler", "barrier") not in (
            "barrier",
            "ready",
        ):
            raise ValueError("Unknown collection scheduler")
        worker_encoding = performance.get("worker_encoding", False)
        reuse_encoded = performance.get("reuse_encoded", False)
        start_rollout = perf_counter()
        target = collector.config["collection"]["min_learner_decisions_per_update"]
        if fixed_specs is None:
            if max_decisions is not None:
                target = min(target, max_decisions)
            if episode_length_hint is None:
                episode_length_hint = self._episode_length_hint
            if not math.isfinite(episode_length_hint) or episode_length_hint <= 0:
                raise ValueError("episode_length_hint must be positive and finite")
        num_envs = collector.num_envs
        if num_envs != sum(collector.worker_capacities):
            raise ValueError("Worker/env count mismatch")
        fixed_offset = 0
        wave_sizes = []
        while decisions < target if fixed_specs is None else fixed_offset < len(fixed_specs):
            if (deadline_reached(deadline)) or (max_decisions is not None and decisions >= max_decisions):
                break
            wave_size = num_envs
            if fixed_specs is None:
                mean_length = decisions / len(episodes) if episodes else episode_length_hint
                wave_size = min(num_envs, max(1, math.ceil((target - decisions) / mean_length)))
            specs = (
                [collector.factory.make(next_episode + i, stage) for i in range(wave_size)]
                if fixed_specs is None
                else fixed_specs[fixed_offset : fixed_offset + num_envs]
            )
            if fixed_specs is None:
                collector.factory.validate_specs(specs)
            wave_sizes.append(len(specs))
            wave = collector.prepare_wave(specs, policy, reuse_encoded, worker_encoding)
            if collector.reply_size:
                self._partial_wave(collector, policy, wave, worker_encoding, times)
                wave.pending.clear()
            while wave.pending:
                ready, refreshed = self._poll_workers(collector, wave, ready_mode, times)
                live = [i for i in refreshed if not wave.slots[i].state["done"]]
                now = perf_counter()
                for i in refreshed:
                    state = wave.slots[i].state
                    if state["done"] and wave.slots[i].completed_at is None:
                        wave.slots[i].builder.finalize(state)
                        wave.slots[i].completed_at = now
                if not live:
                    continue
                wave.decisions += collector.infer_and_select(
                    policy, live, wave, ready, worker_encoding, times
                )
            times["idle_slot_seconds"] += sum(perf_counter() - slot.completed_at for slot in wave.slots)
            if wave.decisions == 0:
                raise RuntimeError("Corpus wave produced no learner decisions")
            episodes.extend(slot.builder.episode for slot in wave.slots)
            decisions += wave.decisions
            next_episode += len(specs)
            fixed_offset += len(specs)
            if monitor:
                monitor.mark("collection", rollout_decisions=decisions)
        times["collection_seconds"] = perf_counter() - start_rollout
        times["inference_batch_mean"] = times["inference_batch_items"] / max(times["inference_batches"], 1)
        if episodes:
            self._episode_length_hint = decisions / len(episodes)
        times.update(
            wave_sizes=wave_sizes,
            admitted_episodes=len(episodes),
            rollout_decisions=decisions,
            target_decisions=target if fixed_specs is None else None,
            rollout_overshoot=max(0, decisions - target) if fixed_specs is None else 0,
        )
        return episodes, next_episode, times


class BarrierWaveScheduler(WaveScheduler):
    def worker_setup(self, reply_size):
        if reply_size:
            raise ValueError("Partial replies require the ready collector")
        return sync_worker, reply_size

    def collect(self, collector, *args):
        return self._collect_waves(collector, *args, ready_mode=False)


class ReadyWaveScheduler(WaveScheduler):
    def worker_setup(self, reply_size):
        return (ready_worker if reply_size else sync_worker), reply_size

    def collect(self, collector, *args):
        return self._collect_waves(collector, *args, ready_mode=True)


class Collector:
    def __init__(
        self,
        config,
        factory,
        device,
        error_dir,
        *,
        cpu_workers=None,
        reply_size=None,
        num_envs=None,
        collection_scheduler=None,
        inference_min_batch=None,
        inference_wait_seconds=None,
        collection_diagnostics=False,
        worker_priority=None,
        opponent_pool=None,
        record_hidden=True,
    ):
        validate_config(config)
        self.config, self.factory, self.device = config, factory, device
        self.temperature = validate_temperature(config.get("policy_temperature", 1.0))
        self.opponent_pool = opponent_pool
        self.record_hidden = record_hidden
        self._slot_staging = None
        self.opponent_temperature = (
            validate_temperature(opponent_pool.temperature) if opponent_pool is not None else self.temperature
        )
        self.error_dir = error_dir
        self.workers = []
        self.shared = []
        self.collection_diagnostics = collection_diagnostics
        performance = config.get("performance", {})
        self.scheduler = collection_scheduler or performance.get("collection_scheduler", "barrier")
        strategy = {
            "barrier": BarrierWaveScheduler,
            "ready": ReadyWaveScheduler,
            "streaming": StreamingScheduler,
        }.get(self.scheduler)
        if strategy is None:
            raise ValueError("Unknown collection scheduler")
        self._strategy: CollectionScheduler = strategy()
        if collection_diagnostics and not self._strategy.supports_diagnostics:
            raise ValueError("Collection diagnostics currently require streaming")
        worker_count = config["collection"]["cpu_workers"] if cpu_workers is None else cpu_workers
        self.num_envs = config["collection"]["num_envs"] if num_envs is None else num_envs
        if (
            not is_int_at_least(worker_count)
            or not is_int_at_least(self.num_envs)
            or worker_count > self.num_envs
        ):
            raise ValueError("Collection workers must fit the environment count")
        quotient, remainder = divmod(self.num_envs, worker_count)
        self.worker_capacities = [quotient + (w < remainder) for w in range(worker_count)]
        self.slot_locations = [
            (w, slot) for w, capacity in enumerate(self.worker_capacities) for slot in range(capacity)
        ]
        self._global_slots = {location: i for i, location in enumerate(self.slot_locations)}
        self.worker_offsets = [self._global_slots[w, 0] for w in range(worker_count)]
        self.envs_per_worker = max(self.worker_capacities)
        self.reply_size = config.get("performance", {}).get("worker_reply_size", 0)
        if cpu_workers is not None or num_envs is not None:
            self.reply_size = min(self.reply_size, self.envs_per_worker)
        if reply_size is not None:
            self.reply_size = reply_size
        self.inference_min_batch = config.get("performance", {}).get("inference_min_batch", 0)
        self.inference_wait_seconds = config.get("performance", {}).get("inference_wait_seconds", 0.0)
        if inference_min_batch is not None:
            self.inference_min_batch = inference_min_batch
        if inference_wait_seconds is not None:
            self.inference_wait_seconds = inference_wait_seconds
        if (
            not is_int_at_least(self.inference_min_batch, 0)
            or not isinstance(self.inference_wait_seconds, (int, float))
            or isinstance(self.inference_wait_seconds, bool)
            or not math.isfinite(self.inference_wait_seconds)
            or self.inference_wait_seconds < 0
        ):
            raise ValueError("Invalid inference batching limits")
        self._wave_id = 0
        if not is_int_at_least(self.reply_size, 0) or self.reply_size > self.envs_per_worker:
            raise ValueError("Invalid performance.worker_reply_size")
        worker_target, self.reply_size = self._strategy.worker_setup(self.reply_size)
        ctx = mp.get_context("spawn")
        # Spawn imports NumPy before the worker constructor. Inherit the math
        # limits now, otherwise BLAS can create a full thread pool per worker.
        math_variables = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        previous_environment = {name: os.environ.get(name) for name in math_variables}
        for name in math_variables:
            os.environ[name] = "1"
        try:
            for _ in range(worker_count):
                parent = child = process = None
                try:
                    parent, child = ctx.Pipe()
                    shared = None
                    if config.get("performance", {}).get("worker_encoding", False):
                        shared = EncodedSlots(
                            self.worker_capacities[len(self.workers)],
                            config.get("performance", {}).get("shared_action_capacity", 4096),
                        )
                    self.shared.append(shared)
                    profile_dir = config.get("performance", {}).get("worker_profile_dir")
                    if profile_dir:
                        Path(profile_dir).mkdir(parents=True, exist_ok=True)
                    profile_path = (
                        str(Path(profile_dir) / f"worker-{len(self.workers)}.prof") if profile_dir else None
                    )
                    process = ctx.Process(
                        target=worker_target,
                        args=(
                            child,
                            profile_path,
                            shared.descriptor if shared else None,
                        ),
                        daemon=True,
                    )
                    process.start()
                    if worker_priority is not None:
                        import psutil

                        psutil.Process(process.pid).nice(worker_priority)
                    self.workers.append((parent, process))
                    parent = process = None  # ownership transferred
                finally:
                    if child is not None:
                        child.close()
                    if parent is not None:
                        parent.close()
                    if process is not None and process.pid is not None:
                        process.terminate()
                        process.join(timeout=5)
        except BaseException:
            self.close()
            raise
        finally:
            for name, value in previous_environment.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value

    def receive_worker(self, index) -> WorkerPayload:
        conn, proc = self.workers[index]
        if not conn.poll(WORKER_TIMEOUT_SECONDS):
            raise TimeoutError(f"Collector worker {proc.pid} timed out")
        kind, payload = conn.recv()
        if kind != "ok":
            write_json(self.error_dir / "worker-error.json", payload)
            raise RuntimeError(payload["traceback"])
        return payload

    def infer_and_select(
        self,
        policy,
        live: list[int],
        wave: CollectionWave,
        workers,
        worker_encoding,
        times,
        *,
        diagnostics=None,
    ) -> int:
        """Evaluate ready observations, record the sampled actions and dispatch them."""
        inference = self._infer_live(policy, live, wave, worker_encoding, times, diagnostics)
        return self._dispatch_actions(workers, live, wave, inference, worker_encoding, diagnostics)

    def _pack_slot_indices(self, indices):
        import torch

        if not str(self.device).startswith("cuda"):
            return torch.tensor(indices, device=self.device)
        if self._slot_staging is None:
            self._slot_staging = torch.empty(self.num_envs, dtype=torch.long, pin_memory=True)
        host = self._slot_staging[: len(indices)]
        host.numpy()[:] = indices
        # The wave's blocking result transfer completes this copy before reuse.
        return host.to(self.device, non_blocking=True)

    def _infer_live(
        self, policy, live: list[int], wave: CollectionWave, worker_encoding, times, diag
    ) -> _InferenceOutput:
        hidden = wave.hidden
        hidden_offset = hidden.shape[1] // 2 if hidden is not None else self.num_envs
        record_hidden = hidden is not None and self.record_hidden

        slots = [wave.slots[index] for index in live]
        states = [slot.state for slot in slots]
        any_opponent = any(state.get("is_opponent") for state in states)

        clock = diag.begin() if diag else None
        if diag:
            for slot in live:
                request = diag.requests[slot]
                request["inference_prepare_ns"] = clock[0]
                diag.add(
                    "observation_to_inference_prepare",
                    clock[0] - request["observation_ready_ns"],
                )
                diag.add("accepted_to_inference_prepare", clock[0] - request["accepted_ns"])
        start = perf_counter()
        non_blocking = self.config.get("performance", {}).get("non_blocking_inputs", False)
        encoded = [
            self.shared[self.slot_locations[i][0]].read(
                self.slot_locations[i][1],
                state,
                copy=not state.get("is_opponent") and slot.builder.reuse_encoded,
            )
            if worker_encoding
            else (slot.opp_encoder if state.get("is_opponent") else slot.encoder).encode(
                state["observation"], state["actions"]
            )
            for i, slot, state in zip(live, slots, states)
        ]

        if not any_opponent:
            return self._infer_learner_batch(
                policy, live, encoded, hidden, record_hidden, non_blocking, times, diag, clock, start
            )
        return self._infer_mixed_batch(
            policy,
            live,
            slots,
            states,
            encoded,
            hidden,
            hidden_offset,
            record_hidden,
            non_blocking,
            times,
            diag,
            clock,
            start,
        )

    def _infer_learner_batch(
        self, policy, live, encoded, hidden, record_hidden, non_blocking, times, diag, clock, start
    ):
        """Keep the homogeneous forward and its single contiguous result copy."""
        import torch

        from civ5rl.core.model import batch_inputs

        batch = batch_inputs(
            encoded,
            self.device,
            non_blocking=non_blocking,
        )
        times["encoding_seconds"] += perf_counter() - start
        if diag:
            diag.end("inference_prepare_h2d", clock)
        start = perf_counter()
        with torch.no_grad():
            if policy.spatial_cuda_graphs:
                torch.compiler.cudagraph_mark_step_begin()
            slots = self._pack_slot_indices(live) if hidden is not None else None
            h = hidden.index_select(1, slots) if hidden is not None else None
            clock, cuda_events = self._begin_forward(diag) if diag else (None, None)
            logits, critic_values, next_hidden = policy(batch, h)
            pieces = [
                self._log_probs(logits, temperature=self.temperature),
                critic_values[:, None],
            ]
            if record_hidden:
                pieces.append(h[0])
            # One D2H synchronization per inference batch, including diagnostic h.
            output = torch.cat(pieces, -1)
            host = self._output_to_host(output, live, diag, clock, cuda_events)
            action_width = logits.shape[1]
            log_probs = host[:, :action_width]
            cpu_critic_values = host[:, action_width]
            old_hidden = host[:, action_width + 1 :] if record_hidden else None
            if next_hidden is not None:
                hidden.index_copy_(1, slots, next_hidden)
        times["inference_seconds"] += perf_counter() - start
        times["inference_batches"] += 1
        times["inference_batch_items"] += len(live)
        return _InferenceOutput(encoded, log_probs, cpu_critic_values, old_hidden)

    def _infer_mixed_batch(
        self,
        policy,
        live,
        live_slots,
        states,
        encoded,
        hidden,
        hidden_offset,
        record_hidden,
        non_blocking,
        times,
        diag,
        clock,
        start,
    ):
        import torch

        from civ5rl.core.model import batch_inputs

        # Mixed learner + neural opponent batching:
        # Learner turns and 'current' opponent turns share one forward on `policy`
        # while keeping their recurrent states in separate halves of one bank.
        main_pos = []
        opp_groups: dict[str, list[int]] = {}
        for j, (slot, state) in enumerate(zip(live_slots, states)):
            if not state.get("is_opponent"):
                main_pos.append(j)
            else:
                opp_id = slot.opponent_id or "current"
                if opp_id == "current":
                    main_pos.append(j)
                else:
                    opp_groups.setdefault(opp_id, []).append(j)

        main_batch = (
            batch_inputs([encoded[j] for j in main_pos], self.device, non_blocking=non_blocking)
            if main_pos
            else None
        )
        opp_batches = {
            opp_id: batch_inputs([encoded[j] for j in pos_list], self.device, non_blocking=non_blocking)
            for opp_id, pos_list in opp_groups.items()
        }
        main_opponents = [bool(states[j].get("is_opponent")) for j in main_pos]
        mixed_temperature = self.opponent_temperature != self.temperature and any(main_opponents)
        slot_indices = None
        if hidden is not None or mixed_temperature:
            slot_indices = self._pack_slot_indices(
                [live[j] + hidden_offset * is_opponent for j, is_opponent in zip(main_pos, main_opponents)]
                + [live[j] + hidden_offset for positions in opp_groups.values() for j in positions]
            )
        times["encoding_seconds"] += perf_counter() - start
        if diag:
            diag.end("inference_prepare_h2d", clock)

        start = perf_counter()
        log_probs_out: list[np.ndarray | None] = [None] * len(live)
        cpu_critic_values = np.zeros(len(live), dtype=np.float32)
        old_hidden = np.zeros((len(live), policy.hidden_size), dtype=np.float32) if record_hidden else None
        with torch.no_grad():
            clock, cuda_events = self._begin_forward(diag) if diag else (None, None)
            main_action_width = 0
            outputs: list[tuple[list[int], torch.Tensor]] = []
            if main_pos:
                if policy.spatial_cuda_graphs:
                    torch.compiler.cudagraph_mark_step_begin()
                slots = slot_indices[: len(main_pos)] if slot_indices is not None else None
                h = hidden.index_select(1, slots) if hidden is not None else None
                logits, critic_values, next_hidden = policy(main_batch, h)
                if next_hidden is not None:
                    hidden.index_copy_(1, slots, next_hidden)
                log_probs = self._log_probs(
                    logits,
                    temperature=self.temperature,
                    slots=slots if mixed_temperature else None,
                    opponent_offset=hidden_offset,
                )
                pieces = [log_probs, critic_values[:, None]]
                if record_hidden:
                    pieces.append(h[0])
                outputs.append((main_pos, torch.cat(pieces, -1)))
                main_action_width = logits.shape[1]
            slot_start = len(main_pos)
            for opp_id, pos_list in opp_groups.items():
                if self.opponent_pool is None:
                    raise RuntimeError(f"Self-play opponent {opp_id!r} requires an OpponentPool on Collector")
                opp_model = self.opponent_pool.model_for(opp_id, policy)
                if opp_model.spatial_cuda_graphs:
                    torch.compiler.cudagraph_mark_step_begin()
                slot_stop = slot_start + len(pos_list)
                o_slots = slot_indices[slot_start:slot_stop] if hidden is not None else None
                slot_start = slot_stop
                h_opp = hidden.index_select(1, o_slots) if hidden is not None else None
                logits, _, next_hidden = opp_model(opp_batches[opp_id], h_opp, compute_value=False)
                if next_hidden is not None:
                    hidden.index_copy_(1, o_slots, next_hidden)
                opp_lp = self._log_probs(logits, temperature=self.opponent_temperature)
                outputs.append((pos_list, opp_lp))
            # Enqueue all policy forward passes before one host-synchronizing D2H copy.
            flat = (
                outputs[0][1].reshape(-1)
                if len(outputs) == 1
                else torch.cat([output.reshape(-1) for _, output in outputs])
            )
            flat_host = self._output_to_host(flat, live, diag, clock, cuda_events)
            offset = 0
            for positions, output in outputs:
                end = offset + output.numel()
                rows = flat_host[offset:end].reshape(output.shape)
                offset = end
                if positions is main_pos:
                    for row, j in zip(rows, positions):
                        log_probs_out[j] = row[:main_action_width]
                        if not states[j].get("is_opponent"):
                            cpu_critic_values[j] = row[main_action_width]
                            if old_hidden is not None:
                                old_hidden[j] = row[main_action_width + 1 :]
                else:
                    for row, j in zip(rows, positions):
                        log_probs_out[j] = row
        times["inference_seconds"] += perf_counter() - start
        times["inference_batches"] += int(bool(main_pos)) + len(opp_groups)
        times["inference_batch_items"] += len(live)
        return _InferenceOutput(encoded, log_probs_out, cpu_critic_values, old_hidden)

    def _log_probs(self, logits, *, temperature, slots=None, opponent_offset=None):
        if slots is None:
            return policy_logits(logits, temperature).log_softmax(-1)
        import torch

        temperatures = torch.where(slots >= opponent_offset, self.opponent_temperature, temperature).to(
            logits.dtype
        )[:, None]
        return (logits / temperatures).log_softmax(-1)

    def _begin_forward(self, diag):
        clock = diag.begin()
        if not str(self.device).startswith("cuda"):
            return clock, None
        import torch

        events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        events[0].record()
        return clock, events

    def _output_to_host(self, output, live, diag, clock, cuda_events):
        if diag:
            if cuda_events:
                cuda_events[1].record()
            diag.end("forward_submit_and_output", clock)
            for slot in live:
                diag.requests[slot]["forward_submit_ns"] = clock[0]
            clock = diag.begin()
        host = output.cpu().numpy()
        if diag:
            at = diag.end("blocking_result_to_cpu", clock)
            for slot in live:
                diag.requests[slot]["host_result_ns"] = at
            if cuda_events:
                # The blocking D2H has completed the timing events.
                diag.add(
                    "cuda_forward_and_output.stream",
                    round(cuda_events[0].elapsed_time(cuda_events[1]) * 1e6),
                )
        return host

    def _dispatch_actions(
        self, ready, live: list[int], wave: CollectionWave, inference: _InferenceOutput, worker_encoding, diag
    ) -> int:
        pending = wave.pending
        wave_decisions = 0
        clock = diag.begin() if diag else None
        dispatch = [[] for _ in self.workers]
        for j, i in enumerate(live):
            slot = wave.slots[i]
            state = slot.state
            logp = inference.log_probs[j][: len(state["actions"])]
            probabilities = np.exp(logp)
            if state.get("is_opponent"):
                chosen = choose(probabilities, slot.opp_rng)
                action = state["actions"][chosen]
                if not worker_encoding:
                    slot.opp_encoder.remember_action(action, state["observation"])
            else:
                chosen = choose(probabilities, slot.rng)
                action = slot.builder.record_step(
                    state,
                    inference.encoded[j],
                    chosen,
                    logp,
                    probabilities,
                    inference.critic_values[j],
                    None if inference.old_hidden is None else inference.old_hidden[j],
                )
                if not worker_encoding:
                    slot.encoder.remember_action(action, state["observation"])
                wave_decisions += 1
            generation = slot.generation
            worker, local_slot = self.slot_locations[i]
            item = WorkerCommand(local_slot, generation, action)
            dispatch[worker].append(item)
        if diag:
            diag.end("select_and_record_actions", clock)
        for w in ready:
            if dispatch[w]:
                payload = dispatch[w]
                if generation is not None:
                    payload = {"wave": wave.wave_id, "commands": payload}
                if diag:
                    requests = [
                        diag.requests.pop(self._global_slots[w, command.slot]) for command in dispatch[w]
                    ]
                    clock = diag.begin()
                    payload["diagnostic_sent_ns"] = clock[0]
                self.workers[w][0].send(("step", payload))
                if diag:
                    diag.end("parent_action_send", clock)
                    for request in requests:
                        diag.add(
                            "host_result_to_action_send",
                            clock[0] - request["host_result_ns"],
                        )
                        diag.sample(
                            dict(
                                kind="request",
                                **request,
                                action_sent_ns=clock[0],
                                worker=w,
                                worker_pid=self.workers[w][1].pid,
                            )
                        )
                if generation is None:
                    pending.add(w)
        return wave_decisions

    def next_wave_id(self):
        self._wave_id += 1
        return self._wave_id

    def prepare_wave(self, specs, policy, reuse_encoded, worker_encoding) -> CollectionWave:
        import torch

        slots = [self._prepare_slot(spec, policy, reuse_encoded, worker_encoding) for spec in specs]
        hidden = (
            torch.zeros((1, 2 * len(specs), policy.hidden_size), device=self.device)
            if policy.is_recurrent
            else None
        )
        wave_id = self.next_wave_id()
        for w, (conn, _) in enumerate(self.workers):
            offset = self.worker_offsets[w]
            if offset >= len(specs):
                break
            conn.send(
                (
                    "reset",
                    {
                        "specs": specs[offset : offset + self.worker_capacities[w]],
                        "variant": policy.variant,
                        "wave": wave_id,
                        "reply_size": self.reply_size,
                    },
                )
            )
        return CollectionWave(
            wave_id,
            slots,
            hidden,
            pending={worker for worker, offset in enumerate(self.worker_offsets) if offset < len(slots)},
        )

    def _prepare_slot(self, spec, policy, reuse_encoded, worker_encoding):
        learner = spec["learner_side"]
        neural_opponent = spec.get("opponent") == "policy"
        return CollectionSlot(
            builder=EpisodeBuilder(
                spec,
                policy.variant,
                reuse_encoded,
                policy.hidden_size if policy.is_recurrent else 0,
                self.temperature,
                record_hidden=self.record_hidden,
            ),
            rng=random.Random(spec["agent_seeds"][learner]),
            opp_rng=random.Random(spec["agent_seeds"][1 - learner]) if neural_opponent else None,
            opponent_id=spec.get("opponent_id") if neural_opponent else None,
            encoder=(
                None if worker_encoding else ObservationEncoder(context_from_spec(spec), policy.variant)
            ),
            opp_encoder=(
                ObservationEncoder(context_from_spec(spec, side=1 - learner), policy.variant)
                if neural_opponent and not worker_encoding
                else None
            ),
        )

    def collect(
        self,
        policy,
        stage,
        next_episode,
        monitor=None,
        fixed_specs=None,
        trusted_manifest=None,
        deadline=None,
        max_decisions=None,
        episode_length_hint=None,
    ):
        if fixed_specs is not None:
            validate_cases(fixed_specs, trusted_manifest)

        return self._strategy.collect(
            self,
            policy,
            stage,
            next_episode,
            monitor,
            fixed_specs,
            deadline,
            max_decisions,
            episode_length_hint,
        )

    def close(self):
        workers, self.workers = self.workers, []
        shared_slots, self.shared = self.shared, []
        for conn, process in workers:
            try:
                try:
                    conn.send(("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
                process.join(timeout=5)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
            finally:
                conn.close()
        for shared in shared_slots:
            if shared is not None:
                shared.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
