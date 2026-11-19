"""Complete-episode waves with ready-worker pipelining; learner owns the GPU."""

import math
import multiprocessing as mp
import os
import random
from dataclasses import dataclass
from multiprocessing.connection import wait
from time import perf_counter
from typing import TYPE_CHECKING, Protocol

import numpy as np

from civ5rl.core.context import context_from_spec
from civ5rl.core.contracts import (
    Episode,
    ScenarioSpec,
    WorkerCommand,
    WorkerPayload,
    WorkerState,
)
from civ5rl.core.distributions import policy_logits, validate_temperature
from civ5rl.core.observation import ObservationEncoder
from civ5rl.engine.collection.workers import ready_worker, sync_worker
from civ5rl.engine.runtime import is_int_at_least
from civ5rl.engine.runtime.budget import deadline_reached
from civ5rl.provenance.common import choose

if TYPE_CHECKING:
    from civ5rl.core.model import Policy


WORKER_TIMEOUT_SECONDS = 300


class EpisodeBuilder:
    """Accumulate one episode while preserving the learner's dictionary format."""

    def __init__(
        self, spec: ScenarioSpec, variant, reuse_encoded, hidden_size, temperature=1.0
    ):
        self.reuse_encoded = reuse_encoded
        self.hidden_size = hidden_size
        self.episode: Episode = {
            "policy_temperature": temperature,
            "spec": spec,
            "observations": [],
            "actions": [],
            "chosen": [],
            "old_logp": [],
            "old_log_probs": [],
            "old_value": [],
            "old_hidden": [],
            "decision_metrics": [],
        }
        if reuse_encoded:
            self.episode.update(encoded=[], encoded_variant=variant)

    def record_step(self, state: WorkerState, encoded, chosen, logp, value, old_hidden):
        from civ5rl.protocols.diagnostic_metrics import decision_metrics

        ep = self.episode
        action = state["actions"][chosen]
        ep["observations"].append(state["observation"])
        if self.reuse_encoded:
            ep["encoded"].append(encoded)
        ep["actions"].append(state["actions"])
        ep["chosen"].append(chosen)
        ep["old_logp"].append(float(logp[chosen]))
        ep["old_log_probs"].append(logp.copy())
        ep["decision_metrics"].append(
            decision_metrics(state["actions"], chosen, np.exp(logp))
        )
        ep["old_value"].append(float(value))
        is_initial_step = len(ep["chosen"]) == 1
        ep["old_hidden"].append(
            None
            if old_hidden is None or is_initial_step
            else old_hidden.reshape(1, 1, self.hidden_size).copy()
        )
        return action

    def finalize(self, state: WorkerState):
        self.episode["result"] = state["result"]
        self.episode["joint_commands"] = state["joint_commands"]
        return self.episode


@dataclass
class _InferenceOutput:
    encoded: list
    log_probs: np.ndarray
    critic_values: np.ndarray
    old_hidden: np.ndarray | None


@dataclass
class _WaveState:
    builders: list
    encoders: list
    rngs: list
    hidden: object
    wave_id: int
    completed_at: list
    states: list
    pending: set
    decisions: int = 0
    generations: list | None = None
    opp_hidden: object = None
    opp_encoders: list | None = None
    opp_rngs: list | None = None
    opponent_ids: list | None = None

    def slot_state(self, i):
        return self.states[i]

    def slot_encoder(self, i):
        return self.encoders[i]

    def slot_opp_encoder(self, i):
        return self.opp_encoders[i]

    def slot_builder(self, i):
        return self.builders[i]

    def slot_rng(self, i):
        return self.rngs[i]

    def slot_opp_rng(self, i):
        return self.opp_rngs[i]

    def slot_opponent_id(self, i):
        return None if self.opponent_ids is None else self.opponent_ids[i]

    def slot_generation(self, i):
        return self.generations[i] if self.generations is not None else None


class CollectionScheduler(Protocol):
    supports_diagnostics: bool

    def worker_setup(self, reply_size: int): ...

    def collect(
        self,
        collector: "Collector",
        policy: "Policy",
        stage,
        next_episode: int,
        monitor,
        fixed_specs,
        deadline,
        max_decisions: int | None,
        episode_length_hint: float | None,
    ) -> tuple[list[Episode], int, dict]: ...


class WaveScheduler:
    supports_diagnostics = False

    def __init__(self):
        self._episode_length_hint = 128.0

    def _poll_workers(self, collector, wave, ready_mode, times):
        pending, states = wave.pending, wave.states
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
                states[offset + j] = state
                refreshed.append(offset + j)
            pending.remove(w)
        times["worker_wait_seconds"] += perf_counter() - start
        return ready, refreshed

    def _partial_wave(self, collector, policy, wave, worker_encoding, times):
        wave.generations = [0] * len(wave.states)
        active_workers = sorted(wave.pending)
        connections = [collector.workers[w][0] for w in active_workers]
        publish_delays, infer_delays = [], []
        completed = 0
        while completed < len(wave.states):
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
                        times["worker_encoding_seconds"] += payload[
                            "worker_encoding_seconds"
                        ]
                        times["worker_request_seconds"] += payload[
                            "worker_request_seconds"
                        ]
                        times["worker_replies"] += 1
                        for item in payload["items"]:
                            slot = item.slot
                            generation = item.generation
                            state = item.state
                            at = item.completed_at
                            i = collector.worker_offsets[w] + slot
                            if (
                                not 0 <= slot < collector.worker_capacities[w]
                                or i >= len(wave.states)
                                or generation != wave.generations[i] + 1
                                or wave.completed_at[i] is not None
                            ):
                                raise RuntimeError("Stale or duplicate slot reply")
                            wave.generations[i] = generation
                            wave.states[i] = state
                            publish_delays.append(payload["published_at"] - at)
                            if state["done"]:
                                wave.builders[i].finalize(state)
                                wave.completed_at[i] = perf_counter()
                                completed += 1
                            else:
                                live.append(i)
                                ready_at.append(at)
                        if w not in ready:
                            ready.append(w)
                remaining = coalesce_until - perf_counter()
                if (
                    len(live) >= collector.inference_min_batch
                    or len(live) + completed == len(wave.states)
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
        while (
            decisions < target
            if fixed_specs is None
            else fixed_offset < len(fixed_specs)
        ):
            if (deadline_reached(deadline)) or (
                max_decisions is not None and decisions >= max_decisions
            ):
                break
            wave_size = num_envs
            if fixed_specs is None:
                mean_length = (
                    decisions / len(episodes) if episodes else episode_length_hint
                )
                wave_size = min(
                    num_envs, max(1, math.ceil((target - decisions) / mean_length))
                )
            specs = (
                [
                    collector.factory.make(next_episode + i, stage)
                    for i in range(wave_size)
                ]
                if fixed_specs is None
                else fixed_specs[fixed_offset : fixed_offset + num_envs]
            )
            if fixed_specs is None:
                collector.factory.validate_specs(specs)
            wave_sizes.append(len(specs))
            wave = _WaveState(
                *collector.prepare_wave(specs, policy, reuse_encoded, worker_encoding),
                completed_at=[None] * len(specs),
                states=[None] * len(specs),
                pending={
                    w
                    for w, offset in enumerate(collector.worker_offsets)
                    if offset < len(specs)
                },
            )
            collector.attach_wave_opponents(wave, specs, policy, worker_encoding)
            if collector.reply_size:
                self._partial_wave(collector, policy, wave, worker_encoding, times)
                wave.pending.clear()
            while wave.pending:
                ready, refreshed = self._poll_workers(
                    collector, wave, ready_mode, times
                )
                live = [i for i in refreshed if not wave.states[i]["done"]]
                now = perf_counter()
                for i in refreshed:
                    state = wave.states[i]
                    if state["done"] and wave.completed_at[i] is None:
                        wave.builders[i].finalize(state)
                        wave.completed_at[i] = now
                if not live:
                    continue
                wave.decisions += collector.infer_and_select(
                    policy, live, wave, ready, worker_encoding, times
                )
            times["idle_slot_seconds"] += sum(
                perf_counter() - t for t in wave.completed_at
            )
            if wave.decisions == 0:
                raise RuntimeError("Corpus wave produced no learner decisions")
            episodes.extend(builder.episode for builder in wave.builders)
            decisions += wave.decisions
            next_episode += len(specs)
            fixed_offset += len(specs)
            if monitor:
                monitor.mark("collection", rollout_decisions=decisions)
        times["collection_seconds"] = perf_counter() - start_rollout
        times["inference_batch_mean"] = times["inference_batch_items"] / max(
            times["inference_batches"], 1
        )
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
    ):
        from civ5rl.engine.collection.streaming import StreamingScheduler
        from civ5rl.provenance.validation import validate_config

        validate_config(config)
        self.config, self.factory, self.device = config, factory, device
        self.temperature = validate_temperature(config.get("policy_temperature", 1.0))
        self.opponent_pool = opponent_pool
        self.opponent_temperature = (
            validate_temperature(opponent_pool.temperature)
            if opponent_pool is not None
            else self.temperature
        )
        self.error_dir = error_dir
        self.workers = []
        self.shared = []
        self.collection_diagnostics = collection_diagnostics
        performance = config.get("performance", {})
        self.scheduler = collection_scheduler or performance.get(
            "collection_scheduler", "barrier"
        )
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
        worker_count = (
            config["collection"]["cpu_workers"] if cpu_workers is None else cpu_workers
        )
        self.num_envs = (
            config["collection"]["num_envs"] if num_envs is None else num_envs
        )
        if (
            not is_int_at_least(worker_count)
            or not is_int_at_least(self.num_envs)
            or worker_count > self.num_envs
        ):
            raise ValueError("Collection workers must fit the environment count")
        quotient, remainder = divmod(self.num_envs, worker_count)
        self.worker_capacities = [
            quotient + (w < remainder) for w in range(worker_count)
        ]
        self.slot_locations = [
            (w, slot)
            for w, capacity in enumerate(self.worker_capacities)
            for slot in range(capacity)
        ]
        self._global_slots = {
            location: i for i, location in enumerate(self.slot_locations)
        }
        self.worker_offsets = [self._global_slots[w, 0] for w in range(worker_count)]
        self.envs_per_worker = max(self.worker_capacities)
        self.reply_size = config.get("performance", {}).get("worker_reply_size", 0)
        if cpu_workers is not None or num_envs is not None:
            self.reply_size = min(self.reply_size, self.envs_per_worker)
        if reply_size is not None:
            self.reply_size = reply_size
        self.inference_min_batch = config.get("performance", {}).get(
            "inference_min_batch", 0
        )
        self.inference_wait_seconds = config.get("performance", {}).get(
            "inference_wait_seconds", 0.0
        )
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
        if (
            not is_int_at_least(self.reply_size, 0)
            or self.reply_size > self.envs_per_worker
        ):
            raise ValueError("Invalid performance.worker_reply_size")
        worker_target, self.reply_size = self._strategy.worker_setup(self.reply_size)
        ctx = mp.get_context("spawn")
        # Spawn imports NumPy before _WorkerRuntime.__init__. Inherit the math
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
                        from civ5rl.engine.runtime.shared import EncodedSlots

                        shared = EncodedSlots(
                            self.worker_capacities[len(self.workers)],
                            config.get("performance", {}).get(
                                "shared_action_capacity", 4096
                            ),
                        )
                    self.shared.append(shared)
                    profile_dir = config.get("performance", {}).get(
                        "worker_profile_dir"
                    )
                    if profile_dir:
                        from pathlib import Path

                        Path(profile_dir).mkdir(parents=True, exist_ok=True)
                    profile_path = (
                        str(Path(profile_dir) / f"worker-{len(self.workers)}.prof")
                        if profile_dir
                        else None
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
        from civ5rl.provenance.common import write_json

        conn, proc = self.workers[index]
        if not conn.poll(WORKER_TIMEOUT_SECONDS):
            raise TimeoutError(f"Collector worker {proc.pid} timed out")
        kind, payload = conn.recv()
        if kind != "ok":
            write_json(self.error_dir / "worker-error.json", payload)
            raise RuntimeError(payload["traceback"])
        return payload

    def infer_and_select(
        self, policy, live, wave, workers, worker_encoding, times, *, diagnostics=None
    ):
        """Evaluate ready observations, record the sampled actions and dispatch them."""
        inference = self._infer_live(
            policy, live, wave, worker_encoding, times, diagnostics
        )
        return self._dispatch_actions(
            workers, live, wave, inference, worker_encoding, diagnostics
        )

    def _infer_live(self, policy, live, wave, worker_encoding, times, diag):
        import torch

        hidden = wave.hidden
        opp_hidden = getattr(wave, "opp_hidden", None)
        any_opponent = any(wave.slot_state(i).get("is_opponent") for i in live)

        from civ5rl.core.model import batch_inputs

        clock = diag.begin() if diag else None
        if diag:
            for slot in live:
                request = diag.requests[slot]
                request["inference_prepare_ns"] = clock[0]
                diag.add(
                    "observation_to_inference_prepare",
                    clock[0] - request["observation_ready_ns"],
                )
                diag.add(
                    "accepted_to_inference_prepare", clock[0] - request["accepted_ns"]
                )
        start = perf_counter()
        non_blocking = self.config.get("performance", {}).get(
            "non_blocking_inputs", False
        )
        encoded = [
            self.shared[self.slot_locations[i][0]].read(
                self.slot_locations[i][1], wave.slot_state(i)
            )
            if worker_encoding
            else (
                wave.slot_opp_encoder(i)
                if wave.slot_state(i).get("is_opponent")
                else wave.slot_encoder(i)
            ).encode(
                wave.slot_state(i)["observation"], wave.slot_state(i)["actions"]
            )
            for i in live
        ]
        if not any_opponent:
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
                slots = torch.tensor(live, device=self.device)
                h = hidden.index_select(1, slots) if hidden is not None else None
                clock = diag.begin() if diag else None
                cuda_events = None
                if diag and str(self.device).startswith("cuda"):
                    cuda_events = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    cuda_events[0].record()
                logits, critic_values, next_hidden = policy(batch, h)
                pieces = [
                    policy_logits(logits, self.temperature).log_softmax(-1),
                    critic_values[:, None],
                ]
                if h is not None:
                    pieces.append(h[0])
                # One D2H synchronization per inference batch, including diagnostic h.
                output = torch.cat(pieces, -1)
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
                        # The existing blocking D2H already completed these events.
                        diag.add(
                            "cuda_forward_and_output.stream",
                            round(cuda_events[0].elapsed_time(cuda_events[1]) * 1e6),
                        )
                action_width = logits.shape[1]
                log_probs = host[:, :action_width]
                cpu_critic_values = host[:, action_width]
                old_hidden = host[:, action_width + 1 :] if h is not None else None
                if next_hidden is not None:
                    hidden.index_copy_(1, slots, next_hidden)
            times["inference_seconds"] += perf_counter() - start
            times["inference_batches"] += 1
            times["inference_batch_items"] += len(live)
            return _InferenceOutput(encoded, log_probs, cpu_critic_values, old_hidden)

        # Mixed learner + neural opponent batching:
        # Learner turns and 'current' opponent turns share one forward on `policy`
        # while keeping strictly isolated recurrent hidden states (`hidden` vs `opp_hidden`).
        main_pos = []
        opp_groups: dict[str, list[int]] = {}
        for j, i in enumerate(live):
            if not wave.slot_state(i).get("is_opponent"):
                main_pos.append(j)
            else:
                opp_id = wave.slot_opponent_id(i) or "current"
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
            opp_id: batch_inputs(
                [encoded[j] for j in pos_list], self.device, non_blocking=non_blocking
            )
            for opp_id, pos_list in opp_groups.items()
        }
        times["encoding_seconds"] += perf_counter() - start
        if diag:
            diag.end("inference_prepare_h2d", clock)

        start = perf_counter()
        log_probs_out: list[np.ndarray | None] = [None] * len(live)
        cpu_critic_values = np.zeros(len(live), dtype=np.float32)
        old_hidden = (
            np.zeros((len(live), policy.hidden_size), dtype=np.float32)
            if hidden is not None
            else None
        )
        with torch.no_grad():
            clock = diag.begin() if diag else None
            cuda_events = None
            if diag and str(self.device).startswith("cuda"):
                cuda_events = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                cuda_events[0].record()
            if main_pos:
                if policy.spatial_cuda_graphs:
                    torch.compiler.cudagraph_mark_step_begin()
                learner_pairs = [
                    (k, live[j])
                    for k, j in enumerate(main_pos)
                    if not wave.slot_state(live[j]).get("is_opponent")
                ]
                opp_pairs = [
                    (k, live[j])
                    for k, j in enumerate(main_pos)
                    if wave.slot_state(live[j]).get("is_opponent")
                ]
                if hidden is not None:
                    h = torch.empty(
                        (1, len(main_pos), policy.hidden_size),
                        device=self.device,
                        dtype=hidden.dtype,
                    )
                    if learner_pairs:
                        l_ks = torch.tensor([k for k, _ in learner_pairs], device=self.device)
                        l_slots = torch.tensor([s for _, s in learner_pairs], device=self.device)
                        h.index_copy_(1, l_ks, hidden.index_select(1, l_slots))
                    if opp_pairs:
                        o_ks = torch.tensor([k for k, _ in opp_pairs], device=self.device)
                        o_slots = torch.tensor([s for _, s in opp_pairs], device=self.device)
                        h.index_copy_(1, o_ks, opp_hidden.index_select(1, o_slots))
                else:
                    h = None
                logits, critic_values, next_hidden = policy(main_batch, h)
                if next_hidden is not None:
                    if learner_pairs:
                        hidden.index_copy_(1, l_slots, next_hidden.index_select(1, l_ks))
                    if opp_pairs:
                        opp_hidden.index_copy_(1, o_slots, next_hidden.index_select(1, o_ks))
                if self.opponent_temperature == self.temperature or not opp_pairs:
                    scaled = policy_logits(logits, self.temperature)
                else:
                    temps = torch.tensor(
                        [
                            self.opponent_temperature
                            if wave.slot_state(live[j]).get("is_opponent")
                            else self.temperature
                            for j in main_pos
                        ],
                        device=self.device,
                        dtype=logits.dtype,
                    )[:, None]
                    scaled = logits / temps
                pieces = [scaled.log_softmax(-1), critic_values[:, None]]
                if h is not None:
                    pieces.append(h[0])
                host = torch.cat(pieces, -1).cpu().numpy()
                action_width = logits.shape[1]
                for k, j in enumerate(main_pos):
                    log_probs_out[j] = host[k, :action_width]
                    if not wave.slot_state(live[j]).get("is_opponent"):
                        cpu_critic_values[j] = host[k, action_width]
                        if old_hidden is not None:
                            old_hidden[j] = host[k, action_width + 1 :]
            for opp_id, pos_list in opp_groups.items():
                if self.opponent_pool is None:
                    raise RuntimeError(
                        f"Self-play opponent {opp_id!r} requires an OpponentPool on Collector"
                    )
                opp_model = self.opponent_pool.model_for(opp_id, policy)
                if opp_model.spatial_cuda_graphs:
                    torch.compiler.cudagraph_mark_step_begin()
                o_slots = torch.tensor([live[j] for j in pos_list], device=self.device)
                h_opp = opp_hidden.index_select(1, o_slots) if opp_hidden is not None else None
                logits, _, next_hidden = opp_model(opp_batches[opp_id], h_opp)
                if next_hidden is not None and opp_hidden is not None:
                    opp_hidden.index_copy_(1, o_slots, next_hidden)
                host_lp = policy_logits(logits, self.opponent_temperature).log_softmax(-1).cpu().numpy()
                for k, j in enumerate(pos_list):
                    log_probs_out[j] = host_lp[k]
            if diag:
                if cuda_events:
                    cuda_events[1].record()
                at = diag.end("forward_submit_and_output", clock)
                for slot in live:
                    diag.requests[slot]["forward_submit_ns"] = clock[0]
                    diag.requests[slot]["host_result_ns"] = at
                if cuda_events:
                    torch.cuda.synchronize()
                    diag.add(
                        "cuda_forward_and_output.stream",
                        round(cuda_events[0].elapsed_time(cuda_events[1]) * 1e6),
                    )
        times["inference_seconds"] += perf_counter() - start
        times["inference_batches"] += int(bool(main_pos)) + len(opp_groups)
        times["inference_batch_items"] += len(live)
        return _InferenceOutput(encoded, log_probs_out, cpu_critic_values, old_hidden)

    def _dispatch_actions(
        self, ready, live, wave, inference: _InferenceOutput, worker_encoding, diag
    ):
        pending = wave.pending
        wave_decisions = 0
        clock = diag.begin() if diag else None
        dispatch = [[] for _ in self.workers]
        for j, i in enumerate(live):
            state = wave.slot_state(i)
            logp = inference.log_probs[j][: len(state["actions"])]
            if state.get("is_opponent"):
                chosen = choose(np.exp(logp), wave.slot_opp_rng(i))
                action = state["actions"][chosen]
                if not worker_encoding:
                    wave.slot_opp_encoder(i).remember_action(action, state["observation"])
            else:
                chosen = choose(np.exp(logp), wave.slot_rng(i))
                action = wave.slot_builder(i).record_step(
                    state,
                    inference.encoded[j],
                    chosen,
                    logp,
                    inference.critic_values[j],
                    None if inference.old_hidden is None else inference.old_hidden[j],
                )
                if not worker_encoding:
                    wave.slot_encoder(i).remember_action(action, state["observation"])
                wave_decisions += 1
            generation = wave.slot_generation(i)
            worker, slot = self.slot_locations[i]
            item = WorkerCommand(slot, generation, action)
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
                        diag.requests.pop(self._global_slots[w, command.slot])
                        for command in dispatch[w]
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

    def attach_wave_opponents(self, wave, specs, policy, worker_encoding):
        import torch

        wave.opponent_ids = [
            s.get("opponent_id") if s.get("opponent") == "policy" else None
            for s in specs
        ]
        wave.opp_rngs = [
            random.Random(s["agent_seeds"][1 - s["learner_side"]])
            if s.get("opponent") == "policy"
            else None
            for s in specs
        ]
        wave.opp_encoders = (
            []
            if worker_encoding
            else [
                ObservationEncoder(
                    context_from_spec(s, side=1 - s["learner_side"]), policy.variant
                )
                if s.get("opponent") == "policy"
                else None
                for s in specs
            ]
        )
        wave.opp_hidden = (
            torch.zeros((1, len(specs), policy.hidden_size), device=self.device)
            if policy.is_recurrent
            else None
        )

    def prepare_wave(self, specs, policy, reuse_encoded, worker_encoding):
        import torch

        builders = [
            EpisodeBuilder(
                s,
                policy.variant,
                reuse_encoded,
                policy.hidden_size if policy.is_recurrent else 0,
                self.temperature,
            )
            for s in specs
        ]
        encoders = (
            []
            if worker_encoding
            else [
                ObservationEncoder(
                    context_from_spec(s),
                    policy.variant,
                )
                for s in specs
            ]
        )
        rngs = [random.Random(s["agent_seeds"][s["learner_side"]]) for s in specs]
        hidden = (
            torch.zeros((1, len(specs), policy.hidden_size), device=self.device)
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
        return builders, encoders, rngs, hidden, wave_id

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
        from civ5rl.provenance import validate_cases

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
