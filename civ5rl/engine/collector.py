"""Complete-episode waves with ready-worker pipelining; learner owns the GPU."""

import math
import multiprocessing as mp
import os
import random
from dataclasses import dataclass
from multiprocessing.connection import wait
from time import perf_counter

import numpy as np

from civ5rl.core.context import context_from_spec
from civ5rl.core.contracts import (
    Episode,
    ScenarioSpec,
    WorkerCommand,
    WorkerPayload,
    WorkerState,
)
from civ5rl.core.observation import ObservationEncoder
from civ5rl.engine.worker import worker as _worker
from civ5rl.provenance.common import choose

WORKER_TIMEOUT_SECONDS = 300


class EpisodeBuilder:
    """Accumulate one episode while preserving the learner's dictionary format."""

    def __init__(self, spec: ScenarioSpec, variant, reuse_encoded, hidden_size):
        self.reuse_encoded = reuse_encoded
        self.hidden_size = hidden_size
        self.episode: Episode = {
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
    completed_at: list
    states: list
    pending: set
    decisions: int = 0
    wave_id: int = 0
    generations: list | None = None


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
    ):
        from civ5rl.provenance.validation import validate_config

        validate_config(config)
        self.config, self.factory, self.device = config, factory, device
        self.error_dir = error_dir
        self.workers = []
        self.shared = []
        performance = config.get("performance", {})
        self.scheduler = collection_scheduler or performance.get(
            "collection_scheduler", "barrier"
        )
        if self.scheduler not in ("barrier", "ready", "streaming"):
            raise ValueError("Unknown collection scheduler")
        worker_count = (
            config["collection"]["cpu_workers"] if cpu_workers is None else cpu_workers
        )
        self.num_envs = (
            config["collection"]["num_envs"] if num_envs is None else num_envs
        )
        if (
            type(worker_count) is not int
            or worker_count < 1
            or type(self.num_envs) is not int
            or self.num_envs < 1
            or worker_count > self.num_envs
            or (self.scheduler != "streaming" and self.num_envs % worker_count)
        ):
            raise ValueError("Collection workers must divide the environment count")
        quotient, remainder = divmod(self.num_envs, worker_count)
        self.worker_capacities = [
            quotient + (w < remainder) for w in range(worker_count)
        ]
        self.slot_locations = [
            (w, slot)
            for w, capacity in enumerate(self.worker_capacities)
            for slot in range(capacity)
        ]
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
            type(self.inference_min_batch) is not int
            or self.inference_min_batch < 0
            or not isinstance(self.inference_wait_seconds, (int, float))
            or isinstance(self.inference_wait_seconds, bool)
            or not math.isfinite(self.inference_wait_seconds)
            or self.inference_wait_seconds < 0
        ):
            raise ValueError("Invalid inference batching limits")
        self._wave_id = 0
        if (
            type(self.reply_size) is not int
            or not 0 <= self.reply_size <= self.envs_per_worker
        ):
            raise ValueError("Invalid performance.worker_reply_size")
        worker_target = _worker
        if self.scheduler == "streaming":
            self.reply_size = max(self.reply_size, 1)
            from civ5rl.engine.streaming_worker import worker as worker_target
        elif self.reply_size:
            if self.scheduler != "ready":
                raise ValueError("Partial replies require the ready collector")
            from civ5rl.engine.stream_worker import worker as worker_target
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
                        from civ5rl.engine.shared import EncodedSlots

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

    def _receive_one(self, index) -> WorkerPayload:
        from civ5rl.provenance.common import write_json

        conn, proc = self.workers[index]
        if not conn.poll(WORKER_TIMEOUT_SECONDS):
            raise TimeoutError(f"Collector worker {proc.pid} timed out")
        kind, payload = conn.recv()
        if kind != "ok":
            write_json(self.error_dir / "worker-error.json", payload)
            raise RuntimeError(payload["traceback"])
        return payload

    def _poll_workers(self, wave, ready_mode, times):
        pending, states = wave.pending, wave.states
        start = perf_counter()
        if ready_mode:
            connections = [self.workers[w][0] for w in sorted(pending)]
            available = wait(connections, timeout=WORKER_TIMEOUT_SECONDS)
            if not available:
                raise TimeoutError("No collector worker replied within 300 seconds")
            # Include all replies ready now; never wait for a slow worker.
            ready = [w for w in sorted(pending) if self.workers[w][0].poll()]
        else:
            ready = sorted(pending)
        refreshed = []
        for w in ready:
            payload = self._receive_one(w)
            times["worker_encoding_seconds"] += payload["worker_encoding_seconds"]
            offset = w * self.envs_per_worker
            for j, state in enumerate(payload["states"]):
                states[offset + j] = state
                refreshed.append(offset + j)
            pending.remove(w)
        times["worker_wait_seconds"] += perf_counter() - start
        return ready, refreshed

    def _infer_live(self, policy, live, wave, worker_encoding, times):
        import torch

        states, encoders, hidden = wave.states, wave.encoders, wave.hidden

        from civ5rl.core.model import batch_inputs

        start = perf_counter()
        encoded = [
            self.shared[self.slot_locations[i][0]].read(
                self.slot_locations[i][1], states[i]
            )
            if worker_encoding
            else encoders[i].encode(states[i]["observation"], states[i]["actions"])
            for i in live
        ]
        batch = batch_inputs(
            encoded,
            self.device,
            non_blocking=self.config.get("performance", {}).get(
                "non_blocking_inputs", False
            ),
        )
        times["encoding_seconds"] += perf_counter() - start
        start = perf_counter()
        with torch.no_grad():
            if policy.spatial_cuda_graphs:
                torch.compiler.cudagraph_mark_step_begin()
            slots = torch.tensor(live, device=self.device)
            h = hidden.index_select(1, slots) if hidden is not None else None
            logits, critic_values, next_hidden = policy(batch, h)
            pieces = [logits.log_softmax(-1), critic_values[:, None]]
            if h is not None:
                pieces.append(h[0])
            # One D2H synchronization per inference batch, including diagnostic h.
            host = torch.cat(pieces, -1).cpu().numpy()
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

    def _dispatch_actions(
        self, ready, live, wave, inference: _InferenceOutput, worker_encoding
    ):
        states, builders, encoders = wave.states, wave.builders, wave.encoders
        pending, rngs = wave.pending, wave.rngs
        wave_decisions = 0
        dispatch = [[] for _ in self.workers]
        for j, i in enumerate(live):
            state = states[i]
            logp = inference.log_probs[j, : len(state["actions"])]
            chosen = choose(np.exp(logp), rngs[i])
            action = builders[i].record_step(
                state,
                inference.encoded[j],
                chosen,
                logp,
                inference.critic_values[j],
                None if inference.old_hidden is None else inference.old_hidden[j],
            )
            if not worker_encoding:
                encoders[i].remember_action(action, state["observation"])
            generation = wave.generations[i] if wave.generations is not None else None
            worker, slot = self.slot_locations[i]
            item = WorkerCommand(slot, generation, action)
            dispatch[worker].append(item)
            wave_decisions += 1
        for w in ready:
            if dispatch[w]:
                payload = dispatch[w]
                if wave.generations is not None:
                    payload = {"wave": wave.wave_id, "commands": payload}
                self.workers[w][0].send(("step", payload))
                if wave.generations is None:
                    pending.add(w)
        return wave_decisions

    def _start_wave(self, specs, policy, reuse_encoded, worker_encoding):
        import torch

        builders = [
            EpisodeBuilder(
                s,
                policy.variant,
                reuse_encoded,
                policy.hidden_size if policy.is_recurrent else 0,
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
        self._wave_id += 1
        for w, (conn, _) in enumerate(self.workers):
            if w * self.envs_per_worker >= len(specs):
                break
            conn.send(
                (
                    "reset",
                    {
                        "specs": specs[
                            w * self.envs_per_worker : (w + 1) * self.envs_per_worker
                        ],
                        "variant": policy.variant,
                        "wave": self._wave_id,
                        "reply_size": self.reply_size,
                    },
                )
            )
        return builders, encoders, rngs, hidden

    def _partial_wave(self, policy, wave, worker_encoding, times):
        wave.wave_id = self._wave_id
        wave.generations = [0] * len(wave.states)
        active_workers = sorted(wave.pending)
        connections = [self.workers[w][0] for w in active_workers]
        publish_delays, infer_delays = [], []
        completed = 0
        while completed < len(wave.states):
            tick = perf_counter()
            if not wait(connections, timeout=WORKER_TIMEOUT_SECONDS):
                raise TimeoutError("No partial collector reply within 300 seconds")
            live, ready, ready_at = [], [], []
            # Briefly combine replies already in flight. Never wait for a slot
            # whose next action depends on this inference, including wave tails.
            coalesce_until = perf_counter() + self.inference_wait_seconds
            while True:
                for w in active_workers:
                    conn = self.workers[w][0]
                    while conn.poll():
                        payload = self._receive_one(w)
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
                            i = w * self.envs_per_worker + slot
                            if (
                                not 0 <= slot < self.envs_per_worker
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
                    len(live) >= self.inference_min_batch
                    or len(live) + completed == len(wave.states)
                    or remaining <= 0
                    or not wait(connections, timeout=remaining)
                ):
                    break
            times["worker_wait_seconds"] += perf_counter() - tick
            if live:
                now = perf_counter()
                infer_delays.extend(now - at for at in ready_at)
                inference = self._infer_live(policy, live, wave, worker_encoding, times)
                wave.decisions += self._dispatch_actions(
                    ready,
                    live,
                    wave,
                    inference,
                    worker_encoding,
                )
        times["worker_publish_delay_seconds"] += sum(publish_delays)
        times["ready_to_infer_dispatch_seconds"] += sum(infer_delays)
        times["readiness_samples"] += len(infer_delays)
        times["worker_publish_delay_max_seconds"] = max(
            times["worker_publish_delay_max_seconds"], max(publish_delays, default=0.0)
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
        from civ5rl.provenance import validate_cases

        if fixed_specs is not None:
            validate_cases(fixed_specs, trusted_manifest)

        if self.scheduler == "streaming":
            from civ5rl.engine.streaming_collector import collect_streaming

            return collect_streaming(
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
        performance = self.config.get("performance", {})
        if performance.get("collection_scheduler", "barrier") not in (
            "barrier",
            "ready",
        ):
            raise ValueError("Unknown collection scheduler")
        ready_mode = self.scheduler == "ready"
        worker_encoding = performance.get("worker_encoding", False)
        reuse_encoded = performance.get("reuse_encoded", False)
        start_rollout = perf_counter()
        target = self.config["collection"]["min_learner_decisions_per_update"]
        if fixed_specs is None:
            if max_decisions is not None:
                target = min(target, max_decisions)
            if episode_length_hint is None:
                episode_length_hint = getattr(self, "_episode_length_hint", 128.0)
            if not math.isfinite(episode_length_hint) or episode_length_hint <= 0:
                raise ValueError("episode_length_hint must be positive and finite")
        num_envs = self.num_envs
        if num_envs != len(self.workers) * self.envs_per_worker:
            raise ValueError("Worker/env count mismatch")
        fixed_offset = 0
        wave_sizes = []
        while (
            decisions < target
            if fixed_specs is None
            else fixed_offset < len(fixed_specs)
        ):
            if (deadline is not None and perf_counter() >= deadline) or (
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
                [self.factory.make(next_episode + i, stage) for i in range(wave_size)]
                if fixed_specs is None
                else fixed_specs[fixed_offset : fixed_offset + num_envs]
            )
            if fixed_specs is None:
                self.factory.validate_specs(specs)
            wave_sizes.append(len(specs))
            wave = _WaveState(
                *self._start_wave(specs, policy, reuse_encoded, worker_encoding),
                completed_at=[None] * len(specs),
                states=[None] * len(specs),
                pending=set(
                    range(
                        (len(specs) + self.envs_per_worker - 1) // self.envs_per_worker
                    )
                ),
            )
            if self.reply_size:
                self._partial_wave(policy, wave, worker_encoding, times)
                wave.pending.clear()
            while wave.pending:
                ready, refreshed = self._poll_workers(wave, ready_mode, times)
                live = [i for i in refreshed if not wave.states[i]["done"]]
                now = perf_counter()
                for i in refreshed:
                    state = wave.states[i]
                    if state["done"] and wave.completed_at[i] is None:
                        wave.builders[i].finalize(state)
                        wave.completed_at[i] = now
                if not live:
                    continue
                inference = self._infer_live(policy, live, wave, worker_encoding, times)
                wave.decisions += self._dispatch_actions(
                    ready,
                    live,
                    wave,
                    inference,
                    worker_encoding,
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
