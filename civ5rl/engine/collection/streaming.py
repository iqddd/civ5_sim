"""Complete-episode streaming with bounded admission and one frozen policy."""

from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from multiprocessing.connection import wait
from queue import Empty, Queue
from threading import Event, Lock, Thread, get_native_id
from time import perf_counter, perf_counter_ns

import numpy as np

from civ5rl.core.context import context_from_spec
from civ5rl.core.observation import ObservationEncoder
from civ5rl.engine.collection.diagnostics import Diagnostics, merge_workers
from civ5rl.engine.collection.state import (
    WORKER_TIMEOUT_SECONDS,
    CollectionSlot,
    CollectionWave,
    EpisodeBuilder,
)
from civ5rl.engine.collection.workers import streaming_worker
from civ5rl.engine.runtime.budget import deadline_reached, resolve_deadline
from civ5rl.provenance.common import write_json


@dataclass(kw_only=True)
class StreamingSlot(CollectionSlot):
    """Own the state and lifecycle rules of a single environment slot in streaming mode."""

    slot_index: int
    generation: int = 0
    episode_id: int | None = None
    idle_since: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.episode_id is not None

    def remaining_decisions(self, mean_length: float) -> float:
        return max(1.0, mean_length - len(self.builder.episode["chosen"]))

    def assign(
        self,
        episode_id: int,
        spec: dict,
        *,
        policy,
        reuse_encoded: bool,
        worker_encoding: bool,
        temperature: float,
        hidden,
        record_hidden=True,
        now: float | None = None,
    ) -> float:
        self.episode_id = episode_id
        self.builder = EpisodeBuilder(
            spec,
            policy.variant,
            reuse_encoded,
            policy.hidden_size if policy.is_recurrent else 0,
            temperature,
            record_hidden=record_hidden,
        )
        self.encoder = (
            None if worker_encoding else ObservationEncoder(context_from_spec(spec), policy.variant)
        )
        self.opponent_id = spec.get("opponent_id") if spec.get("opponent") == "policy" else None
        self.opp_encoder = (
            None
            if (worker_encoding or self.opponent_id is None)
            else ObservationEncoder(context_from_spec(spec, side=1 - spec["learner_side"]), policy.variant)
        )
        self.rng = random.Random(spec["agent_seeds"][spec["learner_side"]])
        self.opp_rng = (
            random.Random(spec["agent_seeds"][1 - spec["learner_side"]])
            if self.opponent_id is not None
            else None
        )
        self.state = None
        # Lifetime rule 1 (episode-scoped): recurrent hidden state is zeroed on every new
        # episode assigned to this slot.
        # Lifetime rule 2 (wave-scoped): self.generation is intentionally NOT reset here and
        # keeps growing monotonically across all episodes until collect_streaming finishes.
        if hidden is not None:
            hidden[:, self.slot_index :: hidden.shape[1] // 2].zero_()
        current_time = perf_counter() if now is None else now
        return current_time - self.idle_since

    def accept_reply(self, item, *, is_ready: bool = False) -> dict:
        if not self.is_active or is_ready or item.generation != self.generation + 1:
            raise RuntimeError("Stale or duplicate streaming reply")
        self.generation = item.generation
        self.state = item.state
        return item.state

    def complete(self, state: dict | None = None, *, now: float | None = None) -> tuple[int, dict]:
        if state is None:
            state = self.state
        episode_id = self.episode_id
        episode = self.builder.finalize(state)
        self.builder = None
        self.encoder = None
        self.opp_encoder = None
        self.rng = None
        self.opp_rng = None
        self.opponent_id = None
        self.episode_id = None
        self.state = None
        self.idle_since = perf_counter() if now is None else now
        return episode_id, episode


class _ReplyPump:
    """Drain the duplex Pipes even while the owner sends large assignments."""

    def __init__(self, workers, diagnostics=False):
        self.connections = {connection: i for i, (connection, _) in enumerate(workers)}
        self.queue = Queue()
        self.stopped = Event()
        self.thread = Thread(target=self._run, name="collection-replies", daemon=True)
        self.diagnostics = None
        self.lock = Lock()
        self.unaccepted = 0
        if diagnostics:
            self.diagnostics = Diagnostics(0)
            self.diagnostics.gauge("received_unaccepted_observations", 0)

    def accepted(self, payload):
        if self.diagnostics:
            count = sum(not item.state["done"] for item in payload["items"])
            with self.lock:
                self.unaccepted -= count
                self.diagnostics.gauge("received_unaccepted_observations", self.unaccepted)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stopped.set()
        # Pipes stay open until the reader exits; the next collect cannot race it.
        self.thread.join()

    def _run(self):
        if self.diagnostics:
            self.diagnostics.tid = get_native_id()
        while self.connections and not self.stopped.is_set():
            for connection in wait(list(self.connections), timeout=0.05):
                worker = self.connections[connection]
                started = perf_counter()
                clock = self.diagnostics.begin() if self.diagnostics else None
                try:
                    kind, payload = connection.recv()
                except (EOFError, OSError) as exc:
                    kind, payload = "error", {"traceback": repr(exc)}
                    del self.connections[connection]
                if self.diagnostics and kind == "ok":
                    received_ns = self.diagnostics.end("parent_receive_decode", clock)
                    payload["diagnostic_received_ns"] = received_ns
                    count = sum(not item.state["done"] for item in payload["items"])
                    with self.lock:
                        self.unaccepted += count
                        self.diagnostics.gauge("received_unaccepted_observations", self.unaccepted)
                self.queue.put((worker, kind, payload, perf_counter(), perf_counter() - started))


class StreamingScheduler:
    supports_diagnostics = True

    def worker_setup(self, reply_size):
        return streaming_worker, max(reply_size, 1)

    def collect(
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
    ):
        started = perf_counter()
        export_seconds = 0
        reports = []
        with _ReplyPump(collector.workers, collector.collection_diagnostics) as replies:
            collection = _StreamingCollection(
                collector,
                replies,
                policy,
                stage,
                next_episode,
                monitor,
                fixed_specs,
                deadline,
                max_decisions,
                episode_length_hint,
            )
            episodes, following, timing, diag = collection.collect()
            if diag:
                if monitor:
                    monitor.mark("diagnostic_export")
                export_started = perf_counter()
                for connection, _ in collector.workers:
                    connection.send(("diagnostics", {"ended_ns": diag.stopped_ns}))
                for _ in collector.workers:
                    _, kind, payload, _, _ = replies.queue.get(timeout=300)
                    if kind != "diagnostics":
                        raise RuntimeError(f"Unexpected diagnostics reply: {kind}")
                    reports.append(payload)
                export_seconds = perf_counter() - export_started
        collection_seconds = perf_counter() - started - export_seconds
        timing["receiver_shutdown_seconds"] = collection_seconds - timing["collection_seconds"]
        timing["collection_seconds"] = collection_seconds
        if diag:
            export_started = perf_counter()
            timing["diagnostics"] = dict(
                schema=1,
                parent=diag.report(ended_ns=diag.stopped_ns),
                receiver=replies.diagnostics.report(ended_ns=diag.stopped_ns, include_process_cpu=False),
                **merge_workers(reports),
            )
            timing["diagnostic_export_seconds"] = export_seconds + perf_counter() - export_started
        return episodes, following, timing


class _StreamingCollection:
    """Own one collection's admission, queues and metrics on the parent thread."""

    def __init__(
        self,
        collector,
        replies,
        policy,
        stage,
        next_episode,
        monitor,
        fixed_specs,
        deadline,
        max_decisions,
        episode_length_hint,
    ):
        import torch

        self.collector = collector
        self.replies = replies
        self.policy = policy
        self.stage = stage
        self.next_episode = next_episode
        self.monitor = monitor
        self.fixed_specs = fixed_specs
        self.deadline = deadline
        self.max_decisions = max_decisions
        if episode_length_hint is None:
            episode_length_hint = 128.0
        if not math.isfinite(episode_length_hint) or episode_length_hint <= 0:
            raise ValueError("episode_length_hint must be positive and finite")
        self.episode_length_hint = episode_length_hint
        self.started = perf_counter()
        policy.eval()
        self.capacity = collector.num_envs
        performance = collector.config.get("performance", {})
        self.worker_encoding = performance.get("worker_encoding", False)
        self.reuse_encoded = performance.get("reuse_encoded", False)
        self.target = collector.config["collection"]["min_learner_decisions_per_update"]
        if max_decisions is not None:
            self.target = max_decisions if fixed_specs is not None else min(self.target, max_decisions)
        wave_id = collector.next_wave_id()
        self.diag = None
        if collector.collection_diagnostics:
            self.diag = Diagnostics(8192, collection=wave_id)
        self.slots = [StreamingSlot(slot_index=i, idle_since=self.started) for i in range(self.capacity)]
        hidden = (
            torch.zeros((1, 2 * self.capacity, policy.hidden_size), device=collector.device)
            if policy.is_recurrent
            else None
        )
        self.wave = CollectionWave(wave_id, self.slots, hidden)
        self.locations = collector.slot_locations
        self.global_slots = collector._global_slots
        for worker, (connection, _) in enumerate(collector.workers):
            connection.send(
                (
                    "reset",
                    {
                        "wave": self.wave.wave_id,
                        "variant": policy.variant,
                        "capacity": collector.worker_capacities[worker],
                        "reply_size": collector.reply_size,
                        "collection_diagnostics": collector.collection_diagnostics,
                    },
                )
            )
        self.times = dict.fromkeys(
            (
                "encoding_seconds",
                "inference_seconds",
                "worker_wait_seconds",
                "worker_receive_seconds",
                "idle_slot_seconds",
                "worker_encoding_seconds",
                "inference_batches",
                "inference_batch_items",
                "worker_request_seconds",
                "worker_replies",
                "worker_publish_delay_seconds",
                "worker_publish_delay_max_seconds",
                "ready_to_infer_dispatch_seconds",
                "readiness_samples",
                "worker_request_slices",
                "worker_request_slice_max_seconds",
            ),
            0,
        )
        self.ready = {}
        self.finished = {}
        self.admitted = 0
        self.total_lengths = 0
        self.completed_count = 0
        self.delays = []
        self.publish_delays = []
        self.batch_sizes = Counter()
        self.admission_closed = None
        self.admission_reason = None
        self.buffered = []

    def collect(self):
        self._admit()
        while any(slot.is_active for slot in self.slots):
            self._accept_available_replies()
            self._admit()
            active_count = sum(slot.is_active for slot in self.slots)
            if not active_count:
                break
            now = perf_counter()
            timeout = self._dispatch_ready(active_count, now)
            if timeout is not None:
                self._wait_for_reply(timeout, now)
        return self._finish()

    def _queue_counts(self, inference=0):
        if self.diag:
            active_count = sum(slot.is_active for slot in self.slots)
            self.diag.gauge("active_slots", active_count)
            self.diag.gauge("ready_inference_slots", len(self.ready))
            self.diag.gauge("inference_batch_slots", inference)
            self.diag.gauge("worker_side_outstanding_slots", active_count - len(self.ready) - inference)
            self.diag.gauge("idle_slots", self.capacity - active_count)
            self.diag.gauge("completed_episodes", self.completed_count)
            if self.admission_closed is not None:
                self.diag.gauge("tail_active_slots", active_count)

    def _close_admission(self, reason):
        if self.admission_closed is None:
            self.admission_closed = perf_counter()
            self.admission_reason = reason
            if self.diag:
                self.diag.sample(
                    {
                        "kind": "admission_closed",
                        "at_ns": perf_counter_ns(),
                        "active": sum(slot.is_active for slot in self.slots),
                        "reason": reason,
                    }
                )

    def _admit(self):
        if self.admission_closed is not None:
            return
        if deadline_reached(self.deadline):
            self._close_admission("deadline")
            return
        if (
            self.fixed_specs is None or self.max_decisions is not None
        ) and self.wave.decisions >= self.target:
            self._close_admission("target")
            return
        free = [slot for slot in self.slots if not slot.is_active]
        active_slots = [slot for slot in self.slots if slot.is_active]
        if self.fixed_specs is None:
            length = (
                self.total_lengths / self.completed_count
                if self.completed_count
                else self.episode_length_hint
            )
            length = max(1.0, length)
            remaining = sum(slot.remaining_decisions(length) for slot in active_slots)
            allowance = math.ceil(max(0, self.target - self.wave.decisions - remaining) / length)
        else:
            allowance = len(self.fixed_specs) - self.admitted
        load = Counter(self.locations[slot.slot_index][0] for slot in active_slots)
        assignments = [[] for _ in self.collector.workers]
        for _ in range(min(len(free), allowance)):
            if deadline_reached(self.deadline):
                self._close_admission("deadline")
                break
            slot = min(
                free,
                key=lambda s: (
                    load[self.locations[s.slot_index][0]],
                    self.locations[s.slot_index][1],
                    self.locations[s.slot_index][0],
                ),
            )
            free.remove(slot)
            worker, local_slot = self.locations[slot.slot_index]
            spec = (
                self.fixed_specs[self.admitted]
                if self.fixed_specs is not None
                else self.collector.factory.make(self.next_episode + self.admitted, self.stage)
            )
            if self.fixed_specs is None:
                self.collector.factory.validate_specs([spec])
            # Admission begins before scenario generation; finish this admitted
            # episode even if generation crosses the deadline.
            self.times["idle_slot_seconds"] += slot.assign(
                self.admitted,
                spec,
                policy=self.policy,
                reuse_encoded=self.reuse_encoded,
                worker_encoding=self.worker_encoding,
                temperature=self.collector.temperature,
                hidden=self.wave.hidden,
                record_hidden=self.collector.record_hidden,
            )
            self.admitted += 1
            load[worker] += 1
            assignments[worker].append((local_slot, spec))
        for worker, items in enumerate(assignments):
            if items:
                self.collector.workers[worker][0].send(
                    ("assign", {"wave": self.wave.wave_id, "items": items})
                )
        if self.fixed_specs is not None and self.admitted == len(self.fixed_specs):
            self._close_admission("cases_exhausted")
        self._queue_counts()

    def _accept_available_replies(self):
        while True:
            try:
                self.buffered.append(self.replies.queue.get_nowait())
            except Empty:
                break
        for worker, kind, payload, received_at, receive_seconds in self.buffered:
            self.times["worker_receive_seconds"] += receive_seconds
            if kind != "ok":
                write_json(self.collector.error_dir / "worker-error.json", payload)
                raise RuntimeError(payload["traceback"])
            if payload["wave"] != self.wave.wave_id:
                raise RuntimeError("Stale streaming collection")
            accepted_ns = perf_counter_ns() if self.diag else None
            self.replies.accepted(payload)
            for key in (
                "worker_encoding_seconds",
                "worker_request_seconds",
                "worker_request_slices",
            ):
                self.times[key] += payload.get(key, 0)
            self.times["worker_request_slice_max_seconds"] = max(
                self.times["worker_request_slice_max_seconds"],
                payload.get("worker_request_slice_max_seconds", 0),
            )
            self.times["worker_replies"] += 1
            for item in payload["items"]:
                slot_index = self.global_slots[(worker, item.slot)]
                slot = self.slots[slot_index]
                state = slot.accept_reply(item, is_ready=slot_index in self.ready)
                if self.diag:
                    metadata = item.diagnostics
                    received_ns = payload["diagnostic_received_ns"]
                    self.diag.add(
                        "observation_encode",
                        metadata["encoded_ready_ns"] - metadata["observation_ready_ns"],
                    )
                    self.diag.add(
                        "encoded_to_publish",
                        payload["diagnostic_publish_ns"] - metadata["encoded_ready_ns"],
                    )
                    self.diag.add(
                        "publish_to_received",
                        received_ns - payload["diagnostic_publish_ns"],
                    )
                    self.diag.add("received_to_accepted", accepted_ns - received_ns)
                self.publish_delays.append(payload["published_at"] - item.completed_at)
                if state["done"]:
                    episode_id, episode = slot.complete(state)
                    self.finished[episode_id] = episode
                    self.total_lengths += len(episode["chosen"])
                    self.completed_count += 1
                else:
                    self.ready[slot_index] = (received_at, item.completed_at)
                    if self.diag:
                        self.diag.requests[slot_index] = dict(
                            **item.diagnostics,
                            received_ns=received_ns,
                            accepted_ns=accepted_ns,
                            publish_ns=payload["diagnostic_publish_ns"],
                        )
                self._queue_counts()
        self.buffered.clear()

    def _dispatch_ready(self, active_count, now):
        if self.ready:
            flush_at = min(at[0] for at in self.ready.values()) + self.collector.inference_wait_seconds
            if (
                len(self.ready) >= self.collector.inference_min_batch
                or len(self.ready) == active_count
                or now >= flush_at
            ):
                live = list(self.ready)
                self.delays.extend(now - self.ready[slot][1] for slot in live)
                self.ready.clear()
                self._queue_counts(len(live))
                self.wave.decisions += self.collector.infer_and_select(
                    self.policy,
                    live,
                    self.wave,
                    sorted({self.locations[slot][0] for slot in live}),
                    self.worker_encoding,
                    self.times,
                    diagnostics=self.diag,
                )
                self.batch_sizes[len(live)] += 1
                self._queue_counts()
                self._admit()
                if self.monitor:
                    self.monitor.mark("collection", rollout_decisions=self.wave.decisions)
                return None
            timeout = max(0, flush_at - now)
        else:
            timeout = WORKER_TIMEOUT_SECONDS
        return timeout

    def _wait_for_reply(self, timeout, now):
        if self.admission_closed is None and self.deadline is not None:
            timeout = min(timeout, max(0, resolve_deadline(self.deadline) - now))
        tick = perf_counter()
        try:
            self.buffered.append(self.replies.queue.get(timeout=timeout))
            available = True
        except Empty:
            available = False
        self.times["worker_wait_seconds"] += perf_counter() - tick
        if not available and timeout == WORKER_TIMEOUT_SECONDS:
            raise TimeoutError("No streaming worker replied within 300 seconds")

    def _finish(self):
        ended = perf_counter()
        if self.diag:
            self.diag.stop()
        for slot in self.slots:
            self.times["idle_slot_seconds"] += ended - slot.idle_since
        self.times.update(
            collection_seconds=ended - self.started,
            inference_batch_mean=self.times["inference_batch_items"]
            / max(self.times["inference_batches"], 1),
            inference_batch_histogram=dict(sorted(self.batch_sizes.items())),
            ready_to_infer_dispatch_seconds=sum(self.delays),
            ready_to_infer_dispatch_p95_seconds=float(np.percentile(self.delays, 95)) if self.delays else 0,
            readiness_samples=len(self.delays),
            worker_publish_delay_seconds=sum(self.publish_delays),
            worker_publish_delay_max_seconds=max(self.publish_delays, default=0),
            drain_seconds=ended - self.admission_closed if self.admission_closed is not None else 0,
            admission_stop_reason=self.admission_reason,
            admitted_episodes=self.admitted,
            target_decisions=self.target
            if self.fixed_specs is None or self.max_decisions is not None
            else None,
            rollout_overshoot=(
                max(0, self.wave.decisions - self.target)
                if self.fixed_specs is None or self.max_decisions is not None
                else 0
            ),
        )
        return (
            [self.finished[i] for i in range(self.admitted)],
            self.next_episode + self.admitted,
            self.times,
            self.diag,
        )
