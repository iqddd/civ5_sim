"""Complete-episode streaming with bounded admission and one frozen policy."""

import math
import random
from collections import Counter
from multiprocessing.connection import wait
from queue import Empty, Queue
from threading import Event, Lock, Thread, get_native_id
from time import perf_counter, perf_counter_ns

import numpy as np

from civ5rl.core.context import context_from_spec
from civ5rl.core.observation import ObservationEncoder


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
            from civ5rl.engine.collection_diagnostics import Diagnostics

            self.diagnostics = Diagnostics(0)
            self.diagnostics.gauge("received_unaccepted_observations", 0)

    def accepted(self, payload):
        if self.diagnostics:
            count = sum(not item.state["done"] for item in payload["items"])
            with self.lock:
                self.unaccepted -= count
                self.diagnostics.gauge(
                    "received_unaccepted_observations", self.unaccepted
                )

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
                        self.diagnostics.gauge(
                            "received_unaccepted_observations", self.unaccepted
                        )
                self.queue.put(
                    (worker, kind, payload, perf_counter(), perf_counter() - started)
                )


def collect_streaming(collector, *args):
    started = perf_counter()
    export_seconds = 0
    reports = []
    with _ReplyPump(collector.workers, collector.collection_diagnostics) as replies:
        episodes, following, timing = _collect_streaming(collector, replies, *args)
        diag = collector._diagnostics
        if diag:
            monitor = args[3]
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
    timing["receiver_shutdown_seconds"] = (
        collection_seconds - timing["collection_seconds"]
    )
    timing["collection_seconds"] = collection_seconds
    if diag:
        from civ5rl.engine.collection_diagnostics import merge_workers

        export_started = perf_counter()
        timing["diagnostics"] = dict(
            schema=1,
            parent=diag.report(ended_ns=diag.stopped_ns),
            receiver=replies.diagnostics.report(
                ended_ns=diag.stopped_ns, include_process_cpu=False
            ),
            **merge_workers(reports),
        )
        timing["diagnostic_export_seconds"] = (
            export_seconds + perf_counter() - export_started
        )
        collector._diagnostics = None
    return episodes, following, timing


def _collect_streaming(
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

    from civ5rl.engine.collector import (
        WORKER_TIMEOUT_SECONDS,
        EpisodeBuilder,
        _WaveState,
    )

    if episode_length_hint is None:
        episode_length_hint = 128.0
    if not math.isfinite(episode_length_hint) or episode_length_hint <= 0:
        raise ValueError("episode_length_hint must be positive and finite")
    started = perf_counter()
    policy.eval()
    capacity = collector.num_envs
    performance = collector.config.get("performance", {})
    worker_encoding = performance.get("worker_encoding", False)
    reuse_encoded = performance.get("reuse_encoded", False)
    target = collector.config["collection"]["min_learner_decisions_per_update"]
    if max_decisions is not None:
        target = (
            max_decisions if fixed_specs is not None else min(target, max_decisions)
        )
    collector._wave_id += 1
    diag = None
    if collector.collection_diagnostics:
        from civ5rl.engine.collection_diagnostics import Diagnostics

        diag = Diagnostics(8192, collection=collector._wave_id)
    collector._diagnostics = diag
    wave = _WaveState(
        builders=[None] * capacity,
        encoders=[None] * capacity,
        rngs=[None] * capacity,
        hidden=(
            torch.zeros((1, capacity, policy.hidden_size), device=collector.device)
            if policy.is_recurrent
            else None
        ),
        completed_at=[None] * capacity,
        states=[None] * capacity,
        pending=set(),
        wave_id=collector._wave_id,
        generations=[0] * capacity,
    )
    locations = collector.slot_locations
    global_slots = {location: i for i, location in enumerate(locations)}
    for worker, (connection, _) in enumerate(collector.workers):
        connection.send(
            (
                "reset",
                {
                    "wave": wave.wave_id,
                    "variant": policy.variant,
                    "capacity": collector.worker_capacities[worker],
                    "reply_size": collector.reply_size,
                    "collection_diagnostics": collector.collection_diagnostics,
                },
            )
        )
    times = dict.fromkeys(
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
    active, ready, finished = {}, {}, {}
    admitted = 0
    total_lengths = completed_count = 0
    idle_since = [started] * capacity
    delays, publish_delays = [], []
    batch_sizes = Counter()
    admission_closed = None
    admission_reason = None
    buffered = []

    def queue_counts(inference=0):
        if diag:
            diag.gauge("active_slots", len(active))
            diag.gauge("ready_inference_slots", len(ready))
            diag.gauge("inference_batch_slots", inference)
            diag.gauge(
                "worker_side_outstanding_slots", len(active) - len(ready) - inference
            )
            diag.gauge("idle_slots", capacity - len(active))
            diag.gauge("completed_episodes", completed_count)
            if admission_closed is not None:
                diag.gauge("tail_active_slots", len(active))

    def close_admission(reason):
        nonlocal admission_closed, admission_reason
        if admission_closed is None:
            admission_closed = perf_counter()
            admission_reason = reason
            if diag:
                diag.sample(
                    {
                        "kind": "admission_closed",
                        "at_ns": perf_counter_ns(),
                        "active": len(active),
                        "reason": reason,
                    }
                )

    def admit():
        nonlocal admitted
        if admission_closed is not None:
            return
        if deadline is not None and perf_counter() >= deadline:
            close_admission("deadline")
            return
        if (
            fixed_specs is None or max_decisions is not None
        ) and wave.decisions >= target:
            close_admission("target")
            return
        free = [slot for slot in range(capacity) if slot not in active]
        if fixed_specs is None:
            length = (
                total_lengths / completed_count
                if completed_count
                else episode_length_hint
            )
            length = max(1.0, length)
            remaining = sum(
                max(1.0, length - len(wave.builders[slot].episode["chosen"]))
                for slot in active
            )
            allowance = math.ceil(max(0, target - wave.decisions - remaining) / length)
        else:
            allowance = len(fixed_specs) - admitted
        load = Counter(locations[slot][0] for slot in active)
        assignments = [[] for _ in collector.workers]
        for _ in range(min(len(free), allowance)):
            if deadline is not None and perf_counter() >= deadline:
                close_admission("deadline")
                break
            slot = min(
                free,
                key=lambda i: (load[locations[i][0]], locations[i][1], locations[i][0]),
            )
            free.remove(slot)
            worker, local_slot = locations[slot]
            spec = (
                fixed_specs[admitted]
                if fixed_specs is not None
                else collector.factory.make(next_episode + admitted, stage)
            )
            if fixed_specs is None:
                collector.factory.validate_specs([spec])
            # Admission begins before scenario generation; finish this admitted
            # episode even if generation crosses the deadline.
            wave.builders[slot] = EpisodeBuilder(
                spec,
                policy.variant,
                reuse_encoded,
                policy.hidden_size if policy.is_recurrent else 0,
                collector.temperature,
            )
            wave.encoders[slot] = (
                None
                if worker_encoding
                else ObservationEncoder(context_from_spec(spec), policy.variant)
            )
            wave.rngs[slot] = random.Random(spec["agent_seeds"][spec["learner_side"]])
            if wave.hidden is not None:
                wave.hidden[:, slot].zero_()
            wave.states[slot] = None
            active[slot] = admitted
            admitted += 1
            load[worker] += 1
            times["idle_slot_seconds"] += perf_counter() - idle_since[slot]
            assignments[worker].append((local_slot, spec))
        for worker, items in enumerate(assignments):
            if items:
                collector.workers[worker][0].send(
                    ("assign", {"wave": wave.wave_id, "items": items})
                )
        if fixed_specs is not None and admitted == len(fixed_specs):
            close_admission("cases_exhausted")
        queue_counts()

    admit()
    while active:
        while True:
            try:
                buffered.append(replies.queue.get_nowait())
            except Empty:
                break
        for worker, kind, payload, received_at, receive_seconds in buffered:
            times["worker_receive_seconds"] += receive_seconds
            if kind != "ok":
                from civ5rl.provenance.common import write_json

                write_json(collector.error_dir / "worker-error.json", payload)
                raise RuntimeError(payload["traceback"])
            if payload["wave"] != wave.wave_id:
                raise RuntimeError("Stale streaming collection")
            accepted_ns = perf_counter_ns() if diag else None
            replies.accepted(payload)
            for key in (
                "worker_encoding_seconds",
                "worker_request_seconds",
                "worker_request_slices",
            ):
                times[key] += payload.get(key, 0)
            times["worker_request_slice_max_seconds"] = max(
                times["worker_request_slice_max_seconds"],
                payload.get("worker_request_slice_max_seconds", 0),
            )
            times["worker_replies"] += 1
            for item in payload["items"]:
                slot = global_slots[(worker, item.slot)]
                if (
                    slot not in active
                    or slot in ready
                    or item.generation != wave.generations[slot] + 1
                ):
                    raise RuntimeError("Stale or duplicate streaming reply")
                wave.generations[slot] = item.generation
                wave.states[slot] = item.state
                if diag:
                    metadata = item.diagnostics
                    received_ns = payload["diagnostic_received_ns"]
                    diag.add(
                        "observation_encode",
                        metadata["encoded_ready_ns"] - metadata["observation_ready_ns"],
                    )
                    diag.add(
                        "encoded_to_publish",
                        payload["diagnostic_publish_ns"] - metadata["encoded_ready_ns"],
                    )
                    diag.add(
                        "publish_to_received",
                        received_ns - payload["diagnostic_publish_ns"],
                    )
                    diag.add("received_to_accepted", accepted_ns - received_ns)
                publish_delays.append(payload["published_at"] - item.completed_at)
                if item.state["done"]:
                    episode = wave.builders[slot].finalize(item.state)
                    finished[active.pop(slot)] = episode
                    total_lengths += len(episode["chosen"])
                    completed_count += 1
                    wave.builders[slot] = wave.encoders[slot] = wave.rngs[slot] = None
                    idle_since[slot] = perf_counter()
                else:
                    ready[slot] = (received_at, item.completed_at)
                    if diag:
                        diag.requests[slot] = dict(
                            **item.diagnostics,
                            received_ns=received_ns,
                            accepted_ns=accepted_ns,
                            publish_ns=payload["diagnostic_publish_ns"],
                        )
                queue_counts()
        buffered.clear()
        admit()
        if not active:
            break
        now = perf_counter()
        if ready:
            flush_at = (
                min(at[0] for at in ready.values()) + collector.inference_wait_seconds
            )
            if (
                len(ready) >= collector.inference_min_batch
                or len(ready) == len(active)
                or now >= flush_at
            ):
                live = list(ready)
                delays.extend(now - ready[slot][1] for slot in live)
                ready.clear()
                queue_counts(len(live))
                inference = collector._infer_live(
                    policy, live, wave, worker_encoding, times
                )
                batch_sizes[len(live)] += 1
                wave.decisions += collector._dispatch_actions(
                    sorted({locations[slot][0] for slot in live}),
                    live,
                    wave,
                    inference,
                    worker_encoding,
                )
                queue_counts()
                admit()
                if monitor:
                    monitor.mark("collection", rollout_decisions=wave.decisions)
                continue
            timeout = max(0, flush_at - now)
        else:
            timeout = WORKER_TIMEOUT_SECONDS
        if admission_closed is None and deadline is not None:
            timeout = min(timeout, max(0, deadline - now))
        tick = perf_counter()
        try:
            buffered.append(replies.queue.get(timeout=timeout))
            available = True
        except Empty:
            available = False
        times["worker_wait_seconds"] += perf_counter() - tick
        if not available and timeout == WORKER_TIMEOUT_SECONDS:
            raise TimeoutError("No streaming worker replied within 300 seconds")
    ended = perf_counter()
    if diag:
        diag.stop()
    for slot in range(capacity):
        times["idle_slot_seconds"] += ended - idle_since[slot]
    times.update(
        collection_seconds=ended - started,
        inference_batch_mean=times["inference_batch_items"]
        / max(times["inference_batches"], 1),
        inference_batch_histogram=dict(sorted(batch_sizes.items())),
        ready_to_infer_dispatch_seconds=sum(delays),
        ready_to_infer_dispatch_p95_seconds=float(np.percentile(delays, 95))
        if delays
        else 0,
        readiness_samples=len(delays),
        worker_publish_delay_seconds=sum(publish_delays),
        worker_publish_delay_max_seconds=max(publish_delays, default=0),
        drain_seconds=ended - admission_closed if admission_closed is not None else 0,
        admission_stop_reason=admission_reason,
        admitted_episodes=admitted,
        target_decisions=target
        if fixed_specs is None or max_decisions is not None
        else None,
        rollout_overshoot=(
            max(0, wave.decisions - target)
            if fixed_specs is None or max_decisions is not None
            else 0
        ),
    )
    return [finished[i] for i in range(admitted)], next_episode + admitted, times
