"""Shared-queue, two-stage paired continuations for rollout-comparison collection.

One worker pool and one inference service run source matches and continuation
matches with no barrier between them. u171 serves every learner decision;
u107 is evaluated only for source-match requests (proposal alternatives). Every
match keeps its own public encoder, GRU state per model and CP instance.

A fork first runs repetitions 0-3 for both alternatives. If the first-group win
difference is zero, the existing filter can no longer accept the fork, so the
second group (repetitions 4-7) is never scheduled. Repetition indices and their
seeds do not depend on completion order, worker identity or batch composition.

Scheduling (execution settings, not scientific config): free workers take, in
order, a source match while running sources stay below the soft source share,
second-group continuations, first-group continuations, newly admitted forks,
then a source match on otherwise idle slots. Ready continuation stock is kept
near ``stock_factor * C`` with C the continuation slots; admitted unfinished
forks never exceed ``max_active_forks``. Before the deadline, admissions are
narrowed by a drain forecast (p90 match duration over the last 256 matches);
at the deadline admission closes and admitted work completes.
"""

from __future__ import annotations

import math
from collections import deque
from concurrent.futures import FIRST_COMPLETED, wait
from multiprocessing import Pipe
from threading import Thread
from time import perf_counter

import torch

from civ5rl.core.agent import RLAgent
from civ5rl.core.model import batch_inputs
from civ5rl.engine.evaluation.executor import (
    EvaluationExecutor,
    prepare_evaluation_policy,
)
from civ5rl.engine.evaluation.proxy import _remote_task
from civ5rl.experiments.comparison.executor import ProposalInference
from civ5rl.experiments.comparison.prefix import SourcePrefixes

FIRST_GROUP = (0, 1, 2, 3)
SECOND_GROUP = (4, 5, 6, 7)


def group_delta(winners, group, side):
    """Wins of alternative 1 minus wins of alternative 0 over one group."""
    return sum(int(winners[(r, 1)] == side) - int(winners[(r, 0)] == side) for r in group)


def two_stage_label(winners, side):
    """Exact two-stage form of the original eight-repetition preference rule.

    ``winners`` maps (repetition, alternative) to the saved winner. Accepted
    forks and weights are identical to the one-stage rule for the same outcomes.
    """
    d1 = group_delta(winners, FIRST_GROUP, side)
    executed = FIRST_GROUP if d1 == 0 else FIRST_GROUP + SECOND_GROUP
    wdl = [
        {
            k: sum(
                ("D" if winners[(r, a)] is None else "W" if winners[(r, a)] == side else "L") == k
                for r in executed
            )
            for k in "WDL"
        }
        for a in (0, 1)
    ]
    if d1 == 0:
        return {
            "label": "first_group_zero",
            "reason": None,
            "deltas": [0, None],
            "accepted": False,
            "better": None,
            "weight": 0.0,
            "executed_repetitions": len(executed),
            "wdl": wdl,
        }
    d2 = group_delta(winners, SECOND_GROUP, side)
    if d2 == 0:
        label, reason = "rejected", "second_group_zero"
    elif d1 * d2 < 0:
        label, reason = "rejected", "opposite_signs"
    else:
        label, reason = "accepted", None
    accepted = label == "accepted"
    return {
        "label": label,
        "reason": reason,
        "deltas": [d1, d2],
        "accepted": accepted,
        "better": (1 if d1 > 0 else 0) if accepted else None,
        "weight": abs(d1 + d2) / 8 if accepted else 0.0,
        "executed_repetitions": 8,
        "wdl": wdl,
    }


class SelectiveProposalInference(ProposalInference):
    """Evaluate the parent on every request and the proposer on source matches."""

    @torch.inference_mode()
    def infer_encoded(self, requests):
        start = perf_counter()
        compact = self.policy.attention_backend == "compact_fp32"
        encoded = [r[2] for r in requests]
        batch = batch_inputs(encoded, self.device, compact_units=compact)
        proposing = [i for i, r in enumerate(requests) if r[0].proposal]
        second = (
            batch
            if len(proposing) == len(requests)
            else batch_inputs([encoded[i] for i in proposing], self.device, compact_units=compact)
            if proposing
            else None
        )
        self.encoding_seconds += perf_counter() - start
        start = perf_counter()
        choices = []
        for member, (inputs, rows) in enumerate(((batch, range(len(requests))), (second, proposing))):
            if inputs is None:
                choices.append(None)
                continue
            policy = self.policies[member]
            agents = [requests[i][0].members[member] for i in rows]
            hidden = self.gather_hidden(policy, agents, member)
            logits, _, last = policy(inputs, hidden)
            self.store_hidden(agents, last, member)
            for j, agent in enumerate(agents):
                owner = requests[rows[j]][0]
                if member == 0 and hasattr(owner, "prefix_frames"):
                    owner.prefix_frames.append(agent.hidden.clone())
            choices.append(logits.argmax(-1))
        acting = any(r[1] != "observe" for r in requests)
        first = choices[0].cpu().tolist() if acting else None
        alternatives = dict(zip(proposing, choices[1].cpu().tolist())) if acting and proposing else {}
        results = []
        for i, (agent, command, _) in enumerate(requests):
            if command == "observe":
                results.append(None)
            elif agent.proposal:
                results.append(
                    {
                        "index": first[i],
                        "events": [{"kind": "proposal", "alternative": alternatives[i]}],
                    }
                )
            else:
                results.append({"index": first[i]})
        self.forward_seconds += perf_counter() - start
        self.batches += 1
        self.items += len(requests)
        self.proposal_items = getattr(self, "proposal_items", 0) + len(proposing)
        return results


class CollectionExecutor(EvaluationExecutor):
    """Persistent pool + inference thread accepting individually submitted matches."""

    def __init__(self, checkpoints, runtime, *, prefix_states=False):
        super().__init__(str(checkpoints[0]), runtime)
        second = prepare_evaluation_policy(
            RLAgent.from_checkpoint(str(checkpoints[1]), "cpu", "argmax").policy,
            self.runtime.device,
        )
        self.inference = SelectiveProposalInference(
            [self.inference.policy, second],
            self.runtime.device,
            hidden_capacity=self.runtime.workers,
        )
        self.kinds = {}
        self.tasks = {}
        self.prefixes = (
            SourcePrefixes(self.inference.policy, checkpoints[0], self.runtime) if prefix_states else None
        )
        self.thread = None

    def configure_agent(self, connection, agent, info):
        agent.proposal = self.kinds.get(connection) == "source"
        if self.prefixes is not None:
            task = self.tasks[connection]
            if agent.proposal:
                agent.prefix_frames = []
                info["capture_prefixes"] = True
            else:
                branch = task[-1]
                hidden, count = self.prefixes.restore(branch["source"], branch["command"])
                agent.members[0].hidden = hidden
                info["prefix_skip"] = count

    def _receive_message(self, connection, command, args):
        if command == "save_prefixes":
            if self.prefixes is None or self.kinds.get(connection) != "source":
                raise ValueError("Unexpected source prefix capture")
            self.prefixes.capture(self.tasks[connection][2], self.connections[connection].prefix_frames)
            connection.send(("ok", None))
            return None
        kind = self.kinds.get(connection)
        result = super()._receive_message(connection, command, args)
        agent = self.connections.get(connection)
        if agent is not None and not hasattr(agent, "proposal"):
            agent.proposal = kind == "source"
        if connection not in self.connections:
            self.kinds.pop(connection, None)
            self.tasks.pop(connection, None)
        return result

    def statistics(self):
        return {
            **super().statistics(),
            "prefix_source_hits": self.prefixes.hits if self.prefixes else 0,
            "prefix_source_prefills": self.prefixes.prefills if self.prefixes else 0,
            "prefix_observations_skipped": self.prefixes.restored_observations if self.prefixes else 0,
        }

    def __enter__(self):
        super().__enter__()
        self.stop.clear()
        self.thread = Thread(target=self._serve, name="collection-inference", daemon=True)
        self.thread.start()
        return self

    def check(self):
        if self.error is not None:
            raise RuntimeError(f"Collection inference service failed:\n{self.error}")

    def submit(self, function, task, kind):
        self.check()
        server, client = Pipe()
        with self.lock:
            self.connections[server] = None
            self.kinds[server] = kind
            self.tasks[server] = task
        future = self._pool.submit(_remote_task, function, task, client)
        return future, client

    def __exit__(self, *args):
        self.stop.set()
        if self.thread is not None:
            self.thread.join()
        self._fail_all("Collection coordinator stopped")
        try:
            super().__exit__(*args)
        finally:
            for connection, agent in list(self.connections.items()):
                self.inference.release_agent(agent)
                connection.close()
            self.connections.clear()
            self.kinds.clear()
            self.tasks.clear()
            for shared in self.shared.values():
                shared.close()
            self.shared.clear()


def p90(values):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.9 * len(ordered)) - 1)]


class Fork:
    """Runtime view of one selected fork; ``record`` is the persisted dict."""

    def __init__(self, record):
        self.record = record
        self.winners = {tuple(map(int, k.split("-"))): v for k, v in record.get("winners", {}).items()}
        self.running = 0
        self.queued = set()

    @property
    def key(self):
        return (self.record["episode"], self.record["index"])

    def missing(self, group):
        return [
            (r, a) for r in group for a in (0, 1) if (r, a) not in self.winners and (r, a) not in self.queued
        ]

    def complete(self, group):
        return all((r, a) in self.winners for r in group for a in (0, 1))

    def set_winner(self, repetition, alternative, winner):
        self.winners[(repetition, alternative)] = winner
        self.record.setdefault("winners", {})[f"{repetition}-{alternative}"] = winner


class Scheduler:
    """Mechanics of the shared queue; the protocol owns tasks, files and state.

    Protocol interface: ``next_source() -> (key, task) | None``,
    ``has_sources(running_sources) -> bool``, ``source_done(key, seconds) -> [record]``,
    ``continuation_task(record, rep, alt) -> task``,
    ``continuation_done(record, rep, alt, seconds) -> winner``,
    ``decide(record, winners)``, ``reserve_seconds() -> float``, ``tick(stats)``.
    """

    def __init__(
        self,
        executor,
        protocol,
        budget,
        *,
        workers,
        source_share=0.25,
        stock_factor=2.0,
        max_active_forks=None,
        buffer_limit=32,
        function=None,
        history=256,
    ):
        self.executor, self.protocol, self.budget = executor, protocol, budget
        self.workers = workers
        self.source_quota = max(1, math.ceil(workers * source_share)) if source_share > 0 else 0
        self.stock_factor = stock_factor
        self.max_active = max_active_forks or workers
        self.buffer_limit = buffer_limit
        self.function = function
        self.buffer = deque()
        self.active = {}
        self.ready = (deque(), deque())  # second group first, then first group
        self.running = {}
        self.source_seconds = deque(maxlen=history)
        self.continuation_seconds = deque(maxlen=history)
        self.idle_worker_seconds = 0.0
        self.started = perf_counter()
        self.completed_continuations = 0
        self.completed_sources = 0
        self.decided = 0
        self.admitted = 0
        self.buffer_empty_at = None
        self.deadline_at = None
        self.forecast = True  # measurements may disable drain-based narrowing
        self.continuation_finish_times = []

    # Restoration ---------------------------------------------------------
    def restore(self, active, buffered):
        for record in active:
            self._activate(Fork(record))
        for record in buffered:
            self.buffer.append(record)

    def add_selected(self, records):
        self.buffer.extend(records)

    def _activate(self, fork):
        self.active[fork.key] = fork
        if not self._advance_fork(fork):
            self._enqueue(fork, FIRST_GROUP, 1)

    def _advance_fork(self, fork):
        if not fork.complete(FIRST_GROUP):
            return False
        if group_delta(fork.winners, FIRST_GROUP, fork.record["side"]) == 0 or fork.complete(SECOND_GROUP):
            self._decide(fork)
        else:
            self._enqueue(fork, SECOND_GROUP, 0)
        return True

    def _enqueue(self, fork, group, priority):
        for item in fork.missing(group):
            fork.queued.add(item)
            self.ready[priority].append((fork, *item))

    # Forecasts -----------------------------------------------------------
    def remaining(self):
        return self.budget.limit - self.budget.elapsed()

    def running_counts(self):
        sources = sum(1 for kind, *_ in self.running.values() if kind == "source")
        return sources, len(self.running) - sources

    def pending_work(self):
        _, continuations = self.running_counts()
        potential = sum(8 for f in self.active.values() if not f.complete(FIRST_GROUP))
        return continuations + len(self.ready[0]) + len(self.ready[1]) + potential

    def drain(self, extra):
        tau = p90(self.continuation_seconds) if len(self.continuation_seconds) >= 16 else None
        if tau is None or not self.forecast:
            return 0.0
        sources, _ = self.running_counts()
        slots = max(1, self.workers - sources)
        return tau * (math.ceil((self.pending_work() + extra) / slots) + 1)

    def deadline(self):
        return self.budget.expired()

    def can_admit(self):
        if not self.buffer or len(self.active) >= self.max_active or self.deadline():
            return False
        return self.remaining() > self.protocol.reserve_seconds() + self.drain(16)

    def can_start_source(self):
        sources, _ = self.running_counts()
        if self.deadline() or not self.protocol.has_sources(sources):
            return False
        if len(self.buffer) >= self.buffer_limit:
            return False
        tau_source = p90(self.source_seconds) if len(self.source_seconds) >= 4 else 0.0
        extra = 64 * (sources + 1) + 16 * len(self.buffer)
        return self.remaining() > self.protocol.reserve_seconds() + tau_source + self.drain(extra)

    # Dispatch ------------------------------------------------------------
    def stock(self):
        _, continuations = self.running_counts()
        return continuations + len(self.ready[0]) + len(self.ready[1])

    def admit(self):
        record = self.buffer.popleft()
        record["status"] = "running"
        self.admitted += 1
        if not self.buffer and self.buffer_empty_at is None:
            self.buffer_empty_at = perf_counter() - self.started
        self._activate(Fork(record))

    def _start(self, kind, payload, task):
        future, client = self.executor.submit(self.function, task, kind)
        self.running[future] = (kind, payload, perf_counter(), client)

    def _start_continuation(self):
        for queue in self.ready:
            if queue:
                fork, repetition, alternative = queue.popleft()
                fork.running += 1
                self._start(
                    "continuation",
                    (fork, repetition, alternative),
                    self.protocol.continuation_task(fork.record, repetition, alternative),
                )
                return True
        return False

    def _start_source(self):
        item = self.protocol.next_source()
        if item is None:
            return False
        key, task = item
        self._start("source", key, task)
        return True

    def fill(self):
        sources, _ = self.running_counts()
        target = self.stock_factor * max(1, self.workers - sources)
        while self.stock() < target and self.can_admit():
            self.admit()
        while len(self.running) < self.workers:
            sources, _ = self.running_counts()
            source_ok = self.can_start_source()
            if source_ok and sources < self.source_quota and self._start_source():
                continue
            if self._start_continuation():
                continue
            if self.can_admit():
                self.admit()
                continue
            if source_ok and self._start_source():
                continue
            break

    # Completion ----------------------------------------------------------
    def _decide(self, fork):
        self.protocol.decide(fork.record, fork.winners)
        self.active.pop(fork.key, None)
        self.decided += 1

    def _finished(self, future):
        kind, payload, started, client = self.running.pop(future)
        client.close()
        future.result()
        seconds = perf_counter() - started
        if kind == "source":
            self.source_seconds.append(seconds)
            self.completed_sources += 1
            self.add_selected(self.protocol.source_done(payload, seconds))
            return
        fork, repetition, alternative = payload
        self.continuation_seconds.append(seconds)
        self.completed_continuations += 1
        self.continuation_finish_times.append((perf_counter() - self.started, seconds))
        fork.running -= 1
        fork.queued.discard((repetition, alternative))
        fork.set_winner(
            repetition,
            alternative,
            self.protocol.continuation_done(fork.record, repetition, alternative, seconds),
        )
        self._advance_fork(fork)

    def stats(self):
        sources, continuations = self.running_counts()
        elapsed = perf_counter() - self.started
        return {
            "elapsed": elapsed,
            "running_sources": sources,
            "running_continuations": continuations,
            "ready": len(self.ready[0]) + len(self.ready[1]),
            "active_forks": len(self.active),
            "buffered_forks": len(self.buffer),
            "admitted": self.admitted,
            "decided": self.decided,
            "completed_continuations": self.completed_continuations,
            "completed_sources": self.completed_sources,
            "idle_worker_fraction": self.idle_worker_seconds / max(1e-9, self.workers * elapsed),
            "continuation_p90_seconds": p90(self.continuation_seconds),
            "source_p90_seconds": p90(self.source_seconds),
            "buffer_empty_at": self.buffer_empty_at,
            "deadline_at": self.deadline_at,
        }

    def run(self):
        last = perf_counter()
        while True:
            self.executor.check()
            if self.deadline_at is None and self.deadline():
                self.deadline_at = perf_counter() - self.started
            self.fill()
            if not self.running:
                break
            done, _ = wait(list(self.running), timeout=0.05, return_when=FIRST_COMPLETED)
            now = perf_counter()
            self.idle_worker_seconds += (self.workers - len(self.running)) * (now - last)
            last = now
            for future in done:
                self._finished(future)
            self.protocol.tick(self.stats())
        return self.stats()
