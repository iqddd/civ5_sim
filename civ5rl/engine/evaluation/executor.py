"""Serve evaluation RPC requests from simulator workers.

Workers encode public observations into shared slots and own selection RNG and
diagnostics. The service owns the policies and per-agent recurrent state. A
reply returns the slot to its worker. Oversized inputs travel in the pipe
without truncation.
"""

from __future__ import annotations

import traceback
from concurrent.futures import FIRST_COMPLETED, wait
from contextlib import nullcontext
from multiprocessing import Pipe
from threading import Event, Lock, Thread
from time import perf_counter

from civ5rl.engine.evaluation.connection import ConnectionWaiter
from civ5rl.engine.evaluation.inference import AgentMember, AgentSession, BatchedInference
from civ5rl.engine.evaluation.proxy import _remote_task
from civ5rl.engine.evaluation.runtime import (
    EvaluationRuntime,
    evaluation_options,
    evaluation_precision,
    runtime_for,
)
from civ5rl.engine.evaluation.session import (
    evaluation_key,
    evaluation_policy_hash,
    evaluation_session,
    evaluation_worker_pool,
    load_evaluation_policy,
    prepare_evaluation_policy,
    started_pool_workers,
)
from civ5rl.engine.runtime.shared import EncodedSlots, pool
from civ5sim.telemetry import measure

_ConnectionWaiter = ConnectionWaiter

__all__ = [
    "AgentMember",
    "AgentSession",
    "BatchedInference",
    "ConnectionWaiter",
    "EvaluationExecutor",
    "EvaluationRuntime",
    "evaluation_key",
    "evaluation_options",
    "evaluation_policy_hash",
    "evaluation_precision",
    "evaluation_session",
    "evaluation_worker_pool",
    "load_evaluation_policy",
    "prepare_evaluation_policy",
    "runtime_for",
]


class EvaluationExecutor:
    def __init__(self, checkpoint, runtime, *, inference_metrics=None):
        self.inference_metrics = inference_metrics
        self.runtime = runtime.resolve()
        # Load through the CPU path so training AMP and compile options do not
        # leak into this FP32 evaluator. Allocation uses the parent's allocator.
        self.inference = BatchedInference(
            load_evaluation_policy(checkpoint, self.runtime.device),
            self.runtime.device,
            hidden_capacity=self.runtime.workers,
        )
        self.connections = {}
        self.shared = {}
        self.lock, self.stop = Lock(), Event()
        self.error = None
        self.wait_seconds = 0.0
        self.wait_timeouts = 0
        self._pool = None
        self._thread = None

    def __enter__(self):
        if self._pool is not None:
            raise RuntimeError("Evaluation session already open")
        self._pool = pool(self.runtime.workers)
        return self

    def __exit__(self, *args):
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def start_service(self, *, thread_name="evaluation-inference"):
        """Start serving pipes without creating or taking ownership of a worker pool."""
        if self._thread is not None:
            raise RuntimeError("Evaluation inference service already running")
        if self.error is not None:
            raise RuntimeError("Failed evaluation session must be closed")
        self.stop.clear()
        thread = Thread(target=self._serve, name=thread_name, daemon=True)
        thread.start()
        self._thread = thread

    def stop_service(self, message="Evaluation coordinator stopped"):
        """Join the serving thread and notify clients before worker shutdown."""
        if self._thread is not None:
            self.stop.set()
            self._thread.join()
            self._thread = None
            self._fail_all(message)

    def close_clients(self):
        """Release server-side pipes, recurrent state, and shared slots after serving."""
        for connection in list(self.connections):
            self._close_client(connection)

    def check_service(self, label="Evaluation"):
        if self.error is not None:
            raise RuntimeError(f"{label} inference service failed:\n{self.error}")

    def register_client(self, connection, task, **metadata):
        """Give the service a pipe and its task metadata before the worker sends requests."""
        with self.lock:
            self.configure_task(connection, task, **metadata)
            self.connections[connection] = None

    def submit_task(self, function, task, **metadata):
        """Submit through the pool owned by this executor's context manager."""
        if self._pool is None:
            raise RuntimeError("Evaluation worker pool is closed")
        return self._submit_task(self._pool, function, task, **metadata)

    def started_workers(self):
        return started_pool_workers(self._pool)

    def _submit_task(self, executor, function, task, **metadata):
        server, client = Pipe()
        try:
            self.register_client(server, task, **metadata)
            return executor.submit(_remote_task, function, task, client), client
        except BaseException:
            # Closing the client wakes the service to release a registered server.
            client.close()
            with self.lock:
                if server not in self.connections:
                    server.close()
            raise

    def _receive(self, connection):
        try:
            command, args = connection.recv()
        except (EOFError, OSError):
            command, args = "close", ()
        return self._receive_message(connection, command, args)

    def configure_agent(self, connection, agent, info):
        """Allow a specialized executor to install task-local reset state."""

    def configure_task(self, connection, task, **metadata):
        """Associate task-local metadata before a worker can reset its agent."""

    def release_task(self, connection):
        """Discard task-local metadata on close, including coordinator shutdown."""

    def _close_client(self, connection):
        with self.lock:
            agent = self.connections.pop(connection, None)
            self.release_task(connection)
        self.inference.release_agent(agent)
        shared = self.shared.pop(connection, None)
        if shared is not None:
            shared.close()
        connection.close()

    def _receive_message(self, connection, command, args):
        if command == "close":
            self._close_client(connection)
            return None
        if command == "reset":
            mode, temperature, metrics = args
            policies = self.inference.policies
            if any(p.variant != policies[0].variant for p in policies):
                raise ValueError("Policy input variants differ")
            self.inference.release_agent(self.connections.get(connection))
            previous = self.shared.pop(connection, None)
            if previous is not None:
                previous.close()
            shared = EncodedSlots(slots=1)
            self.shared[connection] = shared
            agent = AgentSession(mode, temperature, metrics, [AgentMember() for _ in policies])
            self.connections[connection] = agent
            info = {
                "shared": shared.descriptor,
                "variant": policies[0].variant,
                "attack_preview": policies[0].variant_spec.attack_preview,
            }
            self.configure_agent(connection, agent, info)
            connection.send(("ok", info))
            return None
        agent = self.connections[connection]
        if agent is None:
            raise ValueError("Evaluation agent was not reset")
        if command == "selection_mode":
            (mode,) = args
            if mode not in ("argmax", "sample"):
                raise ValueError("Unknown selection mode")
            agent.mode = mode
            connection.send(("ok", None))
            return None
        if command not in {"act", "observe"}:
            raise ValueError(f"Unknown evaluation request {command!r}")
        (state,) = args
        # The blocked producer owns no write access until the batch reply.
        # batch_inputs packs independent arrays before any reply is sent.
        return connection, (
            agent,
            command,
            self.shared[connection].read(0, state, copy=False),
        )

    def _fail_all(self, message):
        with self.lock:
            connections = list(self.connections)
        for connection in connections:
            try:
                connection.send(("error", message))
            except (OSError, EOFError):
                pass

    def _serve(self):
        waiter = None
        try:
            waiter = ConnectionWaiter()
            while not self.stop.is_set():
                with self.lock:
                    connections = list(self.connections)
                if not connections:
                    self.stop.wait(0.001)
                    continue
                ready = waiter.wait(connections, timeout=0.01)
                pending = []
                for connection in ready[: self.runtime.max_batch]:
                    request = self._receive(connection)
                    if request is not None:
                        pending.append(request)
                if not pending:
                    continue
                started = perf_counter()
                until = started + self.runtime.wait_seconds
                while True:
                    waiting = {p[0] for p in pending}
                    with self.lock:
                        target = min(
                            self.runtime.max_batch,
                            self.runtime.workers,
                            len(self.connections),
                        )
                        remaining = [c for c in self.connections if c not in waiting]
                    if len(pending) >= target or not remaining:
                        break
                    timeout = max(0, until - perf_counter())
                    more = waiter.wait(remaining, timeout=timeout)
                    if not more:
                        self.wait_timeouts += int(perf_counter() >= until)
                        break
                    for connection in more[: target - len(pending)]:
                        request = self._receive(connection)
                        if request is not None:
                            pending.append(request)
                    if perf_counter() >= until:
                        break
                self.wait_seconds += perf_counter() - started
                with measure(self.inference_metrics, "inference"):
                    results = self.inference.infer_encoded([r[1] for r in pending])
                for (connection, _), result in zip(pending, results):
                    connection.send(("ok", result))
        except BaseException as exc:  # noqa: BLE001 - deliver thread failures to waiting workers.
            # Keep diagnostic text, not a traceback retaining agents/GPU tensors.
            self.error = traceback.format_exc()
            self._fail_all(f"{type(exc).__name__}: {exc}")
        finally:
            if waiter is not None:
                waiter.close()

    def run(self, function, tasks):
        workers = (
            nullcontext(self._pool)
            if self._pool is not None
            else evaluation_worker_pool(self.runtime.workers)
        )
        with workers as executor:
            yield from self._run(executor, function, tasks)

    def _run(self, executor, function, tasks):
        tasks = iter(tasks)
        self.start_service()
        pending = {}

        def submit():
            try:
                task = next(tasks)
            except StopIteration:
                return False
            future, client = self._submit_task(executor, function, task)
            pending[future] = client
            return True

        try:
            for _ in range(self.runtime.workers):
                if not submit():
                    break
            while pending:
                self.check_service()
                done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future).close()
                    yield future.result()
                    submit()
        finally:
            self.stop_service()
            for client in pending.values():
                client.close()
            self.close_clients()

    def statistics(self):
        """Return gather overhead, timeout count and the concurrency batch ceiling.

        Each worker blocks on one proxy request, so max_batch cannot exceed the
        active connection count or worker count in practice. Stop gathering when
        all connections are pending; wait_seconds is a ceiling, not a required
        delay. batch_wait_seconds includes gather/receive overhead, while
        batch_wait_timeouts counts waits that actually exhausted their deadline.
        """
        return {
            **self.inference.statistics(),
            "batch_wait_seconds": self.wait_seconds,
            "batch_wait_timeouts": self.wait_timeouts,
            "effective_max_batch": min(self.runtime.max_batch, self.runtime.workers),
        }
