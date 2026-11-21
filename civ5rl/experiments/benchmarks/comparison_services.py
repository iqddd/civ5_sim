"""Diagnostic process-isolated inference executor for saved comparison work.

CPU workers remain in one pool. Each inference process owns its models, GRU
states, shared slots and GIL; requests of a match stay on its assigned service.
This is a benchmark alternative, not a training runner or passport migration.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import replace
from multiprocessing import Pipe
from threading import Lock, Thread

from civ5rl.engine.evaluation.executor import evaluation_precision
from civ5rl.engine.evaluation.proxy import _remote_task
from civ5rl.engine.runtime.shared import pool
from civ5rl.experiments.comparison.queue import CollectionExecutor
from civ5rl.provenance.common import configure


def _service(checkpoints, runtime, control, prefix_states=False):
    configure(17, runtime.device)
    with evaluation_precision():
        executor = CollectionExecutor(checkpoints, runtime, prefix_states=prefix_states)
        executor.stop.clear()
        thread = Thread(target=executor._serve, daemon=True)
        thread.start()
        control.send(("ready", None))
        try:
            while True:
                if executor.error:
                    raise RuntimeError(executor.error)
                if not control.poll(0.1):
                    continue
                command, value = control.recv()
                if command == "close":
                    break
                if command == "submit":
                    connection, kind, task = value
                    with executor.lock:
                        executor.connections[connection] = None
                        executor.kinds[connection] = kind
                        executor.tasks[connection] = task
                elif command == "statistics":
                    control.send(("statistics", executor.statistics()))
                else:
                    raise ValueError(command)
        finally:
            executor.stop.set()
            thread.join()
            executor._fail_all("Inference service stopped")
            for connection, agent in list(executor.connections.items()):
                executor.inference.release_agent(agent)
                connection.close()
            for shared in executor.shared.values():
                shared.close()
            control.close()


class ProcessInferenceExecutor:
    def __init__(self, checkpoints, runtime, services, *, prefix_states=False):
        if services < 1 or services > runtime.workers:
            raise ValueError("Invalid inference process count")
        self.checkpoints, self.runtime, self.count = checkpoints, runtime, services
        self.services = []
        self.pool = None
        self.next_service = 0
        self.loads = [0] * services
        self.lock = Lock()
        self.prefix_states = prefix_states

    def __enter__(self):

        context = mp.get_context("spawn")
        try:
            for i in range(self.count):
                parent, child = Pipe()
                capacity = (self.runtime.workers + self.count - 1 - i) // self.count
                runtime = replace(
                    self.runtime, workers=capacity, max_batch=min(self.runtime.max_batch, capacity)
                )
                process = context.Process(
                    target=_service, args=(self.checkpoints, runtime, child, self.prefix_states)
                )
                process.start()
                child.close()
                self.services.append((process, parent))
            for process, control in self.services:
                if not control.poll(300):
                    raise TimeoutError("Inference service startup timed out")
                if control.recv()[0] != "ready":
                    raise RuntimeError("Inference service did not initialize")
            self.pool = pool(self.runtime.workers)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def check(self):
        for process, _ in self.services:
            if not process.is_alive():
                raise RuntimeError(f"Inference process exited: {process.exitcode}")

    def submit(self, function, task, kind):

        self.check()
        server, client = Pipe()
        with self.lock:
            selected = min(
                range(self.count), key=lambda i: (self.loads[i], (i - self.next_service) % self.count)
            )
            self.loads[selected] += 1
            self.next_service = (selected + 1) % self.count
        _, control = self.services[selected]
        control.send(("submit", (server, kind, task)))
        server.close()
        future = self.pool.submit(_remote_task, function, task, client)

        def finished(_):
            with self.lock:
                self.loads[selected] -= 1

        future.add_done_callback(finished)
        return future, client

    def statistics(self):
        self.check()
        for _, control in self.services:
            control.send(("statistics", None))
        rows = []
        for _, control in self.services:
            if not control.poll(300):
                raise TimeoutError("Inference statistics timed out")
            kind, value = control.recv()
            if kind != "statistics":
                raise RuntimeError("Unexpected service response")
            rows.append(value)
        result = {key: sum(row[key] for row in rows) for key in rows[0] if key != "mean_batch"}
        result["mean_batch"] = result["inference_items"] / max(1, result["inference_batches"])
        return result

    def __exit__(self, *args):
        for process, control in self.services:
            try:
                if process.is_alive():
                    control.send(("close", None))
            except (OSError, EOFError):
                pass
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
        for process, control in self.services:
            process.join(15)
            if process.is_alive():
                process.terminate()
                process.join(10)
            control.close()
