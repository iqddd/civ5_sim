"""Shared evaluation inference; workers run ordinary CPU simulator matches.

Workers own public encoding, legal ordering, selection RNG and diagnostics.
Each match uses a reusable EncodedSlots shared-memory slot; Pipe messages carry
dimensions and replies, with inline Encoded fallback for oversized legal lists.
No legal actions are truncated. CUDA policies and per-match GRU states stay in
the main process. Policies of the same variant share encoded actual histories,
but retain independent hidden states. Reset/close/failure release slots and
connections; persistent executor sessions reuse the pool and frozen policies.

Argmax without selection metrics returns only GPU-selected indices. Sampling
and diagnostics receive logits; an ensemble still combines probabilities before
selecting its action. Evaluation compilation uses default mode without CUDA
Graphs; deterministic FP32 evaluation never inherits training AMP/TF32.
"""

import math
import os
import selectors
import sys
import traceback
from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, wait
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from multiprocessing import Pipe
from multiprocessing.connection import wait as wait_connections
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter

import torch

from civ5rl.core.agent import RLAgent
from civ5rl.core.model import Policy, batch_inputs
from civ5rl.core.observation import ordered_actions
from civ5rl.engine.evaluation.proxy import (
    _remote_task,
)
from civ5rl.engine.evaluation.state import RecurrentStates
from civ5rl.engine.runtime import is_int_at_least
from civ5rl.engine.runtime.shared import pool

_options = ContextVar("evaluation_options", default={})
_session = ContextVar("evaluation_session", default=None)


class _EvaluationSession:
    def __init__(self):
        self.policies = OrderedDict()
        self.checkpoint_hashes = {}
        self.workers = None
        self.executor = None

    def close_pool(self):
        if self.executor is not None:
            executor, self.executor = self.executor, None
            executor.shutdown(wait=True, cancel_futures=True)


@contextmanager
def evaluation_session():
    """Keep compiled checkpoints and simulator processes for one sequential run."""
    session = _EvaluationSession()
    token = _session.set(session)
    try:
        yield
    finally:
        _session.reset(token)
        session.close_pool()


@contextmanager
def evaluation_worker_pool(workers):
    session = _session.get()
    if session is None:
        with pool(workers) as executor:
            yield executor
        return
    if session.workers != workers:
        session.close_pool()
        session.workers = workers
    if session.executor is None:
        session.executor = pool(workers)
    try:
        yield session.executor
    except BaseException:
        session.close_pool()
        raise


class _ConnectionWaiter:
    """Keep POSIX pipe registrations across batches; retain Windows pipe waits.

    Only the serving thread owns the selector. Synchronize removals before
    additions using saved integer descriptors: a closed Connection cannot expose
    its old fd, and that number can already belong to a newly submitted match.
    Pending workers cannot write another request until their batch reply, so
    gathering can leave them registered and filter them out of readiness.
    """

    def __init__(self):
        self.selector = selectors.DefaultSelector() if os.name == "posix" else None
        self.registered = {}

    def wait(self, connections, timeout):
        if self.selector is None:
            return wait_connections(connections, timeout=timeout)
        for connection, fd in list(self.registered.items()):
            if connection.closed:
                self.selector.unregister(fd)
                del self.registered[connection]
        for connection in connections:
            if connection not in self.registered:
                fd = connection.fileno()
                self.selector.register(fd, selectors.EVENT_READ, connection)
                self.registered[connection] = fd
        ready = {key.data for key, _ in self.selector.select(timeout)}
        # A pending peer closing its pipe wakes select too; end this gather
        # rather than spinning on its EOF until the batching timeout.
        return [connection for connection in connections if connection in ready]

    def close(self):
        if self.selector is not None:
            self.selector.close()
        self.registered.clear()


@contextmanager
def evaluation_precision():
    """Evaluate in deterministic FP32, then restore the caller's settings."""
    matmul = torch.backends.cuda.matmul.allow_tf32
    cudnn = torch.backends.cudnn.allow_tf32
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    precision = torch.get_float32_matmul_precision()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul
        torch.backends.cudnn.allow_tf32 = cudnn
        torch.set_float32_matmul_precision(precision)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)


@contextmanager
def evaluation_options(**options):
    values = {key: value for key, value in options.items() if value is not None}
    if values:
        EvaluationRuntime(**values).resolve()
    token = _options.set(values)
    try:
        yield
    finally:
        _options.reset(token)


def runtime_for(directory, config, *, cp_only=False):
    from civ5rl.provenance.common import read_json, write_json

    path = Path(directory) / "evaluation-runtime.json"
    required = config.get("evaluation_runtime")
    if path.exists():
        defaults = read_json(path)["runtime"]
    else:
        record = (
            read_json(Path(directory) / "run.json")
            if (Path(directory) / "run.json").exists()
            else None
        )
        historical = (
            record is not None and record.get("evaluation_runtime_default") != "auto"
        ) or (record is None and (Path(directory) / "evaluations.json").exists())
        # The run marker distinguishes a genuinely new experiment from an old
        # partially unpacked run whose evaluations.json is still archived.
        default_workers = int(config.get("workers", 4) or 4)
        if (
            not required
            and not historical
            and "workers" not in _options.get()
            and config.get("self_play", {}).get("enabled")
        ):
            default_workers = _self_play_evaluation_workers()
        defaults = required or dict(
            device="cpu" if historical else "auto",
            workers=default_workers,
        )
    values = {**defaults, **_options.get()}
    if required is not None and values != required:
        raise ValueError(
            "Evaluation runtime is fixed by this experiment; backend/batching changes are not allowed"
        )
    runtime = EvaluationRuntime(**values).resolve()
    if not path.exists():
        write_json(path, {"schema": 1, "runtime": asdict(runtime)})
    return (
        EvaluationRuntime(
            "cpu", runtime.workers, runtime.max_batch, runtime.wait_seconds
        )
        if cp_only
        else runtime
    )


def _self_play_evaluation_workers():
    import psutil

    process = psutil.Process()
    affinity = (
        process.cpu_affinity()
        if hasattr(process, "cpu_affinity")
        else range(os.cpu_count() or 1)
    )
    available = len(affinity)
    if sys.platform == "linux":
        from tools.rl_collection_hardware import cgroup_limits, effective_cpu_budget

        limits = cgroup_limits()
        available = math.floor(
            effective_cpu_budget(affinity, limits["cpu_quota"], limits["cpuset"])
        )
    return max(1, min(24, available - 1))


@dataclass(frozen=True)
class EvaluationRuntime:
    device: str = "auto"
    workers: int = 4
    max_batch: int = 32
    wait_seconds: float = 0.001

    def resolve(self):
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Evaluation device must be auto, cpu or cuda")
        if not is_int_at_least(self.workers) or not is_int_at_least(self.max_batch):
            raise ValueError(
                "Evaluation workers and max_batch must be positive integers"
            )
        if not math.isfinite(self.wait_seconds) or self.wait_seconds < 0:
            raise ValueError("Evaluation wait_seconds must be finite and nonnegative")
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
        return EvaluationRuntime(
            device, self.workers, self.max_batch, self.wait_seconds
        )

    def identity(self):
        value = {
            **asdict(self),
            "precision": "float32",
            "evaluator_version": 1,
            "compiled_spatial_heads": self.device == "cuda",
            "compile_mode": "default" if self.device == "cuda" else None,
            "attention_backend": "compact_fp32"
            if self.device == "cuda"
            else "dense_fp32",
            "public_inputs": "worker_encoder_shared_slots_v1",
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        }
        if self.device == "cuda":
            value.update(
                gpu=torch.cuda.get_device_name(),
                capability=list(torch.cuda.get_device_capability()),
                cuda=torch.version.cuda,
            )
        return value


@dataclass
class AgentMember:
    hidden: torch.Tensor | None = None


@dataclass
class AgentSession:
    mode: str
    temperature: float
    collect_selection_metrics: bool
    members: list[AgentMember]


class BatchedInference:
    """Batch public inputs while retaining independent GRU state per policy."""

    def __init__(self, policy, device, *, hidden_capacity=32):
        self.policies: list[Policy] = [policy]
        self.device = device
        self.hidden_capacity = hidden_capacity
        self.hidden_banks = {}
        self.batches = self.items = 0
        self.forward_seconds = self.encoding_seconds = 0.0

    @property
    def policy(self):
        return self.policies[0]

    def gather_hidden(self, policy, agents, member=0):
        if not policy.is_recurrent:
            return None
        if torch.device(self.device).type == "cuda":
            bank = self.hidden_banks.get(member)
            if bank is None:
                bank = self.hidden_banks[member] = RecurrentStates(
                    policy.hidden_size, self.device, self.hidden_capacity
                )
            return bank.gather(agents)
        return torch.cat(
            [
                a.hidden
                if a.hidden is not None
                else torch.zeros(1, 1, policy.hidden_size, device=self.device)
                for a in agents
            ],
            dim=1,
        )

    def store_hidden(self, agents, hidden, member=0):
        if hidden is None:
            return
        bank = self.hidden_banks.get(member)
        if bank is not None:
            bank.scatter(agents, hidden)
        else:
            for agent, row in zip(agents, hidden.split(1, dim=1)):
                agent.hidden = row.clone()

    def release_agent(self, agent):
        if agent is not None:
            for member in getattr(agent, "members", [agent]):
                RecurrentStates.release(member)

    def combine_encoded_logits(self, logits_all, requests):
        logits = logits_all[0]
        choices = logits.argmax(-1).cpu().tolist()
        detailed = any(
            command != "observe"
            and (agent.mode != "argmax" or agent.collect_selection_metrics)
            for agent, command, _ in requests
        )
        host = logits.cpu().numpy() if detailed else None
        return [
            None
            if command == "observe"
            else dict(index=choices[i])
            if agent.mode == "argmax" and not agent.collect_selection_metrics
            else dict(logits=host[i, : len(encoded.sources)].copy())
            for i, (agent, command, encoded) in enumerate(requests)
        ]

    def _forward_encoded(self, requests):
        start = perf_counter()
        policies = self.policies
        if any(p.variant != policies[0].variant for p in policies):
            raise ValueError(
                "Shared public encoding requires identical policy variants"
            )
        encoded = [r[2] for r in requests]
        batch = batch_inputs(
            encoded,
            self.device,
            non_blocking=True,
            compact_units=self.policy.attention_backend == "compact_fp32",
        )
        self.encoding_seconds += perf_counter() - start
        start = perf_counter()
        logits_all = []
        for member, policy in enumerate(policies):
            agents = [r[0].members[member] for r in requests]
            hidden = self.gather_hidden(policy, agents, member)
            logits, _, last = policy(batch, hidden, compute_value=False)
            logits_all.append(logits)
            self.store_hidden(agents, last, member)
        self.batches += 1
        self.items += len(requests)
        return logits_all, start

    @torch.inference_mode()
    def infer_encoded(self, requests):
        """One public encoding shared by policies; each policy keeps its own GRU."""
        logits_all, start = self._forward_encoded(requests)
        results = (
            self.combine_encoded_logits(logits_all, requests)
            if any(r[1] != "observe" for r in requests)
            else [None] * len(requests)
        )
        self.forward_seconds += perf_counter() - start
        return results

    @torch.inference_mode()
    def forward_encoded_argmax(self, requests) -> torch.Tensor:
        """Enqueue public-encoded forward + argmax on device without host synchronization."""
        logits, start = self._forward_encoded(requests)
        choices = logits[0].argmax(-1)
        self.forward_seconds += perf_counter() - start
        return choices

    @torch.inference_mode()
    def infer(self, requests):
        # requests are (agent, command, observation, legal_actions, forced_action).
        started = perf_counter()
        actions = [ordered_actions(r[3]) for r in requests]
        encoded = [r[0].encoder.encode(r[2], a) for r, a in zip(requests, actions)]
        batch = batch_inputs(
            encoded,
            self.device,
            compact_units=self.policy.attention_backend == "compact_fp32",
        )
        agents = [r[0] for r in requests]
        hidden = self.gather_hidden(self.policy, agents)
        self.encoding_seconds += perf_counter() - started
        started = perf_counter()
        logits, _, next_hidden = self.policy(batch, hidden, compute_value=False)
        fast = [
            r[1] != "observe"
            and r[0].mode == "argmax"
            and not r[0].collect_selection_metrics
            for r in requests
        ]
        need_logits = any(
            r[1] != "observe" and not greedy for r, greedy in zip(requests, fast)
        )
        choices = logits.argmax(-1).cpu().tolist() if any(fast) else None
        host_logits = logits.cpu() if need_logits else None
        self.forward_seconds += perf_counter() - started
        self.store_hidden(agents, next_hidden)
        outputs = []
        for index, (agent, command, observation, _, forced) in enumerate(requests):
            if command == "observe":
                if forced not in actions[index]:
                    raise ValueError("Forced action is not legal")
                agent.encoder.remember_action(forced, observation)
                outputs.append(None)
            else:
                choice = (
                    agent.action_from_index(choices[index], observation, actions[index])
                    if fast[index]
                    else agent.action_from_logits(
                        host_logits[index : index + 1, : len(actions[index])],
                        observation,
                        actions[index],
                    )
                )
                outputs.append((choice, agent.drain_diagnostics()))
        self.batches += 1
        self.items += len(requests)
        return outputs

    def statistics(self):
        return dict(
            inference_batches=self.batches,
            inference_items=self.items,
            mean_batch=self.items / max(1, self.batches),
            forward_seconds=self.forward_seconds,
            encoding_seconds=self.encoding_seconds,
        )


def prepare_evaluation_policy(policy, device):
    """FP32 inference with compiled spatial/heads; no training AMP or graph pools."""
    policy = policy.to(device).eval()
    if str(device).startswith("cuda"):
        policy.configure_execution(attention_backend="compact_fp32")
        policy.enable_compile(dynamic=True, mode="default", spatial_mode="default")
    return policy


def _checkpoint_identity(checkpoint):
    path = Path(checkpoint).resolve()
    stat = path.stat()
    return path, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def evaluation_policy_hash(checkpoint):
    from civ5rl.experiments.ppo_after_imitation import policy_hash

    session = _session.get()
    key = _checkpoint_identity(checkpoint) if session is not None else None
    if session is not None and key in session.checkpoint_hashes:
        return session.checkpoint_hashes[key]
    fingerprint = policy_hash(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
    if session is not None:
        session.checkpoint_hashes[key] = fingerprint
    return fingerprint


def load_evaluation_policy(checkpoint, device) -> Policy:
    """Reuse the two most recent CUDA checkpoints without changing their weights."""
    session = _session.get()
    key = None
    if session is not None and str(device).startswith("cuda"):
        key = (str(device), *_checkpoint_identity(checkpoint))
        cached = session.policies.get(key)
        if cached is not None:
            session.policies.move_to_end(key)
            return cached
    agent = RLAgent.from_checkpoint(checkpoint, "cpu", "argmax")
    policy = prepare_evaluation_policy(agent.policy, device)
    if key is not None:
        session.policies[key] = policy
        if len(session.policies) > 2:
            session.policies.popitem(last=False)
    return policy


class EvaluationExecutor:
    def __init__(self, checkpoint, runtime):
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

    def __enter__(self):
        if self._pool is not None:
            raise RuntimeError("Evaluation session already open")
        self._pool = pool(self.runtime.workers)
        return self

    def __exit__(self, *args):
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    def _receive(self, connection):
        try:
            command, args = connection.recv()
        except (EOFError, OSError):
            command, args = "close", ()
        return self._receive_message(connection, command, args)

    def configure_agent(self, connection, agent, info):
        """Allow a specialized executor to install task-local reset state."""

    def configure_task(self, connection, task):
        """Associate task-local metadata before a worker can reset its agent."""

    def _receive_message(self, connection, command, args):
        if command == "close":
            with self.lock:
                agent = self.connections.pop(connection, None)
            self.inference.release_agent(agent)
            shared = self.shared.pop(connection, None)
            if shared is not None:
                shared.close()
            connection.close()
            return None
        if command == "reset":
            from civ5rl.engine.runtime.shared import EncodedSlots

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
            agent = AgentSession(
                mode, temperature, metrics, [AgentMember() for _ in policies]
            )
            self.connections[connection] = agent
            info = dict(
                shared=shared.descriptor,
                variant=policies[0].variant,
                attack_preview=policies[0].variant_spec.attack_preview,
            )
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
            waiter = _ConnectionWaiter()
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
                results = self.inference.infer_encoded([r[1] for r in pending])
                for (connection, _), result in zip(pending, results):
                    connection.send(("ok", result))
        except BaseException as exc:
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
        if self.error is not None:
            raise RuntimeError("Failed evaluation session must be closed")
        self.stop.clear()
        thread = Thread(target=self._serve, name="evaluation-inference", daemon=True)
        thread.start()
        pending = {}

        def submit():
            try:
                task = next(tasks)
            except StopIteration:
                return False
            server, client = Pipe()
            with self.lock:
                self.connections[server] = None
                self.configure_task(server, task)
            pending[executor.submit(_remote_task, function, task, client)] = client
            return True

        try:
            for _ in range(self.runtime.workers):
                if not submit():
                    break
            while pending:
                if self.error is not None:
                    raise RuntimeError(
                        f"Evaluation inference service failed:\n{self.error}"
                    )
                done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future).close()
                    yield future.result()
                    submit()
        finally:
            self.stop.set()
            thread.join()
            self._fail_all("Evaluation coordinator stopped")
            for client in pending.values():
                client.close()
            for connection, agent in list(self.connections.items()):
                self.inference.release_agent(agent)
                connection.close()
            self.connections.clear()
            for shared in self.shared.values():
                shared.close()
            self.shared.clear()

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
