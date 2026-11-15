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

from concurrent.futures import wait, FIRST_COMPLETED
from contextvars import ContextVar
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from multiprocessing import Pipe
from multiprocessing.connection import wait as wait_connections
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter
from types import SimpleNamespace
import math
import traceback

import torch

from civ5rl.core.agent import RLAgent, ActionSelection
from civ5rl.core.model import batch_inputs
from civ5rl.core.observation import ordered_actions
from civ5rl.engine.pool import pool


_client = ContextVar("evaluation_client", default=None)
_proxies = ContextVar("evaluation_proxies", default=None)
_options = ContextVar("evaluation_options", default={})


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
        record = read_json(Path(directory) / "run.json") if (Path(directory) / "run.json").exists() else None
        historical = (record is not None and record.get("evaluation_runtime_default") != "auto") or (
            record is None and (Path(directory) / "evaluations.json").exists())
        # The run marker distinguishes a genuinely new experiment from an old
        # partially unpacked run whose evaluations.json is still archived.
        defaults = required or dict(device="cpu" if historical else "auto",
                                   workers=int(config.get("workers", 4) or 4))
    values = {**defaults, **_options.get()}
    if required is not None and values != required:
        raise ValueError("Evaluation runtime is fixed by this experiment; backend/batching changes are not allowed")
    runtime = EvaluationRuntime(**values).resolve()
    if not path.exists():
        write_json(path, {"schema": 1, "runtime": asdict(runtime)})
    return EvaluationRuntime("cpu", runtime.workers, runtime.max_batch, runtime.wait_seconds) if cp_only else runtime


@dataclass(frozen=True)
class EvaluationRuntime:
    device: str = "auto"
    workers: int = 4
    max_batch: int = 32
    wait_seconds: float = 0.001

    def resolve(self):
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("Evaluation device must be auto, cpu or cuda")
        if type(self.workers) is not int or self.workers < 1 or type(self.max_batch) is not int or self.max_batch < 1:
            raise ValueError("Evaluation workers and max_batch must be positive integers")
        if not math.isfinite(self.wait_seconds) or self.wait_seconds < 0:
            raise ValueError("Evaluation wait_seconds must be finite and nonnegative")
        device = self.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluation requested but CUDA is unavailable")
        return EvaluationRuntime(device, self.workers, self.max_batch, self.wait_seconds)

    def identity(self):
        from civ5rl.provenance.common import file_sha256
        value = {**asdict(self), "precision": "float32", "service_sha256": file_sha256(Path(__file__)),
                 "compiled_spatial_heads": self.device == 'cuda', "compile_mode": 'default' if self.device == 'cuda' else None,
                 "attention_backend": 'compact_fp32' if self.device == 'cuda' else 'dense_fp32',
                 "public_inputs": 'worker_encoder_shared_slots_v1',
                 "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                 "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                 "cudnn_tf32": torch.backends.cudnn.allow_tf32}
        if self.device == "cuda":
            value.update(gpu=torch.cuda.get_device_name(), capability=list(torch.cuda.get_device_capability()),
                         cuda=torch.version.cuda)
        return value


class ProxyAgent(ActionSelection):
    """Player-only Agent RPC. No model or CUDA state is allocated in a worker."""

    def __init__(self, connection, mode, temperature=1.0, collect_selection_metrics=False):
        self.connection, self.mode = connection, mode
        self.temperature, self.collect_selection_metrics = temperature, collect_selection_metrics
        self.events = []
        self.shared = None
        proxies=_proxies.get()
        if proxies is not None:proxies.append(self)

    def call(self, command, *args):
        self.connection.send((command, args))
        if not self.connection.poll(300):
            raise TimeoutError("Evaluation inference did not reply within 300 seconds")
        status, value = self.connection.recv()
        if status != "ok":
            raise RuntimeError(f"Evaluation inference failed: {value}")
        return value

    def reset(self, context):
        from civ5rl.engine.shared import EncodedSlots
        self.close()
        info = self.call("reset", self.mode, self.temperature, self.collect_selection_metrics)
        self.shared = EncodedSlots(descriptor=info['shared'])
        self.requires_attack_preview = info['attack_preview']
        self.reset_public_state(context, info['variant'])

    def act(self, observation, actions):
        actions=ordered_actions(actions)
        encoded=self.encoder.encode(observation,actions)
        result=self.call('act',self.shared.write(0,encoded))
        action=(self.action_from_index(result['index'],observation,actions) if 'index' in result else
                self.action_from_logits(torch.from_numpy(result['logits'])[None],observation,actions))
        self.events.extend(result.get('events',()))
        return action

    def observe_step(self, observation, actions, forced_action):
        actions=ordered_actions(actions)
        if forced_action not in actions:raise ValueError('Forced action is not legal')
        encoded=self.encoder.encode(observation,actions)
        self.call('observe',self.shared.write(0,encoded))
        self.encoder.remember_action(forced_action,observation)

    def drain_diagnostics(self):
        events, self.events = self.events, []
        return events

    def close(self):
        if self.shared is not None:
            self.shared.close();self.shared=None


def evaluation_agent(checkpoint, mode, *, temperature=1.0, collect_selection_metrics=False):
    connection = _client.get()
    if connection is not None:
        return ProxyAgent(connection, mode, temperature, collect_selection_metrics)
    return RLAgent.from_checkpoint(checkpoint, "cpu", mode, temperature=temperature,
                                   collect_selection_metrics=collect_selection_metrics)


def _remote_task(function, task, connection):
    token = _client.set(connection)
    agents=[]
    proxies_token=_proxies.set(agents)
    try:
        return function(task)
    finally:
        for agent in agents:agent.close()
        _proxies.reset(proxies_token)
        _client.reset(token)
        try:
            connection.send(("close", ()))
        except (OSError, EOFError):
            pass
        connection.close()


class BatchedInference:
    """One policy; every Agent has its own public encoder, RNG and GRU state."""

    def __init__(self, policy, device):
        self.policy, self.device = policy, device
        self.batches = self.items = 0
        self.forward_seconds = self.encoding_seconds = 0.0

    def combine_encoded_logits(self, logits_all, requests):
        logits=logits_all[0]
        choices=logits.argmax(-1).cpu().tolist()
        detailed=any(command!='observe' and (agent.mode!='argmax' or agent.collect_selection_metrics)
                     for agent,command,_ in requests)
        host=logits.cpu().numpy() if detailed else None
        return [None if command=='observe' else
                dict(index=choices[i]) if agent.mode=='argmax' and not agent.collect_selection_metrics else
                dict(logits=host[i,:len(encoded.sources)].copy())
                for i,(agent,command,encoded) in enumerate(requests)]

    @torch.inference_mode()
    def infer_encoded(self, requests):
        """One public encoding shared by policies; each policy keeps its own GRU."""
        start=perf_counter()
        policies=getattr(self,'policies',[self.policy])
        if any(p.variant!=policies[0].variant for p in policies):
            raise ValueError('Shared public encoding requires identical policy variants')
        encoded=[r[2] for r in requests]
        batch=batch_inputs(encoded,self.device,compact_units=self.policy.attention_backend=='compact_fp32')
        self.encoding_seconds+=perf_counter()-start
        start=perf_counter()
        logits_all=[]
        for member,policy in enumerate(policies):
            agents=[r[0].members[member] for r in requests]
            hidden=None
            if policy.is_recurrent:
                hidden=torch.cat([a.hidden if a.hidden is not None else
                                  torch.zeros(1,1,policy.hidden_size,device=self.device) for a in agents],1)
            logits,_,last=policy(batch,hidden)
            logits_all.append(logits)
            if last is not None:
                for i,agent in enumerate(agents):agent.hidden=last[:,i:i+1].clone()
        results=(self.combine_encoded_logits(logits_all,requests) if any(r[1]!='observe' for r in requests)
                 else [None]*len(requests))
        self.forward_seconds+=perf_counter()-start
        self.batches+=1;self.items+=len(requests)
        return results

    @torch.inference_mode()
    def infer(self, requests):
        # requests are (agent, command, observation, legal_actions, forced_action).
        started = perf_counter()
        actions = [ordered_actions(r[3]) for r in requests]
        encoded = [r[0].encoder.encode(r[2], a) for r, a in zip(requests, actions)]
        batch = batch_inputs(encoded, self.device, compact_units=self.policy.attention_backend=='compact_fp32')
        hidden = None
        if self.policy.is_recurrent:
            hidden = torch.cat([r[0].hidden if r[0].hidden is not None else
                                torch.zeros((1, 1, self.policy.hidden_size), device=self.device)
                                for r in requests], dim=1)
        self.encoding_seconds += perf_counter() - started
        started = perf_counter()
        logits, _, next_hidden = self.policy(batch, hidden)
        fast = [r[1]!='observe' and r[0].mode=='argmax' and not r[0].collect_selection_metrics for r in requests]
        need_logits = any(r[1]!='observe' and not greedy for r,greedy in zip(requests,fast))
        choices = logits.argmax(-1).cpu().tolist() if any(fast) else None
        host_logits = logits.cpu() if need_logits else None
        self.forward_seconds += perf_counter() - started
        outputs = []
        for index, (agent, command, observation, _, forced) in enumerate(requests):
            if next_hidden is not None:
                agent.hidden = next_hidden[:, index:index + 1].clone()
            if command == "observe":
                if forced not in actions[index]:
                    raise ValueError("Forced action is not legal")
                agent.encoder.remember_action(forced, observation)
                outputs.append(None)
            else:
                choice = (agent.action_from_index(choices[index],observation,actions[index]) if fast[index] else
                          agent.action_from_logits(host_logits[index:index + 1, :len(actions[index])], observation, actions[index]))
                outputs.append((choice, agent.drain_diagnostics()))
        self.batches += 1
        self.items += len(requests)
        return outputs

    def statistics(self):
        return dict(inference_batches=self.batches, inference_items=self.items,
                    mean_batch=self.items / max(1, self.batches), forward_seconds=self.forward_seconds,
                    encoding_seconds=self.encoding_seconds)


def prepare_evaluation_policy(policy, device):
    """FP32 inference with compiled spatial/heads; no training AMP or graph pools."""
    policy=policy.to(device).eval()
    if str(device).startswith('cuda'):
        policy.configure_execution(attention_backend='compact_fp32')
        policy.enable_compile(dynamic=True, mode='default', spatial_mode='default')
    return policy


class EvaluationExecutor:
    def __init__(self, checkpoint, runtime):
        self.runtime = runtime.resolve()
        # Load through the CPU path so training AMP and compile options do not
        # leak into this FP32 evaluator. Allocation uses the parent's allocator.
        agent = RLAgent.from_checkpoint(checkpoint, "cpu", "argmax")
        self.inference = BatchedInference(prepare_evaluation_policy(agent.policy,self.runtime.device), self.runtime.device)
        self.connections = {}
        self.shared = {}
        self.lock, self.stop = Lock(), Event()
        self.error = None
        self.wait_seconds = 0.0
        self.wait_timeouts = 0
        self._pool = None

    def __enter__(self):
        if self._pool is not None:
            raise RuntimeError('Evaluation session already open')
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
        if command == "close":
            with self.lock:
                self.connections.pop(connection, None)
            shared=self.shared.pop(connection,None)
            if shared is not None:shared.close()
            connection.close()
            return None
        if command == "reset":
            from civ5rl.engine.shared import EncodedSlots
            mode, temperature, metrics = args
            policies=getattr(self.inference,'policies',[self.inference.policy])
            if any(p.variant!=policies[0].variant for p in policies):raise ValueError('Policy input variants differ')
            previous=self.shared.pop(connection,None)
            if previous is not None:previous.close()
            shared=EncodedSlots(slots=1)
            self.shared[connection]=shared
            agent=SimpleNamespace(mode=mode,temperature=temperature,collect_selection_metrics=metrics,
                                  members=[SimpleNamespace(hidden=None) for p in policies])
            self.connections[connection] = agent
            connection.send(("ok",dict(shared=shared.descriptor,variant=policies[0].variant,
                                       attack_preview=policies[0].variant_spec.attack_preview)))
            return None
        agent = self.connections[connection]
        if agent is None:
            raise ValueError("Evaluation agent was not reset")
        if command not in {"act", "observe"}:
            raise ValueError(f"Unknown evaluation request {command!r}")
        (state,)=args
        return connection, (agent, command, self.shared[connection].read(0,state))

    def _fail_all(self, message):
        with self.lock:
            connections = list(self.connections)
        for connection in connections:
            try:
                connection.send(("error", message))
            except (OSError, EOFError):
                pass

    def _serve(self):
        try:
            while not self.stop.is_set():
                with self.lock:
                    connections = list(self.connections)
                if not connections:
                    self.stop.wait(0.001)
                    continue
                ready = wait_connections(connections, timeout=0.01)
                pending = []
                for connection in ready[:self.runtime.max_batch]:
                    request = self._receive(connection)
                    if request is not None:
                        pending.append(request)
                if not pending:
                    continue
                started = perf_counter()
                until = started + self.runtime.wait_seconds
                while True:
                    with self.lock:
                        target=min(self.runtime.max_batch,self.runtime.workers,len(self.connections))
                        remaining = [c for c in self.connections if c not in {p[0] for p in pending}]
                    if len(pending)>=target or not remaining:break
                    timeout = max(0, until - perf_counter())
                    more = wait_connections(remaining, timeout=timeout) if remaining else []
                    if not more:
                        self.wait_timeouts+=int(perf_counter()>=until)
                        break
                    for connection in more[:target - len(pending)]:
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

    def run(self, function, tasks):
        tasks = iter(tasks)
        if self.error is not None:
            raise RuntimeError('Failed evaluation session must be closed')
        self.stop.clear()
        executor = self._pool or pool(self.runtime.workers)
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
            pending[executor.submit(_remote_task, function, task, client)] = client
            return True

        try:
            for _ in range(self.runtime.workers):
                if not submit():
                    break
            while pending:
                if self.error is not None:
                    raise RuntimeError(f"Evaluation inference service failed:\n{self.error}")
                done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.pop(future).close()
                    yield future.result()
                    submit()
        finally:
            self.stop.set()
            thread.join()
            self._fail_all("Evaluation coordinator stopped")
            if self._pool is None:
                executor.shutdown(wait=True, cancel_futures=True)
            for client in pending.values():
                client.close()
            for connection in list(self.connections):
                connection.close()
            self.connections.clear()
            for shared in self.shared.values():shared.close()
            self.shared.clear()

    def statistics(self):
        """Return gather overhead, timeout count and the concurrency batch ceiling.

        Each worker blocks on one proxy request, so max_batch cannot exceed the
        active connection count or worker count in practice. Stop gathering when
        all connections are pending; wait_seconds is a ceiling, not a required
        delay. batch_wait_seconds includes gather/receive overhead, while
        batch_wait_timeouts counts waits that actually exhausted their deadline.
        """
        return {**self.inference.statistics(), "batch_wait_seconds": self.wait_seconds,
                "batch_wait_timeouts":self.wait_timeouts,
                "effective_max_batch":min(self.runtime.max_batch,self.runtime.workers)}
