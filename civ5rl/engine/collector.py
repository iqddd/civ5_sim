"""Complete-episode waves with ready-worker pipelining; learner owns the GPU."""

import multiprocessing as mp
import random
import traceback
from multiprocessing.connection import wait
from time import perf_counter

import numpy as np

from civ5rl.core.context import context_for
from civ5rl.core.contracts import Episode, ScenarioSpec, WorkerPayload, WorkerState
from civ5rl.core.env import DecisionEnv
from civ5rl.core.observation import ObservationEncoder
from civ5rl.provenance.common import choose
from civ5sim.engine import step_batch


def _worker(connection, profile_path=None, shared_descriptor=None):
    # This process never imports torch or initializes CUDA.
    import os

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    envs = []
    encoders = []
    previous = []
    shared = None
    if shared_descriptor is not None:
        from civ5rl.engine.shared import EncodedSlots

        shared = EncodedSlots(descriptor=shared_descriptor)
    reported_done = []
    profiler = None
    if profile_path is not None:
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()
    try:
        while True:
            kind, payload = connection.recv()
            if kind == "close":
                break
            if kind == "reset":
                for env in envs:
                    env.close()
                envs = []
                specs = payload["specs"] if isinstance(payload, dict) else payload
                reported_done = [False] * len(specs)
                encoders = []
                for spec in specs:
                    envs.append(DecisionEnv(spec))
                    if shared is not None:
                        encoders.append(
                            ObservationEncoder(
                                context_for(
                                    spec["scenario"],
                                    spec["armies"],
                                    spec["learner_side"],
                                    spec["max_rounds"],
                                    0,
                                ),
                                payload["variant"],
                            )
                        )
            elif kind == "step":
                selected = [(envs[i], action) for i, action in payload]
                for env, action in selected:
                    env.prepare(action)
                start = perf_counter()
                results = step_batch(
                    [e.battle for e, _ in selected], [a for _, a in selected]
                )
                elapsed = (perf_counter() - start) / max(len(selected), 1)
                for (env, action), result in zip(selected, results):
                    env.timings["simulation_seconds"] += elapsed
                    env.applied(action, result)
                if shared is not None:
                    for i, action in payload:
                        encoders[i].remember_action(action, previous[i]["observation"])
            states = []
            encoding_seconds = 0.0
            for i, env in enumerate(envs):
                state = {"done": True} if reported_done[i] else env.request()
                reported_done[i] = state["done"]
                if shared is not None and not state["done"]:
                    start = perf_counter()
                    state.update(
                        shared.write(
                            i,
                            encoders[i].encode(state["observation"], state["actions"]),
                        )
                    )
                    encoding_seconds += perf_counter() - start
                states.append(state)
            previous = states
            connection.send(
                ("ok", {"states": states, "worker_encoding_seconds": encoding_seconds})
            )
    except EOFError:
        pass
    except BaseException:  # noqa: BLE001 -- worker boundary must preserve reproduction
        connection.send(
            (
                "error",
                {
                    "traceback": traceback.format_exc(),
                    "episodes": [e.reproduction() for e in envs],
                },
            )
        )
    finally:
        if profiler is not None:
            profiler.disable()
            profiler.dump_stats(profile_path)
        for env in envs:
            env.close()
        if shared is not None:
            shared.close()
        connection.close()


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
        ep["old_hidden"].append(
            None
            if old_hidden is None or len(ep["chosen"]) == 1
            else old_hidden.reshape(1, 1, self.hidden_size).copy()
        )
        return action

    def finalize(self, state: WorkerState):
        self.episode["result"] = state["result"]
        self.episode["joint_commands"] = state["joint_commands"]
        return self.episode


class Collector:
    def __init__(self, config, factory, device, error_dir):
        from civ5rl.provenance.validation import validate_config

        validate_config(config)
        self.config, self.factory, self.device = config, factory, device
        self.error_dir = error_dir
        self.workers = []
        self.shared = []
        self.envs_per_worker = config["collection"]["envs_per_worker"]
        ctx = mp.get_context("spawn")
        try:
            for _ in range(config["collection"]["cpu_workers"]):
                parent = child = process = None
                try:
                    parent, child = ctx.Pipe()
                    shared = None
                    if config.get("performance", {}).get("worker_encoding", False):
                        from civ5rl.engine.shared import EncodedSlots

                        shared = EncodedSlots(
                            self.envs_per_worker,
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
                        target=_worker,
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

    def _receive_one(self, index) -> WorkerPayload:
        from civ5rl.provenance.common import write_json

        conn, proc = self.workers[index]
        if not conn.poll(300):
            raise TimeoutError(f"Collector worker {proc.pid} timed out")
        kind, payload = conn.recv()
        if kind != "ok":
            write_json(self.error_dir / "worker-error.json", payload)
            raise RuntimeError(payload["traceback"])
        return payload

    def _poll_workers(self, pending, states, ready_mode, times):
        start = perf_counter()
        if ready_mode:
            connections = [self.workers[w][0] for w in sorted(pending)]
            available = wait(connections, timeout=300)
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

    def _infer_live(
        self, policy, live, states, encoders, hidden, worker_encoding, times
    ):
        import torch

        from civ5rl.core.model import batch_inputs

        start = perf_counter()
        encoded = [
            self.shared[i // self.envs_per_worker].read(
                i % self.envs_per_worker, states[i]
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
            slots = torch.tensor(live, device=self.device)
            h = hidden.index_select(1, slots) if hidden is not None else None
            logits, values, next_hidden = policy(batch, h)
            log_probs = logits.log_softmax(-1)
            # One D2H synchronization per inference batch, including diagnostic h.
            pieces = [log_probs, values[:, None]]
            if h is not None:
                pieces.append(h[0])
            host = torch.cat(pieces, -1).cpu().numpy()
            action_width = logits.shape[1]
            log_probs, values = host[:, :action_width], host[:, action_width]
            old_hidden = host[:, action_width + 1 :]
            if next_hidden is not None:
                hidden.index_copy_(1, slots, next_hidden)
        times["inference_seconds"] += perf_counter() - start
        times["inference_batches"] += 1
        times["inference_batch_items"] += len(live)
        return encoded, log_probs, values, old_hidden, h

    def _dispatch_actions(
        self,
        ready,
        pending,
        live,
        states,
        builders,
        encoded,
        log_probs,
        values,
        old_hidden,
        h,
        rngs,
        worker_encoding,
        encoders,
    ):
        wave_decisions = 0
        dispatch = [[] for _ in self.workers]
        for j, i in enumerate(live):
            state = states[i]
            logp = log_probs[j, : len(state["actions"])]
            chosen = choose(np.exp(logp), rngs[i])
            action = builders[i].record_step(
                state,
                encoded[j],
                chosen,
                logp,
                values[j],
                None if h is None else old_hidden[j],
            )
            if not worker_encoding:
                encoders[i].remember_action(action, state["observation"])
            dispatch[i // self.envs_per_worker].append(
                (i % self.envs_per_worker, action)
            )
            wave_decisions += 1
        for w in ready:
            if dispatch[w]:
                self.workers[w][0].send(("step", dispatch[w]))
                pending.add(w)
        return wave_decisions

    def collect(
        self,
        policy,
        stage,
        next_episode,
        monitor=None,
        fixed_specs=None,
        trusted_manifest=None,
    ):
        import torch

        from civ5rl.provenance import validate_cases

        if fixed_specs is not None:
            validate_cases(fixed_specs, trusted_manifest)

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
        }
        performance = self.config.get("performance", {})
        if performance.get("collection_scheduler", "barrier") not in (
            "barrier",
            "ready",
        ):
            raise ValueError("Unknown collection scheduler")
        ready_mode = performance.get("collection_scheduler", "barrier") == "ready"
        worker_encoding = performance.get("worker_encoding", False)
        reuse_encoded = performance.get("reuse_encoded", False)
        start_rollout = perf_counter()
        target = self.config["collection"]["min_learner_decisions_per_update"]
        num_envs = self.config["collection"]["num_envs"]
        if num_envs != len(self.workers) * self.envs_per_worker:
            raise ValueError("Worker/env count mismatch")
        fixed_offset = 0
        while (
            decisions < target
            if fixed_specs is None
            else fixed_offset < len(fixed_specs)
        ):
            specs = (
                [self.factory.make(next_episode + i, stage) for i in range(num_envs)]
                if fixed_specs is None
                else fixed_specs[fixed_offset : fixed_offset + num_envs]
            )
            if len(specs) != num_envs:
                raise ValueError("Evaluation cases must fill a complete wave")
            if fixed_specs is None:
                self.factory.validate_specs(specs)
            builders = [
                EpisodeBuilder(
                    s,
                    policy.variant,
                    reuse_encoded,
                    policy.hidden_size if policy.variant == "C" else 0,
                )
                for s in specs
            ]
            encoders = (
                []
                if worker_encoding
                else [
                    ObservationEncoder(
                        context_for(
                            s["scenario"],
                            s["armies"],
                            s["learner_side"],
                            s["max_rounds"],
                            0,
                        ),
                        policy.variant,
                    )
                    for s in specs
                ]
            )
            rngs = [random.Random(s["agent_seeds"][s["learner_side"]]) for s in specs]
            hidden = (
                torch.zeros((1, num_envs, policy.hidden_size), device=self.device)
                if policy.variant == "C"
                else None
            )
            completed_at = [None] * num_envs
            states = [None] * num_envs
            pending = set(range(len(self.workers)))
            for w, (conn, _) in enumerate(self.workers):
                conn.send(
                    (
                        "reset",
                        {
                            "specs": specs[
                                w * self.envs_per_worker : (w + 1)
                                * self.envs_per_worker
                            ],
                            "variant": policy.variant,
                        },
                    )
                )
            wave_decisions = 0
            while pending:
                ready, refreshed = self._poll_workers(
                    pending, states, ready_mode, times
                )
                live = [i for i in refreshed if not states[i]["done"]]
                now = perf_counter()
                for i in refreshed:
                    state = states[i]
                    if state["done"] and completed_at[i] is None:
                        builders[i].finalize(state)
                        completed_at[i] = now
                if not live:
                    continue
                encoded, log_probs, values, old_hidden, h = self._infer_live(
                    policy, live, states, encoders, hidden, worker_encoding, times
                )
                wave_decisions += self._dispatch_actions(
                    ready,
                    pending,
                    live,
                    states,
                    builders,
                    encoded,
                    log_probs,
                    values,
                    old_hidden,
                    h,
                    rngs,
                    worker_encoding,
                    encoders,
                )
            times["idle_slot_seconds"] += sum(perf_counter() - t for t in completed_at)
            if wave_decisions == 0:
                raise RuntimeError("Corpus wave produced no learner decisions")
            episodes.extend(builder.episode for builder in builders)
            decisions += wave_decisions
            next_episode += num_envs
            fixed_offset += num_envs
            if monitor:
                monitor.mark("collection", rollout_decisions=decisions)
        times["collection_seconds"] = perf_counter() - start_rollout
        times["inference_batch_mean"] = times["inference_batch_items"] / max(
            times["inference_batches"], 1
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
