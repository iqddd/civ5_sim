"""CPU-only worker publishing bounded groups of independently ready episodes."""

import os
import traceback
from collections import deque
from time import perf_counter

from civ5rl.core.context import context_from_spec
from civ5rl.core.env import DecisionEnv
from civ5rl.core.observation import ObservationEncoder
from civ5sim.engine import step_batch


def worker(connection, profile_path=None, shared_descriptor=None):
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    envs, encoders, previous = [], [], []
    offered, done, generations = [], [], []
    pending = deque()
    wave_id, reply_size = None, None
    shared = None
    profiler = None
    if shared_descriptor is not None:
        from civ5rl.engine.shared import EncodedSlots

        shared = EncodedSlots(descriptor=shared_descriptor)
    if profile_path is not None:
        import cProfile

        profiler = cProfile.Profile()
        profiler.enable()

    def command(kind, payload):
        nonlocal envs, encoders, previous, offered, done, generations, wave_id, reply_size
        if kind == "close":
            return False
        if kind == "reset":
            if pending or any(offered):
                raise RuntimeError("Reset before all episode replies completed")
            for env in envs:
                env.close()
            envs, encoders = [], []
            wave_id, reply_size = payload["wave"], payload["reply_size"]
            for spec in payload["specs"]:
                envs.append(DecisionEnv(spec))
                if shared is not None:
                    encoders.append(
                        ObservationEncoder(context_from_spec(spec), payload["variant"])
                    )
            previous = [None] * len(envs)
            offered = [False] * len(envs)
            done = [False] * len(envs)
            generations = [0] * len(envs)
            pending.extend(range(len(envs)))
        elif kind == "step":
            if payload["wave"] != wave_id:
                raise RuntimeError("Stale episode wave")
            commands = payload["commands"]
            indices = [i for i, _, _ in commands]
            if len(set(indices)) != len(indices):
                raise RuntimeError("Duplicate slot in learner commands")
            for i, generation, action in commands:
                if (
                    not 0 <= i < len(envs)
                    or not offered[i]
                    or done[i]
                    or generation != generations[i]
                ):
                    raise RuntimeError("Stale learner action or slot ownership violation")
                if action not in previous[i]["actions"]:
                    raise RuntimeError("Learner action not in offered legal list")
                envs[i].prepare(action)
            start = perf_counter()
            results = step_batch(
                [envs[i].battle for i in indices], [a for _, _, a in commands]
            )
            elapsed = (perf_counter() - start) / max(len(commands), 1)
            for (i, _, action), result in zip(commands, results):
                envs[i].timings["simulation_seconds"] += elapsed
                envs[i].applied(action, result)
                if shared is not None:
                    encoders[i].remember_action(action, previous[i]["observation"])
                offered[i] = False
                pending.append(i)
        else:
            raise ValueError(f"Unknown worker command: {kind!r}")
        return True

    try:
        running = True
        while running:
            if not pending:
                running = command(*connection.recv())
            while running and connection.poll():
                running = command(*connection.recv())
            if not running:
                break
            items = []
            encoding_seconds = request_seconds = 0.0
            for _ in range(min(reply_size, len(pending))):
                i = pending.popleft()
                if offered[i] or done[i]:
                    raise RuntimeError("Slot advanced before its previous action")
                start = perf_counter()
                state = envs[i].request()
                request_seconds += perf_counter() - start
                done[i] = state["done"]
                if shared is not None and not state["done"]:
                    start = perf_counter()
                    state.update(
                        shared.write(
                            i, encoders[i].encode(state["observation"], state["actions"])
                        )
                    )
                    encoding_seconds += perf_counter() - start
                offered[i] = not state["done"]
                generations[i] += 1
                previous[i] = state
                items.append((i, generations[i], state, perf_counter()))
            if items:
                connection.send(
                    (
                        "ok",
                        {
                            "wave": wave_id,
                            "items": items,
                            "published_at": perf_counter(),
                            "worker_encoding_seconds": encoding_seconds,
                            "worker_request_seconds": request_seconds,
                        },
                    )
                )
    except EOFError:
        pass
    except BaseException:
        connection.send(
            (
                "error",
                {
                    "traceback": traceback.format_exc(),
                    "episodes": [env.reproduction() for env in envs],
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
