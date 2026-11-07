"""CPU-only worker publishing bounded groups of independently ready episodes."""

from collections import deque
from time import perf_counter

from civ5rl.core.contracts import WorkerStepItem
from civ5rl.engine.worker_runtime import _WorkerRuntime


def worker(connection, profile_path=None, shared_descriptor=None):
    runtime = _WorkerRuntime(connection, profile_path, shared_descriptor)
    offered, done, generations = [], [], []
    pending = deque()
    wave_id, reply_size = None, None

    def command(kind, payload):
        nonlocal offered, done, generations
        nonlocal wave_id, reply_size
        if kind == "close":
            return False
        if kind == "warmup":
            if pending or any(offered):
                raise RuntimeError("Warmup requires an idle worker")
            connection.send(("warmup", runtime.warmup(payload)))
            return True
        if kind == "reset":
            if pending or any(offered):
                raise RuntimeError("Reset before all episode replies completed")
            wave_id, reply_size = payload["wave"], payload["reply_size"]
            runtime.reset(payload["specs"], payload["variant"])
            offered = [False] * len(runtime.envs)
            done = [False] * len(runtime.envs)
            generations = [0] * len(runtime.envs)
            pending.extend(range(len(runtime.envs)))
        elif kind == "step":
            if payload["wave"] != wave_id:
                raise RuntimeError("Stale episode wave")
            commands = payload["commands"]
            indices = [i for i, _, _ in commands]
            if len(set(indices)) != len(indices):
                raise RuntimeError("Duplicate slot in learner commands")

            def validate(item):
                i, generation, action = item
                if (
                    not 0 <= i < len(runtime.envs)
                    or not offered[i]
                    or done[i]
                    or generation != generations[i]
                ):
                    raise RuntimeError(
                        "Stale learner action or slot ownership violation"
                    )
                if action not in runtime.previous[i]["actions"]:
                    raise RuntimeError("Learner action not in offered legal list")

            def applied(item):
                offered[item.slot] = False
                pending.append(item.slot)

            runtime.step(
                commands, interleave_history=True, validate=validate, applied=applied
            )
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
            if not pending:
                continue
            items = []
            encoding_seconds = request_seconds = 0.0
            for _ in range(min(reply_size, len(pending))):
                i = pending.popleft()
                if offered[i] or done[i]:
                    raise RuntimeError("Slot advanced before its previous action")
                start = perf_counter()
                state = runtime.envs[i].request()
                request_seconds += perf_counter() - start
                done[i] = state["done"]
                encoding_seconds += runtime.encode(i, state)
                offered[i] = not state["done"]
                generations[i] += 1
                runtime.previous[i] = state
                items.append(WorkerStepItem(i, generations[i], state, perf_counter()))
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
    except BaseException:  # noqa: BLE001 -- propagate worker failures to parent
        runtime.send_error()
    finally:
        runtime.close()
