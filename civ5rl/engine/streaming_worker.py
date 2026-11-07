"""CPU-only round-robin execution with reusable, episode-local slots."""

from collections import deque
from time import perf_counter

from civ5rl.core.contracts import WorkerStepItem
from civ5rl.engine.worker_runtime import _WorkerRuntime


def worker(connection, profile_path=None, shared_descriptor=None):
    runtime = _WorkerRuntime(connection, profile_path, shared_descriptor)
    pending = deque()
    offered, done, generations = [], [], []
    collection_id = variant = None
    reply_size = 1
    request_seconds = encoding_seconds = 0.0
    slices = 0
    max_slice_seconds = 0.0

    def command(kind, payload):
        nonlocal collection_id, variant, reply_size, offered, done, generations
        if kind == "close":
            return False
        if kind == "reset":
            if pending or any(offered):
                raise RuntimeError("Reset before streaming episodes drained")
            runtime.reset([], payload["variant"])
            capacity = payload["capacity"]
            runtime.envs = [None] * capacity
            runtime.encoders = [None] * capacity
            runtime.previous = [None] * capacity
            offered = [False] * capacity
            done = [True] * capacity
            generations = [0] * capacity
            collection_id = payload["wave"]
            variant = payload["variant"]
            reply_size = payload["reply_size"]
        elif kind == "assign":
            if payload["wave"] != collection_id:
                raise RuntimeError("Stale streaming assignment")
            for slot, spec in payload["items"]:
                if not done[slot] or offered[slot] or slot in pending:
                    raise RuntimeError("Assignment overwrites an active episode")
                runtime.assign(slot, spec, variant)
                done[slot] = False
                pending.append(slot)
        elif kind == "step":
            if payload["wave"] != collection_id:
                raise RuntimeError("Stale streaming action")
            commands = payload["commands"]
            if len({item.slot for item in commands}) != len(commands):
                raise RuntimeError("Duplicate streaming command")

            def validate(item):
                slot, generation, action = item
                if (
                    not 0 <= slot < len(offered)
                    or not offered[slot]
                    or done[slot]
                    or generation != generations[slot]
                    or action not in runtime.previous[slot]["actions"]
                ):
                    raise RuntimeError("Stale action or slot ownership violation")

            def applied(item):
                offered[item.slot] = False
                pending.append(item.slot)

            runtime.step(
                commands, interleave_history=True, validate=validate, applied=applied
            )
        else:
            raise ValueError(f"Unknown streaming command: {kind!r}")
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
            # A group is bounded by CPU portions, not by ready observations.
            # Flush partial groups so reply_size never holds a ready slot hostage.
            for _ in range(min(reply_size, len(pending))):
                slot = pending.popleft()
                started = perf_counter()
                state = runtime.envs[slot].request_slice()
                elapsed = perf_counter() - started
                request_seconds += elapsed
                max_slice_seconds = max(max_slice_seconds, elapsed)
                slices += 1
                if state is None:
                    pending.append(slot)
                    continue
                done[slot] = state["done"]
                encoding_seconds += runtime.encode(slot, state)
                offered[slot] = not state["done"]
                generations[slot] += 1
                runtime.previous[slot] = state
                items.append(
                    WorkerStepItem(slot, generations[slot], state, perf_counter())
                )
            if items:
                connection.send(
                    (
                        "ok",
                        {
                            "wave": collection_id,
                            "items": items,
                            "published_at": perf_counter(),
                            "worker_encoding_seconds": encoding_seconds,
                            "worker_request_seconds": request_seconds,
                            "worker_request_slices": slices,
                            "worker_request_slice_max_seconds": max_slice_seconds,
                        },
                    )
                )
                request_seconds = encoding_seconds = 0.0
                slices = 0
                max_slice_seconds = 0.0
    except EOFError:
        pass
    except BaseException:  # noqa: BLE001 -- propagate worker failures to parent
        runtime.send_error()
    finally:
        runtime.close()
