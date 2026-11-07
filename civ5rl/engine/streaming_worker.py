"""CPU-only round-robin execution with reusable, episode-local slots."""

from collections import deque
from time import perf_counter, perf_counter_ns

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
    diag = None
    queued_at, ready_at = {}, {}

    def enqueue(slot):
        pending.append(slot)
        if diag:
            queued_at[slot] = perf_counter_ns()
            diag.gauge("cpu_pending_slots", len(pending))

    def receive():
        clock = diag.begin() if diag else None
        kind, payload = connection.recv()
        if diag and kind not in ("diagnostics", "reset", "close"):
            diag.end("worker_receive", clock)
        elif diag and kind == "diagnostics":
            # The final control message is outside Collection. Keep only the
            # idle wall interval before the parent's boundary; no CPU estimate
            # is invented for a partial clock interval.
            diag.add(
                "worker_idle_until_collection_end.wall",
                max(0, payload["ended_ns"] - clock[0]),
            )
        return command(kind, payload)

    def command(kind, payload):
        nonlocal collection_id, variant, reply_size, offered, done, generations, diag
        if kind == "close":
            return False
        if kind == "warmup":
            if pending or any(offered):
                raise RuntimeError("Warmup requires an idle worker")
            connection.send(("warmup", runtime.warmup(payload)))
            return True
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
            if payload.get("collection_diagnostics"):
                from civ5rl.engine.collection_diagnostics import Diagnostics

                diag = Diagnostics(128, collection=collection_id)
                diag.gauge("cpu_pending_slots", 0)
            else:
                diag = None
            queued_at.clear()
            ready_at.clear()
        elif kind == "diagnostics":
            if pending or any(offered):
                raise RuntimeError("Diagnostics requested before collection drained")
            diag.stop(payload["ended_ns"], cpu_checkpoint=diag.last_work_checkpoint)
            connection.send(("diagnostics", diag))
        elif kind == "assign":
            if payload["wave"] != collection_id:
                raise RuntimeError("Stale streaming assignment")
            for slot, spec in payload["items"]:
                if not done[slot] or offered[slot] or slot in pending:
                    raise RuntimeError("Assignment overwrites an active episode")
                runtime.assign(slot, spec, variant)
                runtime.envs[slot].collection_diagnostics = diag
                done[slot] = False
                enqueue(slot)
        elif kind == "step":
            if payload["wave"] != collection_id:
                raise RuntimeError("Stale streaming action")
            commands = payload["commands"]
            received_ns = perf_counter_ns() if diag else None
            if diag:
                for item in commands:
                    diag.add(
                        "action_transport", received_ns - payload["diagnostic_sent_ns"]
                    )
                    diag.add(
                        "observation_to_action_received",
                        received_ns - ready_at[item.slot],
                    )
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
                if diag:
                    at = perf_counter_ns()
                    diag.sample(
                        {
                            "kind": "action_applied",
                            "key": (
                                collection_id,
                                runtime.envs[item.slot].spec["episode_id"],
                                item.slot,
                                item.generation,
                            ),
                            "observation_ready_ns": ready_at[item.slot],
                            "action_sent_ns": payload["diagnostic_sent_ns"],
                            "action_received_ns": received_ns,
                            "action_applied_ns": at,
                        }
                    )
                    diag.add("action_receive_to_applied", at - received_ns)
                enqueue(item.slot)

            clock = diag.begin() if diag else None
            runtime.step(
                commands, interleave_history=True, validate=validate, applied=applied
            )
            if diag:
                diag.end("action_apply", clock)
        else:
            raise ValueError(f"Unknown streaming command: {kind!r}")
        return True

    try:
        running = True
        while running:
            if not pending:
                running = receive()
            while running and connection.poll():
                running = receive()
            if not running:
                break
            items = []
            # A group is bounded by CPU portions, not by ready observations.
            # Flush partial groups so reply_size never holds a ready slot hostage.
            for _ in range(min(reply_size, len(pending))):
                slot = pending.popleft()
                if diag:
                    diag.gauge("cpu_pending_slots", len(pending))
                    diag.gauge("cpu_running_slots", 1)
                    diag.add("cpu_local_queue", perf_counter_ns() - queued_at.pop(slot))
                    env = runtime.envs[slot]
                    env.diagnostic_key = (
                        collection_id,
                        env.spec["episode_id"],
                        slot,
                        generations[slot] + 1,
                    )
                    clock = diag.begin()
                started = perf_counter()
                state = runtime.envs[slot].request_slice()
                state_ready_ns = diag.end("request_slice", clock) if diag else None
                if diag:
                    diag.gauge("cpu_running_slots", 0)
                elapsed = perf_counter() - started
                request_seconds += elapsed
                max_slice_seconds = max(max_slice_seconds, elapsed)
                slices += 1
                if state is None:
                    enqueue(slot)
                    continue
                done[slot] = state["done"]
                clock = diag.begin() if diag else None
                encoding_seconds += runtime.encode(slot, state)
                if diag:
                    encoded_ns = diag.end("worker_encode", clock)
                    ready_at[slot] = state_ready_ns
                offered[slot] = not state["done"]
                generations[slot] += 1
                runtime.previous[slot] = state
                items.append(
                    WorkerStepItem(
                        slot,
                        generations[slot],
                        state,
                        perf_counter(),
                        {
                            "observation_ready_ns": state_ready_ns,
                            "encoded_ready_ns": encoded_ns,
                            "key": runtime.envs[slot].diagnostic_key,
                        }
                        if diag
                        else None,
                    )
                )
            if items:
                clock = diag.begin() if diag else None
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
                            "diagnostic_publish_ns": clock[0] if diag else None,
                        },
                    )
                )
                if diag:
                    diag.end("worker_send", clock)
                request_seconds = encoding_seconds = 0.0
                slices = 0
                max_slice_seconds = 0.0
            if (
                diag
                and not pending
                and not any(offered)
                and not hasattr(diag, "stopped_ns")
            ):
                # A later refill will replace this checkpoint. At final drain
                # it freezes CPU before receiving/serializing diagnostic RPCs.
                diag.checkpoint()
    except EOFError:
        pass
    except BaseException:  # noqa: BLE001 -- propagate worker failures to parent
        runtime.send_error()
    finally:
        runtime.close()
