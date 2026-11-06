"""CPU-only synchronous worker for complete episode waves."""

from civ5rl.engine.worker_runtime import _WorkerRuntime


def worker(connection, profile_path=None, shared_descriptor=None):
    runtime = _WorkerRuntime(connection, profile_path, shared_descriptor)
    reported_done = []
    try:
        while True:
            kind, payload = connection.recv()
            if kind == "close":
                break
            if kind == "reset":
                specs = payload["specs"] if isinstance(payload, dict) else payload
                runtime.reset(
                    specs, payload["variant"] if runtime.shared is not None else None
                )
                reported_done = [False] * len(specs)
            elif kind == "step":
                runtime.step(payload, interleave_history=False)
            else:
                raise ValueError(f"Unknown worker command: {kind!r}")
            states = []
            encoding_seconds = 0.0
            for i, env in enumerate(runtime.envs):
                state = {"done": True} if reported_done[i] else env.request()
                reported_done[i] = state["done"]
                encoding_seconds += runtime.encode(i, state)
                states.append(state)
            runtime.previous = states
            connection.send(
                ("ok", {"states": states, "worker_encoding_seconds": encoding_seconds})
            )
    except EOFError:
        pass
    except BaseException:
        runtime.send_error()
    finally:
        runtime.close()
