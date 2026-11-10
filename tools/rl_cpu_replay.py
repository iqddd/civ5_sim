"""CPU-only fixed-action replay; spawn workers never import PyTorch."""

import multiprocessing as mp
import os
import pickle
import sys

_cases = None


def _initialize(path, source_root, ready):
    global _cases
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = "1"
    sys.path.insert(0, source_root)
    with open(path, "rb") as stream:
        _cases = pickle.load(stream)
    from civ5rl.core.env import DecisionEnv  # noqa: F401 -- load before timing

    if "torch" in sys.modules:
        raise RuntimeError("CPU replay worker imported torch")
    ready.put(os.getpid())


def replay_case(case):
    from civ5rl.core.env import DecisionEnv

    env = DecisionEnv(case["spec"])
    try:
        for action in case["commands"]:
            state = env.request()
            if state["done"] or action not in state["actions"]:
                raise AssertionError("Recorded learner command is no longer legal")
            env.step(action)
        state = env.request()
        if not state["done"]:
            raise AssertionError("Replay did not finish the complete episode")
        return {
            "commands": state["joint_commands"],
            "result": {k: v for k, v in state["result"].items() if k != "timings"},
        }
    finally:
        env.close()


def _replay_index(index):
    return replay_case(_cases[index])


class CPUReplay:
    """Preload cases once, keep workers alive across warmup and timed trials."""

    def __init__(self, episodes, workers, directory, source_root):
        self.cases = [
            {
                "spec": ep["spec"],
                "commands": [row[int(chosen)] for row, chosen in zip(ep["actions"], ep["chosen"])],
            }
            for ep in episodes
        ]
        self.expected = [
            {
                "commands": ep["joint_commands"],
                "result": {k: v for k, v in ep["result"].items() if k != "timings"},
            }
            for ep in episodes
        ]
        self.pool = None
        self.ready = None
        if workers > 1:
            path = directory / "cpu-cases.pkl"
            with path.open("wb") as stream:
                pickle.dump(self.cases, stream, protocol=5)
            context = mp.get_context("spawn")
            self.ready = context.Queue()
            self.pool = context.Pool(
                workers,
                _initialize,
                (str(path.resolve()), str(source_root.resolve()), self.ready),
            )
            try:
                for _ in range(workers):
                    self.ready.get(timeout=120)
            except BaseException:
                self.close(terminate=True)
                raise

    def run(self):
        if self.pool:
            return self.pool.map(_replay_index, range(len(self.cases)), chunksize=1)
        return [replay_case(case) for case in self.cases]

    def verify(self, actual):
        if actual != self.expected:
            raise AssertionError("CPU replay changed episode order, commands or outcomes")
        from civ5sim.engine import digest

        return {
            "commands_hash": digest([row["commands"] for row in actual]),
            "results_hash": digest([row["result"] for row in actual]),
            "commands": sum(len(row["commands"]) for row in actual),
        }

    def close(self, terminate=False):
        if self.pool:
            if terminate:
                self.pool.terminate()
            else:
                self.pool.close()
            self.pool.join()
            self.pool = None
        if self.ready:
            self.ready.close()
            self.ready.join_thread()
            self.ready = None
