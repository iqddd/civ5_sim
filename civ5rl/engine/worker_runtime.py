"""CPU-only operations shared by the two worker schedulers."""

import os
import traceback
from time import perf_counter

from civ5rl.core.context import context_from_spec
from civ5rl.core.contracts import ScenarioSpec, WorkerCommand, WorkerState
from civ5rl.core.env import DecisionEnv
from civ5rl.core.observation import ObservationEncoder
from civ5sim.engine import step_batch


class _WorkerRuntime:
    def __init__(self, connection, profile_path=None, shared_descriptor=None):
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"
        self.connection = connection
        self.envs = []
        self.encoders = []
        self.previous = []
        self.shared = None
        self.profiler = None
        self.profile_path = profile_path
        if shared_descriptor is not None:
            from civ5rl.engine.shared import EncodedSlots

            self.shared = EncodedSlots(descriptor=shared_descriptor)
        if profile_path is not None:
            import cProfile

            self.profiler = cProfile.Profile()
            self.profiler.enable()

    def reset(self, specs: list[ScenarioSpec], variant):
        for env in self.envs:
            env.close()
        self.envs = []
        self.encoders = []
        for spec in specs:
            self.envs.append(DecisionEnv(spec))
            if self.shared is not None:
                self.encoders.append(
                    ObservationEncoder(context_from_spec(spec), variant)
                )
        self.previous = [None] * len(self.envs)

    def step(
        self,
        commands: list[WorkerCommand],
        *,
        interleave_history: bool,
        validate=None,
        applied=None,
    ):
        for command in commands:
            if validate is not None:
                validate(command)
            self.envs[command.slot].prepare(command.action)
        started = perf_counter()
        results = step_batch(
            [self.envs[command.slot].battle for command in commands],
            [command.action for command in commands],
        )
        elapsed = (perf_counter() - started) / max(len(commands), 1)
        for command, result in zip(commands, results):
            env = self.envs[command.slot]
            env.timings["simulation_seconds"] += elapsed
            env.applied(command.action, result)
            if interleave_history:
                self.remember(command)
            if applied is not None:
                applied(command)
        if not interleave_history:
            for command in commands:
                self.remember(command)

    def remember(self, command: WorkerCommand):
        if self.shared is not None:
            self.encoders[command.slot].remember_action(
                command.action, self.previous[command.slot]["observation"]
            )

    def encode(self, slot: int, state: WorkerState) -> float:
        if self.shared is None or state["done"]:
            return 0.0
        started = perf_counter()
        state.update(
            self.shared.write(
                slot, self.encoders[slot].encode(state["observation"], state["actions"])
            )
        )
        return perf_counter() - started

    def send_error(self):
        self.connection.send(
            (
                "error",
                {
                    "traceback": traceback.format_exc(),
                    "episodes": [env.reproduction() for env in self.envs],
                },
            )
        )

    def close(self):
        if self.profiler is not None:
            self.profiler.disable()
            self.profiler.dump_stats(self.profile_path)
        for env in self.envs:
            env.close()
        if self.shared is not None:
            self.shared.close()
        self.connection.close()
