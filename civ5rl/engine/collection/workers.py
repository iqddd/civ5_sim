"""CPU-only operations shared by the collection workers."""

from __future__ import annotations

import cProfile
import os
import traceback
from collections import deque
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter, perf_counter_ns

from civ5rl.core.context import context_from_spec
from civ5rl.core.contracts import (
    ScenarioSpec,
    WorkerCommand,
    WorkerState,
    WorkerStepItem,
)
from civ5rl.core.env import DecisionEnv
from civ5rl.core.observation import ObservationEncoder
from civ5rl.engine.collection.diagnostics import Diagnostics
from civ5rl.engine.runtime.shared import EncodedSlots
from civ5sim.engine import step_batch


@dataclass
class WorkerSlotState:
    env: object = None
    encoder: object = None
    opp_encoder: object = None
    previous: object = None
    offered: bool = False
    done: bool = False
    generation: int = 0


class BaseWorker:
    """Own CPU simulators and optional shared input slots."""

    def __init__(self, connection, profile_path=None, shared_descriptor=None):
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            os.environ[name] = "1"
        self.connection = connection
        self.slots = []
        self.shared = None
        self.profiler = None
        self.profile_path = profile_path
        if shared_descriptor is not None:
            self.shared = EncodedSlots(descriptor=shared_descriptor)
        if profile_path is not None:
            self.profiler = cProfile.Profile()
            self.profiler.enable()

    def reset(self, specs: list[ScenarioSpec], variant):
        for slot in self.slots:
            if slot.env is not None:
                slot.env.close()
        self.slots = []
        for index, spec in enumerate(specs):
            self.slots.append(WorkerSlotState())
            self.assign(index, spec, variant)

    def warmup(self, spec: ScenarioSpec):
        """Exercise CP once in an owned temporary episode, outside collection."""
        spec = deepcopy(spec)
        spec["learner_side"] = 1  # Initial side zero belongs to CP in this episode.
        spec["opponent"] = "cp"
        spec.pop("opponent_id", None)
        if not spec.get("opponent_search"):
            spec.pop("opponent_search", None)
        started = perf_counter()
        env = DecisionEnv(spec)
        try:
            if not callable(getattr(env.opponent, "act_with_planning", None)):
                raise TypeError("Worker warmup requires a CP opponent")
            env.request_slice()
            if env.commands != 1:
                raise RuntimeError("Worker warmup did not execute one CP command")
            calls = env.commands
        finally:
            env.close()
        return {
            "pid": os.getpid(),
            "cp_calls": calls,
            "seconds": perf_counter() - started,
        }

    def assign(self, slot: int, spec: ScenarioSpec, variant):
        state = self.slots[slot]
        if state.env is not None:
            state.env.close()
        state.env = DecisionEnv(spec)
        state.env.attack_preview = variant == "C2P"
        state.encoder = (
            ObservationEncoder(context_from_spec(spec), variant) if self.shared is not None else None
        )
        state.opp_encoder = (
            ObservationEncoder(context_from_spec(spec, side=1 - spec["learner_side"]), variant)
            if self.shared is not None and spec.get("opponent") == "policy"
            else None
        )
        state.previous = None

    def step(
        self,
        commands: list[WorkerCommand],
        *,
        interleave_history: bool,
        validate=None,
        applied=None,
    ):
        prepared = []
        for command in commands:
            if validate is not None:
                validate(command)
            prepared.append(self.slots[command.slot].env.prepare_step(command.action))
        started = perf_counter()
        results = step_batch(
            [self.slots[command.slot].env.battle for command in commands],
            [command.action for command in commands],
        )
        elapsed = (perf_counter() - started) / max(len(commands), 1)
        for command, step, result in zip(commands, prepared, results):
            env = self.slots[command.slot].env
            env.timings["simulation_seconds"] += elapsed
            env.finish_step(step, result)
            if interleave_history:
                self.remember(command)
            if applied is not None:
                applied(command)
        if not interleave_history:
            for command in commands:
                self.remember(command)

    def remember(self, command: WorkerCommand):
        if self.shared is not None:
            slot = self.slots[command.slot]
            encoder = slot.opp_encoder if slot.previous and slot.previous.get("is_opponent") else slot.encoder
            encoder.remember_action(command.action, slot.previous["observation"])

    def encode(self, slot: int, state: WorkerState) -> float:
        if self.shared is None or state["done"]:
            return 0.0
        started = perf_counter()
        entry = self.slots[slot]
        encoder = entry.opp_encoder if state.get("is_opponent") else entry.encoder
        state.update(
            self.shared.write(
                slot,
                encoder.encode(state["observation"], state["actions"]),
            )
        )
        return perf_counter() - started

    def validate_command(self, item, *, streaming=False):
        slots = self.slots
        slot, generation, action = item
        if (
            not 0 <= slot < len(slots)
            or not slots[slot].offered
            or slots[slot].done
            or generation != slots[slot].generation
        ):
            raise RuntimeError(
                "Stale action or slot ownership violation"
                if streaming
                else "Stale learner action or slot ownership violation"
            )
        if action not in self.slots[slot].previous["actions"]:
            raise RuntimeError(
                "Stale action or slot ownership violation"
                if streaming
                else "Learner action not in offered legal list"
            )

    def send_error(self):
        self.connection.send(
            (
                "error",
                {
                    "traceback": traceback.format_exc(),
                    "episodes": [slot.env.reproduction() for slot in self.slots if slot.env is not None],
                },
            )
        )

    def run(self):
        try:
            self.run_loop()
        except EOFError:
            pass
        except BaseException:  # noqa: BLE001 -- report worker failure before closing IPC
            self.send_error()
        finally:
            self.close()

    def close(self):
        if self.profiler is not None:
            self.profiler.disable()
            self.profiler.dump_stats(self.profile_path)
        for slot in self.slots:
            if slot.env is not None:
                slot.env.close()
        if self.shared is not None:
            self.shared.close()
        self.connection.close()


class SyncWaveWorker(BaseWorker):
    """Process one complete synchronous reply for each wave command."""

    def run_loop(self):
        runtime = self
        connection = self.connection
        while True:
            kind, payload = connection.recv()
            if kind == "close":
                break
            if kind == "reset":
                specs = payload["specs"] if isinstance(payload, dict) else payload
                runtime.reset(
                    specs,
                    payload.get("variant") if isinstance(payload, dict) else None,
                )
            elif kind == "step":
                runtime.step(payload, interleave_history=False)
            else:
                raise ValueError(f"Unknown worker command: {kind!r}")
            states = []
            encoding_seconds = 0.0
            for i, slot in enumerate(runtime.slots):
                env = slot.env
                state = {"done": True} if slot.done else env.request()
                slot.done = state["done"]
                encoding_seconds += runtime.encode(i, state)
                states.append(state)
            for slot, state in zip(runtime.slots, states):
                slot.previous = state
            connection.send(
                (
                    "ok",
                    {"states": states, "worker_encoding_seconds": encoding_seconds},
                )
            )


def sync_worker(connection, profile_path=None, shared_descriptor=None):
    SyncWaveWorker(connection, profile_path, shared_descriptor).run()


class QueuedWorker(BaseWorker):
    """Share ownership and action handling for workers with partial replies."""

    streaming_actions = False
    duplicate_action_error = "Duplicate slot in learner commands"

    def receive(self):
        return self.command(*self.connection.recv())

    def poll_commands(self):
        """Block only when no slot already has pending work."""
        if not self.pending and not self.receive():
            return False
        while self.connection.poll():
            if not self.receive():
                return False
        return True

    def is_busy(self):
        return bool(self.pending) or any(slot.offered for slot in self.slots)

    def handle_control(self, kind, payload):
        """Handle close and warmup commands common to both queued workers."""
        if kind == "close":
            return False
        if kind == "warmup":
            if self.is_busy():
                raise RuntimeError("Warmup requires an idle worker")
            self.connection.send(("warmup", self.warmup(payload)))
            return True
        return None

    def require_distinct_slots(self, commands):
        if len({item.slot for item in commands}) != len(commands):
            raise RuntimeError(self.duplicate_action_error)

    def apply_commands(self, commands):
        self.step(
            commands,
            interleave_history=True,
            validate=self._validate_action,
            applied=self.applied,
        )

    def _validate_action(self, item):
        self.validate_command(item, streaming=self.streaming_actions)

    def offer(self, index, state):
        slot = self.slots[index]
        slot.offered = not state["done"]
        slot.generation += 1
        slot.previous = state
        return slot.generation


class ReadyWaveWorker(QueuedWorker):
    """Advance fixed wave slots to their next complete observation."""

    def applied(self, item):
        self.slots[item.slot].offered = False
        self.pending.append(item.slot)

    def command(self, kind, payload):
        control = self.handle_control(kind, payload)
        if control is not None:
            return control
        if kind == "reset":
            if self.is_busy():
                raise RuntimeError("Reset before all episode replies completed")
            self.wave_id, self.reply_size = payload["wave"], payload["reply_size"]
            self.reset(payload["specs"], payload["variant"])
            self.pending.extend(range(len(self.slots)))
        elif kind == "step":
            if payload["wave"] != self.wave_id:
                raise RuntimeError("Stale episode wave")
            commands = payload["commands"]
            self.require_distinct_slots(commands)
            self.apply_commands(commands)
        else:
            raise ValueError(f"Unknown worker command: {kind!r}")
        return True

    def run_loop(self):
        self.slots = []
        self.pending = deque()
        self.wave_id, self.reply_size = None, None

        while self.poll_commands():
            if not self.pending:
                continue
            items = []
            encoding_seconds = request_seconds = 0.0
            for _ in range(min(self.reply_size, len(self.pending))):
                i = self.pending.popleft()
                if self.slots[i].offered or self.slots[i].done:
                    raise RuntimeError("Slot advanced before its previous action")
                start = perf_counter()
                state = self.slots[i].env.request()
                request_seconds += perf_counter() - start
                self.slots[i].done = state["done"]
                encoding_seconds += self.encode(i, state)
                generation = self.offer(i, state)
                items.append(WorkerStepItem(i, generation, state, perf_counter()))
            if items:
                self.connection.send(
                    (
                        "ok",
                        {
                            "wave": self.wave_id,
                            "items": items,
                            "published_at": perf_counter(),
                            "worker_encoding_seconds": encoding_seconds,
                            "worker_request_seconds": request_seconds,
                        },
                    )
                )


def ready_worker(connection, profile_path=None, shared_descriptor=None):
    ReadyWaveWorker(connection, profile_path, shared_descriptor).run()


class StreamingWorker(QueuedWorker):
    """Advance a slot by one CPU slice and requeue unfinished requests."""

    streaming_actions = True
    duplicate_action_error = "Duplicate streaming command"

    def applied(self, item):
        self.slots[item.slot].offered = False
        if self.diag:
            at = perf_counter_ns()
            self.diag.sample(
                {
                    "kind": "action_applied",
                    "key": (
                        self.collection_id,
                        self.slots[item.slot].env.spec["episode_id"],
                        item.slot,
                        item.generation,
                    ),
                    "observation_ready_ns": self.ready_at[item.slot],
                    "action_sent_ns": self.action_sent_ns,
                    "action_received_ns": self.action_received_ns,
                    "action_applied_ns": at,
                }
            )
            self.diag.add("action_receive_to_applied", at - self.action_received_ns)
        self.enqueue(item.slot)

    def enqueue(self, slot):
        self.pending.append(slot)
        if self.diag:
            self.queued_at[slot] = perf_counter_ns()
            self.diag.gauge("cpu_pending_slots", len(self.pending))

    def receive(self):
        clock = self.diag.begin() if self.diag else None
        kind, payload = self.connection.recv()
        if self.diag and kind not in ("diagnostics", "reset", "close"):
            self.diag.end("worker_receive", clock)
        elif self.diag and kind == "diagnostics":
            # The final control message is outside Collection. Keep only the
            # idle wall interval before the parent's boundary; no CPU estimate
            # is invented for a partial clock interval.
            self.diag.add(
                "worker_idle_until_collection_end.wall",
                max(0, payload["ended_ns"] - clock[0]),
            )
        return self.command(kind, payload)

    def command(self, kind, payload):
        control = self.handle_control(kind, payload)
        if control is not None:
            return control
        if kind == "reset":
            if self.is_busy():
                raise RuntimeError("Reset before streaming episodes drained")
            self.reset([], payload["variant"])
            capacity = payload["capacity"]
            self.slots = [WorkerSlotState(done=True) for _ in range(capacity)]
            self.collection_id = payload["wave"]
            self.variant = payload["variant"]
            self.reply_size = payload["reply_size"]
            if payload.get("collection_diagnostics"):
                self.diag = Diagnostics(128, collection=self.collection_id)
                self.diag.gauge("cpu_pending_slots", 0)
            else:
                self.diag = None
            self.queued_at.clear()
            self.ready_at.clear()
        elif kind == "diagnostics":
            if self.is_busy():
                raise RuntimeError("Diagnostics requested before collection drained")
            self.diag.stop(payload["ended_ns"], cpu_checkpoint=self.diag.last_work_checkpoint)
            self.connection.send(("diagnostics", self.diag))
        elif kind == "assign":
            if payload["wave"] != self.collection_id:
                raise RuntimeError("Stale streaming assignment")
            for slot, spec in payload["items"]:
                if not self.slots[slot].done or self.slots[slot].offered or slot in self.pending:
                    raise RuntimeError("Assignment overwrites an active episode")
                self.assign(slot, spec, self.variant)
                self.slots[slot].env.collection_diagnostics = self.diag
                self.slots[slot].done = False
                self.enqueue(slot)
        elif kind == "step":
            if payload["wave"] != self.collection_id:
                raise RuntimeError("Stale streaming action")
            commands = payload["commands"]
            received_ns = perf_counter_ns() if self.diag else None
            if self.diag:
                for item in commands:
                    self.diag.add("action_transport", received_ns - payload["diagnostic_sent_ns"])
                    self.diag.add(
                        "observation_to_action_received",
                        received_ns - self.ready_at[item.slot],
                    )
            self.require_distinct_slots(commands)

            self.action_sent_ns = payload.get("diagnostic_sent_ns")
            self.action_received_ns = received_ns
            clock = self.diag.begin() if self.diag else None
            self.apply_commands(commands)
            if self.diag:
                self.diag.end("action_apply", clock)
        else:
            raise ValueError(f"Unknown streaming command: {kind!r}")
        return True

    def run_loop(self):
        self.pending = deque()
        self.slots = []
        self.collection_id = self.variant = None
        self.reply_size = 1
        request_seconds = encoding_seconds = 0.0
        slices = 0
        max_slice_seconds = 0.0
        self.diag = None
        self.queued_at, self.ready_at = {}, {}

        while self.poll_commands():
            items = []
            # A group is bounded by CPU portions, not by ready observations.
            # Flush partial groups so reply_size never holds a ready slot hostage.
            for _ in range(min(self.reply_size, len(self.pending))):
                slot = self.pending.popleft()
                if self.diag:
                    self.diag.gauge("cpu_pending_slots", len(self.pending))
                    self.diag.gauge("cpu_running_slots", 1)
                    self.diag.add(
                        "cpu_local_queue",
                        perf_counter_ns() - self.queued_at.pop(slot),
                    )
                    env = self.slots[slot].env
                    env.diagnostic_key = (
                        self.collection_id,
                        env.spec["episode_id"],
                        slot,
                        self.slots[slot].generation + 1,
                    )
                    clock = self.diag.begin()
                started = perf_counter()
                state = self.slots[slot].env.request_slice()
                state_ready_ns = self.diag.end("request_slice", clock) if self.diag else None
                if self.diag:
                    self.diag.gauge("cpu_running_slots", 0)
                elapsed = perf_counter() - started
                request_seconds += elapsed
                max_slice_seconds = max(max_slice_seconds, elapsed)
                slices += 1
                if state is None:
                    self.enqueue(slot)
                    continue
                self.slots[slot].done = state["done"]
                clock = self.diag.begin() if self.diag else None
                encoding_seconds += self.encode(slot, state)
                if self.diag:
                    encoded_ns = self.diag.end("worker_encode", clock)
                    self.ready_at[slot] = state_ready_ns
                generation = self.offer(slot, state)
                items.append(
                    WorkerStepItem(
                        slot,
                        generation,
                        state,
                        perf_counter(),
                        {
                            "observation_ready_ns": state_ready_ns,
                            "encoded_ready_ns": encoded_ns,
                            "key": self.slots[slot].env.diagnostic_key,
                        }
                        if self.diag
                        else None,
                    )
                )
            if items:
                clock = self.diag.begin() if self.diag else None
                self.connection.send(
                    (
                        "ok",
                        {
                            "wave": self.collection_id,
                            "items": items,
                            "published_at": perf_counter(),
                            "worker_encoding_seconds": encoding_seconds,
                            "worker_request_seconds": request_seconds,
                            "worker_request_slices": slices,
                            "worker_request_slice_max_seconds": max_slice_seconds,
                            "diagnostic_publish_ns": clock[0] if self.diag else None,
                        },
                    )
                )
                if self.diag:
                    self.diag.end("worker_send", clock)
                request_seconds = encoding_seconds = 0.0
                slices = 0
                max_slice_seconds = 0.0
            if self.diag and not self.is_busy() and not hasattr(self.diag, "stopped_ns"):
                # A later refill will replace this checkpoint. At final drain
                # it freezes CPU before receiving/serializing diagnostic RPCs.
                self.diag.checkpoint()


def streaming_worker(connection, profile_path=None, shared_descriptor=None):
    StreamingWorker(connection, profile_path, shared_descriptor).run()
