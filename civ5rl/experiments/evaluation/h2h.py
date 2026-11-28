"""Deterministic FP32 head-to-head evaluation and dual-line milestone selection."""

from __future__ import annotations

import platform
import shutil
import sys
import traceback
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, wait
from multiprocessing import Pipe
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter

import numpy as np

from civ5rl.engine.evaluation.matches import combat_metrics, run_saved_match
from civ5rl.engine.evaluation.proxy import ProxyAgent
from civ5rl.engine.runtime.shared import EncodedSlots
from civ5rl.provenance.common import ROOT, configure, file_sha256, read_json, write_json
from civ5rl.reporting.evaluation_audit import h2h_world_bootstrap
from civ5sim.engine import digest


def _validate_h2h_case(case):
    if (
        case["split"] != "validation"
        or digest({k: v for k, v in case.items() if k != "case_hash"}) != case["case_hash"]
    ):
        raise ValueError("Invalid validation case")


def _h2h_worker_task(task, cand_conn, ref_conn):
    """Run one policy-vs-policy match in a worker with two isolated ProxyAgents."""
    config, case, cand_path, ref_path, directory = task
    directory = Path(directory)
    _validate_h2h_case(case)
    side = case["learner_side"]
    if cand_conn is not None and ref_conn is not None:
        cand_agent = ProxyAgent(cand_conn, "argmax")
        ref_agent = ProxyAgent(ref_conn, "argmax")
    else:
        from civ5rl.core.agent import RLAgent

        configure(17, "cpu")
        cand_agent = (
            ProxyAgent(cand_conn, "argmax")
            if cand_conn is not None
            else RLAgent.from_checkpoint(cand_path, "cpu", "argmax")
        )
        ref_agent = (
            ProxyAgent(ref_conn, "argmax")
            if ref_conn is not None
            else RLAgent.from_checkpoint(ref_path, "cpu", "argmax")
        )
    agents = [None, None]
    agents[side], agents[1 - side] = cand_agent, ref_agent
    write_json(directory / "case.json", case)
    started = perf_counter()
    try:
        outcome = run_saved_match(
            case,
            agents,
            directory,
            max_calls_per_turn=config["environment"]["max_calls_per_turn"],
        )
        if outcome.result["status"] != "completed":
            raise RuntimeError(str(outcome.result["error"]))
        additional_metrics = combat_metrics(case, outcome.replay, outcome.final)
        metrics = {
            "status": "completed",
            "case_id": case["case_id"],
            "side": side,
            "parent_world": case["parent_world"],
            "winner": outcome.result["winner"],
            "reason": outcome.result["reason"],
            "rounds": outcome.result["completed_rounds"],
            "commands": len(outcome.replay["steps"]),
            "seconds": perf_counter() - started,
            "remaining_hp": outcome.result["remaining_hp"],
            "case_hash": case["case_hash"],
            **additional_metrics,
        }
        write_json(directory / "metrics.json", metrics)
        return metrics
    except BaseException:
        (directory / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        for agent, conn in ((cand_agent, cand_conn), (ref_agent, ref_conn)):
            if isinstance(agent, ProxyAgent):
                agent.close()
            if conn is not None:
                # The coordinator owns the original endpoint across matches.
                conn.close()


def _h2h_cpu_task(task):
    return _h2h_worker_task(task, None, None)


class DualPolicyExecutor:
    """Serve two frozen FP32 policies concurrently with strictly isolated GRU states."""

    def __init__(self, candidate_checkpoint, reference_checkpoint, runtime, *, identical=False):
        from civ5rl.engine.evaluation.executor import (
            BatchedInference,
            load_evaluation_policy,
        )

        self.runtime = runtime.resolve()
        self.identical = identical
        capacity = max(2, self.runtime.workers * (2 if identical else 1))
        self.cand_inference = BatchedInference(
            load_evaluation_policy(candidate_checkpoint, self.runtime.device),
            self.runtime.device,
            hidden_capacity=capacity,
        )
        if identical:
            self.ref_inference = self.cand_inference
        else:
            self.ref_inference = BatchedInference(
                load_evaluation_policy(reference_checkpoint, self.runtime.device),
                self.runtime.device,
                hidden_capacity=capacity,
            )
        self.connections = {}  # server_conn -> (role, agent_or_None)
        self.shared = {}
        self.lock, self.stop = Lock(), Event()
        self.error = None
        self.wait_seconds = 0.0
        self.wait_timeouts = 0
        self.result_transfer_seconds = 0.0

    def _inference_for(self, role):
        return self.cand_inference if role == "candidate" else self.ref_inference

    def _receive(self, connection):
        try:
            command, args = connection.recv()
        except (EOFError, OSError):
            command, args = "close", ()
        with self.lock:
            entry = self.connections.get(connection)
        if entry is None:
            return None
        role, agent = entry
        inference = self._inference_for(role)
        if command == "close":
            with self.lock:
                self.connections.pop(connection, None)
            inference.release_agent(agent)
            shared = self.shared.pop(connection, None)
            if shared is not None:
                shared.close()
            connection.close()
            return None
        if command == "reset":
            from civ5rl.engine.evaluation.executor import AgentMember, AgentSession

            mode, temperature, metrics = args
            if mode != "argmax" or metrics:
                raise ValueError("H2H evaluation requires argmax without selection metrics")
            inference.release_agent(agent)
            shared = self.shared.get(connection)
            if shared is None:
                shared = self.shared[connection] = EncodedSlots(slots=1)
            new_agent = AgentSession(mode, temperature, metrics, [AgentMember() for _ in inference.policies])
            with self.lock:
                self.connections[connection] = (role, new_agent)
            info = {
                "shared": shared.descriptor,
                "variant": inference.policies[0].variant,
                "attack_preview": inference.policies[0].variant_spec.attack_preview,
            }
            connection.send(("ok", info))
            return None
        if agent is None:
            raise ValueError("H2H evaluation agent was not reset")
        if command not in {"act", "observe"}:
            raise ValueError(f"Unknown H2H evaluation request {command!r}")
        (state,) = args
        return (
            connection,
            role,
            (agent, command, self.shared[connection].read(0, state, copy=False)),
        )

    def _fail_all(self, message):
        with self.lock:
            connections = list(self.connections)
        for connection in connections:
            try:
                connection.send(("error", message))
            except (OSError, EOFError):
                pass

    def _serve(self):
        import torch

        from civ5rl.engine.evaluation.connection import ConnectionWaiter

        waiter = None
        try:
            waiter = ConnectionWaiter()
            while not self.stop.is_set():
                with self.lock:
                    connections = list(self.connections)
                if not connections:
                    self.stop.wait(0.001)
                    continue
                ready = waiter.wait(connections, timeout=0.01)
                pending = []
                for connection in ready[: self.runtime.max_batch]:
                    request = self._receive(connection)
                    if request is not None:
                        pending.append(request)
                if not pending:
                    continue
                started = perf_counter()
                until = started + self.runtime.wait_seconds
                while True:
                    waiting = {p[0] for p in pending}
                    with self.lock:
                        # Each active worker has 2 pipes, only 1 can be active at a time per match.
                        active_matches = max(1, len(self.connections) // 2)
                        target = min(self.runtime.max_batch, self.runtime.workers, active_matches)
                        remaining = [c for c in self.connections if c not in waiting]
                    if len(pending) >= target or not remaining:
                        break
                    timeout = max(0, until - perf_counter())
                    more = waiter.wait(remaining, timeout=timeout)
                    if not more:
                        self.wait_timeouts += int(perf_counter() >= until)
                        break
                    for connection in more[: target - len(pending)]:
                        request = self._receive(connection)
                        if request is not None:
                            pending.append(request)
                    if perf_counter() >= until:
                        break
                self.wait_seconds += perf_counter() - started
                if self.identical or len({r[1] for r in pending}) == 1:
                    groups = [pending]
                else:
                    groups = [[r for r in pending if r[1] == role] for role in ("candidate", "reference")]
                choices = [
                    self._inference_for(group[0][1]).forward_encoded_argmax(
                        [request for _, _, request in group]
                    )
                    for group in groups
                ]
                started = perf_counter()
                packed = choices[0] if len(choices) == 1 else torch.cat(choices)
                host_choices = iter(packed.cpu().tolist())
                self.result_transfer_seconds += perf_counter() - started
                for group in groups:
                    for connection, _, (_, command, _) in group:
                        choice = next(host_choices)
                        connection.send(("ok", None if command == "observe" else {"index": choice}))
        except BaseException as exc:  # noqa: BLE001 -- relay terminal failures from the inference thread
            self.error = traceback.format_exc()
            self._fail_all(f"{type(exc).__name__}: {exc}")
        finally:
            if waiter is not None:
                waiter.close()

    def run(self, tasks):
        from civ5rl.engine.evaluation.executor import evaluation_worker_pool

        with evaluation_worker_pool(self.runtime.workers) as executor:
            yield from self._run(executor, tasks)

    def _run(self, executor, tasks):
        tasks = iter(tasks)
        self.stop.clear()
        thread = Thread(target=self._serve, name="h2h-evaluation-inference", daemon=True)
        thread.start()
        pending = {}
        clients = []

        def submit(lane=None):
            try:
                task = next(tasks)
            except StopIteration:
                return False
            if lane is None:
                c_srv, c_cli = Pipe()
                r_srv, r_cli = Pipe()
                with self.lock:
                    self.connections[c_srv] = ("candidate", None)
                    self.connections[r_srv] = ("reference", None)
                lane = (c_cli, r_cli)
                clients.extend(lane)
            # A future finishes after the worker closes its handles and views.
            # The next match resets both agents before accessing the same slots.
            future = executor.submit(_h2h_worker_task, task, *lane)
            pending[future] = lane
            return True

        try:
            for _ in range(self.runtime.workers):
                if not submit():
                    break
            while pending:
                if self.error is not None:
                    raise RuntimeError(f"H2H evaluation inference failed:\n{self.error}")
                done, _ = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                for future in done:
                    lane = pending.pop(future)
                    yield future.result()
                    submit(lane)
        finally:
            self.stop.set()
            thread.join()
            self._fail_all("H2H evaluation coordinator stopped")
            for connection in clients:
                connection.close()
            for connection, (role, agent) in list(self.connections.items()):
                self._inference_for(role).release_agent(agent)
                connection.close()
            self.connections.clear()
            for shared in self.shared.values():
                shared.close()
            self.shared.clear()

    def statistics(self):
        cand_stats = self.cand_inference.statistics()
        ref_stats = {} if self.identical else self.ref_inference.statistics()
        batches = cand_stats["inference_batches"] + ref_stats.get("inference_batches", 0)
        items = cand_stats["inference_items"] + ref_stats.get("inference_items", 0)
        return {
            "inference_batches": batches,
            "inference_items": items,
            "mean_batch": items / max(1, batches),
            "forward_seconds": (
                cand_stats["forward_seconds"]
                + ref_stats.get("forward_seconds", 0.0)
                + self.result_transfer_seconds
            ),
            "encoding_seconds": (cand_stats["encoding_seconds"] + ref_stats.get("encoding_seconds", 0.0)),
            "batch_wait_seconds": self.wait_seconds,
            "batch_wait_timeouts": self.wait_timeouts,
        }


def h2h_evaluation_key(config, plan, out, candidate_path, reference_path, purpose, runtime_identity):
    import torch

    from civ5rl.engine.evaluation.executor import evaluation_policy_hash
    from civ5rl.engine.evaluation.runtime import EVALUATOR_VERSION

    cand_hash = evaluation_policy_hash(out / candidate_path)
    ref_hash = evaluation_policy_hash(out / reference_path)
    evaluator = digest(
        {
            "version": EVALUATOR_VERSION,
            "kind": "h2h_argmax_fp32",
            "platform": platform.system(),
            "torch": str(torch.__version__),
            "kernel": file_sha256(
                ROOT / "build" / ("combat.dll" if sys.platform == "win32" else "libcombat.so")
            ),
        }
    )
    key = digest(
        {
            "purpose": purpose,
            "candidate": cand_hash,
            "reference": ref_hash,
            "mode": "argmax",
            "cases": [c["case_hash"] for c in plan[purpose]],
            "evaluator": evaluator,
            "runtime": runtime_identity,
        }
    )
    return cand_hash, ref_hash, key


def evaluate_h2h(config, plan, out, candidate_path, reference_path, purpose):
    """Run mirrored deterministic FP32 argmax head-to-head series vs reference_path."""
    from civ5rl.engine.evaluation.executor import evaluation_precision

    with evaluation_precision():
        return _evaluate_h2h_impl(config, plan, out, candidate_path, reference_path, purpose)


def _h2h_execution_plan(tasks, *, same_checkpoint):
    """Serve both logical sides from one physical match for an identical policy."""
    pending = [task for task in tasks if not (Path(task[4]) / "metrics.json").exists()]
    if not same_checkpoint:
        return pending, []
    by_side = {(task[1]["case_id"], task[1]["learner_side"]): task for task in tasks}
    pending_dirs = {task[4] for task in pending}
    simulate, mirror = [], []
    for task in pending:
        _, case, _, _, target = task
        side = case["learner_side"]
        partner = by_side.get((case["case_id"], 1 - side))
        if partner is None or {**partner[1], "learner_side": side, "case_hash": case["case_hash"]} != case:
            simulate.append(task)
            continue
        if side == 0 and partner[4] in pending_dirs:
            simulate.append(task)
        else:
            _validate_h2h_case(case)
            mirror.append((Path(partner[4]), Path(target), case))
    return simulate, mirror


def _materialize_h2h_mirror(source, target, case):
    started = perf_counter()
    metrics = read_json(source / "metrics.json")
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "case.json", case)
    for name in ("replay.json", "final.json", "result.json"):
        shutil.copyfile(source / name, target / name)
    side = case["learner_side"]
    metrics.update(
        side=side,
        case_hash=case["case_hash"],
        army_size=len(case["armies"][side]),
        seconds=perf_counter() - started,
    )
    write_json(target / "metrics.json", metrics)


def _evaluate_h2h_impl(config, plan, out, candidate_path, reference_path, purpose):
    from civ5rl.engine.evaluation.executor import evaluation_worker_pool, runtime_for
    from civ5rl.protocols.turn_reset import validate_case

    out = Path(out)
    cases = plan[purpose]
    for case in cases:
        validate_case(case, config)
    runtime = runtime_for(out, config)
    runtime_identity = runtime.identity()
    cand_hash, ref_hash, key = h2h_evaluation_key(
        config, plan, out, candidate_path, reference_path, purpose, runtime_identity
    )
    cache_path = out / "evaluations-h2h.json"
    cache = read_json(cache_path) if cache_path.exists() else {}
    if key in cache:
        return cache[key]
    started = perf_counter()
    relative = f"evaluation/h2h_{purpose}/{cand_hash[:16]}-vs-{ref_hash[:16]}/{key[:16]}/argmax"
    directory = out / relative
    tasks = [
        (
            config,
            case,
            str(out / candidate_path),
            str(out / reference_path),
            str(directory / f"case-{case['case_id']:03}-side-{case['learner_side']}"),
        )
        for case in cases
    ]
    tasks, mirrors = _h2h_execution_plan(
        tasks,
        same_checkpoint=(out / candidate_path).resolve() == (out / reference_path).resolve(),
    )
    inference_stats = None
    if runtime.device == "cuda" and tasks:
        executor = DualPolicyExecutor(
            str(out / candidate_path),
            str(out / reference_path),
            runtime,
            identical=(cand_hash == ref_hash),
        )
        for count, _ in enumerate(executor.run(tasks), 1):
            if count % runtime.workers == 0 or count == len(tasks):
                print(f"H2H {purpose}: {count}/{len(tasks)} new matches", flush=True)
        inference_stats = executor.statistics()
    elif tasks:
        with evaluation_worker_pool(runtime.workers) as executor:
            for first in range(0, len(tasks), runtime.workers):
                list(executor.map(_h2h_cpu_task, tasks[first : first + runtime.workers]))
                print(
                    f"H2H {purpose}: {min(first + runtime.workers, len(tasks))}/{len(tasks)} new matches",
                    flush=True,
                )
    for source, target, case in mirrors:
        _materialize_h2h_mirror(source, target, case)
    bootstrap = h2h_world_bootstrap(
        directory,
        len(cases),
        clusters=16 if len(cases) >= 64 else None,
    )
    rows = [read_json(p) for p in directory.glob("*/metrics.json")]
    slices = {}
    for field in ("map_size", "army_bin", "army_size", "placement", "side"):
        slices[field] = {}
        for value in sorted({r[field] for r in rows}, key=str):
            selected = [r for r in rows if r[field] == value]
            wdl = Counter(
                "D" if r["winner"] is None else "W" if r["winner"] == r["side"] else "L" for r in selected
            )
            slices[field][str(value)] = {
                "matches": len(selected),
                **{k: wdl[k] for k in ("W", "D", "L")},
                "delta_wins": int(wdl["W"] - wdl["L"]),
                "mean_rounds": float(np.mean([r["rounds"] for r in selected])),
            }
    summary = {
        k: bootstrap[k]
        for k in (
            "matches",
            "W",
            "D",
            "L",
            "delta_wins",
            "delta",
            "win_rate",
            "draw_rate",
            "loss_rate",
        )
    }
    result = {
        "checkpoint": candidate_path,
        "reference": reference_path,
        "policy_hash": cand_hash,
        "reference_policy_hash": ref_hash,
        "purpose": purpose,
        "seconds": perf_counter() - started,
        "runtime": runtime_identity,
        "inference": inference_stats,
        "directory": relative,
        "summary": summary,
        "bootstrap": bootstrap,
        "slices": slices,
    }
    cache[key] = result
    write_json(cache_path, cache)
    print(
        f"H2H {purpose}: W/D/L={summary['W']}/{summary['D']}/{summary['L']} "
        f"(delta_wins={summary['delta_wins']:+d}, delta={summary['delta'] * 100:+.2f}pp, "
        f"CI95=[{bootstrap['CI95'][0] * 100:+.2f}; {bootstrap['CI95'][1] * 100:+.2f}]pp); wall={result['seconds']:.1f}s",
        flush=True,
    )
    return result
