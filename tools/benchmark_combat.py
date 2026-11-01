"""Repeatable kernel/Python boundary benchmarks, not an RL training run."""

import argparse
import ctypes as C
import json
import platform
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from civ5sim import Action, Battle, step_batch
from civ5sim import _native as N
from civ5sim.engine import KINDS


def scenario(count, fog, random_damage, rough):
    m = {
        "schema": 1,
        "width": 16,
        "height": 16,
        "row_parity": 0,
        "wrap_x": False,
        "metadata": {},
        "tiles": [
            {
                "terrain": "grass",
                "elevation": "hills" if rough and i % 7 == 0 else "flat",
                "feature": "forest" if rough and i % 5 == 0 else None,
                "rivers": [-1, -1, -1],
            }
            for i in range(256)
        ],
    }
    types = ("warrior", "archer", "chariot_archer", "spearman", "horseman")
    units = [
        {
            "type": types[i % 5],
            "side": side,
            "tile": [1 + 11 * side + (i // 12), 1 + i % 12],
        }
        for side in (0, 1)
        for i in range(count)
    ]
    return {
        "schema": 1,
        "kind": "civ5sim.scenario",
        "map": m,
        "units": units,
        "fog": fog,
        "random_damage": random_damage,
        "seed": 42,
    }


def measure(fn, n):
    for _ in range(min(n, 10)):
        fn()
    start = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = time.perf_counter() - start
    return {
        "iterations": n,
        "seconds": elapsed,
        "per_second": n / elapsed,
        "microseconds_each": elapsed * 1e6 / n,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--out", type=Path, default=ROOT / "artifacts/combat/benchmark.json")
    args = p.parse_args()
    if args.iterations < 1:
        p.error("iterations must be positive")
    records = []
    for count in (10, 30):
        for fog, rand, rough in ((False, False, False), (True, True, True)):
            s = scenario(count, fog, rand, rough)
            start = time.perf_counter()
            with Battle(s) as b:
                record = {
                    "units_per_side": count,
                    "fog": fog,
                    "random_damage": rand,
                    "rough": rough,
                    "initialization_ms": (time.perf_counter() - start) * 1000,
                }
                raw_end = N.Action(5, -1, -1)
                record["native_end_turn_via_ctypes"] = measure(
                    lambda command=raw_end: b._lib.sim_step(b._handle, command),
                    args.iterations * 10,
                )
                record["python_end_turn"] = measure(
                    lambda: b.step(Action("end_turn")), args.iterations
                )
                record["packed_available_actions"] = measure(
                    b.action_buffer, args.iterations
                )
                record["python_available_actions"] = measure(
                    b.available_actions, args.iterations
                )
                record["python_observe"] = measure(b.observe, args.iterations)
                record["packed_observe"] = measure(
                    b.observation_buffer, args.iterations
                )
                visibility = (C.c_int * 256)()
                record["native_visibility_copy"] = measure(
                    lambda output=visibility: b._lib.sim_visibility(
                        b._handle, 0, output
                    ),
                    args.iterations,
                )
                clones = [b.clone() for _ in range(32)]
                try:
                    commands = [Action("end_turn")] * len(clones)
                    record["python_batch32_end_turn"] = measure(
                        lambda states=clones, commands=commands: step_batch(
                            states, commands
                        ),
                        args.iterations,
                    )
                finally:
                    for clone in clones:
                        clone.close()
                record["snapshot_json_bytes"] = len(json.dumps(b.snapshot()))
                record["native_bytes_including_geometry"] = b._lib.sim_memory(
                    b._handle, 1
                )
                record["native_bytes_per_clone"] = b._lib.sim_memory(b._handle, 0)
                record["python_action_buffer_bytes"] = C.sizeof(b._action_buffer)
                with Battle(s) as collector:
                    rng = random.Random(1729)
                    attack_count, move_count, turn_count, swap_count, resets = (
                        0,
                        0,
                        0,
                        0,
                        0,
                    )
                    start = time.perf_counter()
                    for _ in range(args.iterations):
                        collector.observation_buffer()
                        buffer, n = collector.action_buffer()
                        # Exercise battle commands, not just repeated turn resets.
                        attacks = [i for i in range(n) if buffer[i].kind in (1, 2)]
                        index = rng.choice(attacks) if attacks else rng.randrange(n)
                        command = buffer[index]
                        result = collector.step(
                            Action(KINDS[command.kind], command.unit, command.target)
                        )
                        if not result.ok:
                            raise RuntimeError(f"Collector benchmark: {result.error}")
                        attack_count += command.kind in (1, 2)
                        move_count += command.kind == 0
                        turn_count += command.kind == 5
                        swap_count += command.kind == 6
                        if any(collector.army_status()["destroyed"]):
                            collector.reset()
                            resets += 1
                    elapsed = time.perf_counter() - start
                    record["mixed_collector"] = {
                        "steps": args.iterations,
                        "seconds": elapsed,
                        "steps_per_second": args.iterations / elapsed,
                        "attacks": attack_count,
                        "moves": move_count,
                        "end_turns": turn_count,
                        "swaps": swap_count,
                        "external_resets_after_army_elimination": resets,
                    }
                records.append(record)
    # Moving visibility updates, with resets outside timing.
    with Battle(scenario(10, True, True, True)) as b:
        original = b._units()
        info = b._info()
        # Accumulate pure per-command time; restore in native memory each trial.
        samples = {}
        for label, command in (("move_with_visibility", N.Action(0, 0, 34)),):
            elapsed = 0.0
            for _ in range(args.iterations):
                b._lib.sim_reset(b._handle, original, len(original), C.byref(info))
                start = time.perf_counter()
                result = b._lib.sim_step(b._handle, command)
                elapsed += time.perf_counter() - start
                if result.code:
                    raise RuntimeError(f"Invalid benchmark command: {result.code}")
            samples[label] = {
                "per_second": args.iterations / elapsed,
                "microseconds_each": elapsed * 1e6 / args.iterations,
            }
    duel = scenario(1, True, True, False)
    duel["units"][0]["tile"] = [6, 6]
    duel["units"][1]["tile"] = [7, 6]
    with Battle(duel) as b:
        original, info = b._units(), b._info()
        elapsed = 0.0
        for _ in range(args.iterations):
            b._lib.sim_reset(b._handle, original, len(original), C.byref(info))
            start = time.perf_counter()
            outcome = b._lib.sim_step(b._handle, N.Action(1, 0, 103))
            elapsed += time.perf_counter() - start
            if outcome.code:
                raise RuntimeError("Invalid melee benchmark")
        samples["random_melee"] = {
            "per_second": args.iterations / elapsed,
            "microseconds_each": elapsed * 1e6 / args.iterations,
        }
    result = {
        "platform": platform.platform(),
        "python": sys.version,
        "processor": platform.processor(),
        "rules": N.library().sim_rules().decode(),
        "records": records,
        "command_samples": samples,
        "notes": "CPU only. End-turn timings are not full combat rollout throughput. Batch rate is batches/second.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
