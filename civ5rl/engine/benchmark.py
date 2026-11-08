"""Bounded discarded-update measurements and deterministic update comparisons."""

import copy
import inspect
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from civ5rl.engine.ppo import update
from civ5rl.experiments.imitation_helpers import CompilationClock
from civ5rl.experiments.ppo_artifacts import save_rollout
from civ5rl.provenance.common import file_sha256
from civ5rl.provenance.rng import capture_rng, restore_rng


def _hashes(paths):
    return {str(Path(p).resolve()): file_sha256(p) for p in paths}


def _compare(a, b, *, rtol, atol, path="result"):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=rtol, atol=atol, msg=path)
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.dtype != b.dtype:
            raise AssertionError(f"{path}: dtype {a.dtype} != {b.dtype}")
        if np.issubdtype(a.dtype, np.inexact) and np.issubdtype(b.dtype, np.inexact):
            np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=path)
        else:
            np.testing.assert_array_equal(a, b, err_msg=path)
    elif isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            raise AssertionError(f"{path}: keys differ")
        for key in a:
            _compare(a[key], b[key], rtol=rtol, atol=atol, path=f"{path}.{key}")
    elif isinstance(a, (tuple, list)) and isinstance(b, type(a)):
        if len(a) != len(b):
            raise AssertionError(f"{path}: lengths differ")
        for i, (x, y) in enumerate(zip(a, b)):
            _compare(x, y, rtol=rtol, atol=atol, path=f"{path}[{i}]")
    elif isinstance(a, float) and isinstance(b, (int, float)):
        if not math.isclose(a, b, rel_tol=rtol, abs_tol=atol):
            raise AssertionError(f"{path}: {a} != {b}")
    elif a != b:
        raise AssertionError(f"{path}: {a} != {b}")


def assert_update_parity(
    fn_a, fn_b, *, reset, shuffle_rng=None, protected_inputs=(), rtol=3e-5, atol=2e-7
):
    """Run ABBA after reset; callbacks return model, optimizer, gradients, sampled_kl, commands.

    reset must restore model and Adam to their initial state; this helper restores
    Python, NumPy, torch, CUDA and optional shuffle RNG before every callback. Callbacks must return
    independent snapshots of gradients (including any deliberately absent ones).
    Commands compare exactly, all remaining numeric tensors use tolerances.
    """
    expected = {"model", "optimizer", "gradients", "sampled_kl", "commands"}
    hashes = _hashes(protected_inputs)
    rng = capture_rng(shuffle_rng)
    results = []
    try:
        for fn in (fn_a, fn_b, fn_b, fn_a):
            reset()
            restore_rng(rng, shuffle_rng)
            result = copy.deepcopy(fn())
            if set(result) != expected:
                raise ValueError(
                    f"Parity callback must return exactly {sorted(expected)}"
                )
            results.append(result)
            if _hashes(protected_inputs) != hashes:
                raise AssertionError("Protected input changed during parity run")
        reference = results[0]
        for result in results[1:]:
            if result["commands"] != reference["commands"]:
                raise AssertionError("Trajectory commands differ")
            for key in expected - {"commands"}:
                _compare(reference[key], result[key], rtol=rtol, atol=atol, path=key)
    finally:
        reset()
        restore_rng(rng, shuffle_rng)
        if _hashes(protected_inputs) != hashes:
            raise AssertionError("Protected input changed during parity run")
    return {"passed": True, "order": "ABBA", "protected_sha256": hashes}


def run_full_cycle_benchmark(
    *,
    policy,
    optimizer,
    shuffle_rng,
    collector,
    config,
    specs,
    manifest,
    output,
    device,
    repeats=3,
    protected_inputs=(),
    monitor=None,
):
    """One warmup and N fixed-case collect/save/update trials, restoring all state."""
    if repeats < 1 or not specs:
        raise ValueError("Need at least one measurement and one fixed case")
    from civ5sim.engine import digest

    output = Path(output).resolve()
    paths = [Path(p).resolve() for p in protected_inputs]
    if any(p == output or p.is_relative_to(output) for p in paths):
        raise ValueError("Output directory contains a protected input")
    hashes = _hashes(paths)
    initial_model = copy.deepcopy(policy.state_dict())
    initial_optimizer = copy.deepcopy(optimizer.state_dict())
    initial_rng = capture_rng(shuffle_rng)
    initial_training = policy.training
    initial_gradients = [
        None if p.grad is None else p.grad.detach().clone() for p in policy.parameters()
    ]
    rows = []

    def restore():
        policy.load_state_dict(initial_model)
        optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
        policy.train(initial_training)
        for parameter, gradient in zip(policy.parameters(), initial_gradients):
            parameter.grad = None if gradient is None else gradient.clone()
        restore_rng(initial_rng, shuffle_rng)

    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)

    try:
        for trial in range(repeats + 1):
            restore()
            sync()
            start = perf_counter()
            if monitor:
                monitor.mark(
                    "warmup_collection" if trial == 0 else "collection", trial=trial
                )
            episodes, _, timings = collector.collect(
                policy, None, 0, fixed_specs=specs, trusted_manifest=manifest
            )
            sync()
            collected = perf_counter()
            if monitor:
                monitor.mark(
                    "warmup_save" if trial == 0 else "rollout_save", trial=trial
                )
            record = save_rollout(episodes, output / f"trial-{trial}", 1)
            saved = perf_counter()
            if monitor:
                monitor.mark("warmup_ppo" if trial == 0 else "ppo", trial=trial)
            with CompilationClock() as compilation:
                compilation.monitor = monitor
                metrics = update(
                    policy, optimizer, episodes, config, 0, device, shuffle_rng
                )
                sync()
            stop = perf_counter()
            trajectory_hash = digest(
                [
                    {
                        "commands": ep["joint_commands"],
                        "result": {
                            k: v for k, v in ep["result"].items() if k != "timings"
                        },
                    }
                    for ep in episodes
                ]
            )
            if rows and trajectory_hash != rows[0]["trajectory_sha256"]:
                raise AssertionError(
                    "Fixed-case trajectory or outcome changed between trials"
                )
            rows.append(
                {
                    "trial": trial,
                    "warmup": trial == 0,
                    "full_seconds": stop - start,
                    "collection_seconds": collected - start,
                    "save_seconds": saved - collected,
                    "ppo_seconds": stop - saved,
                    "compilation_in_ppo_seconds": compilation.seconds(saved, stop),
                    "accepted_adam_steps": metrics["optimizer_steps"],
                    "decisions": sum(len(e["chosen"]) for e in episodes),
                    "rollout_sha256": record["sha256"],
                    "trajectory_sha256": trajectory_hash,
                    "collection_timing": timings,
                }
            )
            if _hashes(paths) != hashes:
                raise AssertionError("Protected input changed during benchmark")
    finally:
        restore()
        if _hashes(paths) != hashes:
            raise AssertionError("Protected input changed during benchmark")
    measured = rows[1:]
    seconds = sum(row["full_seconds"] for row in measured)
    steps = sum(row["accepted_adam_steps"] for row in measured)
    return {
        "trials": rows,
        "protected_sha256": hashes,
        "seconds_per_accepted_adam_amortized": seconds / steps if steps else None,
    }


def run_replay_benchmark(
    *,
    policy,
    episodes,
    implementations,
    config,
    device,
    repeats=1,
    warmup=1,
    protected_inputs=(),
    monitor=None,
    rtol=3e-5,
    atol=2e-7,
):
    """Compare fixed-weight replay in ABBA order; no collection or optimizer.

    Each implementation owns its RolloutCache and function globals. Only the
    selected module differs; model, dependencies, cases and precision are fixed.
    All metrics must agree, and model parameters must remain bitwise unchanged.
    """
    from civ5rl.core.distributions import (
        require_rollout_temperature,
        validate_temperature,
    )
    from civ5sim.engine import digest

    if (
        type(repeats) is not int
        or repeats < 1
        or type(warmup) is not int
        or warmup < 1
        or not episodes
        or not 1 <= len(implementations) <= 2
    ):
        raise ValueError(
            "Need saved episodes, one/two implementations and positive repeats/warmup"
        )
    temperature = validate_temperature(config.get("policy_temperature", 1.0))
    require_rollout_temperature(episodes, temperature)
    # Check support before any warmup so an old T=1 backend cannot be silently
    # presented as the baseline for a cold-policy rollout.
    for name, module in implementations.items():
        parameters = inspect.signature(module.measure_replay).parameters
        if temperature != 1.0 and "temperature" not in parameters:
            raise ValueError(
                f"{name}: Git replay implementation lacks policy temperature support"
            )
        if "full_distribution" not in parameters:
            raise ValueError(
                f"{name}: Replay implementation lacks full-distribution diagnostics"
            )

    hashes = _hashes(protected_inputs)
    initial_model = copy.deepcopy(policy.state_dict())
    initial_gradients = [
        None if p.grad is None else p.grad.detach().clone() for p in policy.parameters()
    ]
    initial_training = policy.training
    initial_rng = capture_rng()
    options = config.get("performance", {})
    sequence_batch = options.get("kl_sequences_per_batch", 8)
    caches = {}
    records = []
    reference = None

    def reset():
        policy.load_state_dict(initial_model)
        policy.eval()
        for p, gradient in zip(policy.parameters(), initial_gradients):
            p.grad = None if gradient is None else gradient.clone()
        restore_rng(initial_rng)

    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize(device)

    def call(name, *, warming, index):
        nonlocal reference
        if monitor:
            monitor.mark("replay_preparation", trial=index)
        reset()
        module = implementations[name]
        kwargs = {"full_distribution": True}
        if "temperature" in inspect.signature(module.measure_replay).parameters:
            kwargs["temperature"] = temperature
        if monitor:
            monitor.mark(("warmup_" if warming else "replay_") + name, trial=index)
        sync()
        with CompilationClock() as compilation:
            compilation.monitor = monitor
            started = perf_counter()
            metrics = module.measure_replay(
                policy, caches[name], sequence_batch, **kwargs
            )
            sync()
            stopped = perf_counter()
        # Validation/hashing is deliberately outside the measured replay interval.
        if monitor:
            monitor.mark("replay_validation", trial=index)
        for key, value in metrics.items():
            if isinstance(value, (float, int)) and not math.isfinite(value):
                raise AssertionError(f"{name}: non-finite replay metric {key}")
        if reference is None:
            reference = copy.deepcopy(metrics)
        else:
            _compare(reference, metrics, rtol=rtol, atol=atol, path=f"replay.{name}")
        for key, value in policy.state_dict().items():
            if not torch.equal(value, initial_model[key]):
                raise AssertionError(f"Replay modified model weights: {key}")
        if _hashes(protected_inputs) != hashes:
            raise AssertionError(
                "Protected source/input changed during replay benchmark"
            )
        if not warming:
            records.append(
                {
                    "implementation": name,
                    "trial": index,
                    "seconds": stopped - started,
                    "compilation_seconds": compilation.seconds(started, stopped),
                    "metrics": copy.deepcopy(metrics),
                }
            )

    names = list(implementations)
    order = [names[0], names[1], names[1], names[0]] if len(names) == 2 else names
    try:
        for name, module in implementations.items():
            if monitor:
                monitor.mark("replay_preparation", implementation=name)
            reset()
            caches[name] = module.RolloutCache(
                copy.deepcopy(episodes),
                policy.variant,
                device,
                resident=options.get("resident_rollout", True),
            )
            if str(device).startswith("cuda") and options.get(
                "ppo_optimized_inputs", False
            ):
                caches[name].enable_compiled_gather()
                caches[name].enable_async_transfers()
            if options.get("ppo_cached_replay", False):
                caches[name].enable_cached_replay()
            for iteration in range(warmup):
                call(name, warming=True, index=iteration)
        for repeat in range(repeats):
            for name in order:
                call(name, warming=False, index=repeat)
        results = {}
        for name in names:
            values = [r["seconds"] for r in records if r["implementation"] == name]
            results[name] = {
                "passes": len(values),
                "mean_seconds": float(np.mean(values)),
                "median_seconds": float(np.median(values)),
                "min_seconds": min(values),
                "max_seconds": max(values),
            }
        return {
            "order": order * repeats,
            "warmup_passes_per_implementation": warmup,
            "warmup_excluded": True,
            "parity_verified": len(names) == 2,
            "rtol": rtol,
            "atol": atol,
            "policy_temperature": temperature,
            "episodes": len(episodes),
            "decisions": sum(len(ep["chosen"]) for ep in episodes),
            "joint_commands_sha256": digest([ep["joint_commands"] for ep in episodes]),
            "Adam_steps": 0,
            "new_matches": 0,
            "protected_sha256": hashes,
            "results": results,
            "trials": records,
        }
    finally:
        reset()
        policy.train(initial_training)
        if _hashes(protected_inputs) != hashes:
            raise AssertionError(
                "Protected source/input changed during replay benchmark"
            )
