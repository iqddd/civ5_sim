"""Post-compilation BC training benchmark, NVML utilization, and Kineto trace."""

import gzip
import json
import math
import os
import random
import statistics
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from civ5rl.engine.imitation import label_minibatches
from civ5rl.engine.imitation_performance import backward_minibatch_async
from civ5rl.engine.imitation_stream import PreparedDemonstrations, prepare_inputs
from civ5rl.engine.resources import ResourceMonitor, analyze_telemetry, integrate_gpu_interval
from civ5rl.engine.training import FrozenParamsCheck, bc_optimizer_step
from civ5rl.experiments.imitation_helpers import CompilationClock, adam, load_policy
from civ5rl.experiments.imitation_target_scale import (
    _chunk_priority_order,
    load_config as load_target_bc_config,
)
from civ5rl.provenance.common import configure, file_sha256, read_json, seed_for, write_json
from civ5rl.provenance.rng import capture_rng, restore_rng


def _merge_intervals(intervals):
    if not intervals:
        return 0.0, []
    ordered = sorted(intervals)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1][1] = end
        else:
            merged.append([start, end])
    total = sum(end - start for start, end in merged)
    return total, merged


def summarize_kineto_trace(trace_path, wall_seconds):
    opener = gzip.open if str(trace_path).endswith(".gz") else open
    with opener(trace_path, "rt", encoding="utf-8") as stream:
        payload = json.load(stream)
    events = payload.get("traceEvents", [])

    kernel_intervals = []
    memcpy_intervals = []
    memset_intervals = []
    kernel_by_name = {}
    memcpy_by_name = {}
    cuda_runtime_by_name = {}
    annotations = {}

    for ev in events:
        if ev.get("ph") != "X":
            continue
        dur = float(ev.get("dur", 0.0))
        ts = float(ev.get("ts", 0.0))
        cat = ev.get("cat", "")
        name = ev.get("name", "")
        if cat == "kernel":
            kernel_intervals.append((ts, ts + dur))
            rec = kernel_by_name.setdefault(name, {"count": 0, "us": 0.0})
            rec["count"] += 1
            rec["us"] += dur
        elif cat == "gpu_memcpy":
            memcpy_intervals.append((ts, ts + dur))
            rec = memcpy_by_name.setdefault(name, {"count": 0, "us": 0.0})
            rec["count"] += 1
            rec["us"] += dur
        elif cat == "gpu_memset":
            memset_intervals.append((ts, ts + dur))
        elif cat == "cuda_runtime":
            rec = cuda_runtime_by_name.setdefault(name, {"count": 0, "us": 0.0})
            rec["count"] += 1
            rec["us"] += dur
        elif cat == "user_annotation":
            rec = annotations.setdefault(name, {"count": 0, "us": 0.0})
            rec["count"] += 1
            rec["us"] += dur

    kernel_us, merged_kernels = _merge_intervals(kernel_intervals)
    memcpy_us, _ = _merge_intervals(memcpy_intervals)
    memset_us, _ = _merge_intervals(memset_intervals)
    any_gpu_us, merged_gpu = _merge_intervals(
        [*kernel_intervals, *memcpy_intervals, *memset_intervals]
    )

    gaps_us = [
        merged_gpu[i + 1][0] - merged_gpu[i][1]
        for i in range(len(merged_gpu) - 1)
        if merged_gpu[i + 1][0] > merged_gpu[i][1]
    ]
    wall_us = wall_seconds * 1e6

    top_kernels = sorted(
        ({"name": k, "count": v["count"], "ms": v["us"] / 1000.0} for k, v in kernel_by_name.items()),
        key=lambda item: item["ms"],
        reverse=True,
    )[:20]
    top_runtime = sorted(
        ({"name": k, "count": v["count"], "ms": v["us"] / 1000.0} for k, v in cuda_runtime_by_name.items()),
        key=lambda item: item["ms"],
        reverse=True,
    )[:20]
    memcpy_summary = sorted(
        ({"name": k, "count": v["count"], "ms": v["us"] / 1000.0} for k, v in memcpy_by_name.items()),
        key=lambda item: item["ms"],
        reverse=True,
    )
    annotation_summary = sorted(
        ({"name": k, "count": v["count"], "ms": v["us"] / 1000.0} for k, v in annotations.items()),
        key=lambda item: item["ms"],
        reverse=True,
    )

    return {
        "wall_ms": wall_us / 1000.0,
        "gpu_kernel_busy_ms": kernel_us / 1000.0,
        "gpu_kernel_busy_pct_of_wall": 100.0 * kernel_us / wall_us if wall_us else 0.0,
        "gpu_memcpy_ms": memcpy_us / 1000.0,
        "gpu_memset_ms": memset_us / 1000.0,
        "gpu_any_engine_busy_ms": any_gpu_us / 1000.0,
        "gpu_any_engine_busy_pct_of_wall": 100.0 * any_gpu_us / wall_us if wall_us else 0.0,
        "gpu_idle_gaps": {
            "count": len(gaps_us),
            "total_ms": sum(gaps_us) / 1000.0,
            "gaps_over_100us_count": sum(1 for g in gaps_us if g >= 100.0),
            "gaps_over_100us_ms": sum(g for g in gaps_us if g >= 100.0) / 1000.0,
            "gaps_over_1ms_count": sum(1 for g in gaps_us if g >= 1000.0),
            "gaps_over_1ms_ms": sum(g for g in gaps_us if g >= 1000.0) / 1000.0,
            "max_gap_ms": max(gaps_us, default=0.0) / 1000.0,
        },
        "user_annotations": annotation_summary,
        "memcpy_breakdown": memcpy_summary,
        "top_cuda_runtime": top_runtime,
        "top_kernels": top_kernels,
    }


def _execute_bc_steps(
    *,
    policy,
    optimizer,
    cache,
    batches,
    microbatch,
    steps_log_path,
    compilation,
    use_pipeline=True,
    annotate=False,
    on_step_completed=None,
):
    """Execute a list of (spans, label_count) minibatches using the active BC runner."""
    from civ5rl.engine import imitation_performance as ip

    if use_pipeline and hasattr(ip, "run_bc_minibatch_stream"):
        return ip.run_bc_minibatch_stream(
            policy=policy,
            optimizer=optimizer,
            cache=cache,
            batches=batches,
            micro_size=microbatch,
            max_gradient_norm=0.5,
            steps_log_path=steps_log_path,
            compilation=compilation,
            annotate=annotate,
            on_step_completed=on_step_completed,
        )

    def backward(p, c, s, n):
        return backward_minibatch_async(p, c, s, n, microbatch)

    records = []
    steps = 0
    for spans, n in batches:
        start = perf_counter()
        before = dict(cache.counters)
        if annotate:
            with record_function("bc::step_total"):
                with record_function("bc::optimizer_step"):
                    ce, norm = bc_optimizer_step(
                        policy,
                        optimizer,
                        backward,
                        cache,
                        spans,
                        n,
                        max_gradient_norm=0.5,
                        asynchronous=True,
                    )
                with record_function("bc::sync_and_bookkeeping"):
                    torch.cuda.synchronize()
                    ce = float(ce)
                    norm = float(norm)
                    if not math.isfinite(ce) or not math.isfinite(norm):
                        raise FloatingPointError("Non-finite BC loss or gradient norm")
                    steps += 1
                    counters = {int(s["step"]) for s in optimizer.state.values()}
                    if counters != {steps}:
                        raise ValueError("BC Adam counters differ from completed steps")
                    record = {
                        "step": steps,
                        "epoch": 1,
                        "labels": n,
                        "ce": ce,
                        "gradient_norm": norm,
                        "seconds": perf_counter() - start,
                        "prefix_positions": cache.counters["prefix_positions"]
                        - before.get("prefix_positions", 0),
                        "history_positions": cache.counters["history_positions"]
                        - before.get("history_positions", 0),
                        "compile_seconds": compilation.seconds(start, perf_counter())
                        if compilation is not None
                        else 0.0,
                    }
                    records.append(record)
                if steps_log_path is not None:
                    with steps_log_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(record) + "\n")
                if on_step_completed is not None:
                    on_step_completed(record, steps)
        else:
            ce, norm = bc_optimizer_step(
                policy,
                optimizer,
                backward,
                cache,
                spans,
                n,
                max_gradient_norm=0.5,
                asynchronous=True,
            )
            torch.cuda.synchronize()
            ce = float(ce)
            norm = float(norm)
            if not math.isfinite(ce) or not math.isfinite(norm):
                raise FloatingPointError("Non-finite BC loss or gradient norm")
            steps += 1
            counters = {int(s["step"]) for s in optimizer.state.values()}
            if counters != {steps}:
                raise ValueError("BC Adam counters differ from completed steps")
            record = {
                "step": steps,
                "epoch": 1,
                "labels": n,
                "ce": ce,
                "gradient_norm": norm,
                "seconds": perf_counter() - start,
                "prefix_positions": cache.counters["prefix_positions"]
                - before.get("prefix_positions", 0),
                "history_positions": cache.counters["history_positions"]
                - before.get("history_positions", 0),
                "compile_seconds": compilation.seconds(start, perf_counter())
                if compilation is not None
                else 0.0,
            }
            records.append(record)
            if steps_log_path is not None:
                with steps_log_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record) + "\n")
            if on_step_completed is not None:
                on_step_completed(record, steps)
    return records


@contextmanager
def _instrument_for_trace(policy, cache, optimizer):
    """Attach temporary record_function scopes for Kineto attribution only."""
    import civ5rl.engine.imitation_performance as ip

    orig_get = cache.get
    orig_get_arrays = getattr(cache, "get_arrays", None)
    orig_pack = getattr(cache, "_pack_cpu_spans", None)
    orig_batch = cache.batch
    orig_gather = getattr(cache, "gather_spatial_inputs", None)
    orig_encode = policy.encode_spatial
    orig_seq = policy.sequence_memory
    orig_heads = policy.score_heads
    orig_prefixes = ip.prefixes
    orig_opt_step = optimizer.step
    orig_clip = torch.nn.utils.clip_grad_norm_

    def wrapped_get(*a, **kw):
        with record_function("bc::cache_get"):
            return orig_get(*a, **kw)

    def wrapped_get_arrays(*a, **kw):
        with record_function("bc::cache_get_arrays"):
            return orig_get_arrays(*a, **kw)

    def wrapped_pack(*a, **kw):
        with record_function("bc::pack_cpu_spans"):
            return orig_pack(*a, **kw)

    def wrapped_batch(*a, **kw):
        mode = "grad" if torch.is_grad_enabled() else "nograd"
        with record_function(f"bc::cache_batch[{mode}]"):
            return orig_batch(*a, **kw)

    def wrapped_gather(*a, **kw):
        with record_function("bc::gather_spatial_inputs"):
            return orig_gather(*a, **kw)

    def wrapped_encode(*a, **kw):
        mode = "grad" if torch.is_grad_enabled() else "nograd"
        with record_function(f"bc::encode_spatial[{mode}]"):
            return orig_encode(*a, **kw)

    def wrapped_seq(*a, **kw):
        mode = "grad" if torch.is_grad_enabled() else "nograd"
        with record_function(f"bc::sequence_memory[{mode}]"):
            return orig_seq(*a, **kw)

    def wrapped_heads(*a, **kw):
        with record_function("bc::score_heads"):
            return orig_heads(*a, **kw)

    def wrapped_prefixes(*a, **kw):
        with record_function("bc::prefixes"):
            return orig_prefixes(*a, **kw)

    def wrapped_opt_step(*a, **kw):
        with record_function("bc::adam_step"):
            return orig_opt_step(*a, **kw)

    def wrapped_clip(*a, **kw):
        with record_function("bc::clip_grad_norm"):
            return orig_clip(*a, **kw)

    cache.get = wrapped_get
    if orig_get_arrays is not None:
        cache.get_arrays = wrapped_get_arrays
    if orig_pack is not None:
        cache._pack_cpu_spans = wrapped_pack
    cache.batch = wrapped_batch
    if orig_gather is not None:
        cache.gather_spatial_inputs = wrapped_gather
    policy.encode_spatial = wrapped_encode
    policy.sequence_memory = wrapped_seq
    policy.score_heads = wrapped_heads
    ip.prefixes = wrapped_prefixes
    optimizer.step = wrapped_opt_step
    torch.nn.utils.clip_grad_norm_ = wrapped_clip
    try:
        yield
    finally:
        cache.get = orig_get
        if orig_get_arrays is not None:
            cache.get_arrays = orig_get_arrays
        if orig_pack is not None:
            cache._pack_cpu_spans = orig_pack
        cache.batch = orig_batch
        if orig_gather is not None:
            cache.gather_spatial_inputs = orig_gather
        policy.encode_spatial = orig_encode
        policy.sequence_memory = orig_seq
        policy.score_heads = orig_heads
        ip.prefixes = orig_prefixes
        optimizer.step = orig_opt_step
        torch.nn.utils.clip_grad_norm_ = orig_clip


def run_bc_benchmark(args, parser):
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    run_dir = (args.run or Path(load_target_bc_config(config_path)["output"])).resolve()
    collection_path = run_dir / "collection.json"
    if not config_path.is_file() or not checkpoint_path.is_file() or not collection_path.is_file():
        parser.error("BC benchmark requires existing --config, --checkpoint, and RUN/collection.json")

    out = args.out.resolve()
    if out.suffix.lower() != ".json":
        out = out / "bc-benchmark.json"
    if out.exists():
        parser.error("benchmark output already exists; choose a new result path")
    if any(out == p or out.is_relative_to(p) for p in (run_dir,)):
        parser.error("benchmark output must be outside the scientific run directory")

    scratch_dir = (args.scratch_dir or (Path(".tmp") / "bc-bench-scratch")).resolve()
    if any(scratch_dir == p or scratch_dir.is_relative_to(p) for p in (run_dir,)):
        parser.error("scratch_dir must be outside the scientific run directory")

    protected = [config_path, checkpoint_path, collection_path]
    before_hashes = {str(p): file_sha256(p) for p in protected}

    config = load_target_bc_config(config_path)
    data = read_json(collection_path)
    if data.get("status") != "complete":
        parser.error("RUN/collection.json is not complete")

    os.environ["CIV5RL_SCRATCH_DIR"] = str(scratch_dir)
    prep_workers = args.prep_workers or min(10, os.cpu_count() or 4)
    input_preparation = prepare_inputs(
        run_dir, data["entries"], workers=prep_workers, scratch_dir=scratch_dir
    )
    prepared_entries = input_preparation["entries"]

    configure(args.seed, args.device)
    policy = load_policy(config, checkpoint_path, args.device)
    base_model_state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    frozen = FrozenParamsCheck(policy.value)

    fixed_limit = min(config["cache_bytes"] // 16, 256 * 1024**2)
    cache_limit = config["cache_bytes"] - fixed_limit
    cache = PreparedDemonstrations(
        run_dir, prepared_entries, args.device, cache_limit, scratch_dir=scratch_dir
    )
    if not args.legacy_loop:
        cache.spatial_cell_padding = policy.spatial_cell_padding
    chunks = cache.chunks()

    shuffle = random.Random(
        seed_for(
            17, 0, "shuffle", namespace=config["experiment"]["shuffle_namespace"]
        )
    )
    all_minibatches = []
    epoch_idx = 0
    target_batches = args.warmup + args.steps + max(args.trace_steps + 5, 16)
    while len(all_minibatches) < target_batches:
        ep_rng = (
            shuffle
            if epoch_idx == 0
            else random.Random(
                seed_for(
                    17,
                    epoch_idx,
                    "shuffle",
                    namespace=config["experiment"]["shuffle_namespace"],
                )
            )
        )
        epoch_order = list(chunks)
        ep_rng.shuffle(epoch_order)
        all_minibatches.extend(label_minibatches(epoch_order, cache.masks))
        epoch_idx += 1

    priority_order = _chunk_priority_order(data, chunks)
    priority_probes = list(label_minibatches(priority_order, cache.masks))[:4]
    warmup_count = min(args.warmup, len(all_minibatches))
    measured_count = min(args.steps, len(all_minibatches) - warmup_count)
    if measured_count < 1:
        parser.error("Not enough minibatches in epoch for requested --warmup and --steps")

    warmup_batches = [*priority_probes, *all_minibatches[:warmup_count]]
    measured_batches = all_minibatches[warmup_count : warmup_count + measured_count]

    out.parent.mkdir(parents=True, exist_ok=True)
    telemetry_dir = out.parent / (out.stem + "-telemetry")
    steps_log = out.parent / (out.stem + "-steps.jsonl")

    configure(args.seed, args.device)
    rng_snapshot = capture_rng(shuffle)

    microbatch = config.get("microbatch_sequences", 8)

    with (
        CompilationClock() as compilation,
        ResourceMonitor(telemetry_dir, interval=0.5) as monitor,
    ):
        compilation.monitor = monitor
        monitor.mark("warmup")
        policy.load_state_dict(base_model_state)
        optimizer = adam(policy, config, args.device)
        _execute_bc_steps(
            policy=policy,
            optimizer=optimizer,
            cache=cache,
            batches=warmup_batches,
            microbatch=microbatch,
            steps_log_path=None,
            compilation=compilation,
            use_pipeline=not args.legacy_loop,
            annotate=False,
        )
        # Pre-compile full 32-block spatial CUDAGraphTree chains and singleton head
        # specializations during warmup so post-compilation measurement has zero compiles.
        if str(args.device).startswith("cuda"):
            from civ5rl.core.schema import (
                ACTION_DIM,
                GLOBAL_DIM,
                HEX_DIRECTIONS,
                PREV_ACTION_DIM,
                TILE_DIM,
            )

            num_cells = max(policy.spatial_cell_padding, 256)
            for s_rows in (1024, 1023, 31):
                dummy_sp = {
                    "cells": torch.zeros((s_rows, num_cells, TILE_DIM), device=args.device),
                    "cell_mask": torch.ones((s_rows, num_cells), dtype=torch.bool, device=args.device),
                    "neighbors": torch.full((s_rows, num_cells, HEX_DIRECTIONS), -1, dtype=torch.long, device=args.device),
                    "globals": torch.zeros((s_rows, GLOBAL_DIM), device=args.device),
                    "previous": torch.zeros((s_rows, PREV_ACTION_DIM), device=args.device),
                }
                with torch.no_grad():
                    if policy.spatial_cuda_graphs:
                        torch.compiler.cudagraph_mark_step_begin()
                    policy.encode_spatial(dummy_sp)
                if policy.spatial_cuda_graphs:
                    torch.compiler.cudagraph_mark_step_begin()
                sx, sc = policy.encode_spatial(dummy_sp)
                (sx.float().sum() + sc.float().sum()).backward()
                optimizer.zero_grad(set_to_none=True)

            if getattr(policy, "_heads_runner", None) is not None:
                for n_rows, n_acts in ((1, 4), (4, 1), (1, 1)):
                    dummy_x = torch.zeros(
                        (n_rows, num_cells, policy.hex_dim),
                        dtype=torch.bfloat16,
                        device=args.device,
                        requires_grad=True,
                    )
                    dummy_h = torch.zeros(
                        (n_rows, policy.hidden_size),
                        device=args.device,
                        requires_grad=True,
                    )
                    dummy_b = {
                        "sources": torch.zeros((n_rows, n_acts), dtype=torch.long, device=args.device),
                        "targets": torch.zeros((n_rows, n_acts), dtype=torch.long, device=args.device),
                        "action_features": torch.zeros((n_rows, n_acts, ACTION_DIM), device=args.device),
                        "action_mask": torch.ones((n_rows, n_acts), dtype=torch.bool, device=args.device),
                    }
                    lg, vl = policy.score_heads(dummy_b, dummy_x, dummy_h)
                    (lg.sum() + vl.sum()).backward()
                optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

        # Reset model, optimizer, RNG, and optionally CPU cache for clean measurement.
        policy.load_state_dict(base_model_state)
        optimizer = adam(policy, config, args.device)
        restore_rng(rng_snapshot, shuffle)
        if args.cold_cache:
            cache.clear()
        cache.counters.clear()

        start_Compile_intervals = len(compilation.intervals)
        gru_runner = getattr(policy, "_memory_runner", None)
        start_gru_captures = getattr(gru_runner, "captures", 0)
        monitor.mark("iterative_training")
        start_elapsed = monitor.sample()["elapsed_seconds"]
        wall_start = perf_counter()
        records = _execute_bc_steps(
            policy=policy,
            optimizer=optimizer,
            cache=cache,
            batches=measured_batches,
            microbatch=microbatch,
            steps_log_path=steps_log,
            compilation=compilation,
            use_pipeline=not args.legacy_loop,
            annotate=False,
        )
        torch.cuda.synchronize()
        wall_seconds = perf_counter() - wall_start
        stop_elapsed = monitor.sample()["elapsed_seconds"]
        gru_captures_in_measured = getattr(gru_runner, "captures", 0) - start_gru_captures
        monitor.mark("finished")
        frozen.verify()

    telemetry_analysis = analyze_telemetry(
        telemetry_dir / "resources.jsonl", active_phase="iterative_training", window=60
    )
    samples = [
        json.loads(line)
        for line in (telemetry_dir / "resources.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    phases = [
        json.loads(line)
        for line in (telemetry_dir / "phases.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    integrated = integrate_gpu_interval(
        samples,
        start_elapsed,
        stop_elapsed,
        session_id=samples[0]["session_id"],
        sample_interval=0.5,
        phase_events=phases,
        active_phase="iterative_training",
    )
    steady_tail_60s = (
        integrate_gpu_interval(
            samples,
            max(start_elapsed, stop_elapsed - 65.0),
            stop_elapsed,
            session_id=samples[0]["session_id"],
            sample_interval=0.5,
            phase_events=phases,
            active_phase="iterative_training",
        )
        if stop_elapsed - start_elapsed >= 60.0
        else None
    )

    parity = None
    if args.verify_parity:
        parity_steps = min(args.parity_steps, len(measured_batches))
        parity_batches = measured_batches[:parity_steps]

        policy.load_state_dict(base_model_state)
        opt_ref = adam(policy, config, args.device)
        restore_rng(rng_snapshot, shuffle)
        cache.clear()
        cache.counters.clear()
        ref_records = _execute_bc_steps(
            policy=policy,
            optimizer=opt_ref,
            cache=cache,
            batches=parity_batches,
            microbatch=microbatch,
            steps_log_path=None,
            compilation=None,
            use_pipeline=False,
            annotate=False,
        )
        ref_weights = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

        policy.load_state_dict(base_model_state)
        opt_new = adam(policy, config, args.device)
        restore_rng(rng_snapshot, shuffle)
        cache.clear()
        cache.counters.clear()
        new_records = _execute_bc_steps(
            policy=policy,
            optimizer=opt_new,
            cache=cache,
            batches=parity_batches,
            microbatch=microbatch,
            steps_log_path=None,
            compilation=None,
            use_pipeline=True,
            annotate=False,
        )
        new_weights = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}

        max_ce_diff = max(
            abs(a["ce"] - b["ce"]) for a, b in zip(ref_records, new_records, strict=True)
        )
        max_norm_diff = max(
            abs(a["gradient_norm"] - b["gradient_norm"])
            for a, b in zip(ref_records, new_records, strict=True)
        )
        max_param_diff = max(
            float((ref_weights[k] - new_weights[k]).abs().max().item())
            for k in ref_weights
        )
        exact_weights = all(
            torch.equal(ref_weights[k], new_weights[k]) for k in ref_weights
        )
        parity = {
            "steps": parity_steps,
            "max_ce_abs_diff": max_ce_diff,
            "max_grad_norm_abs_diff": max_norm_diff,
            "max_param_abs_diff": max_param_diff,
            "exact_bitwise_weights": exact_weights,
        }

    trace_summary = None
    trace_file = None
    if args.trace_steps > 0:
        lead_in = 3
        need_total = lead_in + args.trace_steps + (0 if args.legacy_loop else 1)
        trace_start = warmup_count + measured_count
        trace_batches = all_minibatches[trace_start : trace_start + need_total]
        if len(trace_batches) < need_total:
            trace_batches = measured_batches[: min(need_total, len(measured_batches))]
        trace_file = out.parent / (out.stem + "-trace.json.gz")
        policy.load_state_dict(base_model_state)
        opt_trace = adam(policy, config, args.device)
        restore_rng(rng_snapshot, shuffle)
        if args.cold_cache:
            cache.clear()
        torch.cuda.synchronize()
        active_window = {"t0": None, "t1": None}
        sched = torch.profiler.schedule(
            wait=lead_in - 1, warmup=1, active=args.trace_steps, repeat=1
        )
        with (
            _instrument_for_trace(policy, cache, opt_trace),
            profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=sched,
                record_shapes=False,
            ) as prof,
        ):
            def _on_trace_step(rec, step_idx):
                now_t = perf_counter()
                if step_idx == lead_in:
                    active_window["t0"] = now_t
                elif step_idx == lead_in + args.trace_steps:
                    active_window["t1"] = now_t
                prof.step()

            _execute_bc_steps(
                policy=policy,
                optimizer=opt_trace,
                cache=cache,
                batches=trace_batches,
                microbatch=microbatch,
                steps_log_path=None,
                compilation=None,
                use_pipeline=not args.legacy_loop,
                annotate=True,
                on_step_completed=_on_trace_step,
            )
            torch.cuda.synchronize()
        trace_wall = (
            active_window["t1"] - active_window["t0"]
            if active_window["t0"] is not None and active_window["t1"] is not None
            else 0.001
        )
        prof.export_chrome_trace(str(trace_file))
        trace_summary = summarize_kineto_trace(trace_file, trace_wall)
        write_json(out.parent / (out.stem + "-trace-summary.json"), trace_summary)

    for p in protected:
        if file_sha256(p) != before_hashes[str(p)]:
            raise AssertionError(f"Benchmark modified protected file: {p}")

    step_times = [r["seconds"] for r in records]
    compile_in_measured = sum(r["compile_seconds"] for r in records)
    total_labels = sum(r["labels"] for r in records)

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "run_dir": str(run_dir),
        "scratch_dir": str(scratch_dir),
        "legacy_loop": bool(args.legacy_loop),
        "cold_cache": bool(args.cold_cache),
        "warmup_steps": len(warmup_batches),
        "measured_steps": len(records),
        "measured_labels": total_labels,
        "wall_seconds": wall_seconds,
        "compile_seconds_in_measured": compile_in_measured,
        "compile_events_in_measured": len(compilation.intervals) - start_Compile_intervals,
        "gru_graph_captures_in_measured": gru_captures_in_measured,
        "step_seconds": {
            "mean": wall_seconds / len(records),
            "recorded_mean": statistics.mean(step_times),
            "median": statistics.median(step_times),
            "p95": sorted(step_times)[int(0.95 * (len(step_times) - 1))],
            "min": min(step_times),
            "max": max(step_times),
        },
        "labels_per_second": total_labels / wall_seconds,
        "steps_per_second": len(records) / wall_seconds,
        "cache_statistics": cache.statistics(),
        "nvml_integrated": integrated,
        "nvml_steady_tail_65s": steady_tail_60s,
        "nvml_metrics": telemetry_analysis["metrics"],
        "nvml_60s_windows_count": len(telemetry_analysis["windows"]),
        "nvml_60s_mean": (
            statistics.mean(w["core_average"] for w in telemetry_analysis["windows"])
            if telemetry_analysis["windows"]
            else None
        ),
        "nvml_60s_last": (
            telemetry_analysis["windows"][-1]["core_average"]
            if telemetry_analysis["windows"]
            else None
        ),
        "parity": parity,
        "trace_file": str(trace_file) if trace_file else None,
        "trace_summary": trace_summary,
        "protected_inputs_unchanged": True,
    }
    write_json(out, result)
    print(
        json.dumps(
            {
                "output": str(out),
                "wall_seconds": round(wall_seconds, 3),
                "mean_step_ms": round(1000.0 * wall_seconds / len(records), 2),
                "labels_per_second": round(total_labels / wall_seconds, 1),
                "compile_seconds_in_measured": round(compile_in_measured, 4),
                "nvml_core_mean_pct": integrated["gpu_core_load_percent_mean"],
                "nvml_60s_last_pct": result["nvml_60s_last"],
                "trace_kernel_busy_pct": (
                    round(trace_summary["gpu_kernel_busy_pct_of_wall"], 2)
                    if trace_summary
                    else None
                ),
                "parity": parity,
            },
            indent=2,
        )
    )
    return result
