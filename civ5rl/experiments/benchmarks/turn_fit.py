"""Measure whole-turn fit steps on a disposable copy of the configured workload."""

from __future__ import annotations

import random
from pathlib import Path
from statistics import mean
from time import perf_counter

import torch
from torch.profiler import ProfilerActivity, profile

from civ5rl.engine.comparison.turn_fit import configure_policy, parent_references
from civ5rl.engine.comparison.turn_inputs import TurnHistories
from civ5rl.engine.comparison.turn_training import (
    control_kl,
    hidden_states_at,
    logical_step,
    pack_batch,
    prepare_anchor_subset,
    prepare_preferences,
)
from civ5rl.engine.runtime.telemetry import ResourceMonitor
from civ5rl.engine.runtime.transactions import AdamSnapshot, atomic_torch_save, cpu_snapshot
from civ5rl.experiments.benchmarks.profiling import store_trace, top_cuda_kernels
from civ5rl.experiments.comparison.turn_data import build_dataset, history_inputs
from civ5rl.experiments.comparison.turn_fit import load_config
from civ5rl.provenance.common import ROOT, configure, seed_for, write_json
from civ5rl.provenance.rng import capture_rng


def benchmark(args, parser):
    """Discard target-host full-size steps; report preparation, warmup, update and KL separately."""

    if args.steps < 1 or args.warmup < 0 or args.trace_steps < 0 or args.history_microbatch < 1:
        parser.error("steps/microbatch must be positive, warmup/trace-steps nonnegative")
    cfg = load_config(args.config)
    out = Path(args.out).resolve()
    if out == (ROOT / cfg["output"]).resolve() or out.exists() and any(out.iterdir()):
        parser.error("Use a separate empty benchmark output")
    out.mkdir(parents=True, exist_ok=True)
    configure(cfg["seed"], "cuda")
    started = perf_counter()
    dataset = build_dataset(cfg)
    training = cfg["training"]
    if training["positions_per_step"] > len(dataset["positions"]):
        parser.error("training.positions_per_step exceeds the number of prepared positions")
    order = list(range(len(dataset["positions"])))
    random.Random(seed_for(cfg["seed"], 0, "positions", namespace=cfg["namespace"])).shuffle(order)
    workload = {
        **dataset,
        "positions": [dataset["positions"][i] for i in order[: training["positions_per_step"]]],
    }
    anchor_order = list(range(len(dataset["anchors"])))
    random.Random(seed_for(cfg["seed"], 0, "anchors", namespace=cfg["namespace"])).shuffle(anchor_order)
    anchors = [dataset["anchors"][i] for i in anchor_order[: training["anchors_per_step"]]]
    with history_inputs(cfg, workload) as (sources, roots, branches):
        histories = TurnHistories(out, sources, roots, branches, dataset["anchors"])
        preferences = prepare_preferences(histories, workload["positions"])
        chosen_keys = sorted(histories.targets)
        anchor_subset = prepare_anchor_subset(
            histories, dataset["anchors"], preferences.training_observations
        )
        histories.prepare([*chosen_keys, *map(tuple, dataset["anchors"])])
        batch = pack_batch(
            preferences.rows, "cuda", histories=histories, branch_steps=preferences.branch_steps
        )
        policy = configure_policy(str(ROOT / cfg["parents"][0]), compile_model=args.compile)
        optimizer = torch.optim.Adam(
            [p for p in policy.parameters() if p.requires_grad],
            lr=training["initial_lr"],
            betas=(0.9, 0.999),
            eps=1e-5,
            weight_decay=0,
        )
        parent, train_refs, control_refs = parent_references(
            policy, histories, workload, batch.keys, out, cfg, branch_steps=preferences.branch_steps
        )
        reference = torch.stack([parent[k] for k in batch.keys])
        needed_anchor_bp = [(ep, idx - 128) for ep, idx in set(map(tuple, anchors)) if idx >= 128]
        anchor_bp_h = hidden_states_at(policy, histories, needed_anchor_bp)
        anchor_prefixes = {
            (ep, idx): anchor_bp_h[(ep, idx - 128)] for ep, idx in set(map(tuple, anchors)) if idx >= 128
        }
        needed_branch_bp = [bk for pos in batch.positions for bk in pos.boundary_keys if bk is not None]
        branch_prefixes = hidden_states_at(
            policy, histories, needed_branch_bp, branch_steps=preferences.branch_steps
        )
        torch.cuda.synchronize()
        preparation = perf_counter() - started
        snapshot = AdamSnapshot(policy, optimizer)
        rows = []
        warmup = []

        def step():
            torch.cuda.synchronize()
            phase_started = perf_counter()
            logical_step(
                policy,
                optimizer,
                histories,
                batch,
                anchors,
                reference,
                train_refs,
                microbatch=args.history_microbatch,
                anchor_prefixes=anchor_prefixes,
                branch_prefixes=branch_prefixes,
            )
            torch.cuda.synchronize()
            update = perf_counter() - phase_started
            phase_started = perf_counter()
            kl, _ = control_kl(
                policy,
                histories,
                dataset["anchors"],
                control_refs,
                anchor_subset=anchor_subset,
                collect_prefixes=anchors,
                collected_prefixes={},
            )
            torch.cuda.synchronize()
            control = perf_counter() - phase_started
            phase_started = perf_counter()

            for _ in range(2):
                atomic_torch_save(
                    cpu_snapshot(
                        {
                            "model": policy.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "rng": capture_rng(),
                        }
                    ),
                    out / "discarded-working.pt",
                )
            return {
                "step_seconds": update,
                "full_kl_seconds": control,
                "working_saves_seconds": perf_counter() - phase_started,
                "full_kl": kl,
            }

        with ResourceMonitor(out / "telemetry") as monitor:
            monitor.mark("warmup")
            for _ in range(args.warmup):
                warmup.append(step())
                snapshot.restore()
            monitor.mark("measured_steps")
            for _ in range(args.steps):
                rows.append(step())
                snapshot.restore()
            kernels = []
            if args.trace_steps:
                monitor.mark("trace")
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
                    for _ in range(args.trace_steps):
                        step()
                        snapshot.restore()
                trace.export_chrome_trace(str(out / "trace.json"))
                kernels = top_cuda_kernels(trace, 20)
        cycle = mean(r["step_seconds"] + r["full_kl_seconds"] + r["working_saves_seconds"] for r in rows)
        cold = sum(r["step_seconds"] + r["full_kl_seconds"] + r["working_saves_seconds"] for r in warmup)
        accepted_steps = min(
            training["max_accepted_steps"],
            training["max_epochs"] * (len(order) // training["positions_per_step"]),
        )
        worst_case_attempts = accepted_steps * (training["max_retries"] + 1)
        result = {
            "status": "complete",
            "discarded": True,
            "positions_per_step": len(workload["positions"]),
            "anchors_per_step": len(anchors),
            "control_anchors": len(dataset["anchors"]),
            "target_accepted_steps": accepted_steps,
            "preparation_seconds": preparation,
            "warmup": warmup,
            "measured": rows,
            "mean_cycle_seconds": cycle,
            "forecast_accepted_seconds": preparation + cold + accepted_steps * cycle,
            "forecast_every_retry_seconds": preparation + cold + worst_case_attempts * cycle,
            "budget_seconds": cfg["training_seconds"],
            "top_kernels": kernels,
            "note": "Fixed-shape discarded steps, two working saves per attempt. Forecast adds warmup conservatively; cleanup and actual KL rejection frequency remain uncertain.",
        }
        result["trace_archive"] = store_trace(out)
        write_json(out / "benchmark.json", result)
        print(result, flush=True)
        return result
