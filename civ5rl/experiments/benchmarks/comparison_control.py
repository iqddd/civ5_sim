"""Bounded, disposable timing of full-corpus comparison attempts."""

from __future__ import annotations

import gc
import json
import shutil
import statistics
import tempfile
from pathlib import Path
from time import perf_counter

import torch
from torch.profiler import ProfilerActivity, profile

from civ5rl.engine.comparison.control import initial_state
from civ5rl.engine.comparison.heads import HeadFeatures, configure_heads
from civ5rl.engine.comparison.storage import restore_tree
from civ5rl.engine.comparison.training import (
    Histories,
    configure_policy,
    full_kl,
    logical_step,
    prepare_references,
)
from civ5rl.engine.runtime.telemetry import ResourceMonitor
from civ5rl.engine.runtime.transactions import AdamSnapshot, atomic_torch_save
from civ5rl.experiments.comparison.rollout_fit import load_config
from civ5rl.provenance.checkpoint import verify_frozen_passport
from civ5rl.provenance.common import ROOT, configure, file_sha256, read_json, seed_for, write_json
from civ5rl.provenance.rng import capture_rng


def benchmark(args, parser):
    """Measure disposable full work separately from the small correctness smoke."""

    if args.config is None:
        parser.error("Full-corpus comparison requires --config")
    cfg = load_config(args.config)
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Use an empty benchmark output")
    if args.steps < 1 or args.warmup < 1:
        parser.error("Require warmup and measured steps")
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / cfg["collection_input"]["run"]
    for name, sha in cfg["collection_input"]["sha256"].items():
        if file_sha256(source / name) != sha:
            raise ValueError(f"Collection input changed: {name}")
    verify_frozen_passport(source)
    data = read_json(source / "dataset.json")
    options = cfg["training"]
    started = perf_counter()
    configure(cfg["seed"], "cuda")
    with tempfile.TemporaryDirectory(prefix="comparison-full-bench-", dir=ROOT / ".tmp") as tmp:
        scratch = Path(tmp)
        shutil.copytree(source / "sources.storage", scratch / "sources.storage")
        restore_tree(scratch, "sources")
        head_only = options["mode"] == "frozen_heads_step_kl"
        if head_only:
            policy = configure_heads(cfg["parents"][0])
        else:
            policy = configure_policy(cfg["parents"][0])
        optimizer = torch.optim.Adam(
            [p for p in policy.parameters() if p.requires_grad],
            lr=options["initial_lr"],
            betas=(0.9, 0.999),
            eps=1e-5,
            weight_decay=0,
        )
        keys = [(p["episode"], p["index"]) for p in data["pairs"]] + [tuple(k) for k in data["anchors"]]
        limits = {}
        for episode, index in keys:
            limits[episode] = max(limits.get(episode, 0), index + 1)
        histories = Histories(scratch, source_hashes=data["source_hashes"], limits=limits)
        histories.prepare(keys, device="cuda")
        if head_only:
            features = HeadFeatures(policy, histories, data)
            features.prepare(cfg["parents"][0])
        else:
            refs = prepare_references(policy, histories, data["anchors"], scratch / "references.pt")
        state = initial_state(
            data["anchors"], options, seed_for(cfg["seed"], 0, "anchors", namespace=cfg["namespace"])
        )
        anchors = [data["anchors"][i] for i in state.anchor_order[: options["anchors_per_step"]]]
        cold = perf_counter() - started
        snapshot = AdamSnapshot(policy, optimizer)
        rows = []
        with ResourceMonitor(out / "telemetry") as monitor:
            for index in range(args.warmup + args.steps):
                snapshot.restore()
                monitor.mark("warmup" if index < args.warmup else "full_corpus_attempt")
                begin = perf_counter()
                if head_only:
                    features.update(optimizer, data["pairs"], anchors)
                else:
                    logical_step(policy, optimizer, histories, data["pairs"], anchors, refs["mixed"])
                step = perf_counter() - begin
                begin = perf_counter()
                if head_only:
                    kl, learning = features.measure()
                else:
                    kl = full_kl(policy, histories, data["anchors"], refs["fp32"])
                control = perf_counter() - begin
                begin = perf_counter()
                atomic_torch_save(
                    {"model": policy.state_dict(), "optimizer": optimizer.state_dict(), "rng": capture_rng()},
                    scratch / "discarded.pt",
                )
                save = perf_counter() - begin
                begin = perf_counter()
                snapshot.restore()
                rollback = perf_counter() - begin
                rows.append(
                    {
                        "warmup": index < args.warmup,
                        "step_seconds": step,
                        "kl_seconds": control,
                        "save_seconds": save,
                        "rollback_seconds": rollback,
                        "full_kl": kl,
                    }
                )
        if head_only:
            snapshot.restore()

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
                features.update(optimizer, data["pairs"], anchors)
            trace.export_chrome_trace(str(out / "heads-trace.json"))
            kernels = sorted(
                (
                    {"name": e.key, "cuda_us": e.device_time_total, "count": e.count}
                    for e in trace.key_averages()
                    if e.device_time_total
                ),
                key=lambda e: e["cuda_us"],
                reverse=True,
            )[:20]
            write_json(out / "top-kernels.json", kernels)
            snapshot.restore()
        measured = [r for r in rows if not r["warmup"]]
        cycle = statistics.mean(r["step_seconds"] + r["kl_seconds"] + 2 * r["save_seconds"] for r in measured)
        warmup = sum(
            r["step_seconds"] + r["kl_seconds"] + r["save_seconds"] + r["rollback_seconds"]
            for r in rows
            if r["warmup"]
        )
        accepted_steps = options["max_accepted_steps"]
        attempt_limit = accepted_steps * (1 + options["max_retries"])
        estimate = cold + warmup + accepted_steps * cycle
        result = {
            "discarded": True,
            "pairs": len(data["pairs"]),
            "training_anchors": len(anchors),
            "control_anchors": len(data["anchors"]),
            "cold_preparation_seconds": cold,
            "warmup_seconds": warmup,
            "measurements": rows,
            "planned_accepted_steps": accepted_steps,
            "maximum_attempts": attempt_limit,
            "estimated_planned_steps_seconds": estimate,
            "estimated_maximum_attempts_seconds": cold
            + warmup
            + attempt_limit * (cycle + statistics.mean(r["rollback_seconds"] for r in measured)),
            "training_seconds": cfg["training_seconds"],
            "fits_nominal": estimate <= cfg["training_seconds"],
            "limitation": "Measured batch and hardware only; retries and final packing can add time.",
        }
        if head_only:
            result.update(
                feature_cache=features.statistics,
                feature_parity=features.parity,
                learning_after_discarded_step=learning,
                trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad),
            )
        del histories

        gc.collect()
    write_json(out / "summary.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result
