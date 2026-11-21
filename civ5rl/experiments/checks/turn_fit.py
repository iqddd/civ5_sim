"""Discardable FP32 correctness smoke on real turn histories, not a timing forecast."""

from __future__ import annotations

import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

from civ5rl.engine.comparison.turn_fit import TurnTrainingState, configure_policy, parent_references
from civ5rl.engine.comparison.turn_inputs import TurnHistories
from civ5rl.engine.comparison.turn_training import (
    control_kl,
    current_chosen,
    logical_step,
    pack_batch,
    preference_loss,
    prepare_anchor_subset,
    prepare_preferences,
)
from civ5rl.engine.runtime.telemetry import ResourceMonitor
from civ5rl.engine.runtime.transactions import AdamSnapshot, atomic_torch_save, cpu_snapshot
from civ5rl.experiments.benchmarks.profiling import store_trace, top_cuda_kernels
from civ5rl.experiments.comparison.turn_data import build_dataset, history_inputs, read_json_lines
from civ5rl.provenance.common import ROOT, read_json, write_json
from civ5rl.provenance.rng import capture_rng, restore_rng


def smoke(cfg, out, *, microbatch=2, compile_model=True):
    out = Path(out)
    if any(p.name not in ("run.lock", "derived") for p in out.iterdir()):
        raise ValueError("Use a fresh smoke directory")
    started = perf_counter()
    manifest = read_json(ROOT / cfg["screening_run"] / "positions.json")
    options = cfg["training"]
    positions_per_step = options["positions_per_step"]
    steps_per_epoch = len(manifest["positions"]) // positions_per_step
    if steps_per_epoch < 1:
        raise ValueError("training.positions_per_step exceeds the number of saved positions")
    review = {tuple(p["pos"]): p for p in read_json_lines(ROOT / cfg["review"])}
    selected = {
        tuple(p["position_key"])
        for p in read_json(ROOT / cfg["screening_run"] / "derived/turn-review/candidates.json")["rows"]
    }
    eligible = []
    for i, p in enumerate(manifest["positions"]):
        row = review[tuple(p["key"])]
        if len({r["wdl"] == "W" for r in row["rows"]}) < 2:
            continue
        cost = (p["history_index"] + 8) * sum(r["free"] for r in row["rows"])
        eligible.append((cost, i, tuple(p["key"]) in selected))
    ids = [
        min(x for x in eligible if x[2])[1],
        min(x for x in eligible if not x[2] and manifest["positions"][x[1]]["history_index"] > 0)[1],
    ]
    dataset = build_dataset(cfg, position_indices=ids, anchor_limit=32)
    write_json(out / "dataset.json", dataset)
    with history_inputs(cfg, dataset) as (sources, roots, branches):
        histories = TurnHistories(out, sources, roots, branches, dataset["anchors"])
        preferences = prepare_preferences(histories, dataset["positions"])
        chosen_keys = sorted(histories.targets)
        anchor_subset = prepare_anchor_subset(
            histories, dataset["anchors"], preferences.training_observations
        )
        histories.prepare([*chosen_keys, *map(tuple, dataset["anchors"])])
        batch = pack_batch(
            preferences.rows, "cuda", histories=histories, branch_steps=preferences.branch_steps
        )
        policy = configure_policy(str(ROOT / cfg["parents"][0]), compile_model=compile_model)
        optimizer = torch.optim.Adam(
            [p for p in policy.parameters() if p.requires_grad],
            lr=options["initial_lr"],
            betas=(0.9, 0.999),
            eps=1e-5,
            weight_decay=0,
        )
        parent, training_refs, control_refs = parent_references(
            policy, histories, dataset, batch.keys, out, cfg, branch_steps=preferences.branch_steps
        )
        reference = torch.stack([parent[k] for k in batch.keys])
        before = current_chosen(policy, histories, batch.keys, branch_steps=preferences.branch_steps)
        baseline = float(
            preference_loss(before, reference, batch.indices, batch.mask, batch.signs, batch.weights)
        )
        initial_kl, _ = control_kl(
            policy, histories, dataset["anchors"], control_refs, anchor_subset=anchor_subset
        )
        assert abs(initial_kl) < 1e-6 and float((before - reference).abs().max()) < 1e-5
        value = {k: v.clone() for k, v in policy.value.state_dict().items()}
        snapshot = AdamSnapshot(policy, optimizer)

        with ResourceMonitor(out / "telemetry") as monitor:
            monitor.mark("smoke_optimizer")
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
                metrics = logical_step(
                    policy,
                    optimizer,
                    histories,
                    batch,
                    dataset["anchors"],
                    reference,
                    training_refs,
                    microbatch=microbatch,
                )
            trace.export_chrome_trace(str(out / "trace.json"))
        after = current_chosen(policy, histories, batch.keys, branch_steps=preferences.branch_steps)
        final = float(
            preference_loss(after, reference, batch.indices, batch.mask, batch.signs, batch.weights)
        )
        scores = ((after - reference)[batch.indices] * batch.mask).sum(-1) / batch.mask.sum(-1).clamp_min(1)
        margins = batch.signs * (scores[:, 0] - scores[:, 1])
        final_kl, learning = control_kl(
            policy, histories, dataset["anchors"], control_refs, anchor_subset=anchor_subset
        )
        assert final < baseline and bool((margins > 0).any()), (
            "Real winning executions must gain relative preference"
        )
        assert all(torch.equal(v, policy.value.state_dict()[k]) for k, v in value.items())

        resumed = TurnTrainingState(
            steps=1,
            epoch=1 // steps_per_epoch,
            cursor=1 % steps_per_epoch * positions_per_step,
            mode="turn_preferences_step_kl",
            lr=options["initial_lr"],
            metadata={"steps_per_epoch": steps_per_epoch, "positions_per_step": positions_per_step},
        )
        accepted = cpu_snapshot(
            {
                **resumed.to_checkpoint_dict(),
                "model": policy.state_dict(),
                "optimizer": optimizer.state_dict(),
                "rng": capture_rng(),
            }
        )
        atomic_torch_save(accepted, out / "smoke-resume.pt")

        random.random()
        np.random.random()
        torch.rand(())
        torch.rand((), device="cuda")
        snapshot.restore()
        assert not optimizer.state
        assert all(torch.equal(current, saved) for current, saved in snapshot.model)
        restored = capture_rng()
        assert restored["python_rng"] == snapshot.rng["python_rng"]
        assert np.array_equal(restored["numpy_rng"][1], snapshot.rng["numpy_rng"][1])
        assert torch.equal(restored["torch_rng"], snapshot.rng["torch_rng"])
        assert all(torch.equal(a, b) for a, b in zip(restored["cuda_rng"], snapshot.rng["cuda_rng"]))
        loaded = torch.load(out / "smoke-resume.pt", map_location="cpu", weights_only=False)
        policy.load_state_dict(loaded["model"])
        optimizer.load_state_dict(loaded["optimizer"])
        restore_rng(loaded["rng"])
        state = TurnTrainingState.from_checkpoint_dict(loaded)
        assert (state.steps, state.epoch, state.cursor) == (resumed.steps, resumed.epoch, resumed.cursor)
        assert all(int(s["step"]) == 1 for s in optimizer.state.values())
        assert all(torch.equal(v.cpu(), loaded["model"][k]) for k, v in policy.state_dict().items())
        result = {
            "status": "passed",
            "discarded": True,
            "positions": ids,
            "dataset": dataset["counts"],
            "precision": "deterministic_cuda_fp32_no_amp_no_tf32",
            "initial_full_kl": initial_kl,
            "final_full_kl": final_kl,
            "pair_loss_before": baseline,
            "pair_loss_after": final,
            "winning_margins_positive": int((margins > 0).sum()),
            "value_weights_unchanged": True,
            "rollback_restored": True,
            "rng_restored": True,
            "checkpoint_kept_cursor": True,
            "loaded_adam_steps": 1,
            "nonempty_prefix_tested": True,
            "seconds": perf_counter() - started,
            "metrics": metrics,
            "learning": learning,
            "top_kernels": top_cuda_kernels(trace),
            "performance_note": "Correctness smoke; shared-device NVML can include a concurrently running game. No full-fit ETA inferred.",
        }
        result["trace_archive"] = store_trace(out)
        write_json(out / "precheck.json", result)
        print(result, flush=True)
        return result
