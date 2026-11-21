"""Short, discarded end-to-end evidence for rollout comparison preparation."""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import torch

from civ5rl.deploy.archive import pack
from civ5rl.engine.comparison.storage import restore_tree
from civ5rl.engine.comparison.training import (
    Histories,
    configure_policy,
    epoch_fit_options,
    fit,
    full_kl,
    logical_step,
    pair_loss,
    prepare_references,
    target_logits,
)
from civ5rl.engine.runtime.telemetry import ResourceMonitor
from civ5rl.engine.runtime.transactions import AdamSnapshot, atomic_torch_save
from civ5rl.experiments.checks.comparison_control import continuation_smoke, smoke
from civ5rl.experiments.comparison.rollout_fit import collect, prepare
from civ5rl.experiments.imitation.helpers import assert_equal
from civ5rl.provenance.common import configure, file_sha256, read_json, write_json
from civ5rl.provenance.rng import capture_rng, restore_rng
from civ5rl.reporting.comparison_benchmarks import write_precheck_report


def check_fit_gate(cfg, out, example, anchors):
    """Exercise rollback, terminal resume and publication recovery in isolation."""

    with tempfile.TemporaryDirectory(prefix="comparison-smoke-") as temporary:
        tiny = Path(temporary)
        source = out / "sources" / str(example["episode"])
        target_folder = tiny / "sources" / str(example["episode"])
        target_folder.mkdir(parents=True)
        for name in ("case.json", "history.json"):
            shutil.copy2(source / name, target_folder / name)
        # Deliberately cross the KL threshold on this disposable example.
        options = {
            **cfg,
            "training": {"max_epochs": 1, "pairs_per_step": 1, "initial_lr": 1e-4, "kl_limit": 1e-12},
        }
        dataset = {"pairs": [example], "anchors": anchors}
        identity = {"synthetic_smoke": True}
        fit(options, tiny, dataset, identity)
        actual_gate = read_json(tiny / "training.json")
        if (
            actual_gate["stop_reason"] != "kl_limit"
            or actual_gate["changed"]
            or actual_gate["training_steps"] != 0
        ):
            raise AssertionError(f"Full KL rollback failed: {actual_gate}")
        gate_before = deepcopy(actual_gate)
        fit(options, tiny, dataset, identity)
        if read_json(tiny / "training.json") != gate_before:
            raise AssertionError("Terminal resume changed training")
        (tiny / "candidate.pt").unlink()
        (tiny / "training.json").unlink()
        fit(options, tiny, dataset, identity)
        if read_json(tiny / "training.json")["training_steps"] != 0:
            raise AssertionError("Publication recovery reapplied training")
    return actual_gate


def precheck(cfg, out):
    if cfg.get("fit_continuation"):
        return continuation_smoke(cfg, out)
    if cfg.get("training", {}).get("mode") in ("full_corpus_step_kl", "frozen_heads_step_kl"):
        return smoke(cfg, out)

    out = Path(out).resolve()
    options = epoch_fit_options(cfg)
    logical_pairs = options["pairs_per_step"]
    frozen_collection = bool(cfg.get("collection_input"))
    configure(cfg["seed"], "cuda")
    prepare(cfg, out)
    if not frozen_collection:
        fork_count = min(2, cfg["max_forks"])
        collect(cfg, out, limit=fork_count)
    data = read_json(out / "dataset.json")
    if not frozen_collection and data["completed_forks"] != fork_count:
        raise RuntimeError(f"Precheck needs {fork_count} complete real forks")
    collection = read_json(out / "collection.json")

    restore_tree(out, "sources")
    histories = Histories(out)
    # Time a complete BPTT window, rather than extrapolating six-decision forks.
    candidates = []
    for episode in collection["sources"]:
        episode_rows = histories.raw(episode)
        candidates.extend((episode, i) for i, r in enumerate(episode_rows) if len(r["actions"]) > 1)
    if not candidates:
        raise RuntimeError("Smoke requires a free training decision")
    episode, target = next(((e, i) for e, i in candidates if i >= 127), max(candidates, key=lambda k: k[1]))
    history, _ = histories.get(episode)
    rows = histories.raw(episode)
    row = rows[target]
    # Explicitly artificial preference, never inserted into dataset.json/labels.
    example = {
        "episode": episode,
        "index": target,
        "better": (row["chosen"] + 1) % len(row["actions"]),
        "worse": row["chosen"],
        "weight": 0.5,
        "kind": "synthetic_smoke",
    }
    anchors = [(example["episode"], example["index"])]
    policy = configure_policy(cfg["parents"][0])
    started = perf_counter()
    references = prepare_references(policy, histories, anchors, out / "smoke-references-128.pt")
    reference_seconds = perf_counter() - started
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad],
        lr=options["initial_lr"],
        betas=(0.9, 0.999),
        eps=1e-5,
    )

    def difference():
        with torch.no_grad():
            logits = target_logits(policy, history, example["index"])
            return float(logits[example["better"]] - logits[example["worse"]])

    before = difference()
    started = perf_counter()

    with ResourceMonitor(out / "training-telemetry") as monitor:
        monitor.mark("comparative_step")
        metrics = logical_step(
            policy,
            optimizer,
            histories,
            [example] * logical_pairs,
            anchors * (2 * logical_pairs),
            references["mixed"],
        )
    step_seconds = perf_counter() - started
    after = difference()
    if not after > before:
        raise AssertionError("Synthetic preference did not increase")
    steps = {int(s["step"]) for s in optimizer.state.values()}
    if steps != {1}:
        raise AssertionError("Adam did not take exactly one step")
    started = perf_counter()
    kl = full_kl(policy, histories, anchors, references["fp32"])
    kl_seconds = perf_counter() - started
    if not torch.isfinite(torch.tensor(kl)) or kl < -1e-7:
        raise AssertionError(f"Unexpected smoke KL {kl}")
    saved = {
        "model": deepcopy(policy.state_dict()),
        "optimizer": deepcopy(optimizer.state_dict()),
        "rng": capture_rng(),
        "cursor": logical_pairs,
        "steps": 1,
        "synthetic_smoke": True,
    }
    atomic_torch_save(saved, out / "smoke-resume.pt")
    payload = torch.load(out / "smoke-resume.pt", weights_only=False, map_location="cpu")
    # Reuse the compiled execution instance; deserialization restores all mutable
    # training state and does not require another copy of the parent model.
    transaction = AdamSnapshot(policy, optimizer)
    logical_step(policy, optimizer, histories, [example], anchors * 2, references["mixed"])
    transaction.restore()
    assert_equal(policy.state_dict(), payload["model"])
    assert_equal(optimizer.state_dict(), payload["optimizer"])
    assert_equal(capture_rng(), payload["rng"])
    policy.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"])
    assert_equal(policy.state_dict(), payload["model"])
    assert_equal(optimizer.state_dict(), payload["optimizer"])
    assert_equal(capture_rng(), payload["rng"])
    actual_gate = check_fit_gate(cfg, out, example, anchors)
    # A short real kernel trace, outside the timed logical step. No optimizer change.
    trace = out / "training-trace.json"
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        optimizer.zero_grad(set_to_none=True)

        logits = target_logits(policy, history, example["index"], gradient=True)
        pair_loss(logits, example["better"], example["worse"], 0.5).backward()
    profiler.export_chrome_trace(str(trace))

    kernel_times = Counter()
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernel_times[event.name] += event.device_time_total
    kernels = [{"name": name, "cuda_us": duration} for name, duration in kernel_times.most_common(15)]
    samples = [
        json.loads(line)
        for line in (out / "training-telemetry/resources.jsonl").read_text().splitlines()
        if line.strip()
    ]
    samples = [
        s
        for s in samples
        if s.get("phase") == "comparative_step" and s.get("gpu_core_load_percent") is not None
    ]
    nvml = {
        name: sum(s[name] for s in samples if s.get(name) is not None)
        / sum(s.get(name) is not None for s in samples)
        for name in (
            "gpu_core_load_percent",
            "gpu_power_watts",
            "gpu_power_percent_tdp",
        )
        if any(s.get(name) is not None for s in samples)
    }
    result = {
        "status": "passed",
        "purpose": "preparation_only",
        "reused_frozen_collection": frozen_collection,
        "real_selected_forks": data["selected_forks"],
        "real_completed_forks": data["completed_forks"],
        "real_accepted_pairs": data["accepted_pairs"],
        "real_rejections": data["rejections"],
        "synthetic_smoke": True,
        "logical_batch": {
            "pairs": logical_pairs,
            "anchors": 2 * logical_pairs,
            "seconds": step_seconds,
            "history_length": example["index"] + 1,
            "bptt": min(128, example["index"] + 1),
            "repeated_history": True,
        },
        "preference_before": before,
        "preference_after": after,
        "loss": metrics,
        "full_control_kl": kl,
        "would_accept_step": kl <= options["kl_limit"],
        "control_seconds": kl_seconds,
        "reference_seconds": reference_seconds,
        "reference_timing": references["_timing"],
        "save_resume_exact": True,
        "adam_rng_rollback_exact": True,
        "epoch_gate_rollback": actual_gate,
        "step_reapplied_on_resume": False,
        "maximum_planned_step_seconds": options["max_epochs"]
        * math.ceil((len(data["pairs"]) if frozen_collection else cfg["max_forks"]) / logical_pairs)
        * step_seconds,
        "training_budget_seconds": cfg["training_seconds"],
        "forecast_limitation": "Repeated real history: not representative if shorter than 128; excludes larger anchor set and diverse prefixes. No full run authorized.",
        "top_kernels": kernels,
        "nvml_mean": nvml,
        "parents_unchanged": [file_sha256(p) == sha for p, sha in zip(cfg["parents"], cfg["parent_sha256"])],
    }
    write_json(out / "precheck.json", result)
    write_precheck_report(out)
    # Discard synthetic updated weights/Adam. Evidence preserves metrics, not a candidate.
    (out / "smoke-resume.pt").unlink()

    pack(
        out,
        out.parent / (out.name + "-packet"),
        token="comparison-precheck",
        direction="download",
        force=True,
    )
    print(json.dumps(result, indent=2), flush=True)
    return result
