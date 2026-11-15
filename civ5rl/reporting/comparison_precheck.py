"""Short, discarded end-to-end evidence for rollout comparison preparation."""

from copy import deepcopy
from pathlib import Path
from time import perf_counter
import json
import torch
from civ5rl.engine.comparison_training import (
    Histories,
    configure_policy,
    prepare_references,
    logical_step,
    target_logits,
    full_kl,
    atomic_torch_save,
)
from civ5rl.engine.optimizer_transaction import AdamSnapshot
from civ5rl.provenance.common import configure, read_json, write_json, file_sha256
from civ5rl.provenance.rng import capture_rng, restore_rng
from civ5rl.experiments.imitation_helpers import assert_equal


def check_fit_gate(cfg, out, example, anchors):
    """Exercise rollback, terminal resume and publication recovery in isolation."""
    import tempfile, shutil
    from civ5rl.engine.comparison_training import fit

    with tempfile.TemporaryDirectory(prefix="comparison-smoke-") as temporary:
        tiny = Path(temporary)
        source = out / "sources" / str(example["episode"])
        target_folder = tiny / "sources" / str(example["episode"])
        target_folder.mkdir(parents=True)
        for name in ("case.json", "history.json"):
            shutil.copy2(source / name, target_folder / name)
        fit(
            cfg, tiny, dict(pairs=[example], anchors=anchors), {"synthetic_smoke": True}
        )
        actual_gate = read_json(tiny / "training.json")
        if (
            actual_gate["stop_reason"] != "kl_limit"
            or actual_gate["changed"]
            or actual_gate["training_steps"] != 0
        ):
            raise AssertionError(f"Full KL rollback failed: {actual_gate}")
        gate_before = deepcopy(actual_gate)
        fit(
            cfg, tiny, dict(pairs=[example], anchors=anchors), {"synthetic_smoke": True}
        )
        if read_json(tiny / "training.json") != gate_before:
            raise AssertionError("Terminal resume changed training")
        (tiny / "candidate.pt").unlink()
        (tiny / "training.json").unlink()
        fit(
            cfg, tiny, dict(pairs=[example], anchors=anchors), {"synthetic_smoke": True}
        )
        if read_json(tiny / "training.json")["training_steps"] != 0:
            raise AssertionError("Publication recovery reapplied training")
    return actual_gate


def precheck(cfg, out):
    from civ5rl.experiments.rollout_comparison import prepare, collect

    out = Path(out).resolve()
    configure(17, "cuda")
    prepare(cfg, out)
    collect(cfg, out, limit=2)
    data = read_json(out / "dataset.json")
    if data["completed_forks"] != 2:
        raise RuntimeError("Precheck needs two complete real forks")
    collection = read_json(out / "collection.json")
    from civ5rl.reporting.comparison_storage import restore_tree

    restore_tree(out, "sources")
    histories = Histories(out)
    # Time a complete BPTT window, rather than extrapolating six-decision forks.
    candidates = []
    for episode in collection["sources"]:
        _, episode_rows = histories.get(episode)
        candidates.extend(
            (episode, i) for i, r in enumerate(episode_rows) if len(r["actions"]) > 1
        )
    if not candidates:
        raise RuntimeError("Smoke requires a free training decision")
    episode, target = next(
        ((e, i) for e, i in candidates if i >= 127), max(candidates, key=lambda k: k[1])
    )
    history, rows = histories.get(episode)
    row = rows[target]
    # Explicitly artificial preference, never inserted into dataset.json/labels.
    example = dict(
        episode=episode,
        index=target,
        better=(row["chosen"] + 1) % len(row["actions"]),
        worse=row["chosen"],
        weight=0.5,
        kind="synthetic_smoke",
    )
    anchors = [(example["episode"], example["index"])]
    policy = configure_policy(cfg["parents"][0])
    started = perf_counter()
    references = prepare_references(
        policy, histories, anchors, out / "smoke-references-128.pt"
    )
    reference_seconds = perf_counter() - started
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad],
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-5,
    )

    def difference():
        with torch.no_grad():
            logits = target_logits(policy, history, example["index"])
            return float(logits[example["better"]] - logits[example["worse"]])

    before = difference()
    started = perf_counter()
    from civ5rl.engine.resources import ResourceMonitor

    with ResourceMonitor(out / "training-telemetry") as monitor:
        monitor.mark("comparative_step")
        metrics = logical_step(
            policy,
            optimizer,
            histories,
            [example] * 32,
            anchors * 64,
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
    saved = dict(
        model=deepcopy(policy.state_dict()),
        optimizer=deepcopy(optimizer.state_dict()),
        rng=capture_rng(),
        cursor=32,
        steps=1,
        synthetic_smoke=True,
    )
    atomic_torch_save(saved, out / "smoke-resume.pt")
    payload = torch.load(
        out / "smoke-resume.pt", weights_only=False, map_location="cpu"
    )
    # Reuse the compiled execution instance; deserialization restores all mutable
    # training state and does not require another copy of the parent model.
    transaction = AdamSnapshot(policy, optimizer)
    logical_step(
        policy, optimizer, histories, [example], anchors * 2, references["mixed"]
    )
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
        from civ5rl.engine.comparison_training import pair_loss

        logits = target_logits(policy, history, example["index"], gradient=True)
        pair_loss(logits, example["better"], example["worse"], 0.5).backward()
    profiler.export_chrome_trace(str(trace))
    from collections import Counter

    kernel_times = Counter()
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            kernel_times[event.name] += event.device_time_total
    kernels = [
        dict(name=name, cuda_us=duration)
        for name, duration in kernel_times.most_common(15)
    ]
    samples = [
        json.loads(line)
        for line in (out / "training-telemetry/resources.jsonl")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    samples = [
        s
        for s in samples
        if s.get("phase") == "comparative_step"
        and s.get("gpu_core_load_percent") is not None
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
    result = dict(
        status="passed",
        purpose="preparation_only",
        real_selected_forks=data["selected_forks"],
        real_completed_forks=data["completed_forks"],
        real_accepted_pairs=data["accepted_pairs"],
        real_rejections=data["rejections"],
        synthetic_smoke=True,
        logical_batch=dict(
            pairs=32,
            anchors=64,
            seconds=step_seconds,
            history_length=example["index"] + 1,
            bptt=min(128, example["index"] + 1),
            repeated_history=True,
        ),
        preference_before=before,
        preference_after=after,
        loss=metrics,
        full_control_kl=kl,
        would_accept_step=kl <= 0.015,
        control_seconds=kl_seconds,
        reference_seconds=reference_seconds,
        reference_timing=references["_timing"],
        save_resume_exact=True,
        adam_rng_rollback_exact=True,
        epoch_gate_rollback=actual_gate,
        step_reapplied_on_resume=False,
        maximum_64_steps_seconds=64 * step_seconds,
        training_budget_seconds=1800,
        forecast_limitation="Repeated real history: not representative if shorter than 128; excludes larger anchor set and diverse prefixes. No full run authorized.",
        top_kernels=kernels,
        nvml_mean=nvml,
        parents_unchanged=[
            file_sha256(p) == sha
            for p, sha in zip(cfg["parents"], cfg["parent_sha256"])
        ],
    )
    write_json(out / "precheck.json", result)
    (out / "report.md").write_text(
        "# Rollout comparison preparation\n\nPrepared, not launched. Two real forks, 32 verified continuation replays. Synthetic smoke trained only a disposable copy; never part of scientific data.\n\n```json\n"
        + json.dumps(result, indent=2)
        + "\n```\n",
        encoding="utf-8",
    )
    # Discard synthetic updated weights/Adam. Evidence preserves metrics, not a candidate.
    (out / "smoke-resume.pt").unlink()
    from civ5rl.deploy.archive import pack

    pack(
        out,
        out.parent / (out.name + "-packet"),
        token="comparison-precheck",
        direction="download",
        force=True,
    )
    print(json.dumps(result, indent=2), flush=True)
    return result
