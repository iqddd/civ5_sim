"""Bounded CPU/GPU phase measurements for the rollout-comparison protocol."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
from pathlib import Path
from collections import defaultdict, Counter
from copy import deepcopy
from time import perf_counter, process_time
import json
import statistics
import torch
from civ5rl.provenance.common import read_json, write_json, configure, file_sha256
from civ5rl.engine.resources import ResourceMonitor
from civ5rl.engine.evaluation import (
    EvaluationExecutor,
    EvaluationRuntime,
    evaluation_precision,
)
from civ5rl.experiments.comparison_collection import ProposalExecutor, match_task
from civ5rl.engine.comparison_training import (
    Histories,
    configure_policy,
    prepare_references,
    logical_step,
    target_logits,
    pair_loss,
    training_runtime,
)
from civ5rl.engine.optimizer_transaction import AdamSnapshot
from civ5rl.core.observation import ObservationEncoder
from civ5rl.deploy.archive import unpack


def measured_match(task):
    # Per-worker wall scopes include RPC waiting; process CPU is reported separately.
    import civ5rl.experiments.comparison_collection as module

    original = module.run_match
    scopes = []
    prefix = 0.0

    def run(*args, **kwargs):
        nonlocal prefix
        hook = kwargs.get("before_decision")

        def timed_hook(*args):
            nonlocal prefix
            start = perf_counter()
            result = hook(*args)
            prefix += perf_counter() - start
            return result

        if hook is not None:
            kwargs["before_decision"] = timed_hook
        return original(*args, **kwargs, metrics=scopes.append)

    module.run_match = run
    wall = perf_counter()
    cpu = process_time()
    try:
        result = match_task(task)
    finally:
        module.run_match = original
    grouped = defaultdict(float)
    for row in scopes:
        grouped[f"{row['kind']}/side{row['side']}"] += row["ns"] / 1e9
    write_json(
        Path(task[2]) / "measurement.json",
        dict(
            wall_seconds=perf_counter() - wall,
            cpu_seconds=process_time() - cpu,
            scopes=dict(grouped),
            prefix_hook_seconds=prefix,
            learner_side=task[1]["learner_side"],
        ),
    )
    return result


class ServiceMeter:
    def __init__(self, executor):
        self.executor = executor
        self.call = executor.inference.infer
        self.encode = ObservationEncoder.encode
        self.encode_seconds = 0.0
        self.wall = 0.0
        self.calls = 0

    def __enter__(self):
        def encode(agent, *args, **kwargs):
            start = perf_counter()
            result = self.encode(agent, *args, **kwargs)
            self.encode_seconds += perf_counter() - start
            return result

        def infer(requests):
            start = perf_counter()
            result = self.call(requests)
            self.wall += perf_counter() - start
            self.calls += 1
            return result

        ObservationEncoder.encode = encode
        self.executor.inference.infer = infer
        return self

    def __exit__(self, *args):
        ObservationEncoder.encode = self.encode
        self.executor.inference.infer = self.call

    def result(self):
        return dict(
            service_wall_seconds=self.wall,
            public_encoding_seconds=self.encode_seconds,
            calls=self.calls,
        )


def inference_trace(cfg, source, out):
    """Three saved ready batches, replayed on the profiling thread (no CP time)."""
    from civ5rl.core.agent import RLAgent
    from civ5rl.core.context import context_from_spec
    from civ5rl.experiments.comparison_collection import ProposalInference
    from civ5sim import Action

    folders = sorted((out / "matches/source").iterdir())[:8]
    with evaluation_precision():
        from civ5rl.engine.evaluation import prepare_evaluation_policy

        policies = [
            prepare_evaluation_policy(
                RLAgent.from_checkpoint(p, "cpu", "argmax").policy, "cuda"
            )
            for p in cfg["parents"]
        ]
        agents = []
        histories = []
        for folder in folders:
            case = read_json(folder / "case.json")
            members = []
            for policy in policies:
                member = RLAgent(policy, "cuda", "argmax")
                member.reset(
                    context_from_spec(
                        case, seed=case["agent_seeds"][case["learner_side"]]
                    )
                )
                members.append(member)
            members[0].members = members
            agents.append(members[0])
            histories.append(read_json(folder / "history.json"))
        backend = ProposalInference(policies, "cuda")

        def infer(index):
            requests = []
            for agent, history in zip(agents, histories):
                if index >= len(history):
                    continue
                row = history[index]
                actions = [Action(**a) for a in row["actions"]]
                requests.append(
                    (
                        agent,
                        "observe",
                        row["observation"],
                        actions,
                        actions[row["chosen"]],
                    )
                )
            backend.infer(requests)

        for index in range(20):
            infer(index)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            for index in range(20, 23):
                infer(index)
        profile.export_chrome_trace(str(out / "inference-trace.json"))


def trace_summary(path):
    events = read_json(path)["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("dur", 0) > 0]
    if not kernels:
        return {"error": "No CUDA kernels recorded"}
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
    end = intervals[0][0]
    busy = 0.0
    gaps = []
    for start, stop in intervals:
        if start > end:
            gaps.append(start - end)
            busy += stop - start
        elif stop > end:
            busy += stop - end
        end = max(end, stop)
    times = Counter()
    runtime = Counter()
    copies = {}
    for e in events:
        if e.get("cat") == "kernel":
            times[e["name"]] += e.get("dur", 0) / 1000
        elif "cuda_runtime" in e.get("cat", ""):
            runtime[e["name"]] += e.get("dur", 0) / 1000
        elif e.get("cat") == "gpu_memcpy":
            row = copies.setdefault(e["name"], dict(count=0, bytes=0, ms=0.0))
            row["count"] += 1
            row["bytes"] += e.get("args", {}).get("bytes", 0)
            row["ms"] += e.get("dur", 0) / 1000
    window = end - intervals[0][0]
    return dict(
        kernel_count=len(kernels),
        kernel_window_ms=window / 1000,
        kernel_union_ms=busy / 1000,
        kernel_busy_in_window_percent=100 * busy / window,
        gaps_over_1ms=sum(g > 1000 for g in gaps),
        largest_gap_ms=max(gaps, default=0) / 1000,
        top_kernels=[dict(name=n, ms=t) for n, t in times.most_common(12)],
        top_runtime=[dict(name=n, ms=t) for n, t in runtime.most_common(8)],
        copies=copies,
    )


def finish_report(out):
    result = read_json(out / "summary.json")
    traces = {
        name: trace_summary(out / (name + "-trace.json"))
        for name in ("training", "inference")
    }
    write_json(out / "trace-summary.json", traces)
    rows = [
        "# CPU/GPU rollout-comparison benchmark",
        "",
        "Local RTX4060Ti (165W), CUDA deterministic FP32 inference / mixed BF16 spatial training. No scientific training or evaluation; all measured Adam steps are discarded. Startup/warmup excluded.",
        "",
        "| Phase | Wall | GPU Core | Memory Controller | Power | CPU logical cores |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in [
        ("source_collection", "8 source matches,171+107"),
        ("continuations", "16 continuations,171"),
        ("training", "32 pairs+64 anchors per batch"),
    ]:
        r = result[key]
        m = r["nvml_cpu"]
        seconds = r.get("seconds", r.get("mean_seconds"))
        rows.append(
            f"| {label} | {seconds:.2f}s | {m['gpu_core_load_percent']['mean']:.1f}% | {m['gpu_memory_controller_load_percent']['mean']:.1f}% | {m['gpu_power_watts']['mean']:.1f}W | {m['cpu_tree_percent']['mean'] / 100:.2f} |"
        )
    rows += [
        "",
        "Memory Controller Load measures activity, not occupied VRAM or percent of peak GB/s. CPU cores are process-tree CPU percent/100, not hardware occupancy. Collection CP worker scopes and inference service overlap and must not be added to phase wall time. CP search is nested inside act.",
        "",
        "The source workload is dominated by central inference service wall time: "
        + str(result["source_collection"]["service"])
        + ". Continuations: "
        + str(result["continuations"]["service"])
        + ". Worker model-act scopes include RPC waiting; worker process CPU time separates actual CPU consumption from waiting.",
        "",
        f"Training uses {result['training']['unique_targets']} distinct target positions across eight episodes/maps10/12/16. {len(result['training']['samples'])} timed batches follow warmup; samples and NVML are in summary.json/telemetry. Timed training excludes reference preparation and full KL checks. Each batch starts from the same restored model/Adam.",
        "",
        "Kineto traces are short, separate profiled fragments and include profiler overhead: their GPU busy ratio is not the unprofiled phase utilization. Inspect top kernels, CPU launch gaps, DMA and synchronization times in trace-summary.json to attribute the remaining cost.",
        "",
        "Local system background CPU load varied; inspect cpu_system_percent before extrapolating to another machine. The measured GPU phases are shorter than10minutes and were not extended to meet utilization policy.",
        "",
        "The measurement uses the recorded compiled runtime, compact attention and batched independent histories. Logical batch, training budget and precision are unchanged; the execution strategy is recorded in identity.json.",
        "",
        "CPU-only cold/warm scenario generation of eight cases: "
        + str(read_json(out / "factory.json"))
        + ". This cost is separate from saved-case match timing.",
        "",
        "Raw match JSON and Kineto traces are stored in the sibling PPMd packet after verified packing; summary, telemetry and this report remain readable.",
    ]
    (out / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def archive_evidence(out):
    from civ5rl.deploy.archive import pack, _extract, _inventory
    import tempfile, shutil

    out = Path(out).resolve()
    packet = out.with_name(out.name + "-packet")
    manifest = pack(
        out, packet, token="comparison-performance", direction="download", force=True
    )
    with tempfile.TemporaryDirectory(
        prefix="verify-comparison-performance-"
    ) as temporary:
        _extract(packet, Path(temporary), manifest)
    if _inventory(out) != (manifest["files"], manifest["directories"]):
        raise RuntimeError("Measurement changed during packing")
    for name in ("matches", "training-trace.json", "inference-trace.json"):
        target = (out / name).resolve()
        if not target.is_relative_to(out) or target == out:
            raise ValueError("Unsafe evidence path")
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    write_json(
        out / "archive-receipt.json",
        dict(
            status="verified",
            packet=str(packet),
            files=len(manifest["files"]),
            bytes=sum(p.stat().st_size for p in packet.rglob("*") if p.is_file()),
        ),
    )


def run_benchmark(args, parser):
    from civ5rl.experiments.rollout_comparison import load_config

    cfg = load_config(args.config or Path("configs/rl-c2p-rollout-comparison.json"))
    if args.run is None:
        parser.error("comparison benchmark requires --run PRECHECK_DATA")
    if args.steps < 1 or args.warmup < 0:
        parser.error("Invalid step count")
    source = args.run.resolve()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Use an empty output")
    if out.is_relative_to(source) or source.is_relative_to(out):
        parser.error("Output must be separate from source")
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / "scratch"
    scratch.mkdir()
    for name in ("sources", "continuations"):
        if (source / (name + ".storage")).exists():
            unpack(
                source / (name + ".storage"),
                scratch / name,
                token="comparison-" + name,
                direction="download",
            )
        else:
            import shutil

            shutil.copytree(source / name, scratch / name)
    configure(17, "cuda")
    # CPU-only scenario generation is measured separately from saved-case matches.
    from civ5rl.protocols.ppo_target_scale import TargetFactory

    factory_config = deepcopy(read_json(source / "match-config.json"))
    factory_config["experiment"].update(
        start_episode=cfg["start_episode"],
        training_namespace=cfg["namespace"] + "_training",
    )
    base = read_json(source / "manifest.json")["base"]
    cold = TargetFactory(factory_config, base)
    generation = []
    for label in ("cold", "warm"):
        factory = cold if label == "cold" else TargetFactory(factory_config, base)
        if label == "warm":
            factory.maps = cold.maps
            factory.visibility_conflicts = cold.visibility_conflicts
        start = perf_counter()
        cpu = process_time()
        cases = [factory.make(cfg["start_episode"] + i) for i in range(8)]
        generation.append(
            dict(
                mode=label,
                cases=len(cases),
                wall_seconds=perf_counter() - start,
                cpu_seconds=process_time() - cpu,
            )
        )
    write_json(out / "factory.json", generation)
    import psutil, pynvml

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    hardware = dict(
        gpu=torch.cuda.get_device_name(),
        cpu=__import__("platform").processor(),
        logical_cpus=psutil.cpu_count(),
        gpu_limit_watts=pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000,
    )
    pynvml.nvmlShutdown()
    write_json(
        out / "identity.json",
        dict(
            hardware=hardware,
            config=cfg,
            parents=[file_sha256(p) for p in cfg["parents"]],
            training_runtime=training_runtime(),
            evaluation_runtime=EvaluationRuntime("cuda", 8, 32, 0.001).identity(),
            purpose="diagnostic only; discarded Adam steps",
        ),
    )
    config = read_json(source / "match-config.json")
    collection = read_json(source / "collection.json")
    episodes = collection["sources"][:8]
    runtime = EvaluationRuntime("cuda", 8, 32, 0.001)
    results = {}
    with (
        ResourceMonitor(out / "telemetry", interval=0.25) as monitor,
        evaluation_precision(),
    ):
        monitor.mark("source_startup")
        with ProposalExecutor(cfg["parents"], runtime) as executor:
            tasks = [
                (
                    config,
                    read_json(scratch / "sources" / str(e) / "case.json"),
                    str(out / "matches" / "warmup" / str(e)),
                    cfg["parents"][0],
                    None,
                )
                for e in episodes
            ]
            list(executor.run(measured_match, tasks))
            monitor.mark("source_collection")
            start = perf_counter()
            tasks = [
                (
                    config,
                    t[1],
                    str(out / "matches" / "source" / str(e)),
                    cfg["parents"][0],
                    None,
                )
                for e, t in zip(episodes, tasks)
            ]
            with ServiceMeter(executor) as service:
                list(executor.run(measured_match, tasks))
            results["source_collection"] = dict(
                seconds=perf_counter() - start,
                service=service.result(),
                matches=len(tasks),
            )
            monitor.mark("source_cleanup")
        monitor.mark("continuation_startup")
        fork = collection["forks"][0]
        case = read_json(scratch / "sources" / str(fork["episode"]) / "case.json")

        def continuation_tasks(label):
            tasks = []
            for j in range(8):
                for a in range(2):
                    branch = read_json(
                        scratch
                        / "continuations"
                        / str(fork["id"])
                        / f"{j}-{a}"
                        / "branch.json"
                    )
                    branch["source"] = str(
                        (scratch / "sources" / str(fork["episode"])).resolve()
                    )
                    tasks.append(
                        (
                            config,
                            case,
                            str(out / "matches" / label / f"{j}-{a}"),
                            cfg["parents"][0],
                            branch,
                        )
                    )
            return tasks

        with EvaluationExecutor(cfg["parents"][0], runtime) as executor:
            list(
                executor.run(measured_match, continuation_tasks("continuation-warmup"))
            )
            monitor.mark("continuations")
            start = perf_counter()
            with ServiceMeter(executor) as service:
                list(executor.run(measured_match, continuation_tasks("continuations")))
            results["continuations"] = dict(
                seconds=perf_counter() - start, service=service.result(), matches=16
            )
            monitor.mark("continuation_cleanup")
        monitor.mark("training_prepare")
        histories = Histories(scratch, capacity=8)
        pairs = []
        anchors = []
        for episode in episodes:
            _, rows = histories.get(episode)
            eligible = [
                i for i, r in enumerate(rows) if i >= 127 and len(r["actions"]) > 1
            ]
            if not eligible:
                eligible = [i for i, r in enumerate(rows) if len(r["actions"]) > 1]
            indices = [
                eligible[min(len(eligible) - 1, int(k * len(eligible) / 12))]
                for k in range(12)
            ]
            for index in indices[:4]:
                row = rows[index]
                pairs.append(
                    dict(
                        episode=episode,
                        index=index,
                        better=(row["chosen"] + 1) % len(row["actions"]),
                        worse=row["chosen"],
                        weight=0.5,
                    )
                )
            anchors.extend((episode, i) for i in indices[4:])
        policy = configure_policy(cfg["parents"][0])
        references = prepare_references(
            policy, histories, anchors, out / "references.pt"
        )
        optimizer = torch.optim.Adam(
            [p for p in policy.parameters() if p.requires_grad],
            lr=1e-5,
            betas=(0.9, 0.999),
            eps=1e-5,
        )
        initial = AdamSnapshot(policy, optimizer)
        timings = []
        for step in range(args.warmup + args.steps):
            initial.restore()
            monitor.mark(
                "training_warmup" if step < args.warmup else "training", update=step
            )
            start = perf_counter()
            metrics = logical_step(
                policy, optimizer, histories, pairs, anchors, references["mixed"]
            )
            seconds = perf_counter() - start
            print(f"Comparison batch {step + 1}: {seconds:.3f}s", flush=True)
            if step >= args.warmup:
                timings.append(dict(seconds=seconds, **metrics))
        initial.restore()
        monitor.mark("trace")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            optimizer.zero_grad(set_to_none=True)
            row = pairs[-1]
            history, _ = histories.get(row["episode"])
            logits = target_logits(policy, history, row["index"], gradient=True)
            pair_loss(logits, row["better"], row["worse"], 0.5).backward()
        profiler.export_chrome_trace(str(out / "training-trace.json"))
        initial.restore()
        results["training"] = dict(
            logical_pairs=len(pairs),
            logical_anchors=len(anchors),
            samples=timings,
            mean_seconds=statistics.mean(r["seconds"] for r in timings),
            median_seconds=statistics.median(r["seconds"] for r in timings),
            unique_targets=len(
                set((p["episode"], p["index"]) for p in pairs) | set(anchors)
            ),
            map_sizes=[
                read_json(scratch / "sources" / str(e) / "case.json")["map_size"]
                for e in episodes
            ],
            synthetic_preferences=True,
        )
        monitor.mark("finished")
    rows = [
        json.loads(line)
        for line in (out / "telemetry/resources.jsonl").read_text().splitlines()
    ]
    for phase in ("source_collection", "continuations", "training"):
        samples = [r for r in rows if r["phase"] == phase]
        names = (
            "gpu_core_load_percent",
            "gpu_memory_controller_load_percent",
            "gpu_power_watts",
            "gpu_power_percent_tdp",
            "cpu_tree_percent",
            "cpu_system_percent",
            "vram_used_bytes",
        )
        results[phase]["nvml_cpu"] = {
            name: dict(
                mean=statistics.mean(
                    r[name] for r in samples if r.get(name) is not None
                ),
                max=max(r[name] for r in samples if r.get(name) is not None),
            )
            for name in names
            if any(r.get(name) is not None for r in samples)
        }
        results[phase]["samples"] = (
            samples if phase != "training" else results[phase]["samples"]
        )
    for phase, label in [
        ("source_collection", "source"),
        ("continuations", "continuations"),
    ]:
        measurements = [
            read_json(p) for p in (out / "matches" / label).glob("*/measurement.json")
        ]
        cpu = defaultdict(float)
        for r in measurements:
            cpu["worker_cpu_seconds"] += r["cpu_seconds"]
            cpu["worker_wall_seconds"] += r["wall_seconds"]
            cpu["prefix_hook_seconds"] += r["prefix_hook_seconds"]
            side = r["learner_side"]
            for name, value in r["scopes"].items():
                kind, s = name.split("/")
                who = "model" if s == f"side{side}" else "cp"
                cpu[kind + "/" + who] += value
        results[phase]["worker_scopes"] = dict(cpu)
    results["training"]["parent_hashes_unchanged"] = [
        file_sha256(p) == expected
        for p, expected in zip(cfg["parents"], cfg["parent_sha256"])
    ]
    if not all(results["training"]["parent_hashes_unchanged"]):
        raise RuntimeError("Parent checkpoint changed")
    write_json(out / "summary.json", results)
    inference_trace(cfg, source, out)
    finish_report(out)
    # Scratch is a copy of already verified inputs; final evidence is separately packed.
    import shutil

    if scratch.resolve().is_relative_to(out):
        shutil.rmtree(scratch)
    archive_evidence(out)
    print(
        json.dumps(
            {
                k: {x: v for x, v in r.items() if x not in ("samples",)}
                for k, r in results.items()
            },
            indent=2,
        ),
        flush=True,
    )
    return results
