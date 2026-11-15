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

    # This replacement is process-wide within the worker and is always restored.
    original_run_match = module.run_match
    original_verify = module._verify
    original_encode = ObservationEncoder.encode
    _, case, match_output, _, _ = task
    learner_side = case["learner_side"]
    scopes = []
    prefix_hook_seconds = 0.0
    verify_seconds = encoding_seconds = 0.0

    def measured_verify(*args,**kwargs):
        nonlocal verify_seconds
        start=perf_counter()
        try:return original_verify(*args,**kwargs)
        finally:verify_seconds+=perf_counter()-start

    def measured_encode(*args,**kwargs):
        nonlocal encoding_seconds
        start=perf_counter()
        try:return original_encode(*args,**kwargs)
        finally:encoding_seconds+=perf_counter()-start

    def run_match_with_measurements(*args, **kwargs):
        nonlocal prefix_hook_seconds
        hook = kwargs.get("before_decision")

        def timed_hook(*args):
            nonlocal prefix_hook_seconds
            start = perf_counter()
            result = hook(*args)
            prefix_hook_seconds += perf_counter() - start
            return result

        if hook is not None:
            kwargs["before_decision"] = timed_hook
        return original_run_match(*args, **kwargs, metrics=scopes.append)

    module.run_match = run_match_with_measurements
    module._verify = measured_verify
    ObservationEncoder.encode = measured_encode
    wall_started = perf_counter()
    cpu_started = process_time()
    try:
        result = match_task(task)
    finally:
        module.run_match = original_run_match
        module._verify = original_verify
        ObservationEncoder.encode = original_encode
    grouped = defaultdict(float)
    for row in scopes:
        grouped[f"{row['kind']}/side{row['side']}"] += row["ns"] / 1e9
    write_json(
        Path(match_output) / "measurement.json",
        dict(
            wall_seconds=perf_counter() - wall_started,
            cpu_seconds=process_time() - cpu_started,
            scopes=dict(grouped),
            prefix_hook_seconds=prefix_hook_seconds,
            learner_side=learner_side,
            replay_verification_seconds=verify_seconds,
            public_encoding_seconds=encoding_seconds,
        ),
    )
    return result


class ServiceMeter:
    def __init__(self, executor):
        self.executor = executor
        self.call = executor.inference.infer_encoded
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
        self.executor.inference.infer_encoded = infer
        return self

    def __exit__(self, *args):
        ObservationEncoder.encode = self.encode
        self.executor.inference.infer_encoded = self.call

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

        def prepare_requests(index):
            requests = []
            for agent, history in zip(agents, histories):
                if index >= len(history):
                    continue
                row = history[index]
                actions = [Action(**a) for a in row["actions"]]
                encoded=agent.encoder.encode(row['observation'],actions)
                agent.encoder.remember_action(actions[row['chosen']],row['observation'])
                requests.append((agent,'observe',encoded))
            return requests

        for index in range(20):
            backend.infer_encoded(prepare_requests(index))
        ready=[prepare_requests(index) for index in range(20,23)]
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            for requests in ready:
                backend.infer_encoded(requests)
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


TELEMETRY_FIELDS = (
    "gpu_core_load_percent",
    "gpu_memory_controller_load_percent",
    "gpu_power_watts",
    "gpu_power_percent_tdp",
    "cpu_tree_percent",
    "cpu_system_percent",
    "vram_used_bytes",
)


def aggregate_telemetry(rows, phase):
    """Return phase samples and measured means/maxima; missing values stay unknown."""
    samples = [row for row in rows if row.get("phase") == phase]
    aggregates = {}
    for name in TELEMETRY_FIELDS:
        values = [row[name] for row in samples if row.get(name) is not None]
        aggregates[name] = (
            {"mean": statistics.mean(values), "max": max(values)}
            if values else {"mean": None, "max": None}
        )
    return samples, aggregates


def format_measurement(aggregates, name, *, scale=1, suffix="", precision=1):
    value = aggregates.get(name, {}).get("mean")
    return "n/a" if value is None else f"{value / scale:.{precision}f}{suffix}"


def finish_report(out):
    result = read_json(out / "summary.json")
    hardware = read_json(out / "identity.json")["hardware"]
    factory_results = read_json(out / "factory.json")
    map_sizes = result["training"]["map_sizes"]
    traces = {
        name: trace_summary(out / (name + "-trace.json"))
        for name in ("training", "inference")
    }
    write_json(out / "trace-summary.json", traces)
    rows = [
        "# CPU/GPU rollout-comparison benchmark",
        "",
        f"Local {hardware['gpu']} ({hardware['gpu_limit_watts']:g}W limit), CUDA deterministic FP32 inference / mixed BF16 spatial training. No scientific training or evaluation; all measured Adam steps are discarded. Startup/warmup excluded.",
        "",
        "| Phase | Wall | GPU Core | Memory Controller | Power | CPU logical cores |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in [
        (
            "source_collection",
            f"{result['source_collection']['matches']} source matches",
        ),
        ("continuations", f"{result['continuations']['matches']} continuations"),
        (
            "training",
            f"{result['training']['logical_pairs']} pairs+{result['training']['logical_anchors']} anchors per batch",
        ),
    ]:
        r = result[key]
        m = r["nvml_cpu"]
        seconds = r.get("seconds", r.get("mean_seconds"))
        rows.append(
            f"| {label} | {seconds:.2f}s | {format_measurement(m, 'gpu_core_load_percent', suffix='%')} | {format_measurement(m, 'gpu_memory_controller_load_percent', suffix='%')} | {format_measurement(m, 'gpu_power_watts', suffix='W')} | {format_measurement(m, 'cpu_tree_percent', scale=100, precision=2)} |"
        )
    rows += [
        "",
        "Memory Controller Load measures activity, not occupied VRAM or percent of peak GB/s. CPU cores are process-tree CPU percent/100, not hardware occupancy. Collection CP worker scopes and inference service overlap and must not be added to phase wall time. CP search is nested inside act.",
        "",
        "Source inference service measurements: "
        + str(result["source_collection"]["service"])
        + ". Continuations: "
        + str(result["continuations"]["service"])
        + ". Worker model-act scopes include RPC waiting; worker process CPU time separates actual CPU consumption from waiting.",
        "",
        f"Training uses {result['training']['unique_targets']} distinct target positions across {len(map_sizes)} episodes; map sizes: {', '.join(str(size) for size in sorted(set(map_sizes)))}. {len(result['training']['step_samples'])} timed batches follow warmup; samples and NVML are in summary.json/telemetry. Timed training excludes reference preparation and full KL checks. Each batch starts from the same restored model/Adam.",
        "",
        "Kineto traces are short, separate profiled fragments and include profiler overhead: their GPU busy ratio is not the unprofiled phase utilization. Inspect top kernels, CPU launch gaps, DMA and synchronization times in trace-summary.json to attribute the remaining cost.",
        "",
        "Inspect cpu_system_percent before extrapolating to another machine. Assess each iterative GPU phase against the utilization policy using its measured duration.",
        "",
        "The measurement uses the recorded compiled runtime, compact attention and batched independent histories. Logical batch, training budget and precision are unchanged; the execution strategy is recorded in identity.json.",
        "",
        "CPU-only cold/warm scenario generation: "
        + str(factory_results)
        + ". This cost is separate from saved-case match timing.",
        "",
        "Raw match JSON and Kineto traces are stored in the sibling PPMd packet after verified packing; summary, telemetry and this report remain readable.",
    ]
    (out / "report.md").write_text("\n".join(rows) + "\n", encoding="utf-8")


def archive_evidence(out):
    from civ5rl.reporting.storage import pack_verified
    import shutil

    out = Path(out).resolve()
    packet = out.with_name(out.name + "-packet")
    manifest = pack_verified(
        out,
        packet,
        token="comparison-performance",
        temporary_prefix="verify-comparison-performance-",
        changed_message="Measurement changed during packing",
    )
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


def measure_source_collection(cfg, config, episodes, runtime, scratch, out, monitor):
    monitor.mark("source_startup")
    with ProposalExecutor(cfg["parents"], runtime) as executor:
        episode_cases = [
            (episode, read_json(scratch / "sources" / str(episode) / "case.json"))
            for episode in episodes
        ]
        warmup_tasks = [
            (
                config,
                case,
                str(out / "matches" / "warmup" / str(episode)),
                cfg["parents"][0],
                None,
            )
            for episode, case in episode_cases
        ]
        list(executor.run(measured_match, warmup_tasks))
        tasks = [
            (
                config,
                case,
                str(out / "matches" / "source" / str(episode)),
                cfg["parents"][0],
                None,
            )
            for episode, case in episode_cases
        ]
        monitor.mark("source_collection")
        start = perf_counter()
        with ServiceMeter(executor) as service:
            list(executor.run(measured_match, tasks))
        result = dict(
            seconds=perf_counter() - start,
            service=service.result(),
            matches=len(tasks),
        )
        monitor.mark("source_cleanup")
    return result


def measure_continuations(cfg, config, collection, runtime, scratch, out, monitor):
    monitor.mark("continuation_startup")
    fork = collection["forks"][0]
    case = read_json(scratch / "sources" / str(fork["episode"]) / "case.json")

    def continuation_tasks(label):
        tasks = []
        for repetition in range(8):
            for action_index in range(2):
                branch = read_json(
                    scratch
                    / "continuations"
                    / str(fork["id"])
                    / f"{repetition}-{action_index}"
                    / "branch.json"
                )
                branch["source"] = str(
                    (scratch / "sources" / str(fork["episode"])).resolve()
                )
                tasks.append(
                    (
                        config,
                        case,
                        str(out / "matches" / label / f"{repetition}-{action_index}"),
                        cfg["parents"][0],
                        branch,
                    )
                )
        return tasks

    with EvaluationExecutor(cfg["parents"][0], runtime) as executor:
        warmup_tasks = continuation_tasks("continuation-warmup")
        tasks = continuation_tasks("continuations")
        list(executor.run(measured_match, warmup_tasks))
        monitor.mark("continuations")
        start = perf_counter()
        with ServiceMeter(executor) as service:
            list(executor.run(measured_match, tasks))
        result = dict(
            seconds=perf_counter() - start, service=service.result(), matches=len(tasks)
        )
        monitor.mark("continuation_cleanup")
    return result


def measure_training(cfg, episodes, scratch, out, monitor, warmup, steps):
    monitor.mark("training_prepare")
    histories = Histories(scratch)
    pairs = []
    anchors = []
    for episode in episodes:
        rows = histories.raw(episode)
        eligible = [i for i, r in enumerate(rows) if i >= 127 and len(r["actions"]) > 1]
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
    references = prepare_references(policy, histories, anchors, out / "references.pt")
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad],
        lr=1e-5,
        betas=(0.9, 0.999),
        eps=1e-5,
    )
    initial = AdamSnapshot(policy, optimizer)
    timings = []
    for step in range(warmup + steps):
        initial.restore()
        monitor.mark("training_warmup" if step < warmup else "training", update=step)
        start = perf_counter()
        metrics = logical_step(
            policy, optimizer, histories, pairs, anchors, references["mixed"]
        )
        seconds = perf_counter() - start
        print(f"Comparison batch {step + 1}: {seconds:.3f}s", flush=True)
        if step >= warmup:
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
    result = dict(
        logical_pairs=len(pairs),
        logical_anchors=len(anchors),
        step_samples=timings,
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
    return result


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
        results["source_collection"] = measure_source_collection(
            cfg, config, episodes, runtime, scratch, out, monitor
        )
        results["continuations"] = measure_continuations(
            cfg, config, collection, runtime, scratch, out, monitor
        )
        results["training"] = measure_training(
            cfg, episodes, scratch, out, monitor, args.warmup, args.steps
        )
    rows = [
        json.loads(line)
        for line in (out / "telemetry/resources.jsonl").read_text().splitlines()
    ]
    for phase in ("source_collection", "continuations", "training"):
        telemetry_samples, aggregates = aggregate_telemetry(rows, phase)
        results[phase]["telemetry_samples"] = telemetry_samples
        results[phase]["nvml_cpu"] = aggregates
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
                k: {x: v for x, v in r.items() if x not in ("telemetry_samples", "step_samples")}
                for k, r in results.items()
            },
            indent=2,
        ),
        flush=True,
    )
    return results
