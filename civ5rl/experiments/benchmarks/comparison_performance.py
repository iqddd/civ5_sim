"""Bounded CPU/GPU phase measurements for the rollout-comparison protocol."""

from __future__ import annotations

import os
import platform

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import json
import shutil
import statistics
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from time import perf_counter, process_time

import psutil
import pynvml
import torch

from civ5rl.core.agent import RLAgent
from civ5rl.core.context import context_from_spec
from civ5rl.deploy.archive import unpack
from civ5rl.engine.comparison.training import (
    Histories,
    configure_policy,
    epoch_fit_options,
    logical_step,
    pair_loss,
    prepare_references,
    target_logits,
    training_runtime,
)
from civ5rl.engine.evaluation.executor import (
    EvaluationExecutor,
    EvaluationRuntime,
    evaluation_precision,
    prepare_evaluation_policy,
)
from civ5rl.engine.runtime.storage import pack_verified
from civ5rl.engine.runtime.telemetry import ResourceMonitor
from civ5rl.engine.runtime.transactions import AdamSnapshot
from civ5rl.experiments.benchmarks.comparison_measurement import measured_match
from civ5rl.experiments.benchmarks.profiling import trace_summary
from civ5rl.experiments.comparison.executor import ProposalExecutor, ProposalInference
from civ5rl.experiments.comparison.rollout_fit import load_config
from civ5rl.protocols.ppo_target_scale import TargetFactory
from civ5rl.provenance.common import ROOT, configure, file_sha256, read_json, write_json
from civ5rl.reporting.comparison_benchmarks import write_performance_report
from civ5sim import Action


class ServiceMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.wall = 0.0
        self.calls = 0

    def __call__(self, sample):
        self.wall += sample["ns"] / 1e9
        self.calls += 1

    def result(self):
        return {
            "service_wall_seconds": self.wall,
            "calls": self.calls,
        }


def inference_trace(cfg, runtime, out):
    """Three saved ready batches, replayed on the profiling thread (no CP time)."""

    folders = sorted((out / "matches/source").iterdir())[: runtime.max_batch]
    with evaluation_precision():
        policies = [
            prepare_evaluation_policy(RLAgent.from_checkpoint(p, "cpu", "argmax").policy, "cuda")
            for p in cfg["parents"]
        ]
        agents = []
        histories = []
        for folder in folders:
            case = read_json(folder / "case.json")
            members = []
            for policy in policies:
                member = RLAgent(policy, "cuda", "argmax")
                member.reset(context_from_spec(case, seed=case["agent_seeds"][case["learner_side"]]))
                members.append(member)
            members[0].members = members
            agents.append(members[0])
            histories.append(read_json(folder / "history.json"))
        backend = ProposalInference(policies, "cuda", hidden_capacity=len(folders))

        def prepare_requests(index):
            requests = []
            for agent, history in zip(agents, histories):
                if index >= len(history):
                    continue
                row = history[index]
                actions = [Action(**a) for a in row["actions"]]
                encoded = agent.encoder.encode(row["observation"], actions)
                agent.encoder.remember_action(actions[row["chosen"]], row["observation"])
                requests.append((agent, "observe", encoded))
            return requests

        stops = max(map(len, histories))
        warmup = min(20, max(0, stops - 3))
        for index in range(warmup):
            backend.infer_encoded(prepare_requests(index))
        ready = [prepare_requests(index) for index in range(warmup, min(stops, warmup + 3))]
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profile:
            for requests in ready:
                backend.infer_encoded(requests)
        profile.export_chrome_trace(str(out / "inference-trace.json"))


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
            {"mean": statistics.mean(values), "max": max(values)} if values else {"mean": None, "max": None}
        )
    return samples, aggregates


def archive_evidence(out):

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
        {
            "status": "verified",
            "packet": str(packet),
            "files": len(manifest["files"]),
            "bytes": sum(p.stat().st_size for p in packet.rglob("*") if p.is_file()),
        },
    )


def measure_source_collection(cfg, config, episodes, runtime, scratch, out, monitor):
    monitor.mark("source_startup")
    service = ServiceMeter()
    with ProposalExecutor(cfg["parents"], runtime, inference_metrics=service) as executor:
        episode_cases = [
            (episode, read_json(scratch / "sources" / str(episode) / "case.json")) for episode in episodes
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
        service.reset()
        list(executor.run(measured_match, tasks))
        result = {
            "seconds": perf_counter() - start,
            "service": service.result(),
            "matches": len(tasks),
        }
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
                branch["source"] = str((scratch / "sources" / str(fork["episode"])).resolve())
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

    service = ServiceMeter()
    with EvaluationExecutor(cfg["parents"][0], runtime, inference_metrics=service) as executor:
        warmup_tasks = continuation_tasks("continuation-warmup")
        tasks = continuation_tasks("continuations")
        list(executor.run(measured_match, warmup_tasks))
        monitor.mark("continuations")
        start = perf_counter()
        service.reset()
        list(executor.run(measured_match, tasks))
        result = {"seconds": perf_counter() - start, "service": service.result(), "matches": len(tasks)}
        monitor.mark("continuation_cleanup")
    return result


def measure_training(cfg, episodes, scratch, out, monitor, warmup, steps):
    monitor.mark("training_prepare")
    histories = Histories(scratch)
    options = epoch_fit_options(cfg)
    eligible_cases = []
    for episode in episodes:
        rows = histories.raw(episode)
        eligible = [i for i, row in enumerate(rows) if len(row["actions"]) > 1]
        if eligible:
            eligible_cases.append((episode, rows, [i for i in eligible if i >= 127] or eligible))
    if not eligible_cases:
        raise ValueError("Saved cases contain no training decisions with multiple legal actions")
    per_case, extra = divmod(options["pairs_per_step"], len(eligible_cases))
    pairs = []
    anchors = []
    for slot, (episode, rows, eligible) in enumerate(eligible_cases):
        count = per_case + (slot < extra)
        indices = [eligible[k * len(eligible) // (3 * count)] for k in range(3 * count)]
        for index in indices[:count]:
            row = rows[index]
            pairs.append(
                {
                    "episode": episode,
                    "index": index,
                    "better": (row["chosen"] + 1) % len(row["actions"]),
                    "worse": row["chosen"],
                    "weight": 0.5,
                }
            )
        anchors.extend((episode, i) for i in indices[count:])
    policy = configure_policy(cfg["parents"][0])
    references = prepare_references(policy, histories, anchors, out / "references.pt")
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad],
        lr=options["initial_lr"],
        betas=(0.9, 0.999),
        eps=1e-5,
    )
    initial = AdamSnapshot(policy, optimizer)
    timings = []
    for step in range(warmup + steps):
        initial.restore()
        monitor.mark("training_warmup" if step < warmup else "training", update=step)
        start = perf_counter()
        metrics = logical_step(policy, optimizer, histories, pairs, anchors, references["mixed"])
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
    result = {
        "logical_pairs": len(pairs),
        "logical_anchors": len(anchors),
        "step_samples": timings,
        "mean_seconds": statistics.mean(r["seconds"] for r in timings),
        "median_seconds": statistics.median(r["seconds"] for r in timings),
        "unique_targets": len(set((p["episode"], p["index"]) for p in pairs) | set(anchors)),
        "map_sizes": [read_json(scratch / "sources" / str(e) / "case.json")["map_size"] for e in episodes],
        "synthetic_preferences": True,
    }
    monitor.mark("finished")
    return result


def run_benchmark(args, parser):

    cfg = load_config(args.config or Path("configs/rl-c2p-rollout-comparison.json"))
    cfg = {**cfg, "parents": [str(ROOT / parent) for parent in cfg["parents"]]}
    if args.run is None:
        parser.error("comparison benchmark requires --run PRECHECK_DATA")
    case_limit = 8 if args.cases is None else args.cases
    if args.steps < 1 or args.warmup < 0 or case_limit < 1:
        parser.error("Invalid step or case count")
    workers = cfg.get("workers", 8) if args.workers is None else args.workers
    runtime = EvaluationRuntime("cuda", workers, args.max_batch, args.wait_seconds).resolve()
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
            shutil.copytree(source / name, scratch / name)
    configure(cfg["seed"], "cuda")
    # CPU-only scenario generation is measured separately from saved-case matches.

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
        cases = [factory.make(cfg["start_episode"] + i) for i in range(case_limit)]
        generation.append(
            {
                "mode": label,
                "cases": len(cases),
                "wall_seconds": perf_counter() - start,
                "cpu_seconds": process_time() - cpu,
            }
        )
    write_json(out / "factory.json", generation)

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    hardware = {
        "gpu": torch.cuda.get_device_name(),
        "cpu": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "gpu_limit_watts": pynvml.nvmlDeviceGetEnforcedPowerLimit(handle) / 1000,
    }
    pynvml.nvmlShutdown()
    write_json(
        out / "identity.json",
        {
            "hardware": hardware,
            "config": cfg,
            "parents": [file_sha256(p) for p in cfg["parents"]],
            "training_runtime": training_runtime(cfg),
            "evaluation_runtime": runtime.identity(),
            "purpose": "diagnostic only; discarded Adam steps",
        },
    )
    config = read_json(source / "match-config.json")
    collection = read_json(source / "collection.json")
    episodes = collection["sources"][:case_limit]
    if not episodes:
        raise ValueError("Benchmark input has no saved source cases")
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
        results["training"] = measure_training(cfg, episodes, scratch, out, monitor, args.warmup, args.steps)
    rows = [json.loads(line) for line in (out / "telemetry/resources.jsonl").read_text().splitlines()]
    for phase in ("source_collection", "continuations", "training"):
        telemetry_samples, aggregates = aggregate_telemetry(rows, phase)
        results[phase]["telemetry_samples"] = telemetry_samples
        results[phase]["nvml_cpu"] = aggregates
    for phase, label in [
        ("source_collection", "source"),
        ("continuations", "continuations"),
    ]:
        measurements = [read_json(p) for p in (out / "matches" / label).glob("*/measurement.json")]
        cpu = defaultdict(float)
        for measurement in measurements:
            cpu["worker_cpu_seconds"] += measurement["cpu_seconds"]
            cpu["worker_wall_seconds"] += measurement["wall_seconds"]
            cpu["prefix_hook_seconds"] += measurement["prefix_hook_seconds"]
            cpu["replay_verification_seconds"] += measurement["replay_verification_seconds"]
            cpu["public_encoding_seconds"] += measurement["public_encoding_seconds"]
            side = measurement["learner_side"]
            for name, value in measurement["scopes"].items():
                kind, scope_side = name.split("/")
                who = "model" if scope_side == f"side{side}" else "cp"
                cpu[kind + "/" + who] += value
        results[phase]["worker_scopes"] = dict(cpu)
    results["training"]["parent_hashes_unchanged"] = [
        file_sha256(p) == expected for p, expected in zip(cfg["parents"], cfg["parent_sha256"])
    ]
    if not all(results["training"]["parent_hashes_unchanged"]):
        raise RuntimeError("Parent checkpoint changed")
    write_json(out / "summary.json", results)
    inference_trace(cfg, runtime, out)
    traces = {name: trace_summary(out / f"{name}-trace.json") for name in ("training", "inference")}
    write_json(out / "trace-summary.json", traces)
    write_performance_report(out)
    # Scratch is a copy of already verified inputs; final evidence is separately packed.

    if scratch.resolve().is_relative_to(out):
        shutil.rmtree(scratch)
    archive_evidence(out)
    print(
        json.dumps(
            {
                k: {x: v for x, v in measurement.items() if x not in ("telemetry_samples", "step_samples")}
                for k, measurement in results.items()
            },
            indent=2,
        ),
        flush=True,
    )
    return results
