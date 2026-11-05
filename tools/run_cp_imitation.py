"""CP imitation phases; benchmark stops after eight matches and latency probes."""

import argparse
import json
import multiprocessing as mp
import os
import random
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from civ5rl.protocols.imitation import (
    file_hash,
    load_episode,
    make_case,
    match_task,
    prepare_manifest,
    task_for,
    worker_setup,
)
from civ5rl.provenance.common import read_json, write_json
from civ5sim.engine import digest


def initialize(config, out):
    import torch

    from civ5rl.core.model import Policy
    from civ5rl.provenance import checkpoint
    from civ5rl.provenance.common import configure

    out = Path(out)
    manifest = prepare_manifest(config)
    path = out / "manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Imitation manifest mismatch")
    write_json(path, manifest)
    if (out / "passport.json").exists():
        identity = read_json(out / "passport.json")
        if read_json(out / "resolved-config.json") != config:
            raise ValueError("Imitation config mismatch")
        with tempfile.TemporaryDirectory() as tmp:
            current = checkpoint.passport(tmp, config, manifest)
        if not checkpoint.same_execution_identity(identity, current):
            raise ValueError("Imitation source/corpus/environment mismatch")
    else:
        identity = checkpoint.passport(out, config, manifest)
    configure(17)
    if not (out / "checkpoints/initial.pt").exists():
        policy = Policy("C")
        for parameter in policy.value.parameters():
            parameter.requires_grad_(False)
        optimizer = adam(policy, config, "cpu")
        checkpoint.save(
            out / "checkpoints/initial.pt",
            policy,
            optimizer,
            {"epoch": 0, "labels": 0},
            random.Random(18),
            config,
            identity,
        )
    initial = torch.load(
        out / "checkpoints/initial.pt", weights_only=False, map_location="cpu"
    )
    if initial["variant"] != "C" or initial["state"]["epoch"] != 0:
        raise ValueError("Unexpected initial BC checkpoint")
    return manifest, identity


def adam(policy, config, device):
    import torch

    values = dict(config["optimizer"])
    values["betas"] = tuple(values["betas"])
    return torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad],
        **values,
        fused=str(device).startswith("cuda"),
    )


def load_policy(config, path, device):
    import torch

    from civ5rl.core.model import Policy

    saved = torch.load(path, weights_only=False, map_location="cpu")
    policy = Policy("C").to(device)
    policy.load_state_dict(saved["model"])
    policy.enable_amp(str(device).startswith("cuda"), spatial_batch_size=32)
    if config["performance"].get("inverse_routing", False):
        policy.enable_inverse_routing()
    if config["performance"].get("optimized_routing", False):
        policy.enable_inverse_routing()
        policy.enable_action_routing()
        policy.enable_maximum_routing()
    if str(device).startswith("cuda") and config["performance"]["compile"]:
        policy.enable_compile(
            spatial_mode=config["performance"].get("spatial_compile_mode")
        )
    for parameter in policy.value.parameters():
        parameter.requires_grad_(False)
    if str(device).startswith("cuda"):
        if config["performance"].get("coalesced_spatial", False):
            policy.enable_coalesced_spatial()
        policy.coalesced_accumulation = config["performance"].get("coalesced_accumulation", False)
        if config["performance"].get("recurrent_cuda_graphs", False):
            policy.enable_recurrent_graphs(
                max_shapes=config["performance"].get("recurrent_graph_max_shapes", 64)
            )
    return policy


def pool(workers):
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=worker_setup,
    )


def benchmark(config, manifest, out, workers):
    from civ5rl.engine.resources import ResourceMonitor

    destination = out / "benchmark/collection.json"
    if destination.exists():
        return read_json(destination)
    jobs = [
        task_for(config, manifest, "training", i, out / f"data/training/match-{i:06}")
        for i in range(8)
    ]
    # Freeze every case before launching any CP decisions.
    write_json(out / "benchmark/cases.json", [job[2] for job in jobs])
    with ResourceMonitor(out / "benchmark/telemetry") as monitor:
        monitor.mark("CP_generation", workers=workers)
        start = perf_counter()
        with pool(workers) as executor:
            rows = list(executor.map(match_task, jobs))
        seconds = perf_counter() - start
    result = {
        "workers": workers,
        "pool_wall_seconds": seconds,
        "matches": rows,
        "labels": sum(r["labels"] for r in rows),
        "forced": sum(r["forced"] for r in rows),
        "labels_per_second": sum(r["labels"] for r in rows) / seconds,
        "closed_test_evaluated": False,
    }
    write_json(destination, result)
    return result


def latency(config, out):
    import numpy as np
    import torch

    from civ5rl.core.agent import RLAgent
    from civ5rl.provenance.common import configure
    from civ5sim import Action

    episodes = [
        load_episode(out / f"data/training/match-{i:06}/demonstration.json.gz")
        for i in range(8)
    ]
    result = {}
    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            result[device] = {"unavailable": True}
            continue
        configure(17, device)
        agent = RLAgent.from_checkpoint(out / "checkpoints/initial.pt", device=device)
        synchronize = torch.cuda.synchronize if device == "cuda" else lambda: None
        warm_start = perf_counter()
        agent.reset(episodes[0]["context"])
        for row in episodes[0]["decisions"][:8]:
            agent.act(row["observation"], tuple(Action(**a) for a in row["actions"]))
        synchronize()
        warm = perf_counter() - warm_start
        samples = {"forced": [], "free": [], "all": []}
        for episode in episodes:
            agent.reset(episode["context"])
            for row in episode["decisions"]:
                legal = tuple(Action(**a) for a in row["actions"])
                synchronize()
                start = perf_counter()
                agent.act(row["observation"], legal)
                synchronize()
                seconds = perf_counter() - start
                # Replay the teacher's previous command into the encoder; timing
                # includes the real Agent API, not CP or counterfactual simulation.
                agent.encoder.remember_action(legal[row["chosen"]], row["observation"])
                agent.drain_diagnostics()
                samples["free" if row["supervised"] else "forced"].append(seconds)
                samples["all"].append(seconds)
        result[device] = {
            "warmup_seconds": warm,
            "compiled": False,
            "history": "fixed_public_teacher_inputs_with_teacher_previous_action",
            "precision": "mixed_bfloat16" if device == "cuda" else "float32",
            "samples": {
                key: {
                    "count": len(values),
                    "mean_ms": float(np.mean(values) * 1000),
                    "median_ms": float(np.median(values) * 1000),
                    "p95_ms": float(np.quantile(values, 0.95) * 1000),
                }
                if values
                else None
                for key, values in samples.items()
            },
        }
        del agent
    write_json(out / "benchmark/latency.json", result)
    return result


def collect(config, manifest, out, workers):
    from civ5rl.engine.resources import ResourceMonitor

    start = perf_counter()
    totals = {}
    complete = True
    with (
        ResourceMonitor(out / "collection-telemetry") as monitor,
        pool(workers) as executor,
    ):
        monitor.mark("CP_generation", workers=workers)
        for split, target in (
            ("training", config["training_labels"]),
            ("holdout", config["holdout_labels"]),
        ):
            directory = out / "data" / split
            existing = sorted(directory.glob("match-*/metrics.json"))
            labels = 0
            for path in existing:
                case = read_json(path.parent / "case.json")
                if case != make_case(config, manifest, split, case["case_id"]):
                    raise ValueError("Existing demonstration has wrong provenance")
                row = read_json(path)
                if row["status"] != "completed" or row[
                    "demonstration_sha256"
                ] != file_hash(path.parent / "demonstration.json.gz"):
                    raise ValueError("Existing demonstration hash/completion mismatch")
                labels += row["labels"]
            index = max([read_json(p)["case_id"] for p in existing], default=-1) + 1
            while (
                labels < target
                and perf_counter() - start < config["collection_seconds"]
            ):
                jobs = [
                    task_for(config, manifest, split, i, directory / f"match-{i:06}")
                    for i in range(index, index + workers)
                ]
                rows = list(executor.map(match_task, jobs))
                labels += sum(r["labels"] for r in rows)
                index += workers
                print(
                    json.dumps({"split": split, "labels": labels, "target": target}),
                    flush=True,
                )
            totals[split] = {
                "labels": labels,
                "target": target,
                "matches": index,
                "overflow": max(0, labels - target),
            }
            complete &= labels >= target
    result = {
        "status": "complete" if complete else "collection_budget_exhausted",
        "splits": totals,
        "seconds": perf_counter() - start,
    }
    write_json(out / "collection.json", result)
    return result


def demonstrations(config, manifest, out, split):
    episodes = []
    files = {}
    for path in sorted((out / "data" / split).glob("match-*/demonstration.json.gz")):
        case = read_json(path.parent / "case.json")
        metrics = read_json(path.parent / "metrics.json")
        if case != make_case(config, manifest, split, case["case_id"]):
            raise ValueError("Untrusted demonstration case")
        if (
            metrics["status"] != "completed"
            or file_hash(path) != metrics["demonstration_sha256"]
        ):
            raise ValueError("Unfinished/corrupted demonstration")
        episode = load_episode(path)
        if (
            episode["parent_world"] not in manifest["worlds"][split]
            or episode["split"] != split
        ):
            raise ValueError("Demonstration world split leakage")
        episodes.append(episode)
        files[path.relative_to(out).as_posix()] = metrics["demonstration_sha256"]
    return episodes, files


def train(config, manifest, identity, out, device):
    import torch

    from civ5rl.engine.imitation import (
        DemonstrationCache,
        backward_minibatch,
        label_minibatches,
        teacher_metrics,
    )
    from civ5rl.engine.resources import ResourceMonitor
    from civ5rl.provenance import checkpoint
    from civ5rl.provenance.common import configure

    if (out / "training.json").exists():
        raise ValueError(
            "BC training already recorded; use a new experiment for another training run"
        )
    collection = read_json(out / "collection.json")
    if collection["status"] != "complete":
        raise ValueError(
            "Insufficient corpus: refusing automatic training on a smaller sample"
        )
    configure(17, device)
    training, train_files = demonstrations(config, manifest, out, "training")
    holdout, holdout_files = demonstrations(config, manifest, out, "holdout")
    if (
        sum(r["supervised"] for ep in training for r in ep["decisions"]) < 20000
        or sum(r["supervised"] for ep in holdout for r in ep["decisions"]) < 2457
    ):
        raise ValueError("Corpus below fixed label targets")
    write_json(
        out / "training-inputs.json",
        {"training": train_files, "holdout": holdout_files},
    )
    policy = load_policy(config, out / "checkpoints/initial.pt", device)
    optimizer = adam(policy, config, device)
    value_before = {k: v.detach().clone() for k, v in policy.value.state_dict().items()}
    train_cache, holdout_cache = (
        DemonstrationCache(training, device),
        DemonstrationCache(holdout, device),
    )
    rng = random.Random(18)
    elapsed = 0.0
    epochs, steps = [], 0
    initial_holdout = teacher_metrics(policy, holdout_cache)
    with ResourceMonitor(out / "training-telemetry") as monitor:
        for epoch in range(1, 4):
            monitor.mark("BC_epoch", epoch=epoch)
            order = train_cache.chunks()
            rng.shuffle(order)
            batches = list(label_minibatches(order, train_cache.masks))
            sums = labels = 0
            records = []
            completed = True
            for spans, count in batches:
                if elapsed >= config["training_seconds"]:
                    completed = False
                    break
                start = perf_counter()
                optimizer.zero_grad(set_to_none=True)
                ce = backward_minibatch(policy, train_cache, spans, count)
                norm = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(), 0.5, error_if_nonfinite=True
                )
                optimizer.step()
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed += perf_counter() - start
                steps += 1
                sums += ce * count
                labels += count
                records.append(
                    {
                        "Adam_step": steps,
                        "labels": count,
                        "ce": ce,
                        "gradient_norm_pre_clip": float(norm),
                        "seconds": perf_counter() - start,
                    }
                )
            if any(
                not torch.equal(v, value_before[k])
                for k, v in policy.value.state_dict().items()
            ):
                raise RuntimeError("Frozen value head changed")
            row = {
                "epoch": epoch,
                "completed": completed,
                "labels": labels,
                "ce": sums / labels if labels else None,
                "Adam_steps_total": steps,
                "optimizer_steps": records,
                "training_seconds": elapsed,
                "holdout": teacher_metrics(policy, holdout_cache),
            }
            epochs.append(row)
            checkpoint.save(
                out
                / f"checkpoints/{'epoch' if completed else 'partial'}-{epoch:02}.pt",
                policy,
                optimizer,
                {
                    "epoch": epoch if completed else epoch - 1,
                    "labels": labels,
                    "Adam_steps": steps,
                },
                rng,
                config,
                identity,
            )
            write_json(
                out / "training.json",
                {
                    "status": "complete"
                    if completed and epoch == 3
                    else "training_budget_exhausted",
                    "initial_holdout": initial_holdout,
                    "epochs": epochs,
                    "final_checkpoint": f"epoch-{epoch:02}.pt"
                    if completed
                    else (f"epoch-{epoch - 1:02}.pt" if epoch > 1 else None),
                },
            )
            if not completed:
                break


def evaluate(config, manifest, out, workers, device):
    final = read_json(out / "training.json")["final_checkpoint"]
    if not final:
        raise ValueError("No completed trained epoch to evaluate")
    jobs = []
    for case in manifest["validation_cases"]:
        for side in (0, 1):
            paired = {**case, "learner_side": side}
            paired["case_hash"] = digest(
                {k: v for k, v in paired.items() if k != "case_hash"}
            )
            for model, opponent in (
                ("initial", "cp"),
                ("initial", "greedy"),
                ("final", "cp"),
                ("final", "greedy"),
                ("teacher", "greedy"),
            ):
                path = (
                    out
                    / f"evaluation/{model}-{opponent}/case-{case['case_id']:03}-side-{side}"
                )
                contender = (
                    "teacher"
                    if model == "teacher"
                    else str(
                        (
                            out
                            / "checkpoints"
                            / ("initial.pt" if model == "initial" else final)
                        ).resolve()
                    )
                )
                jobs.append(
                    (config, manifest, paired, str(path), contender, opponent, device)
                )
    with pool(workers) as executor:
        rows = list(executor.map(match_task, jobs))
    write_json(
        out / "evaluation.json", {"matches": len(rows), "closed_test_evaluated": False}
    )


def report(config, out):
    from tools.report_cp_imitation import report as render

    render(config, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase",
        choices=("prepare", "benchmark", "collect", "train", "evaluate", "report", "run"),
    )
    parser.add_argument("--config", default="configs/rl-cp-imitation.json")
    parser.add_argument("--out")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cycle", type=int, choices=(1, 2))
    args = parser.parse_args()
    config = read_json(args.config)
    out = Path(args.out or config["output"]).resolve()
    workers = args.workers or config["workers"]
    if workers < 1:
        raise ValueError("workers must be positive")
    if config["kind"] == "rl01.cp_imitation_own_states":
        from civ5rl.engine.imitation_own_states import run

        run(args.phase, config, out, workers, args.device, args.cycle)
        return
    if args.phase == "run" or args.cycle is not None:
        raise ValueError("run/--cycle are supported only for the own-state protocol")
    if config["kind"] == "rl01.cp_imitation_more_data":
        from civ5rl.engine.imitation_more_data import run

        run(args.phase, config, out, workers, args.device)
        return
    if config["kind"] == "rl01.cp_imitation_continuation":
        from civ5rl.engine.imitation_continuation import run

        run(args.phase, config, out, workers, args.device)
        return
    if args.phase == "report":
        report(config, out)
        return
    manifest, identity = initialize(config, out)
    if args.phase == "benchmark":
        benchmark(config, manifest, out, workers)
        latency(config, out)
        write_json(
            out / "status.json",
            {
                "status": "benchmark_complete_stopped",
                "next": "review_cost_report",
                "research_training_started": False,
                "full_evaluation_started": False,
            },
        )
    elif args.phase == "collect":
        collect(config, manifest, out, workers)
    elif args.phase == "train":
        train(config, manifest, identity, out, args.device)
    elif args.phase == "evaluate":
        evaluate(config, manifest, out, workers, args.device)
    report(config, out)


if __name__ == "__main__":
    main()
