"""Initial CP imitation: initialization, data collection, training and evaluation."""

import json
import random
import tempfile
from pathlib import Path
from time import perf_counter

from civ5rl.engine.training import FrozenParamsCheck, bc_optimizer_step
from civ5rl.experiments.imitation_helpers import adam, demonstrations, load_policy, pool
from civ5rl.protocols.imitation import make_case, match_task, prepare_manifest, task_for
from civ5rl.provenance.common import file_sha256, read_json, write_json
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
                ] != file_sha256(path.parent / "demonstration.json.gz"):
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
    frozen_value = FrozenParamsCheck(policy.value)
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
                ce, norm = bc_optimizer_step(
                    policy,
                    optimizer,
                    backward_minibatch,
                    train_cache,
                    spans,
                    count,
                    max_gradient_norm=config["max_gradient_norm"],
                )
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
            frozen_value.verify()
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
