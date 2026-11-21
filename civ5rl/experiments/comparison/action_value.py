"""Offline action-value experiment; no simulator matches or policy updates.

Preparation precedes the soft training timer. A completed epoch is the only
selection boundary. The immutable candidate receipt precedes confirmation;
resumption may finish that calculation but cannot select another checkpoint.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from civ5rl.engine.comparison.action_value import (
    bootstrap,
    fork_targets,
    make_head,
    make_optimizer,
    predict,
    public_action_vector,
    split_worlds,
    summarize,
    world_scores,
)
from civ5rl.engine.comparison.action_value_features import FrozenActionFeatures
from civ5rl.engine.comparison.inputs import Histories
from civ5rl.engine.comparison.storage import readable_tree
from civ5rl.engine.runtime import is_int_at_least
from civ5rl.engine.runtime.control import RunLock
from civ5rl.engine.runtime.transactions import atomic_torch_save
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import ROOT, configure, file_sha256, read_json, write_json
from civ5rl.provenance.rng import capture_rng, restore_rng
from civ5rl.reporting.action_value import report

KIND = "rl01.action_value"
PROTOCOL = {
    "head": [534, 128, 1],
    "activation": "gelu",
    "precision": "cuda_deterministic_fp32_no_amp_no_tf32",
    "optimizer": {
        "name": "AdamW",
        "lr": 3e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay_weights": 1e-4,
        "weight_decay_bias": 0.0,
        "clip_norm": 1.0,
    },
    "shuffle_namespace": "c2p_action_value_shuffle_v1",
    "split_namespace": "c2p_action_value_split_v1",
    "world_counts": [33, 12, 12],
    "evaluation_repetitions": [0, 1, 2, 3],
    "bootstrap": {"samples": 10000, "seed": 20261103},
    "minimum_gain": 0.01,
}


def load_config(path):
    cfg = read_json(path)
    if cfg["kind"] != KIND or cfg["seed"] != 17:
        raise ValueError("Unknown action-value protocol")
    for key in ("epochs", "batch_forks", "training_seconds"):
        if not is_int_at_least(cfg.get(key)):
            raise ValueError(f"{key} must be a positive integer")
    if cfg["protocol"] != PROTOCOL:
        raise ValueError("Unexpected action-value scientific conditions")
    return cfg


def prepare(cfg, out):
    """Copy and verify input evidence, publishing the feature packet last."""
    source = ROOT / cfg["collection_input"]["run"]
    if source.resolve() == out.resolve():
        raise ValueError("Use a separate action-value output")
    for name, sha in cfg["collection_input"]["sha256"].items():
        if file_sha256(source / name) != sha:
            raise ValueError(f"Collection input changed: {name}")
    checkpoint.verify_frozen_passport(source, sources=False)
    if file_sha256(ROOT / cfg["parent"]) != cfg["parent_sha256"]:
        raise ValueError("Frozen parent changed")
    plan = read_json(source / "manifest.json")
    split = split_worlds(plan["base"]["worlds"]["training"])
    if (out / "passport.json").exists():
        identity = checkpoint.verify_execution_passport(out, cfg, split)
    else:
        identity = checkpoint.passport(out, cfg, split)
        write_json(out / "run.json", {"kind": KIND, "config": cfg, "identity": identity})
    if (out / "features.pt").exists() and (out / "data.json").exists():
        if file_sha256(out / "features.pt") != read_json(out / "data.json")["sha256"]:
            raise ValueError("Action-value features changed")
        return
    started = perf_counter()
    write_json(out / "split.json", split)
    write_json(out / "status.json", {"status": "preparing", "phase": "preparation"})
    collection = read_json(source / "collection.json")
    forks = collection["forks"]
    if not collection.get("finalized") or not forks or any(fork["status"] != "completed" for fork in forks):
        raise ValueError("Require a finalized collection with completed forks")
    dataset = read_json(source / "dataset.json")
    if dataset["completed_forks"] != len(forks):
        raise ValueError("Completed fork count differs from the collection dataset")
    target = out / "sources.storage"
    if not target.exists():
        shutil.copytree(source / "sources.storage", target)
    previous = Histories(ROOT / cfg["feature_cache_run"])
    limits = {}
    for episode, index in dataset["anchors"]:
        limits[episode] = max(limits.get(episode, 0), index + 1)
    histories = Histories(
        out,
        source_hashes=dataset["source_hashes"],
        limits=limits,
        cache_dir=previous.directory if previous.directory.exists() else None,
    )
    with readable_tree(out, "sources"):
        features = FrozenActionFeatures(histories, dataset, ROOT / cfg["parent"]).prepare()
        cases, rows = {}, {}
        wanted_by_episode = {}
        for fork in forks:
            wanted_by_episode.setdefault(fork["episode"], set()).add(fork["index"])
        inputs, wins, counts, first, worlds, keys = [], [], [], [], [], []
        split_world_ids = {world for world_ids in split.values() for world in world_ids}
        for fork in forks:
            episode, index = fork["episode"], fork["index"]
            if episode not in cases:
                cases[episode] = read_json(out / "sources" / str(episode) / "case.json")
                raw = histories.raw(episode)
                wanted = wanted_by_episode[episode]
                rows[episode] = {i: r for i, r in enumerate(raw) if i in wanted}
            case = cases[episode]
            world = case["map_entry"]["parent_seed"]
            if world != case["parent_world"] or world not in split_world_ids:
                raise ValueError("Untrusted source world")
            row = rows[episode][index]
            actions = [row["chosen"], row["alternative"]]
            if [row["actions"][a] for a in actions] != fork["actions"] or actions[0] == actions[1]:
                raise ValueError("Fork public action order changed")
            _, location = features.locations[(episode, index)]
            record = features.records[episode]
            vectors = [public_action_vector(record, location, action) for action in actions]
            total, repeats, initial = fork_targets(fork)
            if fork["side"] != case["learner_side"] or total != [v["W"] for v in fork["wdl"]]:
                raise ValueError("Outcome summary differs")
            inputs.append(torch.stack(vectors))
            wins.append(total)
            counts.append(repeats)
            first.append(initial)
            worlds.append(world)
            keys.append([episode, index])
        packet = {
            "inputs": torch.stack(inputs),
            "wins": torch.tensor(wins, dtype=torch.float32),
            "counts": torch.tensor(counts, dtype=torch.float32),
            "first_wins": np.asarray(first),
            "worlds": np.asarray(worlds),
            "keys": keys,
        }
        continuations = int(packet["counts"].sum())
        if packet["inputs"].shape != (len(forks), 2, PROTOCOL["head"][0]) or set(worlds) != split_world_ids:
            raise ValueError("Unexpected action-value corpus dimensions")
        if continuations != dataset["continuations"]:
            raise ValueError("Continuation count differs from the collection dataset")
    atomic_torch_save(packet, out / "features.pt")
    write_json(
        out / "data.json",
        {
            "sha256": file_sha256(out / "features.pt"),
            "forks": len(forks),
            "continuations": continuations,
            "preparation_seconds": perf_counter() - started,
            "cache": features.statistics,
            "split_counts": {name: int(np.isin(packet["worlds"], ws).sum()) for name, ws in split.items()},
            "parent_sha256": cfg["parent_sha256"],
        },
    )
    write_json(out / "status.json", {"status": "prepared", "phase": "training"})


def _train_epoch(head, optimizer, state, order, features, targets, counts, batch_size, after_step):
    epoch_loss = torch.zeros((), dtype=torch.float64, device="cuda")
    epoch_count = torch.zeros_like(epoch_loss)
    for start in range(state["cursor"], len(order), batch_size):
        indices = order[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        logits = head(features[indices].flatten(0, 1)).reshape(-1, 2)
        repeats = counts[indices]
        count = repeats.sum()
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(logits, targets[indices], reduction="none")
            * repeats
        ).sum() / count
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        torch._assert_async(torch.isfinite(norm), "Non-finite action-value gradient")
        optimizer.step()
        if after_step is not None:
            after_step()
        epoch_loss += loss.detach().double() * count
        epoch_count += count
        state["steps"] += 1
        state["cursor"] = min(start + batch_size, len(order))
    return epoch_loss / epoch_count


def train(cfg, out, data, *, after_step=None):
    """Checkpoint at completed epochs; replay an interrupted epoch from its boundary.

    One epoch permutation upload and one loss readback replace per-batch host
    traffic. Confirmation features remain on CPU until candidate confirmation.
    An optional observer runs after each Adam step before the epoch is committed.
    """
    head = make_head()
    optimizer = make_optimizer(head)
    shuffle = random.Random("c2p_action_value_shuffle_v1|17")
    path = out / "training.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        head.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], shuffle)
    else:
        state = {
            "epoch": 0,
            "steps": 0,
            "cursor": 0,
            "order": None,
            "seconds": 0.0,
            "best": None,
            "history": [],
            "complete": False,
            "epoch_started_seconds": None,
        }
    if state["complete"]:
        return
    split = read_json(out / "split.json")
    training = np.flatnonzero(np.isin(data["worlds"], split["training"])).tolist()
    selection = np.flatnonzero(np.isin(data["worlds"], split["selection"]))
    x = data["inputs"][training].cuda()
    n = data["counts"][training].cuda()
    y = (data["wins"][training] / data["counts"][training]).cuda()
    x_sel = data["inputs"][selection].cuda()
    started, base = perf_counter(), state["seconds"]

    def save():
        state.update(
            model=head.state_dict(),
            optimizer=optimizer.state_dict(),
            rng=capture_rng(shuffle),
            seconds=base + perf_counter() - started,
        )
        atomic_torch_save(state, path)
        write_json(
            out / "training.json",
            {k: state[k] for k in ("epoch", "steps", "seconds", "best", "history", "complete")},
        )

    if not path.exists():
        save()
    while state["epoch"] < cfg["epochs"]:
        if state["cursor"] == 0 and base + perf_counter() - started >= cfg["training_seconds"]:
            break
        if state["order"] is None:
            # Both timestamps use cumulative active training time, so an epoch
            # spans resumed processes without counting downtime/preparation.
            state["epoch_started_seconds"] = base + perf_counter() - started
            state["order"] = list(range(len(training)))
            shuffle.shuffle(state["order"])
        order = torch.tensor(state["order"], device="cuda")
        write_json(out / "status.json", {"status": "training", "phase": "training"})
        training_bce = _train_epoch(
            head,
            optimizer,
            state,
            order,
            x,
            y,
            n,
            cfg["batch_forks"],
            after_step,
        )
        logits = predict(head, x_sel)
        metrics = summarize(logits, data["first_wins"][selection], data["worlds"][selection])
        state["epoch"] += 1
        if state["best"] is None or metrics["G"] > state["best"]["G"]:
            atomic_torch_save(
                {
                    "model": head.state_dict(),
                    "epoch": state["epoch"],
                    "selection": metrics,
                },
                out / "best.pt",
            )
            state["best"] = {"epoch": state["epoch"], "G": metrics["G"]}
        state["order"] = None
        state["cursor"] = 0
        state["history"].append(
            {
                "epoch": state["epoch"],
                "seconds": base + perf_counter() - started - state["epoch_started_seconds"],
                "selection": metrics,
                "training_bce": float(training_bce),
            }
        )
        state["epoch_started_seconds"] = None
        state["complete"] = (
            state["epoch"] >= cfg["epochs"] or base + perf_counter() - started >= cfg["training_seconds"]
        )
        save()
        print(
            f"Epoch {state['epoch']}/{cfg['epochs']} selection G={metrics['G']:+.4f}; elapsed={state['seconds']:.1f}s",
            flush=True,
        )
        if state["epoch"] == 1:
            print(
                f"First-epoch forecast: {state['history'][0]['seconds'] * cfg['epochs']:.1f}s for {cfg['epochs']} epochs; soft limit={cfg['training_seconds']}s",
                flush=True,
            )
        if state["complete"]:
            break
    if state["best"] is None:
        raise RuntimeError("No completed epoch to select")
    if not state["complete"]:
        state["complete"] = True
        save()
    freeze_candidate(out, state["best"])


def freeze_candidate(out, best):
    """Publish the immutable selection receipt before any confirmation metric."""
    path = out / "candidate.pt"
    if (out / "selected.json").exists():
        receipt = read_json(out / "selected.json")
        if file_sha256(path) != receipt["sha256"] or receipt["epoch"] != best["epoch"]:
            raise ValueError("Selected candidate changed")
        return receipt
    shutil.copy2(out / "best.pt", path)
    receipt = {**best, "sha256": file_sha256(path)}
    write_json(out / "selected.json", receipt)
    return receipt


def evaluate(cfg, out, data):
    state = torch.load(out / "training.pt", map_location="cpu", weights_only=False)
    if not state["complete"]:
        raise ValueError("Training must finish before confirmation")
    receipt = freeze_candidate(out, state["best"])
    if (out / "result.json").exists():
        return read_json(out / "result.json")
    started = perf_counter()
    write_json(out / "status.json", {"status": "evaluating", "phase": "confirmation"})
    head = make_head()
    head.load_state_dict(torch.load(out / "candidate.pt", map_location="cpu", weights_only=False)["model"])
    ids = np.flatnonzero(np.isin(data["worlds"], read_json(out / "split.json")["confirmation"]))
    logits = predict(head, data["inputs"][ids].cuda())
    wins, worlds = data["first_wins"][ids], data["worlds"][ids]
    scores = world_scores(logits, wins, worlds)
    metrics = summarize(logits, wins, worlds, scores=scores)
    result = {
        "status": "complete",
        "selected": receipt,
        "confirmation": metrics,
        "bootstrap": bootstrap(scores),
        "confirmation_seconds": perf_counter() - started,
        "preparation_seconds": read_json(out / "data.json")["preparation_seconds"],
        "training_seconds": state["seconds"],
    }
    # FP64 export agrees with diagnostic probabilities; policy ranking continues
    # to use logits even when sigmoid saturates at the numerical endpoints.
    probs = torch.as_tensor(logits, dtype=torch.float64).sigmoid().numpy()
    atomic_torch_save(
        {
            "logits": logits,
            "probs": probs,
            "delta_prob": probs[:, 1] - probs[:, 0],
            "chosen_alternative": logits[:, 1] > logits[:, 0],
            "keys": [data["keys"][i] for i in ids],
        },
        out / "confirmation-predictions.pt",
    )
    write_json(out / "result.json", result)
    write_json(out / "status.json", {"status": "complete", "phase": "complete"})
    return result


def run(phase, cfg, out):
    out = Path(out).resolve()
    configure(cfg["seed"], "cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Action-value fitting requires CUDA")
    with RunLock(out):
        prepare(cfg, out)
        if phase == "prepare":
            return
        data = torch.load(out / "features.pt", map_location="cpu", weights_only=False)
        # Cache preparation must not affect the new head's initialization.
        configure(cfg["seed"], "cuda")
        if phase in ("train", "resume"):
            train(cfg, out, data)
        if phase in ("evaluate", "resume"):
            evaluate(cfg, out, data)
        report(out)
