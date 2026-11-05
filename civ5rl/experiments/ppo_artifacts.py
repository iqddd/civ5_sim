"""Bounded PPO artifacts: retain evidence without serializing input caches."""

import os
import pickle
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from civ5rl.provenance.common import file_sha256, read_json, write_json


def pack_rollout(episodes):
    """Intern immutable actions and concatenate ragged behavior distributions."""
    action_ids, action_table, packed = {}, [], []
    transient = {"encoded", "encoded_variant", "old_hidden", "actions", "old_log_probs"}
    for episode in episodes:
        value = {k: v for k, v in episode.items() if k not in transient}
        rows = episode["actions"]
        offsets = np.cumsum([0, *map(len, rows)], dtype=np.int64)
        ids = []
        for row in rows:
            for action in row:
                index = action_ids.get(action)
                if index is None:
                    index = len(action_table)
                    action_ids[action] = index
                    action_table.append(action)
                ids.append(index)
        value["actions"] = (offsets, np.asarray(ids, dtype=np.int32))
        if "old_log_probs" in episode:
            probabilities = episode["old_log_probs"]
            if len(probabilities) != len(rows) or any(
                len(p) != len(a) for p, a in zip(probabilities, rows)
            ):
                raise ValueError("Behavior distributions do not match legal actions")
            if len({p.dtype for p in probabilities}) > 1:
                raise ValueError("Mixed behavior distribution dtypes")
            value["old_log_probs"] = (
                np.concatenate(probabilities) if probabilities else np.empty(0, np.float32)
            )
        packed.append(value)
    return {"format": "rl01.rollout.v2", "action_table": action_table, "episodes": packed}


def unpack_rollout(value):
    if isinstance(value, list):
        return value
    if value.get("format") != "rl01.rollout.v2":
        raise ValueError("Unsupported rollout format")
    table, episodes = value["action_table"], value["episodes"]
    for episode in episodes:
        offsets, indices = episode["actions"]
        if offsets[0] != 0 or offsets[-1] != len(indices) or np.any(np.diff(offsets) < 1):
            raise ValueError("Invalid rollout action offsets")
        if len(offsets) != len(episode["chosen"]) + 1:
            raise ValueError("Rollout decision count mismatch")
        episode["actions"] = [
            tuple(table[int(i)] for i in indices[a:b])
            for a, b in zip(offsets, offsets[1:])
        ]
        if "old_log_probs" in episode:
            probabilities = episode["old_log_probs"]
            if len(probabilities) != len(indices):
                raise ValueError("Rollout behavior distribution count mismatch")
            episode["old_log_probs"] = [
                probabilities[a:b].copy() for a, b in zip(offsets, offsets[1:])
            ]
    return episodes


def load_rollout(path):
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict) and value.get("format") == "rl01.rollout.pickle":
        value = pickle.loads(value["payload"])
    return unpack_rollout(value)


def save_rollout(episodes, directory, update):
    """Public histories reconstruct encoded inputs and GRU prefixes on demand."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("first.pt" if update == 1 else "latest.pt")
    start = perf_counter()
    packed = pack_rollout(episodes)
    prepare_seconds = perf_counter() - start
    tick = perf_counter()
    # C pickle avoids torch's Python persistent_id callback on every public object.
    payload = pickle.dumps(packed, protocol=5)
    serialization_seconds = perf_counter() - tick
    tmp = path.with_suffix(".tmp")
    tick = perf_counter()
    torch.save({"format": "rl01.rollout.pickle", "payload": payload}, tmp)
    os.replace(tmp, path)
    write_seconds = perf_counter() - tick
    tick = perf_counter()
    checksum = file_sha256(path)
    hash_seconds = perf_counter() - tick
    record = {
        "update": update,
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": checksum,
        "episodes": len(episodes),
        "decisions": sum(len(ep["chosen"]) for ep in episodes),
        "omitted_reconstructible_fields": ["encoded", "encoded_variant", "old_hidden"],
        "format": "rl01.rollout.v2",
        "timings": {
            "prepare_seconds": prepare_seconds,
            "serialization_seconds": serialization_seconds,
            "write_seconds": write_seconds,
            "hash_seconds": hash_seconds,
            "total_seconds": perf_counter() - start,
        },
    }
    write_json(path.with_suffix(".json"), record)
    return record


def prune_checkpoints(out, state):
    """Keep initial, evaluated, selected, and the latest two saved boundaries."""
    out = Path(out).resolve()
    directory = out / "checkpoints"
    paths = sorted(directory.glob("update-*.pt"))
    keep = {"initial.pt", Path(state["latest"]).name, Path(state["best"]["path"]).name}
    keep.update(p.name for p in paths[-2:])
    keep.update(Path(e["checkpoint"]).name for e in state["evaluated"].values())
    journal = out / "checkpoint-retention.json"
    removed = read_json(journal) if journal.exists() else []
    for path in paths:
        if path.name in keep:
            continue
        if path.resolve().parent != directory:
            raise ValueError("Checkpoint retention target outside run directory")
        removed.append(
            {"path": path.relative_to(out).as_posix(), "bytes": path.stat().st_size}
        )
        # Persist the intention before deletion; no model or metrics are rewritten.
        write_json(journal, removed)
        path.unlink()
