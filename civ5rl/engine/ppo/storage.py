"""Persist PPO public histories without reconstructible input caches."""

from __future__ import annotations

import pickle
from collections.abc import Hashable, Sequence
from itertools import pairwise
from os import PathLike
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch

from civ5rl.engine.runtime.transactions import atomic_legacy_torch_save
from civ5rl.provenance.common import file_sha256, write_json


def pack_rollout(episodes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Intern immutable actions and concatenate ragged behavior distributions."""
    action_ids: dict[Hashable, int] = {}
    action_table: list[Hashable] = []
    packed: list[dict[str, Any]] = []
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
                len(p) != len(a) for p, a in zip(probabilities, rows, strict=True)
            ):
                raise ValueError("Behavior distributions do not match legal actions")
            if len({p.dtype for p in probabilities}) > 1:
                raise ValueError("Mixed behavior distribution dtypes")
            value["old_log_probs"] = (
                np.concatenate(probabilities) if probabilities else np.empty(0, np.float32)
            )
        packed.append(value)
    return {
        "format": "rl01.rollout.v2",
        "action_table": action_table,
        "episodes": packed,
    }


def unpack_rollout(
    value: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(value, list):
        for episode in value:
            # Legacy lists could persist derived inputs from a different encoder.
            episode.pop("encoded", None)
            episode.pop("encoded_variant", None)
        return value
    if value.get("format") != "rl01.rollout.v2":
        raise ValueError("Unsupported rollout format")
    table = value["action_table"]
    episodes: list[dict[str, Any]] = value["episodes"]
    for episode in episodes:
        offsets, indices = episode["actions"]
        if offsets[0] != 0 or offsets[-1] != len(indices) or np.any(np.diff(offsets) < 1):
            raise ValueError("Invalid rollout action offsets")
        if len(offsets) != len(episode["chosen"]) + 1:
            raise ValueError("Rollout decision count mismatch")
        episode["actions"] = [tuple(table[int(i)] for i in indices[a:b]) for a, b in pairwise(offsets)]
        if "old_log_probs" in episode:
            probabilities = episode["old_log_probs"]
            if len(probabilities) != len(indices):
                raise ValueError("Rollout behavior distribution count mismatch")
            episode["old_log_probs"] = [probabilities[a:b].copy() for a, b in pairwise(offsets)]
    return episodes


def load_rollout(path: str | PathLike[str]) -> list[dict[str, Any]]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(value, dict) and value.get("format") == "rl01.rollout.pickle":
        value = pickle.loads(value["payload"])
    return unpack_rollout(value)


def _measure_rollout(episodes: Sequence[dict[str, Any]], path: Path, update: int) -> dict[str, Any]:
    start = perf_counter()
    packed = pack_rollout(episodes)
    prepare_seconds = perf_counter() - start
    tick = perf_counter()
    # C pickle avoids torch's Python persistent_id callback on every public object.
    payload = pickle.dumps(packed, protocol=5)
    serialization_seconds = perf_counter() - tick
    tick = perf_counter()
    atomic_legacy_torch_save({"format": "rl01.rollout.pickle", "payload": payload}, path)
    write_seconds = perf_counter() - tick
    tick = perf_counter()
    checksum = file_sha256(path)
    hash_seconds = perf_counter() - tick
    size = path.stat().st_size
    episode_count = len(episodes)
    decisions = sum(len(ep["chosen"]) for ep in episodes)
    timings = {
        "prepare_seconds": prepare_seconds,
        "serialization_seconds": serialization_seconds,
        "write_seconds": write_seconds,
        "hash_seconds": hash_seconds,
        "total_seconds": perf_counter() - start,
    }
    return _rollout_metadata(update, path.name, size, checksum, episode_count, decisions, timings)


def _rollout_metadata(
    update: int,
    filename: str,
    size: int,
    checksum: str,
    episodes: int,
    decisions: int,
    timings: dict[str, float],
) -> dict[str, Any]:
    return {
        "update": update,
        "path": filename,
        "bytes": size,
        "sha256": checksum,
        "episodes": episodes,
        "decisions": decisions,
        "omitted_reconstructible_fields": ["encoded", "encoded_variant", "old_hidden"],
        "format": "rl01.rollout.v2",
        "timings": timings,
    }


def save_rollout(
    episodes: Sequence[dict[str, Any]], directory: str | PathLike[str], update: int
) -> dict[str, Any]:
    """Public histories reconstruct encoded inputs and GRU prefixes on demand."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / ("first.pt" if update == 1 else "latest.pt")
    record = _measure_rollout(episodes, path, update)
    write_json(path.with_suffix(".json"), record)
    return record
