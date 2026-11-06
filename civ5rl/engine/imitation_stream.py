"""Bounded episode storage for BC; the existing sequence/loss math is unchanged."""

import sys
import os
from collections import Counter, OrderedDict
from dataclasses import fields, is_dataclass
from time import perf_counter
from typing import NamedTuple, Literal, Any

import numpy as np
import torch

from civ5rl.core.encoded_layout import ENCODED_FIELDS
from civ5rl.core.model import batch_inputs
from civ5rl.core.observation import Encoded
from civ5rl.engine.constants import (
    DEFAULT_CPU_CACHE_BYTES,
    DEFAULT_PREFIX_BLOCK_SIZE,
    PREPARED_VIEW_BYTES_PER_DECISION,
    PREPARED_FIXED_OVERHEAD_BYTES,
)
from civ5rl.engine.imitation import decision_indices, encode_demonstration
from civ5rl.engine.pool import pool
from civ5rl.engine.spans import Span
from civ5rl.protocols.imitation import load_episode
from civ5rl.provenance.common import read_json, write_json


def memory_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return size + (value.nbytes if value.base is not None else 0)
    if isinstance(value, dict):
        return size + sum(
            memory_bytes(k, seen) + memory_bytes(v, seen) for k, v in value.items()
        )
    if isinstance(value, (list, tuple)):
        return size + sum(memory_bytes(v, seen) for v in value)
    if is_dataclass(value):
        return size + sum(
            memory_bytes(getattr(value, f.name), seen) for f in fields(value)
        )
    return size


class CacheRecord(NamedTuple):
    payload: Any
    encoded: list[Encoded]
    size: int


class EpisodeView:
    def __init__(self, owner, representation: Literal["payload", "encoded"]):
        self.owner = owner
        self.representation = representation

    def __len__(self):
        return len(self.owner.entries)

    def __getitem__(self, index):
        payload, encoded = self.owner.get(index)
        return payload if self.representation == "payload" else encoded


class FileDemonstrations:
    """Only immutable public inputs are cached; never recurrent states."""

    def __init__(self, root, entries, device, limit_bytes=DEFAULT_CPU_CACHE_BYTES):
        self.root, self.entries, self.device = root, entries, device
        self.limit_bytes = limit_bytes
        self.masks = [e["supervised"] for e in entries]
        self.episodes, self.encoded = (
            EpisodeView(self, "payload"),
            EpisodeView(self, "encoded"),
        )
        self.lru = OrderedDict()
        self.resident_bytes = self.peak_bytes = 0
        self.counters = Counter()
        self.mode = "training"

    def _lookup(self, index):
        if index in self.lru:
            self.counters["cache_hits"] += 1
            self.lru.move_to_end(index)
            return self.lru[index]
        return None

    def _store(self, index, record: CacheRecord):
        while self.lru and self.resident_bytes + record.size > self.limit_bytes:
            _, previous = self.lru.popitem(last=False)
            self.resident_bytes -= previous.size
        if record.size <= self.limit_bytes:
            self.lru[index] = record
            self.resident_bytes += record.size
            self.peak_bytes = max(self.peak_bytes, self.resident_bytes)

    def get(self, index):
        cached = self._lookup(index)
        if cached is not None:
            return cached.payload, cached.encoded
        started = perf_counter()
        episode = load_episode(self.root / self.entries[index]["path"])
        self.counters["read_seconds"] += perf_counter() - started
        if len(episode["decisions"]) != len(self.masks[index]):
            raise ValueError("Episode length differs from collection index")
        if [r["supervised"] for r in episode["decisions"]] != self.masks[index]:
            raise ValueError("Episode label mask differs from collection index")
        started = perf_counter()
        encoded = encode_demonstration(episode)
        self.counters["encode_seconds"] += perf_counter() - started
        self.counters["cache_misses"] += 1
        size = memory_bytes((episode, encoded))
        record = CacheRecord(episode, encoded, size)
        self._store(index, record)
        return record.payload, record.encoded

    def chunks(self, size=DEFAULT_PREFIX_BLOCK_SIZE):
        return [
            Span(episode_index, start, min(start + size, len(entry["supervised"])))
            for episode_index, entry in enumerate(self.entries)
            for start in range(0, len(entry["supervised"]), size)
        ]

    def batch(self, spans):
        inputs, rows = [], []
        for episode_index, start, stop in spans:
            episode, encoded = self.get(episode_index)
            inputs.extend(encoded[start:stop])
            rows.extend(episode["decisions"][start:stop])
        field = "history_positions" if torch.is_grad_enabled() else "prefix_positions"
        if self.mode != "training":
            field = "diagnostic_positions"
        self.counters[field] += len(rows)
        labels = {
            "indices": torch.tensor(
                [decision_indices(r)[0] for r in rows], device=self.device
            ),
            "mask": torch.tensor(
                [r["supervised"] for r in rows], device=self.device, dtype=torch.bool
            ),
        }
        return (
            batch_inputs(inputs, self.device, non_blocking=True),
            labels,
            [stop - start for _, start, stop in spans],
        )

    def clear(self):
        self.lru.clear()
        self.resident_bytes = 0

    def statistics(self):
        return {
            **self.counters,
            "cached_bytes": self.resident_bytes,
            "peak_cached_bytes": self.peak_bytes,
            "cache_limit_bytes": self.limit_bytes,
        }


def prepare_episode(task):
    """Encode once; store fixed arrays and ragged legal-action arrays losslessly."""
    root, entry, destination = task
    from pathlib import Path

    root, destination = Path(root), Path(destination)
    started = perf_counter()
    episode = load_episode(root / entry["path"])
    encoded = encode_demonstration(episode)
    masks = np.asarray([r["supervised"] for r in episode["decisions"]], dtype=np.bool_)
    if masks.tolist() != entry["supervised"]:
        raise ValueError("Prepared input label mask differs from collection")
    counts = np.asarray([len(x.sources) for x in encoded], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    arrays = {
        name: np.stack([getattr(x, name) for x in encoded])
        for name in (
            field.name
            for field in ENCODED_FIELDS
            if field.name in ("cells", "globals", "previous")
        )
    }
    # Geometry is invariant within an episode. All other features remain per decision.
    if not all(np.array_equal(x.neighbors, encoded[0].neighbors) for x in encoded):
        raise ValueError("Episode geometry changed")
    arrays["neighbors"] = encoded[0].neighbors
    arrays.update(
        {
            name: np.concatenate([getattr(x, name) for x in encoded])
            for name in (
                field.name for field in ENCODED_FIELDS if field.axis == "actions"
            )
        }
    )
    arrays.update(
        offsets=offsets,
        indices=np.asarray(
            [decision_indices(r)[0] for r in episode["decisions"]], dtype=np.int64
        ),
        mask=masks,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
    os.replace(temporary, destination)
    # Conservative array/view overhead is known without recursively walking Python objects.
    size = (
        sum(a.nbytes for a in arrays.values())
        + len(encoded) * PREPARED_VIEW_BYTES_PER_DECISION
        + PREPARED_FIXED_OVERHEAD_BYTES
    )
    return {
        **entry,
        "prepared": destination.relative_to(root).as_posix(),
        "prepared_bytes": size,
        "preparation_seconds": perf_counter() - started,
    }


def prepare_inputs(root, entries, workers=4):
    """One bounded streaming preparation pass, never a second full CPU corpus."""
    directory = root / "prepared-inputs"
    index = directory / "index.json"
    if index.exists():
        saved = read_json(index)
        if saved["schema"] != 2 or [e["path"] for e in saved["entries"]] != [
            e["path"] for e in entries
        ]:
            raise ValueError("Prepared corpus differs")
        return saved
    started = perf_counter()
    prepared = []
    with pool(workers) as executor:
        tasks = [
            (str(root), e, str(directory / f"episode-{i:06}.npz"))
            for i, e in enumerate(entries)
        ]
        # map's iterator is consumed as workers finish; only a few episodes are materialized per worker.
        for i, result in enumerate(executor.map(prepare_episode, tasks)):
            prepared.append(result)
            if (i + 1) % 100 == 0:
                print(
                    f"Prepared immutable BC inputs {i + 1}/{len(entries)}", flush=True
                )
    result = {
        "schema": 2,
        "entries": prepared,
        "seconds": perf_counter() - started,
        "array_bytes_with_view_allowance": sum(e["prepared_bytes"] for e in prepared),
        "source": "public observations; same encoder; lossless numpy arrays; ragged legal lists",
    }
    write_json(index, result)
    return result


class RawEpisodeView:
    """Only diagnostic metrics need the raw action/type dictionaries."""

    def __init__(self, owner):
        self.owner = owner

    def __len__(self):
        return len(self.owner.entries)

    def __getitem__(self, index):
        if self.owner.raw_index != index:
            self.owner.raw_episode = load_episode(
                self.owner.root / self.owner.entries[index]["path"]
            )
            self.owner.raw_index = index
        return self.owner.raw_episode


class PreparedDemonstrations(FileDemonstrations):
    """Same batch inputs, labels and prefixes; no JSON/encoder/size walk in training."""

    def __init__(self, root, entries, device, limit_bytes=DEFAULT_CPU_CACHE_BYTES):
        super().__init__(root, entries, device, limit_bytes)
        self.episodes = RawEpisodeView(self)
        self.raw_index = None
        self.raw_episode = None

    def get(self, index):
        cached = self._lookup(index)
        if cached is not None:
            return cached.payload, cached.encoded
        started = perf_counter()
        with np.load(
            self.root / self.entries[index]["prepared"], allow_pickle=False
        ) as stored:
            arrays = {k: stored[k] for k in stored.files}
        self.counters["prepared_read_seconds"] += perf_counter() - started
        offsets = arrays["offsets"]
        encoded = []
        for i in range(len(arrays["mask"])):
            start, stop = int(offsets[i]), int(offsets[i + 1])
            values = {}
            for field in ENCODED_FIELDS:
                if field.name == "neighbors":
                    values[field.name] = arrays[field.name]
                elif field.axis == "actions":
                    values[field.name] = arrays[field.name][start:stop]
                else:
                    values[field.name] = arrays[field.name][i]
            encoded.append(Encoded(**values))
        if arrays["mask"].tolist() != self.masks[index]:
            raise ValueError("Prepared mask changed")
        size = self.entries[index]["prepared_bytes"]
        record = CacheRecord(arrays, encoded, size)
        self._store(index, record)
        self.counters["cache_misses"] += 1
        return record.payload, record.encoded

    def batch(self, spans):
        inputs = []
        indices = []
        masks = []
        for episode_index, start, stop in spans:
            arrays, encoded = self.get(episode_index)
            inputs.extend(encoded[start:stop])
            indices.extend(arrays["indices"][start:stop])
            masks.extend(arrays["mask"][start:stop])
        field = "history_positions" if torch.is_grad_enabled() else "prefix_positions"
        if self.mode != "training":
            field = "diagnostic_positions"
        self.counters[field] += len(inputs)
        labels = {
            "indices": torch.tensor(np.asarray(indices), device=self.device),
            "mask": torch.tensor(np.asarray(masks, dtype=np.bool_), device=self.device),
        }
        return (
            batch_inputs(inputs, self.device, non_blocking=True),
            labels,
            [stop - start for _, start, stop in spans],
        )

    def clear(self):
        super().clear()
        self.raw_episode = None
        self.raw_index = None
