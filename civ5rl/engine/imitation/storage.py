"""Bounded episode storage for BC; the existing sequence/loss math is unchanged."""

from __future__ import annotations

import os
import sys
from collections import Counter, OrderedDict
from dataclasses import fields, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Generic, Literal, NamedTuple, TypeVar

import numpy as np
import torch

from civ5rl.core.encoded_layout import ENCODED_FIELDS
from civ5rl.core.model import batch_inputs
from civ5rl.core.observation import Encoded, convert_globals_time_to_c2
from civ5rl.core.schema import (
    ACTION_DIM,
    GLOBAL_DIM,
    HEX_DIRECTIONS,
    PREV_ACTION_DIM,
    TILE_DIM,
)
from civ5rl.core.variants import PolicyVariant
from civ5rl.engine.imitation.core import decision_indices, encode_demonstration
from civ5rl.engine.runtime.sequence import (
    DEFAULT_CPU_CACHE_BYTES,
    DEFAULT_PREFIX_BLOCK_SIZE,
    PREPARED_FIXED_OVERHEAD_BYTES,
    PREPARED_VIEW_BYTES_PER_DECISION,
    Span,
)
from civ5rl.engine.runtime.shared import pool
from civ5rl.provenance.common import ROOT, atomic_save_npz, read_gzip_json, read_json, write_json


def _resolve_episode_path(root, path):
    root = Path(root)
    candidate = root / path
    if candidate.is_file():
        return candidate
    fallback = ROOT / path
    if fallback.is_file():
        return fallback
    return candidate


def _entry_key(entry):
    if "entry_id" in entry:
        return (
            entry["entry_id"],
            entry.get("corpus_source"),
            entry.get("match_id"),
            entry.get("side"),
            entry["path"],
        )
    return (entry["path"],)


def _prepared_filename(index, entry):
    if "corpus_source" in entry:
        source = str(entry["corpus_source"])
        match_id = int(entry["match_id"])
        side = int(entry["side"])
        return f"episode-{index:06d}-{source}-match-{match_id:06d}-side-{side}.npz"
    return f"episode-{index:06d}.npz"


def memory_bytes(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    size = sys.getsizeof(value)
    if isinstance(value, np.ndarray):
        return size + (value.nbytes if value.base is not None else 0)
    if isinstance(value, dict):
        return size + sum(memory_bytes(k, seen) + memory_bytes(v, seen) for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return size + sum(memory_bytes(v, seen) for v in value)
    if is_dataclass(value):
        return size + sum(memory_bytes(getattr(value, f.name), seen) for f in fields(value))
    return size


class CacheRecord(NamedTuple):
    payload: Any
    encoded: list[Encoded] | None
    size: int


_Key = TypeVar("_Key")
_Value = TypeVar("_Value")


class ByteBoundedLRU(Generic[_Key, _Value]):
    """Keep the newest entries within a caller-supplied byte budget."""

    def __init__(self, limit_bytes: int):
        self.limit_bytes = limit_bytes
        self.resident_bytes = self.peak_bytes = 0
        self._entries: OrderedDict[_Key, tuple[_Value, int]] = OrderedDict()

    def __contains__(self, key: _Key) -> bool:
        return key in self._entries

    def get(self, key: _Key) -> _Value | None:
        if key not in self._entries:
            return None
        self._entries.move_to_end(key)
        return self._entries[key][0]

    def store(self, key: _Key, value: _Value, size: int) -> None:
        previous = self._entries.pop(key, None)
        if previous is not None:
            self.resident_bytes -= previous[1]
        while self._entries and self.resident_bytes + size > self.limit_bytes:
            _, (_, evicted_bytes) = self._entries.popitem(last=False)
            self.resident_bytes -= evicted_bytes
        if size <= self.limit_bytes:
            self._entries[key] = (value, size)
            self.resident_bytes += size
            self.peak_bytes = max(self.peak_bytes, self.resident_bytes)

    def clear(self) -> None:
        self._entries.clear()
        self.resident_bytes = 0


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
        self.root, self.entries, self.device = Path(root), entries, device
        self.masks = [e["supervised"] for e in entries]
        self.episodes, self.encoded = (
            EpisodeView(self, "payload"),
            EpisodeView(self, "encoded"),
        )
        self.lru = ByteBoundedLRU[int, CacheRecord](limit_bytes)
        self.counters = Counter()
        self.mode = "training"
        self.spatial_cell_padding = 0

    def _lookup(self, index):
        cached = self.lru.get(index)
        if cached is not None:
            self.counters["cache_hits"] += 1
        return cached

    @property
    def resident_bytes(self):
        return self.lru.resident_bytes

    def get(self, index):
        cached = self._lookup(index)
        if cached is not None:
            return cached.payload, cached.encoded
        started = perf_counter()
        episode = read_gzip_json(_resolve_episode_path(self.root, self.entries[index]["path"]))
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
        self.lru.store(index, record, size)
        return record.payload, record.encoded

    def chunks(self, size=DEFAULT_PREFIX_BLOCK_SIZE):
        return [
            Span(episode_index, start, min(start + size, len(entry["supervised"])))
            for episode_index, entry in enumerate(self.entries)
            for start in range(0, len(entry["supervised"]), size)
        ]

    def _record_positions(self, count: int) -> None:
        field = "history_positions" if torch.is_grad_enabled() else "prefix_positions"
        if self.mode != "training":
            field = "diagnostic_positions"
        self.counters[field] += count

    def batch(self, spans):
        inputs, rows = [], []
        for episode_index, start, stop in spans:
            episode, encoded = self.get(episode_index)
            inputs.extend(encoded[start:stop])
            rows.extend(episode["decisions"][start:stop])
        self._record_positions(len(rows))
        labels = {
            "indices": torch.tensor([decision_indices(r)[0] for r in rows], device=self.device),
            "mask": torch.tensor([r["supervised"] for r in rows], device=self.device, dtype=torch.bool),
        }
        return (
            batch_inputs(inputs, self.device, non_blocking=True),
            labels,
            [stop - start for _, start, stop in spans],
        )

    def gather_spatial_inputs(self, spans):
        inputs = []
        for episode_index, start, stop in spans:
            _, encoded = self.get(episode_index)
            inputs.extend(encoded[start:stop])
        self._record_positions(len(inputs))
        return batch_inputs(inputs, self.device, non_blocking=True, spatial_only=True)

    def clear(self):
        self.lru.clear()

    def statistics(self):
        return {
            **self.counters,
            "cached_bytes": self.resident_bytes,
            "peak_cached_bytes": self.lru.peak_bytes,
            "cache_limit_bytes": self.lru.limit_bytes,
        }


def prepare_episode(task):
    """Encode once; store fixed arrays and ragged legal-action arrays losslessly."""
    root, entry, destination = task
    root, destination = Path(root), Path(destination)
    started = perf_counter()
    episode = read_gzip_json(_resolve_episode_path(root, entry["path"]))
    if "match_id" in entry and episode.get("match_id") != entry["match_id"]:
        raise ValueError("Prepared episode match_id differs from collection entry")
    if "side" in entry and episode.get("side") != entry["side"]:
        raise ValueError("Prepared episode side differs from collection entry")
    encoded = encode_demonstration(episode)
    masks = np.asarray([r["supervised"] for r in episode["decisions"]], dtype=np.bool_)
    if masks.tolist() != entry["supervised"]:
        raise ValueError("Prepared input label mask differs from collection")
    counts = np.asarray([len(x.sources) for x in encoded], dtype=np.int64)
    offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(counts)))
    arrays = {
        name: np.stack([getattr(x, name) for x in encoded])
        for name in (field.name for field in ENCODED_FIELDS if field.name in ("cells", "globals", "previous"))
    }
    # Geometry is invariant within an episode. All other features remain per decision.
    if not all(np.array_equal(x.neighbors, encoded[0].neighbors) for x in encoded):
        raise ValueError("Episode geometry changed")
    arrays["neighbors"] = encoded[0].neighbors
    arrays.update(
        {
            name: np.concatenate([getattr(x, name) for x in encoded])
            for name in (field.name for field in ENCODED_FIELDS if field.axis == "actions")
        }
    )
    arrays.update(
        offsets=offsets,
        indices=np.asarray([decision_indices(r)[0] for r in episode["decisions"]], dtype=np.int64),
        mask=masks,
    )
    atomic_save_npz(destination, **arrays)
    # Conservative array/view overhead is known without recursively walking Python objects.
    size = (
        sum(a.nbytes for a in arrays.values())
        + len(encoded) * PREPARED_VIEW_BYTES_PER_DECISION
        + PREPARED_FIXED_OVERHEAD_BYTES
    )
    prepared_rel = (
        destination.relative_to(root).as_posix() if destination.is_relative_to(root) else destination.name
    )
    return {
        **entry,
        "prepared": prepared_rel,
        "prepared_bytes": size,
        "preparation_seconds": perf_counter() - started,
    }


def _prepared_directory(root, scratch_dir=None):
    root = Path(root)
    override = scratch_dir or os.environ.get("CIV5RL_SCRATCH_DIR")
    if override:
        return Path(override) / root.resolve().name / "prepared-inputs"
    return root / "prepared-inputs"


def prepare_inputs(root, entries, workers=4, *, scratch_dir=None):
    """One bounded streaming preparation pass, never a second full CPU corpus."""
    root = Path(root)
    directory = _prepared_directory(root, scratch_dir)
    index = directory / "index.json"
    keys = [_entry_key(e) for e in entries]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate episode entry keys in BC corpus")
    if index.exists():
        saved = read_json(index)
        if saved["schema"] != 2 or [_entry_key(e) for e in saved["entries"]] != keys:
            raise ValueError("Prepared corpus differs")
        return saved
    started = perf_counter()
    prepared = []
    with pool(workers) as executor:
        tasks = [(str(root), e, str(directory / _prepared_filename(i, e))) for i, e in enumerate(entries)]
        # map's iterator is consumed as workers finish; only a few episodes are materialized per worker.
        for i, result in enumerate(executor.map(prepare_episode, tasks)):
            prepared.append(result)
            if (i + 1) % 100 == 0:
                print(f"Prepared immutable BC inputs {i + 1}/{len(entries)}", flush=True)
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
            self.owner.raw_episode = read_gzip_json(
                _resolve_episode_path(self.owner.root, self.owner.entries[index]["path"])
            )
            self.owner.raw_index = index
        return self.owner.raw_episode


def _read_npz_arrays(npz_path):
    started = perf_counter()
    with np.load(npz_path, allow_pickle=False) as stored:
        arrays = {k: stored[k] for k in stored.files}
    return arrays, perf_counter() - started


def _build_encoded_list(arrays):
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
    return encoded


def _pack_spatial_rows(fragments, total_rows, cell_padding, allocate):
    num_cells = max(cell_padding, max(int(arrays["cells"].shape[1]) for arrays, _, _ in fragments))
    values = {
        "cells": allocate((total_rows, num_cells, TILE_DIM), torch.float32),
        "cell_mask": allocate((total_rows, num_cells), torch.bool),
        "neighbors": allocate((total_rows, num_cells, HEX_DIRECTIONS), torch.int64),
        "globals": allocate((total_rows, GLOBAL_DIM), torch.float32),
        "previous": allocate((total_rows, PREV_ACTION_DIM), torch.float32),
    }
    cells, cell_mask, neighbors, global_features, previous = (value.numpy() for value in values.values())

    offset = 0
    for arrays, start, stop in fragments:
        cell_count = arrays["cells"].shape[1]
        end = offset + stop - start
        cells[offset:end, :cell_count] = arrays["cells"][start:stop]
        neighbors[offset:end, :cell_count] = arrays["neighbors"]
        cell_mask[offset:end, :cell_count] = True
        if cell_count < num_cells:
            cells[offset:end, cell_count:] = 0.0
            neighbors[offset:end, cell_count:] = -1
            cell_mask[offset:end, cell_count:] = False
        global_features[offset:end] = arrays["globals"][start:stop]
        previous[offset:end] = arrays["previous"][start:stop]
        offset = end
    return values


def _pack_action_rows(fragments, total_rows, allocate):
    num_actions = max(
        int((arrays["offsets"][start + 1 : stop + 1] - arrays["offsets"][start:stop]).max())
        for arrays, start, stop in fragments
    )
    sources = allocate((total_rows, num_actions), torch.int64)
    targets = allocate((total_rows, num_actions), torch.int64)
    action_mask = allocate((total_rows, num_actions), torch.bool)
    features = allocate((total_rows, num_actions, ACTION_DIM), torch.float32)
    indices = allocate((total_rows,), torch.int64)
    supervision = allocate((total_rows,), torch.bool)

    sources_view = sources.numpy()
    targets_view = targets.numpy()
    action_mask_view = action_mask.numpy()
    features_view = features.numpy()
    indices_view = indices.numpy()
    supervision_view = supervision.numpy()

    sources_view.fill(-1)
    targets_view.fill(-1)
    action_mask_view.fill(False)
    features_view.fill(0.0)

    offset = 0
    for arrays, start, stop in fragments:
        end = offset + stop - start
        indices_view[offset:end] = arrays["indices"][start:stop]
        supervision_view[offset:end] = arrays["mask"][start:stop]
        action_offsets = arrays["offsets"]
        episode_sources = arrays["sources"]
        episode_targets = arrays["targets"]
        episode_features = arrays["action_features"]
        for row, position in enumerate(range(start, stop), start=offset):
            action_start, action_stop = int(action_offsets[position]), int(action_offsets[position + 1])
            count = action_stop - action_start
            if count:
                sources_view[row, :count] = episode_sources[action_start:action_stop]
                targets_view[row, :count] = episode_targets[action_start:action_stop]
                action_mask_view[row, :count] = True
                features_view[row, :count] = episode_features[action_start:action_stop]
        offset = end

    values = {
        "sources": sources,
        "targets": targets,
        "action_features": features,
        "action_mask": action_mask,
    }
    return values, {"indices": indices, "mask": supervision}


class PreparedDemonstrations(FileDemonstrations):
    """Same batch inputs, labels and prefixes; no JSON/encoder/size walk in training."""

    def __init__(
        self,
        root,
        entries,
        device,
        limit_bytes=DEFAULT_CPU_CACHE_BYTES,
        *,
        scratch_dir=None,
        variant: PolicyVariant | str = PolicyVariant.CONTACTS_GRU,
    ):
        super().__init__(root, entries, device, limit_bytes)
        self.variant = PolicyVariant(variant)
        self.prepared_dir = _prepared_directory(root, scratch_dir)
        self.episodes = RawEpisodeView(self)
        self.raw_index = None
        self.raw_episode = None
        self._inflight = {}

    def _npz_path(self, index):
        rel = self.entries[index]["prepared"]
        npz_path = self.root / rel
        if not npz_path.is_file():
            npz_path = self.prepared_dir / Path(rel).name
        return npz_path

    def preload_episodes(self, indices, executor):
        for idx in dict.fromkeys(int(i) for i in indices):
            if idx not in self.lru and idx not in self._inflight:
                self._inflight[idx] = executor.submit(_read_npz_arrays, self._npz_path(idx))

    def get_arrays(self, index):
        cached = self._lookup(index)
        if cached is not None:
            return cached.payload
        inflight = self._inflight.pop(index, None)
        if inflight is not None:
            arrays, elapsed = inflight.result()
        else:
            arrays, elapsed = _read_npz_arrays(self._npz_path(index))
        self.counters["prepared_read_seconds"] += elapsed
        if arrays["mask"].tolist() != self.masks[index]:
            raise ValueError("Prepared mask changed")
        if self.variant == PolicyVariant.CONTACTS_GRU_C2:
            arrays = {
                **arrays,
                "globals": convert_globals_time_to_c2(arrays["globals"]),
            }
        size = self.entries[index]["prepared_bytes"]
        self.lru.store(index, CacheRecord(arrays, None, size), size)
        self.counters["cache_misses"] += 1
        return arrays

    def get(self, index):
        cached = self._lookup(index)
        if cached is not None:
            if cached.encoded is None:
                built = CacheRecord(cached.payload, _build_encoded_list(cached.payload), cached.size)
                self.lru.store(index, built, built.size)
                return built.payload, built.encoded
            return cached.payload, cached.encoded
        arrays = self.get_arrays(index)
        encoded = _build_encoded_list(arrays)
        cached = self.lru.get(index)
        if cached is not None:
            self.lru.store(index, cached._replace(encoded=encoded), cached.size)
        return arrays, encoded

    def pack_cpu_spans(
        self,
        spans,
        *,
        spatial_only: bool,
        pin: bool,
        cell_padding: int = 0,
        alloc=None,
    ):
        if not spans:
            raise ValueError("Each batch item must contain at least one cell")
        fragments = [(self.get_arrays(int(episode)), int(start), int(stop)) for episode, start, stop in spans]
        lengths = [stop - start for _, start, stop in fragments]
        total_rows = sum(lengths)
        if total_rows <= 0 or any(arrays["cells"].shape[1] == 0 for arrays, _, _ in fragments):
            raise ValueError("Each batch item must contain at least one cell")

        def allocate(shape, dtype):
            if alloc is not None:
                return alloc(shape, dtype)
            return torch.empty(shape, dtype=dtype, pin_memory=pin)

        values = _pack_spatial_rows(fragments, total_rows, int(cell_padding), allocate)
        if spatial_only:
            return values, None, lengths
        actions, labels = _pack_action_rows(fragments, total_rows, allocate)
        return {**values, **actions}, labels, lengths

    def gather_spatial_inputs(self, spans):
        is_cuda = str(self.device).startswith("cuda")
        cpu_values, _, lengths = self.pack_cpu_spans(
            spans,
            spatial_only=True,
            pin=is_cuda,
            cell_padding=self.spatial_cell_padding,
        )
        self._record_positions(sum(lengths))
        return {k: v.to(self.device, non_blocking=is_cuda) for k, v in cpu_values.items()}

    def batch(self, spans):
        is_cuda = str(self.device).startswith("cuda")
        cpu_values, cpu_labels, lengths = self.pack_cpu_spans(
            spans,
            spatial_only=False,
            pin=is_cuda,
            cell_padding=self.spatial_cell_padding,
        )
        self._record_positions(sum(lengths))
        return (
            {k: v.to(self.device, non_blocking=is_cuda) for k, v in cpu_values.items()},
            {k: v.to(self.device, non_blocking=is_cuda) for k, v in cpu_labels.items()},
            lengths,
        )

    def clear(self):
        super().clear()
        self._inflight.clear()
        self.raw_episode = None
        self.raw_index = None
