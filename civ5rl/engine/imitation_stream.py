"""Bounded episode storage for BC; the existing sequence/loss math is unchanged."""

import os
import sys
from collections import Counter, OrderedDict
from dataclasses import fields, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, NamedTuple

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
from civ5rl.engine.constants import (
    DEFAULT_CPU_CACHE_BYTES,
    DEFAULT_PREFIX_BLOCK_SIZE,
    PREPARED_FIXED_OVERHEAD_BYTES,
    PREPARED_VIEW_BYTES_PER_DECISION,
)
from civ5rl.engine.imitation import decision_indices, encode_demonstration
from civ5rl.engine.pool import pool
from civ5rl.engine.spans import Span
from civ5rl.protocols.imitation import load_episode
from civ5rl.provenance.common import ROOT, read_json, write_json


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
    encoded: list[Encoded] | None
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
        self.root, self.entries, self.device = Path(root), entries, device
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
        self.spatial_cell_padding = 0

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
        episode = load_episode(
            _resolve_episode_path(self.root, self.entries[index]["path"])
        )
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

    def gather_spatial_inputs(self, spans):
        inputs = []
        for episode_index, start, stop in spans:
            _, encoded = self.get(episode_index)
            inputs.extend(encoded[start:stop])
        field = "history_positions" if torch.is_grad_enabled() else "prefix_positions"
        if self.mode != "training":
            field = "diagnostic_positions"
        self.counters[field] += len(inputs)
        return batch_inputs(
            inputs, self.device, non_blocking=True, spatial_only=True
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
    root, destination = Path(root), Path(destination)
    started = perf_counter()
    episode = load_episode(_resolve_episode_path(root, entry["path"]))
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
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, destination)
    # Conservative array/view overhead is known without recursively walking Python objects.
    size = (
        sum(a.nbytes for a in arrays.values())
        + len(encoded) * PREPARED_VIEW_BYTES_PER_DECISION
        + PREPARED_FIXED_OVERHEAD_BYTES
    )
    prepared_rel = (
        destination.relative_to(root).as_posix()
        if destination.is_relative_to(root)
        else destination.name
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
        tasks = [
            (str(root), e, str(directory / _prepared_filename(i, e)))
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
                _resolve_episode_path(
                    self.owner.root, self.owner.entries[index]["path"]
                )
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
                self._inflight[idx] = executor.submit(
                    _read_npz_arrays, self._npz_path(idx)
                )

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
        self._store(index, CacheRecord(arrays, None, size))
        self.counters["cache_misses"] += 1
        return arrays

    def get(self, index):
        cached = self._lookup(index)
        if cached is not None:
            if cached.encoded is None:
                built = CacheRecord(
                    cached.payload, _build_encoded_list(cached.payload), cached.size
                )
                self.lru[index] = built
                return built.payload, built.encoded
            return cached.payload, cached.encoded
        arrays = self.get_arrays(index)
        encoded = _build_encoded_list(arrays)
        if index in self.lru:
            self.lru[index] = CacheRecord(
                arrays, encoded, self.lru[index].size
            )
        return arrays, encoded

    def _pack_cpu_spans(
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
        ep_arrays = [self.get_arrays(int(ep)) for ep, _, _ in spans]
        lengths = [int(stop - start) for _, start, stop in spans]
        total_rows = sum(lengths)
        if total_rows <= 0 or any(a["cells"].shape[1] == 0 for a in ep_arrays):
            raise ValueError("Each batch item must contain at least one cell")
        num_cells = max(
            max(int(a["cells"].shape[1]) for a in ep_arrays), int(cell_padding)
        )

        def _empty(shape, dtype):
            if alloc is not None:
                return alloc(shape, dtype)
            return torch.empty(shape, dtype=dtype, pin_memory=pin)

        cells_t = _empty((total_rows, num_cells, TILE_DIM), torch.float32)
        cell_mask_t = _empty((total_rows, num_cells), torch.bool)
        neighbors_t = _empty((total_rows, num_cells, HEX_DIRECTIONS), torch.int64)
        globals_t = _empty((total_rows, GLOBAL_DIM), torch.float32)
        previous_t = _empty((total_rows, PREV_ACTION_DIM), torch.float32)

        cells_np = cells_t.numpy()
        cell_mask_np = cell_mask_t.numpy()
        neighbors_np = neighbors_t.numpy()
        globals_np = globals_t.numpy()
        previous_np = previous_t.numpy()

        row = 0
        for a, (_, start, stop) in zip(ep_arrays, spans):
            start, stop = int(start), int(stop)
            length = stop - start
            c = int(a["cells"].shape[1])
            end_row = row + length
            cells_np[row:end_row, :c] = a["cells"][start:stop]
            if c < num_cells:
                cells_np[row:end_row, c:] = 0.0
            neighbors_np[row:end_row, :c] = a["neighbors"]
            if c < num_cells:
                neighbors_np[row:end_row, c:] = -1
            cell_mask_np[row:end_row, :c] = True
            if c < num_cells:
                cell_mask_np[row:end_row, c:] = False
            globals_np[row:end_row] = a["globals"][start:stop]
            previous_np[row:end_row] = a["previous"][start:stop]
            row = end_row

        values = {
            "cells": cells_t,
            "cell_mask": cell_mask_t,
            "neighbors": neighbors_t,
            "globals": globals_t,
            "previous": previous_t,
        }
        if spatial_only:
            return values, None, lengths

        num_actions = max(
            int(
                (
                    a["offsets"][int(start) + 1 : int(stop) + 1]
                    - a["offsets"][int(start) : int(stop)]
                ).max()
            )
            for a, (_, start, stop) in zip(ep_arrays, spans)
        )
        sources_t = _empty((total_rows, num_actions), torch.int64)
        targets_t = _empty((total_rows, num_actions), torch.int64)
        action_mask_t = _empty((total_rows, num_actions), torch.bool)
        features_t = _empty((total_rows, num_actions, ACTION_DIM), torch.float32)
        indices_t = _empty((total_rows,), torch.int64)
        mask_t = _empty((total_rows,), torch.bool)

        sources_np = sources_t.numpy()
        targets_np = targets_t.numpy()
        action_mask_np = action_mask_t.numpy()
        features_np = features_t.numpy()
        indices_np = indices_t.numpy()
        mask_np = mask_t.numpy()

        sources_np.fill(-1)
        targets_np.fill(-1)
        action_mask_np.fill(False)
        features_np.fill(0.0)

        row = 0
        for a, (_, start, stop) in zip(ep_arrays, spans):
            start, stop = int(start), int(stop)
            length = stop - start
            end_row = row + length
            indices_np[row:end_row] = a["indices"][start:stop]
            mask_np[row:end_row] = a["mask"][start:stop]
            offs = a["offsets"]
            src_ep = a["sources"]
            tgt_ep = a["targets"]
            feat_ep = a["action_features"]
            r = row
            for t in range(start, stop):
                a0 = int(offs[t])
                a1 = int(offs[t + 1])
                cnt = a1 - a0
                if cnt:
                    sources_np[r, :cnt] = src_ep[a0:a1]
                    targets_np[r, :cnt] = tgt_ep[a0:a1]
                    action_mask_np[r, :cnt] = True
                    features_np[r, :cnt] = feat_ep[a0:a1]
                r += 1
            row = end_row

        values.update(
            sources=sources_t,
            targets=targets_t,
            action_features=features_t,
            action_mask=action_mask_t,
        )
        labels = {"indices": indices_t, "mask": mask_t}
        return values, labels, lengths

    def _record_positions(self, count: int) -> None:
        field = "history_positions" if torch.is_grad_enabled() else "prefix_positions"
        if self.mode != "training":
            field = "diagnostic_positions"
        self.counters[field] += count

    def gather_spatial_inputs(self, spans):
        is_cuda = str(self.device).startswith("cuda")
        cpu_values, _, lengths = self._pack_cpu_spans(
            spans,
            spatial_only=True,
            pin=is_cuda,
            cell_padding=self.spatial_cell_padding,
        )
        self._record_positions(sum(lengths))
        return {
            k: v.to(self.device, non_blocking=is_cuda) for k, v in cpu_values.items()
        }

    def batch(self, spans):
        is_cuda = str(self.device).startswith("cuda")
        cpu_values, cpu_labels, lengths = self._pack_cpu_spans(
            spans,
            spatial_only=False,
            pin=is_cuda,
            cell_padding=self.spatial_cell_padding,
        )
        self._record_positions(sum(lengths))
        return (
            {
                k: v.to(self.device, non_blocking=is_cuda)
                for k, v in cpu_values.items()
            },
            {
                k: v.to(self.device, non_blocking=is_cuda)
                for k, v in cpu_labels.items()
            },
            lengths,
        )

    def clear(self):
        super().clear()
        self._inflight.clear()
        self.raw_episode = None
        self.raw_index = None
