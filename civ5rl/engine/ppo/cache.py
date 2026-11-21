"""Prepare rollout labels and choose placement for immutable PPO replay inputs."""

from __future__ import annotations

import numpy as np
import torch

from civ5rl.engine.ppo import replay
from civ5rl.engine.ppo.objective import encode_episode, episode_returns, targets
from civ5rl.engine.runtime.sequence import (
    DEFAULT_GPU_RESERVE_BYTES,
    DEFAULT_PINNED_REPLAY_BYTES,
    SequenceBuffer,
    allocation_with_reserve,
)


def _part_tensor_bytes(data, labels, slots) -> int:
    return sum(
        t.numel() * t.element_size()
        for t in (
            *data.values(),
            *labels.values(),
            *((slots,) if slots is not None else ()),
        )
    )


class RolloutCache(SequenceBuffer):
    """Packed rollout inputs, PPO targets and optional replay staging."""

    def __init__(
        self,
        episodes,
        variant,
        device,
        *,
        resident=True,
        cpu_labels=False,
        gamma=1.0,
        gae_lambda=1.0,
    ):
        self.episodes, self.device = episodes, device
        encoded = [encode_episode(episode, variant) for episode in episodes]
        super().__init__(
            device=device, **self._pack_encoded(encoded, device, resident=resident, ragged_actions=True)
        )
        if self.data is not None:
            self.encoded = None
            del encoded
        self.resident = resident
        self.cpu_labels = cpu_labels
        self._prepare_labels(episodes, gamma=gamma, gae_lambda=gae_lambda)
        self._prepare_behavior_distribution(episodes)
        self.resident_bytes = _part_tensor_bytes(self.data or {}, self.labels, None)
        self.spatial_cell_padding = 0
        self.replay_inputs = False
        self._replay_plan: replay.ReplayPlan | None = None
        self._replay_h2d_stream = None
        self._replay_sums_runner = None
        self._categorical_distance_runner = None
        self._pinned_replay = None
        self._pinned_replay_limit = 0
        self.pinned_replay_bytes = 0
        self.pinned_replay_hits = 0
        self.replay_memory_admission = None

    def _prepare_labels(self, episodes, *, gamma, gae_lambda):
        label_device = "cpu" if self.cpu_labels else self.device
        returns = episode_returns(episodes, gamma=gamma)
        advantages = targets(episodes, gamma=gamma, gae_lambda=gae_lambda)
        self.labels = {
            "indices": torch.tensor([v for ep in episodes for v in ep["chosen"]], device=label_device),
            "old_logp": torch.tensor([v for ep in episodes for v in ep["old_logp"]], device=label_device),
            "returns": torch.tensor(
                np.concatenate(returns),
                dtype=torch.float32,
                device=label_device,
            ),
            "advantages": torch.tensor(np.concatenate(advantages), device=label_device),
        }

    def _prepare_behavior_distribution(self, episodes):
        has_full = ["old_log_probs" in ep for ep in episodes]
        if any(has_full) and not all(has_full):
            raise ValueError("Mixed complete/incomplete behavior distributions")
        self.has_behavior_distribution = all(has_full)
        self.ragged_behavior = None
        if not self.has_behavior_distribution:
            return
        rows = [values for ep in episodes for values in ep["old_log_probs"]]
        decision_count = int(self.offsets[-1])
        if self.cpu_labels:
            self._prepare_cpu_behavior(rows, decision_count)
        elif self.data is not None:
            self._prepare_resident_behavior(episodes, rows, decision_count)
        else:
            self._prepare_dense_behavior(rows, decision_count)

    def _prepare_cpu_behavior(self, rows, decision_count):
        """Keep variable-width rows on the host and normalize once for replay."""
        self.ragged_behavior = rows
        if len(self.ragged_behavior) != decision_count:
            raise ValueError("Behavior distribution count differs from decisions")
        normalizers = np.empty(decision_count, dtype=np.float64)
        for i, (values, count) in enumerate(zip(self.ragged_behavior, self.action_counts)):
            if (
                len(values) != count
                or not np.isfinite(values).all()
                or float(values[int(self.labels["indices"][i])]) != float(self.labels["old_logp"][i])
                or not np.isclose(np.exp(values).sum(), 1.0, atol=1e-5, rtol=0)
            ):
                raise ValueError("Invalid ragged behavior distribution")
            normalizers[i] = np.log(np.exp(values, dtype=np.float64).sum())
        self.labels["behavior_log_normalizer"] = torch.from_numpy(normalizers)

    def _prepare_resident_behavior(self, episodes, behavior, decision_count):
        """Store legal behavior entries beside packed inputs on the device."""
        if len(behavior) != decision_count or any(
            len(values) != count or not np.isfinite(values).all()
            for values, count in zip(behavior, self.action_counts)
        ):
            raise ValueError("Invalid resident behavior distribution")
        starts = np.cumsum(np.r_[0, self.action_counts[:-1]])
        values = np.concatenate([*behavior, np.array([-np.inf], dtype=np.float32)], dtype=np.float32)
        totals = np.add.reduceat(np.exp(values[:-1], dtype=np.float64), starts)
        if not np.allclose(totals, 1.0, atol=1e-5, rtol=0):
            raise ValueError("Behavior probabilities do not sum to one")
        chosen = np.fromiter((i for ep in episodes for i in ep["chosen"]), dtype=np.int64)
        if not np.array_equal(
            values[starts + chosen],
            np.asarray([v for ep in episodes for v in ep["old_logp"]], dtype=np.float32),
        ):
            raise ValueError("Recorded chosen logp differs from behavior distribution")
        self.data["old_log_probs"] = torch.from_numpy(values).to(self.device)
        self.labels["behavior_log_normalizer"] = torch.from_numpy(np.log(totals)).to(self.device)

    def _prepare_dense_behavior(self, rows, decision_count):
        """Prepare padded labels for nonresident inputs with device labels."""
        width = int(self.action_counts.max())
        behavior = np.full((decision_count, width), -np.inf, np.float32)
        if len(rows) != decision_count:
            raise ValueError("Behavior distribution count differs from decisions")
        for index, (values, count) in enumerate(zip(rows, self.action_counts)):
            if len(values) != count or not np.isfinite(values).all():
                raise ValueError("Invalid behavior action probabilities")
            behavior[index, :count] = values
        behavior = torch.tensor(behavior, device=self.device)
        selected = behavior.gather(1, self.labels["indices"][:, None])[:, 0]
        if not torch.equal(selected, self.labels["old_logp"]):
            raise ValueError("Recorded chosen logp differs from behavior distribution")
        if not torch.allclose(
            behavior.exp().sum(-1),
            torch.ones(decision_count, device=self.device),
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError("Behavior probabilities do not sum to one")
        self.labels["old_log_probs"] = behavior
        # Padding is -inf. The normalizer is independent of the minibatch width.
        self.labels["behavior_log_normalizer"] = torch.cat(
            [part.double().logsumexp(-1) for part in behavior.split(256)]
        )

    @property
    def replay_mode(self) -> replay.ReplayMode:
        return self._replay_plan.mode if self._replay_plan is not None else "on_demand"

    @property
    def cached_replay_bytes(self) -> int:
        if self._replay_plan is not None and self._replay_plan.mode == "device":
            return self._replay_plan.cached_bytes
        return 0

    def enable_compiled_replay_sums(self):
        if str(self.device).startswith("cuda") and self._replay_sums_runner is None:
            self._replay_sums_runner = replay.compiled_replay_sums()
            if self.has_behavior_distribution:
                self._categorical_distance_runner = replay.compiled_categorical_distance()

    def enable_cached_replay(self):
        """Cache inputs only; GRU state and logits always use current weights."""
        if self.data is None:
            return
        self.enable_compiled_replay_sums()
        self.replay_inputs = True
        self._replay_plan = None

    def enable_pinned_replay(self, limit_bytes=DEFAULT_PINNED_REPLAY_BYTES):
        """Keep immutable replay inputs on CPU; stream each part to the GPU."""
        if self.data is None and self.cpu_labels and str(self.device).startswith("cuda"):
            self._pinned_replay = {}
            self._pinned_replay_limit = limit_bytes
            self._replay_plan = None

    def _build_host_groups(self, sequences_per_batch, *, cpu_groups=None):
        if cpu_groups is None:
            cpu_groups = self._collect_cpu_replay_groups(sequences_per_batch)
        host_groups = []
        for group_size, parts in cpu_groups:
            host_parts = []
            for key, data, labels, lengths, slots, free_count, part_bytes in parts:
                is_pinned = False
                cached = self._pinned_replay.get(key) if self._pinned_replay is not None else None
                if cached is not None:
                    data, labels, lengths = cached
                    is_pinned = True
                elif (
                    self._pinned_replay is not None
                    and self.pinned_replay_bytes + part_bytes <= self._pinned_replay_limit
                ):
                    data = {k: v.pin_memory() for k, v in data.items()}
                    labels = {k: v.pin_memory() for k, v in labels.items()}
                    if slots is not None:
                        slots = slots.pin_memory()
                    self._pinned_replay[key] = data, labels, lengths
                    self.pinned_replay_bytes += part_bytes
                    is_pinned = True
                host_parts.append(replay.HostReplayPart(data, labels, lengths, slots, free_count, is_pinned))
            host_groups.append((group_size, host_parts))
        return host_groups

    def _collect_cpu_replay_groups(self, sequences_per_batch):
        cpu_groups = []
        for group_size, specs in replay.replay_group_specs(self, sequences_per_batch):
            parts = []
            for spans, slots_list, free_count in specs:
                data, labels, lengths = self.batch(spans, device="cpu", min_cells=self.spatial_cell_padding)
                slots = None if slots_list is None else torch.tensor(slots_list, dtype=torch.long)
                part_bytes = _part_tensor_bytes(data, labels, slots)
                parts.append((tuple(spans), data, labels, lengths, slots, free_count, part_bytes))
            cpu_groups.append((group_size, parts))
        return cpu_groups

    def prepare_replay_plan(
        self,
        sequences_per_batch=8,
        *,
        reserve_bytes=2 * DEFAULT_GPU_RESERVE_BYTES,
        prefer_device=False,
    ) -> replay.ReplayPlan:
        """Try real cached replay allocations; fall back only on CUDA OOM."""
        if (
            self._replay_plan is not None
            and self._replay_plan.sequences_per_batch == sequences_per_batch
            and (not prefer_device or self._replay_plan.mode == "device")
        ):
            return self._replay_plan

        specs_by_group = list(replay.replay_group_specs(self, sequences_per_batch))
        if self.replay_inputs or (prefer_device and str(self.device).startswith("cuda")):
            device_plan = self._try_device_plan(specs_by_group, sequences_per_batch, reserve_bytes)
            if device_plan is not None:
                self._replay_plan = device_plan
                return device_plan

        if self._pinned_replay is not None:
            if (
                self._replay_plan is not None
                and self._replay_plan.sequences_per_batch == sequences_per_batch
                and self._replay_plan.mode == "host"
            ):
                self._replay_plan._count_host_hits = True
                return self._replay_plan
            host_groups = self._build_host_groups(sequences_per_batch)
            self._replay_plan = replay.ReplayPlan(
                mode="host",
                sequences_per_batch=sequences_per_batch,
                groups=host_groups,
                _count_host_hits=prefer_device,
            )
            return self._replay_plan

        self._replay_plan = replay.ReplayPlan(
            mode="on_demand",
            sequences_per_batch=sequences_per_batch,
            groups=specs_by_group,
        )
        return self._replay_plan

    def _try_device_plan(self, specs_by_group, sequences_per_batch, reserve_bytes):
        """Allocate actual replay tensors before admitting a resident plan."""
        device_groups, parts, part = [], None, None
        total_bytes = 0
        try:
            with allocation_with_reserve(self.device, reserve_bytes) as admission:
                self.replay_memory_admission = admission
                for group_size, specs in specs_by_group:
                    parts = []
                    device_groups.append((group_size, parts))
                    for part in replay.replay_parts(self, specs):
                        total_bytes += _part_tensor_bytes(part.data, part.labels, part.slots)
                        parts.append(part)
        except torch.cuda.OutOfMemoryError:
            # No exception/traceback is retained in the fallback plan.
            # Drop both completed groups and the most recent local aliases.
            device_groups.clear()
            if parts is not None:
                parts.clear()
            part = None
            self.replay_inputs = False
        else:
            self.replay_inputs = True
            return replay.ReplayPlan(
                mode="device",
                sequences_per_batch=sequences_per_batch,
                groups=device_groups,
                cached_bytes=total_bytes,
            )
        return None

    def stage_gpu_replay(self, sequences_per_batch=8, *, reserve_bytes=2 * DEFAULT_GPU_RESERVE_BYTES):
        """Stage pre-packed immutable replay parts in VRAM when reserve permits."""
        if not str(self.device).startswith("cuda"):
            return None
        return self.prepare_replay_plan(
            sequences_per_batch,
            reserve_bytes=reserve_bytes,
            prefer_device=True,
        )

    def replay_batch(self, spans):
        if self._pinned_replay is None:
            return self.batch(spans, min_cells=self.spatial_cell_padding)
        key = tuple(spans)
        part = self._pinned_replay.get(key)
        if part is None:
            data, labels, lengths = self.batch(spans, device="cpu", min_cells=self.spatial_cell_padding)
            size = sum(v.numel() * v.element_size() for v in (*data.values(), *labels.values()))
            if self.pinned_replay_bytes + size <= self._pinned_replay_limit:
                data = {k: v.pin_memory() for k, v in data.items()}
                labels = {k: v.pin_memory() for k, v in labels.items()}
                self._pinned_replay[key] = data, labels, lengths
                self.pinned_replay_bytes += size
        else:
            data, labels, lengths = part
            self.pinned_replay_hits += 1
        return (
            {k: v.to(self.device, non_blocking=True) for k, v in data.items()},
            {k: v.to(self.device, non_blocking=True) for k, v in labels.items()},
            lengths,
        )

    def batch(self, spans, *, device=None, include_behavior=True, min_cells=0):
        device = self.device if device is None else device
        indices = (
            torch.tensor(
                [
                    i
                    for episode, start, stop in spans
                    for i in range(
                        int(self.offsets[episode]) + start,
                        int(self.offsets[episode]) + stop,
                    )
                ]
            )
            if self.cpu_labels
            else self.indices(spans)
        )
        data = self.gather_inputs(
            spans, indices, device=device, min_cells=min_cells, include_behavior=include_behavior
        )
        if min_cells and "cells" in data and data["cells"].shape[1] < min_cells:
            extra = min_cells - data["cells"].shape[1]
            data = {
                **data,
                "cells": torch.nn.functional.pad(data["cells"], (0, 0, 0, extra)),
                "neighbors": torch.nn.functional.pad(data["neighbors"], (0, 0, 0, extra), value=-1),
                "cell_mask": torch.nn.functional.pad(data["cell_mask"], (0, extra), value=False),
            }
        selected = indices.cpu() if self.cpu_labels else indices
        labels = {}
        if "old_log_probs" in data:
            labels["old_log_probs"] = data.pop("old_log_probs")
        for key, value in self.labels.items():
            if key in ("old_log_probs", "behavior_log_normalizer") and not include_behavior:
                continue
            if key == "old_log_probs":
                value = value[:, : data["action_mask"].shape[1]]
            labels[key] = value.index_select(0, selected).to(device)
        if include_behavior and self.ragged_behavior is not None:
            rows = [self.ragged_behavior[i] for i in selected.tolist()]
            behavior = np.full((len(rows), data["action_mask"].shape[1]), -np.inf, np.float32)
            for i, values in enumerate(rows):
                behavior[i, : len(values)] = values
            labels["old_log_probs"] = torch.from_numpy(behavior).to(device)
        return data, labels, [stop - start for _, start, stop in spans]
