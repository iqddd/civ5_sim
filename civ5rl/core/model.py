"""Hex message passing, army memory, and scores over public legal commands."""

from numbers import Integral

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from civ5rl.core.schema import (
    ACTION_DIM,
    ACTION_INPUT_DIM,
    ATTACK_PREVIEW_DIM,
    C2_ATTN_HEADS,
    C2_DISTANCE_BUCKETS,
    CONTEXT_INPUT_DIM,
    HEX_DIM,
    HEX_DIRECTIONS,
    HEX_MLP_DIM,
    HIDDEN_SIZE,
    MAX_MAP_AXIS,
    MAX_MAP_CELLS,
    MEMORY_MLP_DIM,
    OPPOSITE_HEX_OFFSET,
    SELF_AND_HEX_DIRECTIONS,
    TILE,
    TILE_DIM,
)
from civ5rl.core.spatial import StaticSpatialGraph, iter_spatial_blocks
from civ5rl.core.variants import PolicyVariant, variant_spec

GRU_GATES = 3


def batch_inputs(
    items,
    device,
    *,
    non_blocking=False,
    spatial_only=False,
    min_cells=0,
    compact_units=False,
    actions_only=False,
    index_dtype=np.int64,
):
    if not items or any(len(item.cells) == 0 for item in items):
        raise ValueError("Each batch item must contain at least one cell")
    batch_size = len(items)
    if actions_only and (spatial_only or compact_units):
        raise ValueError('Action-only batching cannot request spatial inputs')
    values = {}
    if not actions_only:
        num_cells = max(min_cells, max(len(item.cells) for item in items))
        cells = np.zeros((batch_size, num_cells, TILE_DIM), np.float32)
        masks = np.zeros((batch_size, num_cells), bool)
        neighbors = np.full((batch_size, num_cells, HEX_DIRECTIONS), -1, index_dtype)
        for i, x in enumerate(items):
            cell_count = len(x.cells)
            cells[i, :cell_count] = x.cells
            neighbors[i, :cell_count] = x.neighbors
            masks[i, :cell_count] = True
        values = {
            "cells": cells,
            "cell_mask": masks,
            "neighbors": neighbors,
            "globals": np.stack([x.globals for x in items]),
            "previous": np.stack([x.previous for x in items]),
        }
    if compact_units:
        # Only public features determine keys. Shape is chosen on the host,
        # avoiding nonzero()/item() synchronization in the CUDA forward.
        public = masks & ((cells[..., TILE["own_type"]] > 0).any(-1)
                          | (cells[..., TILE["enemy_type"]] > 0).any(-1)
                          | (cells[..., TILE["contact_count"]] > 0).any(-1))
        count = max(2, int(public.sum(-1).max()))
        capacity = 1 << (count - 1).bit_length()
        indices = np.full((batch_size, capacity), -1, np.int64)
        for i, row in enumerate(public):
            selected = np.flatnonzero(row)
            indices[i, :len(selected)] = selected
        values["unit_indices"] = indices
    if not spatial_only:
        num_actions = max(len(item.sources) for item in items)
        sources = np.full((batch_size, num_actions), -1, index_dtype)
        targets = sources.copy()
        action_mask = np.zeros((batch_size, num_actions), bool)
        features = np.zeros((batch_size, num_actions, ACTION_DIM), np.float32)
        has_previews = [item.attack_preview is not None for item in items]
        if any(has_previews) and not all(has_previews):
            raise ValueError("Mixed recorded/missing attack preview inputs")
        previews = (
            np.zeros((batch_size, num_actions, ATTACK_PREVIEW_DIM), np.float32)
            if all(has_previews)
            else None
        )
        for i, item in enumerate(items):
            action_count = len(item.sources)
            sources[i, :action_count] = item.sources
            targets[i, :action_count] = item.targets
            action_mask[i, :action_count] = True
            features[i, :action_count] = item.action_features
            if item.attack_preview is not None:
                previews[i, :action_count] = item.attack_preview
        values.update(
            sources=sources,
            targets=targets,
            action_features=features,
            action_mask=action_mask,
        )
        if previews is not None:
            values["attack_preview"] = previews
    return {k: torch.from_numpy(v).to(device, non_blocking=non_blocking) for k, v in values.items()}


def _duplicate_singleton(value):
    return torch.cat((value, value), 0)


def _pad_singleton_rows(batch):
    return {key: _duplicate_singleton(value) for key, value in batch.items()}


class HexNeighborGather(torch.autograd.Function):
    """Each direction is injective; reverse routing needs no sorted scatter."""

    @staticmethod
    def forward(ctx, projected, neighbors):
        ctx.save_for_backward(neighbors)
        b, n, _, width = projected.shape
        return projected[:, :, 1:].gather(
            1, neighbors.clamp_min(0).unsqueeze(-1).expand(b, n, HEX_DIRECTIONS, width)
        ) * (neighbors >= 0).unsqueeze(-1)

    @staticmethod
    def backward(ctx, gradient):
        (neighbors,) = ctx.saved_tensors
        inverse = torch.cat(
            (
                neighbors[:, :, OPPOSITE_HEX_OFFSET:],
                neighbors[:, :, :OPPOSITE_HEX_OFFSET],
            ),
            2,
        )
        routed = gradient.gather(1, inverse.clamp_min(0).unsqueeze(-1).expand_as(gradient)) * (
            inverse >= 0
        ).unsqueeze(-1)
        return torch.cat((torch.zeros_like(routed[:, :, :1]), routed), 2), None


class ActionEmbeddingGather(torch.autograd.Function):
    """Deterministically reduce action gradients over each cell's contiguous segment."""

    @staticmethod
    def forward(ctx, embeddings, indices):
        ctx.save_for_backward(indices)
        ctx.num_cells = embeddings.shape[1]
        return embeddings.gather(
            1, indices.clamp_min(0).unsqueeze(-1).expand(-1, -1, embeddings.shape[2])
        ) * (indices >= 0).unsqueeze(-1)

    @staticmethod
    def backward(ctx, gradient):
        (indices,) = ctx.saved_tensors
        if indices.shape[1] <= 32:
            # A single small GEMM costs less than sorting short action lists.
            cells = torch.arange(ctx.num_cells, device=indices.device)
            routing = (indices.unsqueeze(-1) == cells).to(gradient.dtype)
            with torch.autocast(gradient.device.type, enabled=False):
                return torch.bmm(routing.transpose(1, 2), gradient), None
        keys, order = indices.sort(dim=1, stable=True)
        boundaries = torch.arange(ctx.num_cells + 1, device=indices.device)
        offsets = torch.searchsorted(
            keys, boundaries.expand(len(indices), -1).contiguous()
        )
        ordered = gradient.gather(1, order.unsqueeze(-1).expand_as(gradient))
        # The first offset skips invalid indices. Empty cells get a zero gradient.
        return torch.segment_reduce(ordered, "sum", offsets=offsets, axis=1, unsafe=True), None


class MapMaximum(torch.autograd.Function):
    """One winner per channel: broadcast its gradient instead of scatter/sort."""

    @staticmethod
    def forward(ctx, values):
        maximum, indices = values.max(1)
        ctx.save_for_backward(indices)
        ctx.num_cells = values.shape[1]
        return maximum

    @staticmethod
    def backward(ctx, gradient):
        (indices,) = ctx.saved_tensors
        cells = torch.arange(ctx.num_cells, device=indices.device)[None, :, None]
        return (cells == indices[:, None, :]) * gradient[:, None, :]


class HexBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(SELF_AND_HEX_DIRECTIONS, HEX_DIM, HEX_DIM))
        self.bias = nn.Parameter(torch.zeros(HEX_DIM))
        self.inverse_routing = False
        self.norm1 = nn.LayerNorm(HEX_DIM, eps=1e-5)
        self.norm2 = nn.LayerNorm(HEX_DIM, eps=1e-5)
        self.mlp = nn.Sequential(nn.Linear(HEX_DIM, HEX_MLP_DIM), nn.GELU(), nn.Linear(HEX_MLP_DIM, HEX_DIM))

    def forward(self, x, neighbors, mask):
        # Project all seven directions in one GEMM, then gather by direction.
        # W_d x_neighbor == (W_d x)_neighbor; no model/parameter change.
        b, n, _ = x.shape
        projected = F.linear(x, self.weight.reshape((SELF_AND_HEX_DIRECTIONS) * HEX_DIM, HEX_DIM)).reshape(
            b, n, SELF_AND_HEX_DIRECTIONS, HEX_DIM
        )
        if self.inverse_routing:
            messages = HexNeighborGather.apply(projected, neighbors).float()
        else:
            batch_index = torch.arange(b, device=x.device)[:, None, None]
            directions = torch.arange(1, SELF_AND_HEX_DIRECTIONS, device=x.device)[None, None, :]
            messages = projected[batch_index, neighbors.clamp_min(0), directions].float()
            messages = messages * (neighbors >= 0).unsqueeze(-1)
        message = (
            projected[:, :, 0].float()
            + messages.sum(2)
            + self.bias
        )
        u = x + F.gelu(self.norm1(message))
        return (u + self.mlp(self.norm2(u))) * mask.unsqueeze(-1)


_HEX_QR_CPU: tuple[torch.Tensor, torch.Tensor] | None = None
_HEX_BUCKETS_CPU: torch.Tensor | None = None


def _hex_qr_cpu_tables():
    global _HEX_QR_CPU
    if _HEX_QR_CPU is None:
        cells = np.arange(MAX_MAP_CELLS, dtype=np.int16)[None, :]
        widths = np.tile(np.arange(2, MAX_MAP_AXIS + 1, dtype=np.int16), 2)[:, None]
        parity = np.repeat(np.arange(2, dtype=np.int16), MAX_MAP_AXIS - 1)[:, None]
        r = cells // widths + parity
        q = cells % widths - ((r - (r & 1)) >> 1)
        _HEX_QR_CPU = torch.from_numpy(q), torch.from_numpy(r)
    return _HEX_QR_CPU


def _hex_bucket_table():
    """One byte per pair for each supported width and row parity."""
    global _HEX_BUCKETS_CPU
    if _HEX_BUCKETS_CPU is None:
        q, r = (coordinates.numpy() for coordinates in _hex_qr_cpu_tables())
        dq = q[:, :, None] - q[:, None, :]
        dr = r[:, :, None] - r[:, None, :]
        distances = (np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) >> 1
        _HEX_BUCKETS_CPU = torch.from_numpy(
            np.minimum(distances, C2_DISTANCE_BUCKETS - 1).astype(np.uint8)
        )
    return _HEX_BUCKETS_CPU


def hex_distance_buckets_from_neighbors(
    neighbors: torch.Tensor,
    cell_mask: torch.Tensor,
    num_buckets: int = C2_DISTANCE_BUCKETS,
    key_indices=None,
    *,
    bucket_table: torch.Tensor | None = None,
    qr_tables: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Exact pairwise hex distance buckets 0..num_buckets-1 from the neighbors graph."""
    _b, n, _ = neighbors.shape
    row_parity = (neighbors[:, 0, 5] >= 0).long()
    w = torch.where(neighbors[:, 0, 5] >= 0, neighbors[:, 0, 5], neighbors[:, 0, 0]).clamp_min(2)
    key_mask = cell_mask if key_indices is None else key_indices >= 0
    valid_pairs = cell_mask[:, :, None] & key_mask[:, None, :]
    if bucket_table is not None and n <= MAX_MAP_CELLS and num_buckets == C2_DISTANCE_BUCKETS:
        gid = row_parity * (MAX_MAP_AXIS - 1) + (w.clamp_max(MAX_MAP_AXIS) - 2)
        if key_indices is None:
            dist = bucket_table[gid, :n, :n]
        else:
            cells = torch.arange(n, device=neighbors.device)[None, :, None]
            dist = bucket_table[gid[:, None, None], cells, key_indices[:, None, :].clamp_min(0)]
        return torch.where(valid_pairs, dist, 0)
    if qr_tables is not None and n <= MAX_MAP_CELLS:
        qs, rs = qr_tables
        gid = row_parity * (MAX_MAP_AXIS - 1) + (w.clamp_max(MAX_MAP_AXIS) - 2)
        q = qs[gid, :n].to(torch.int32)
        r = rs[gid, :n].to(torch.int32)
    else:
        idx = torch.arange(n, device=neighbors.device, dtype=torch.long)[None, :]
        x = idx % w[:, None]
        y = idx // w[:, None]
        r = y + row_parity[:, None]
        q = x - ((r - (r & 1)) >> 1)
    key_q = q if key_indices is None else q.gather(1, key_indices.clamp_min(0))
    key_r = r if key_indices is None else r.gather(1, key_indices.clamp_min(0))
    dq = q[:, :, None] - key_q[:, None, :]
    dr = r[:, :, None] - key_r[:, None, :]
    dist = (dq.abs() + dr.abs() + (dq + dr).abs()) >> 1
    return torch.where(valid_pairs, dist.clamp(0, num_buckets - 1), 0).to(torch.long)


def unit_cell_mask_from_cells(cells: torch.Tensor, cell_mask: torch.Tensor) -> torch.Tensor:
    """Public unit cells: own alive units, visible enemies, or remembered contacts."""
    own_present = (cells[..., TILE["own_type"]] > 0).any(-1)
    enemy_present = (cells[..., TILE["enemy_type"]] > 0).any(-1)
    contact_present = (cells[..., TILE["contact_count"]] > 0).any(-1)
    return cell_mask & (own_present | enemy_present | contact_present)


class BucketBias(torch.autograd.Function):
    """Table lookup with deterministic FP32 reductions, without scatter or one-hot."""

    @staticmethod
    def forward(ctx, table, buckets):
        ctx.save_for_backward(buckets)
        ctx.num_buckets = table.shape[1]
        return table[:, buckets.long()]

    @staticmethod
    def backward(ctx, grad):
        (buckets,) = ctx.saved_tensors
        ids = buckets.reshape(-1)
        # Inductor fuses each mask into its reduction. Avoid a [B*N*N, buckets]
        # one-hot and a huge FP32 GEMM just to compute heads*buckets scalars.
        # Fixed reductions also avoid deterministic indexing_backward's sort.
        values = (grad if grad.dtype == torch.float64 else grad.float()).reshape(grad.shape[0], -1)
        table_grad = torch.stack(
            [torch.where(ids == bucket, values, 0.0).sum(-1)
             for bucket in range(ctx.num_buckets)],
            -1,
        )
        return table_grad.to(grad.dtype), None


class CellUnitAttention(nn.Module):
    """Cross-attention from every map cell to all known unit cells with hex-distance bias."""

    def __init__(self):
        super().__init__()
        self.num_heads = C2_ATTN_HEADS
        self.head_dim = HEX_DIM // C2_ATTN_HEADS
        self.scale = self.head_dim**-0.5
        self.norm = nn.LayerNorm(HEX_DIM, eps=1e-5)
        self.q_proj = nn.Linear(HEX_DIM, HEX_DIM)
        self.kv_proj = nn.Linear(HEX_DIM, 2 * HEX_DIM)
        self.out_proj = nn.Linear(HEX_DIM, HEX_DIM)
        self.distance_bias = nn.Parameter(torch.zeros(C2_ATTN_HEADS, C2_DISTANCE_BUCKETS))
        qs, rs = _hex_qr_cpu_tables()
        self.register_buffer("_qr_q", qs, persistent=False)
        self.register_buffer("_qr_r", rs, persistent=False)
        self.register_buffer("_bucket_table", None, persistent=False)

    def enable_compact_geometry(self, enabled):
        if not enabled:
            self._bucket_table = None
        elif self._bucket_table is None:
            self._bucket_table = _hex_bucket_table().to(self.distance_bias.device)

    def forward_with_weights(self, x, cells, neighbors, cell_mask, key_indices=None):
        b, n, _ = x.shape
        unit_mask = unit_cell_mask_from_cells(cells, cell_mask)
        buckets = hex_distance_buckets_from_neighbors(
            neighbors,
            cell_mask,
            key_indices=key_indices,
            bucket_table=self._bucket_table if key_indices is not None else None,
            qr_tables=(self._qr_q, self._qr_r),
        )
        u = self.norm(x)
        q = self.q_proj(u).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        keys = u if key_indices is None else ActionEmbeddingGather.apply(u, key_indices)
        key_count = keys.shape[1]
        key_mask = unit_mask if key_indices is None else key_indices >= 0
        k, v = self.kv_proj(keys).reshape(b, key_count, 2, self.num_heads, self.head_dim).unbind(2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        with torch.autocast(x.device.type, enabled=False):
            bias = BucketBias.apply(self.distance_bias.float(), buckets).permute(1, 0, 2, 3)
            logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale + bias
            logits = logits.masked_fill(~key_mask[:, None, None, :], -torch.inf)
            has_units = key_mask.any(-1, keepdim=True)[:, None, :, None]
            safe_logits = torch.where(has_units, logits, 0.0)
            weights = torch.where(has_units, safe_logits.softmax(-1), 0.0)
            weights = weights * cell_mask[:, None, :, None]
            context = torch.matmul(weights, v.float())
        merged = context.to(u.dtype).transpose(1, 2).reshape(b, n, HEX_DIM)
        out = (x + self.out_proj(merged)) * cell_mask.unsqueeze(-1)
        return out, weights, unit_mask

    def forward(self, x, cells, neighbors, cell_mask, key_indices=None):
        out, _, _ = self.forward_with_weights(x, cells, neighbors, cell_mask, key_indices)
        return out


class _PolicyExecutor:
    """Dispatch spatial batches, fixed blocks and the coalesced training path.

    Eager and Inductor forwards use the same spatial math. StaticSpatialGraph
    wraps the block forward for selected no-grad callers.
    """

    def __init__(self):
        self.spatial_runner = None
        self.spatial_compile_mode = None
        self.replay_graph = None
        self.heads_runner = None
        self.memory_runner = None

    def compile(
        self,
        policy,
        *,
        dynamic=True,
        mode="default",
        recompile_limit=32,
        spatial_mode=None,
    ):
        # Inference/no-grad replay/training have separate autograd specializations.
        # Keep a bounded cache, and retain fullgraph=True so fallback is never silent.
        self.spatial_compile_mode = spatial_mode or mode
        # Existing step markers are for Inductor-managed graphs.
        policy.spatial_cuda_graphs = self.spatial_compile_mode == "reduce-overhead"
        self.replay_graph = None
        self.spatial_runner = torch.compile(
            policy.spatial,
            dynamic=dynamic,
            mode=self.spatial_compile_mode,
            fullgraph=True,
            recompile_limit=recompile_limit,
        )
        self.heads_runner = torch.compile(
            policy.heads,
            dynamic=dynamic,
            mode=mode,
            fullgraph=True,
            recompile_limit=recompile_limit,
        )

    def enable_recurrent_graphs(self, memory_module, *, max_shapes=64):
        from civ5rl.core.training_graphs import RecurrentGraphs

        self.memory_runner = RecurrentGraphs(memory_module, max_shapes=max_shapes)

    def spatial_replay_runner(self, policy):
        if (
            self.spatial_compile_mode != "default"
            or self.spatial_runner is None
            or not policy.amp_enabled
            or (policy.spatial_batch_size, policy.spatial_cell_padding) != (32, 256)
        ):
            return None
        # Check storage once per PPO update. Optimizer steps and rollback update
        # it in place; a model move or parameter replacement needs a new graph.
        storage = tuple((p.device, p.dtype, p.data_ptr()) for p in policy.parameters())
        if self.replay_graph is None or self.replay_graph[0] != storage:
            self.replay_graph = storage, StaticSpatialGraph(self.spatial_runner, 32, 256)
        return self.replay_graph[1]

    def encode_spatial(
        self, policy, batch, *, coalesced=None, return_cells=True, spatial_runner=None
    ):
        extra = policy.spatial_cell_padding - batch["cells"].shape[1]
        if extra > 0:
            # Keep BF16 GEMM geometry independent of which map sizes happen
            # to share a collector/replay batch. Padding never enters pooling.
            batch = {
                **batch,
                "cells": F.pad(batch["cells"], (0, 0, 0, extra)),
                "neighbors": F.pad(batch["neighbors"], (0, 0, 0, extra), value=-1),
                "cell_mask": F.pad(batch["cell_mask"], (0, extra), value=False),
            }
        with policy.autocast(batch["cells"]):
            return self._encode_spatial(
                policy, batch, coalesced=coalesced, return_cells=return_cells,
                spatial_runner=spatial_runner,
            )

    def _encode_spatial(self, policy, batch, *, coalesced, return_cells, spatial_runner):
        use_coalesced = policy.coalesced_spatial if coalesced is None else coalesced
        if use_coalesced and torch.is_grad_enabled():
            x, context = self._coalesced_spatial(policy, batch)
        elif policy.spatial_batch_size:
            return self._fixed_spatial_blocks(
                batch, policy.spatial_batch_size, spatial_runner or policy.spatial_runner,
                return_cells=return_cells,
            )
        else:
            x, context = self._spatial_batch(policy, batch)
        return (x if return_cells else None), context

    @staticmethod
    def _coalesced_spatial(policy, batch):
        if not policy.spatial_batch_size:
            raise ValueError("Coalesced spatial training requires a fixed spatial block")
        from civ5rl.core.training_graphs import CoalescedSpatial

        parameters = tuple(
            p
            for name, p in policy.named_parameters()
            if name.startswith(("projection.", "blocks.", "unit_attention.", "context."))
            and p.requires_grad
        )
        return CoalescedSpatial.apply(policy, batch, *parameters)

    @staticmethod
    def _fixed_spatial_blocks(batch, size, runner, *, return_cells):
        # Fixed BF16 GEMM geometry keeps action selection, replay and training
        # consistent. Discard padded rows before memory and loss.
        if (
            isinstance(runner, StaticSpatialGraph)
            and not torch.is_grad_enabled()
            and len(batch["cells"])
        ):
            return runner.forward_blocks(batch, size, return_cells=return_cells)
        maps, contexts = [], []
        for part, count in iter_spatial_blocks(batch, size):
            x, context = runner(part)
            if return_cells:
                maps.append(x[:count])
            contexts.append(context[:count])
        return torch.cat(maps, 0) if return_cells else None, torch.cat(contexts, 0)

    def _spatial_batch(self, policy, batch):
        if self.spatial_runner is not None and len(batch["cells"]) == 1:
            x, context = self.spatial_runner(_pad_singleton_rows(batch))
            return x[:1], context[:1]
        return policy.spatial_runner(batch)

    def score_heads(self, policy, batch, x, h, *, compute_value):
        with policy.autocast(x):
            return self._score_heads(policy, batch, x, h, compute_value=compute_value)

    def _score_heads(self, policy, batch, x, h, *, compute_value):
        if self.heads_runner is None:
            return policy.heads(batch, x, h, compute_value=compute_value)
        action_keys = ("sources", "targets", "action_features", "action_mask")
        if policy.variant_spec.attack_preview:
            action_keys += ("attack_preview",)
        action_batch = {key: batch[key] for key in action_keys}
        num_rows, num_actions = len(x), action_batch["sources"].shape[1]
        if num_rows == 1:
            action_batch = _pad_singleton_rows(action_batch)
            x, h = _duplicate_singleton(x), _duplicate_singleton(h)
        if num_actions == 1:
            action_batch = dict(action_batch)
            for key in action_keys:
                value = action_batch[key]
                fill = -1 if key in ("sources", "targets") else 0
                action_batch[key] = torch.cat((value, torch.full_like(value, fill)), 1)
        logits, value = self.heads_runner(action_batch, x, h, compute_value=compute_value)
        return logits[:num_rows, :num_actions], value[:num_rows] if compute_value else None

    def run_recurrent_memory(self, policy, context_embedding, lengths, hidden, *, return_sequence):
        # CUDA autocast can select FP16 for cuDNN RNN even inside BF16 autocast.
        # Keep both GRU arithmetic and stored army memory explicitly in FP32.
        with torch.autocast(context_embedding.device.type, enabled=False):
            memory = self.memory_runner or policy.memory
            hidden = None if hidden is None else hidden.float()
            context = context_embedding.float()
            length = lengths[0]
            if all(other == length for other in lengths):
                padded = context.reshape(len(lengths), length, -1).transpose(0, 1).contiguous()
                output, last = memory(padded, hidden)
                valid = output.transpose(0, 1).reshape_as(context) if return_sequence else None
                return valid, last
            output, _ = memory(nn.utils.rnn.pad_sequence(context.split(lengths)), hidden)
        valid = (
            torch.cat([output[:length, i] for i, length in enumerate(lengths)])
            if return_sequence else None
        )
        last = torch.stack([output[length - 1, i] for i, length in enumerate(lengths)])[None]
        return valid, last


class ActionAttention(nn.Module):
    """Residual attention across the legal commands of one decision."""

    num_heads = 4

    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(HEX_DIM)
        self.q_proj = nn.Linear(HEX_DIM, HEX_DIM)
        self.k_proj = nn.Linear(HEX_DIM, HEX_DIM)
        self.v_proj = nn.Linear(HEX_DIM, HEX_DIM)
        self.out_proj = nn.Linear(HEX_DIM, HEX_DIM)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, actions, mask):
        batch_size, count, _ = actions.shape
        normalized = self.norm(actions)
        shape = (batch_size, count, self.num_heads, HEX_DIM // self.num_heads)
        q, k, v = (
            project(normalized).reshape(shape).transpose(1, 2)
            for project in (self.q_proj, self.k_proj, self.v_proj)
        )
        # Give all-padding rows a defined softmax. Their result is discarded.
        keys = (mask | ~mask.any(dim=1, keepdim=True))[:, None, None, :]
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=keys)
        update = self.out_proj(attended.transpose(1, 2).reshape(batch_size, count, HEX_DIM))
        return actions + update.masked_fill(~mask[..., None], 0)


class Policy(nn.Module):
    def __init__(self, variant: PolicyVariant | str = PolicyVariant.CONTACTS_GRU):
        super().__init__()
        self.variant_spec = variant_spec(variant)
        self.variant = self.variant_spec.variant.value
        self.hex_dim = HEX_DIM
        self.hidden_size = HIDDEN_SIZE
        self.projection = nn.Linear(TILE_DIM, HEX_DIM)
        self.blocks = nn.ModuleList([HexBlock() for _ in range(self.variant_spec.hex_blocks)])
        self.unit_attention = CellUnitAttention() if self.variant_spec.uses_unit_attention else None
        self.context = nn.Sequential(
            nn.Linear(CONTEXT_INPUT_DIM, HIDDEN_SIZE),
            nn.GELU(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.GELU(),
        )
        self.memory = (
            nn.GRU(HIDDEN_SIZE, HIDDEN_SIZE)
            if self.variant_spec.is_recurrent
            else nn.Sequential(
                nn.Linear(HIDDEN_SIZE, MEMORY_MLP_DIM),
                nn.GELU(),
                nn.Linear(MEMORY_MLP_DIM, HIDDEN_SIZE),
                nn.GELU(),
            )
        )
        self.action = nn.Sequential(nn.Linear(ACTION_INPUT_DIM, HEX_DIM), nn.GELU(), nn.Linear(HEX_DIM, 1))
        self.value = nn.Sequential(nn.Linear(HIDDEN_SIZE, HEX_DIM), nn.GELU(), nn.Linear(HEX_DIM, 1))
        self.activation_checkpointing = False
        self._executor = _PolicyExecutor()
        self.amp_enabled = False
        self.spatial_batch_size = 0
        self.spatial_cell_padding = 0
        self.attention_backend = "dense_fp32"
        self.action_routing = False
        self.maximum_routing = False
        self.spatial_cuda_graphs = False
        self.coalesced_spatial = False
        self.coalesced_accumulation = False
        self._initialize()
        if self.variant_spec.attack_preview:
            self.attack_preview = nn.Linear(ATTACK_PREVIEW_DIM, HEX_DIM, bias=False)
            nn.init.zeros_(self.attack_preview.weight)
        if self.variant_spec.action_context:
            self.action_attention = ActionAttention()
            self.value_actions = nn.Linear(2 * HEX_DIM, HEX_DIM, bias=False)
            nn.init.zeros_(self.value_actions.weight)

    @property
    def is_recurrent(self) -> bool:
        return self.variant_spec.is_recurrent

    @property
    def spatial_runner(self):
        return self._executor.spatial_runner or self.spatial

    @property
    def _spatial_runner(self):
        return self._executor.spatial_runner

    @_spatial_runner.setter
    def _spatial_runner(self, runner):
        self._executor.spatial_runner = runner
        self._executor.replay_graph = None

    @property
    def _heads_runner(self):
        return self._executor.heads_runner

    @_heads_runner.setter
    def _heads_runner(self, runner):
        self._executor.heads_runner = runner

    @property
    def _memory_runner(self):
        return self._executor.memory_runner

    @_memory_runner.setter
    def _memory_runner(self, runner):
        self._executor.memory_runner = runner

    def enable_amp(self, enabled=True, *, spatial_batch_size=0):
        if enabled and (
            next(self.parameters()).device.type != "cuda"
            or not torch.cuda.is_bf16_supported(including_emulation=False)
        ):
            raise ValueError("BF16 AMP requires a CUDA device with native BF16 support")
        if type(spatial_batch_size) is not int or spatial_batch_size < 0:
            raise ValueError("Spatial batch size must be a nonnegative integer")
        if self.coalesced_spatial and (not enabled or not spatial_batch_size):
            raise ValueError("Coalesced spatial training requires a fixed spatial block")
        self.amp_enabled = enabled
        self.spatial_batch_size = spatial_batch_size if enabled else 0

    def configure_execution(
        self,
        *,
        inverse_routing=None,
        action_routing=None,
        maximum_routing=None,
        coalesced_spatial=None,
        activation_checkpointing=None,
        attention_backend=None,
    ):
        """Set optional execution paths without changing model parameters."""
        options = (
            inverse_routing,
            action_routing,
            maximum_routing,
            coalesced_spatial,
            activation_checkpointing,
        )
        if any(value is not None and type(value) is not bool for value in options):
            raise ValueError("Execution options must be boolean")
        if coalesced_spatial and not self.spatial_batch_size:
            raise ValueError("Coalesced spatial training requires a fixed spatial block")
        self._executor.replay_graph = None
        if attention_backend is not None:
            if attention_backend not in ("dense_fp32", "compact_fp32"):
                raise ValueError("Unknown attention backend")
            self.attention_backend = attention_backend
            if self.unit_attention is not None:
                self.unit_attention.enable_compact_geometry(attention_backend == "compact_fp32")
        if inverse_routing is not None:
            for block in self.blocks:
                block.inverse_routing = inverse_routing
        for name, value in (
            ("action_routing", action_routing),
            ("maximum_routing", maximum_routing),
            ("coalesced_spatial", coalesced_spatial),
            ("activation_checkpointing", activation_checkpointing),
        ):
            if value is not None:
                setattr(self, name, value)

    def enable_inverse_routing(self):
        self.configure_execution(inverse_routing=True)

    def enable_action_routing(self):
        self.configure_execution(action_routing=True)

    def enable_maximum_routing(self):
        self.configure_execution(maximum_routing=True)

    def enable_coalesced_spatial(self):
        self.configure_execution(coalesced_spatial=True)

    def enable_recurrent_graphs(self, *, max_shapes=64):
        self._executor.enable_recurrent_graphs(self.memory, max_shapes=max_shapes)

    def autocast(self, tensor):
        return torch.autocast(tensor.device.type, dtype=torch.bfloat16, enabled=self.amp_enabled)

    def enable_compile(self, *, dynamic=True, mode="default", recompile_limit=32, spatial_mode=None):
        """Compile tensor-heavy regions; cuDNN GRU stays eager, weights stay portable."""
        self._executor.compile(
            self,
            dynamic=dynamic,
            mode=mode,
            recompile_limit=recompile_limit,
            spatial_mode=spatial_mode,
        )

    def spatial_replay_runner(self):
        return self._executor.spatial_replay_runner(self)

    def encode_spatial(self, batch, *, coalesced=None, return_cells=True, spatial_runner=None):
        return self._executor.encode_spatial(
            self, batch, coalesced=coalesced, return_cells=return_cells,
            spatial_runner=spatial_runner,
        )

    def score_heads(self, batch, x, h, *, compute_value=True):
        return self._executor.score_heads(self, batch, x, h, compute_value=compute_value)

    def _check_sequences(self, batch_size, lengths, hidden):
        if (
            not lengths
            or any(
                not isinstance(length, Integral) or isinstance(length, bool) or length <= 0
                for length in lengths
            )
            or sum(lengths) != batch_size
        ):
            raise ValueError("Sequence lengths must be positive and sum to the batch size")
        if self.is_recurrent:
            self._check_hidden(hidden, len(lengths))

    def sequence_memory(self, context_embedding, lengths, hidden=None, *, return_sequence=True):
        self._check_sequences(len(context_embedding), lengths, hidden)
        if not self.is_recurrent:
            if not return_sequence:
                return None, None
            with self.autocast(context_embedding):
                return self.memory(context_embedding).float(), None
        return self._executor.run_recurrent_memory(
            self, context_embedding, lengths, hidden, return_sequence=return_sequence
        )

    def forward_sequences(
        self, batch, lengths, hidden=None, *, memory_only=False, compute_value=True,
        spatial_runner=None, action_groups=(),
    ):
        self._check_sequences(len(batch["cells"]), lengths, hidden)
        x, context_embedding = self.encode_spatial(
            batch, return_cells=not memory_only, spatial_runner=spatial_runner
        )
        h, last = self.sequence_memory(context_embedding, lengths, hidden, return_sequence=not memory_only)
        if memory_only:
            return last
        if action_groups:
            logits, value = self._score_action_groups(batch, x, h, action_groups, compute_value=compute_value)
        else:
            logits, value = self.score_heads(batch, x, h, compute_value=compute_value)
        return logits, value, last

    def _score_action_groups(self, batch, x, h, groups, *, compute_value):
        """Score independent replay rows at their local width and restore row order."""
        scores = h.new_full(batch["action_mask"].shape, -torch.inf)
        values = h.new_empty(len(h)) if compute_value else None
        keys = ("sources", "targets", "action_features", "action_mask", "attack_preview")
        for rows, width in groups:
            part = {key: batch[key][:, :width].index_select(0, rows) for key in keys if key in batch}
            logits, value = self.score_heads(
                part, x.index_select(0, rows), h.index_select(0, rows), compute_value=compute_value,
            )
            scores[:, :width].index_copy_(0, rows, logits)
            if compute_value:
                values.index_copy_(0, rows, value)
        return scores, values

    def _initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, HexBlock):
                for w in module.weight:
                    nn.init.xavier_uniform_(w)
        if self.is_recurrent:
            nn.init.xavier_uniform_(self.memory.weight_ih_l0)
            for w in self.memory.weight_hh_l0.chunk(GRU_GATES):
                nn.init.orthogonal_(w)
            nn.init.zeros_(self.memory.bias_ih_l0)
            nn.init.zeros_(self.memory.bias_hh_l0)
        nn.init.orthogonal_(self.action[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value[-1].weight, gain=1)

    def spatial(self, batch):
        mask = batch["cell_mask"]
        # BF16 rounding in the first projection is amplified by four residual
        # blocks and recurrent replay. Keep the small precision boundaries FP32;
        # the expensive map-block GEMMs still run under BF16 autocast.
        with torch.autocast(batch["cells"].device.type, enabled=False):
            x = F.gelu(self.projection(batch["cells"].float())) * mask.unsqueeze(-1)
        for block in self.blocks:
            if self.activation_checkpointing and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(block, x, batch["neighbors"], mask, use_reentrant=False)
            else:
                x = block(x, batch["neighbors"], mask)
        if self.unit_attention is not None:
            keys = batch["unit_indices"] if self.attention_backend == "compact_fp32" else None
            if self.activation_checkpointing and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(
                    self.unit_attention,
                    x,
                    batch["cells"],
                    batch["neighbors"],
                    mask,
                    keys,
                    use_reentrant=False,
                )
            else:
                x = self.unit_attention(x, batch["cells"], batch["neighbors"], mask, keys)
        mean = x.sum(1) / mask.sum(1, keepdim=True)
        masked = x.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        maximum = MapMaximum.apply(masked) if self.maximum_routing else masked.max(1).values
        with torch.autocast(x.device.type, enabled=False):
            context_embedding = self.context(
                torch.cat((mean, maximum, batch["globals"], batch["previous"]), -1).float()
            )
        return x, context_embedding

    def heads(self, batch, x, h, *, compute_value=True, return_value_features=False):
        b, n, _ = x.shape
        offsets = torch.arange(b, device=x.device)[:, None] * n
        flat = x.reshape(b * n, self.hex_dim)
        embeddings = []
        for key in ("sources", "targets"):
            idx = batch[key]
            if self.action_routing:
                embeddings.append(ActionEmbeddingGather.apply(x, idx))
                continue
            embeddings.append(
                flat[(idx.clamp_min(0) + offsets).reshape(-1)].reshape(b, -1, self.hex_dim)
                * (idx >= 0).unsqueeze(-1)
            )
        action_input = torch.cat(
            (
                h[:, None].expand(-1, batch["sources"].shape[1], -1),
                *embeddings,
                batch["action_features"],
            ),
            -1,
        )
        # Preserve action probabilities across different rollout/replay batches.
        with torch.autocast(h.device.type, enabled=False):
            mask = batch["action_mask"]
            if self.variant_spec.attack_preview:
                first = self.action[0](action_input.float()) + self.attack_preview(batch["attack_preview"].float())
                actions = self.action[1](first)
                if self.variant_spec.action_context:
                    actions = self.action_attention(actions, mask)
                scores = self.action[2](actions)
            else:
                scores = self.action(action_input.float())
            logits = scores.squeeze(-1).masked_fill(~mask, -torch.inf)
            value = None
            if compute_value and self.variant_spec.action_context:
                detached = actions.detach()
                mean = detached.masked_fill(~mask[..., None], 0).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
                maximum = detached.masked_fill(~mask[..., None], -torch.inf).amax(1)
                maximum = maximum.masked_fill(~mask.any(1, keepdim=True), 0)
                context = torch.cat((mean, maximum), dim=-1)
                value = (torch.cat((h.float(), context), dim=-1) if return_value_features
                         else self._value_with_context(h.float(), context))
            elif compute_value:
                value = self.value(h.float()).squeeze(-1)
        return logits, value

    def value_from_features(self, features):
        if not self.variant_spec.action_context:
            return self.value(features).squeeze(-1)
        hidden, context = features.split((self.hidden_size, 2 * self.hex_dim), dim=-1)
        return self._value_with_context(hidden, context)

    def _value_with_context(self, hidden, context):
        summary = self.value_actions(context)
        return self.value[2](self.value[1](self.value[0](hidden) + summary)).squeeze(-1)

    def _check_hidden(self, hidden: torch.Tensor | None, batch_size: int) -> None:
        if hidden is not None and tuple(hidden.shape) != (
            1,
            batch_size,
            self.hidden_size,
        ):
            raise ValueError(f"Hidden state must have shape (1, {batch_size}, {self.hidden_size})")

    def forward(self, batch, hidden=None, *, sequence=False, compute_value=True):
        if not self.is_recurrent:
            return self.forward_sequences(batch, [len(batch["cells"])], hidden, compute_value=compute_value)
        self._check_hidden(hidden, 1 if sequence else len(batch["cells"]))
        x, context_embedding = self.encode_spatial(batch)
        # sequence=True: items are consecutive decisions of one episode.
        recurrent_input = context_embedding[:, None] if sequence else context_embedding[None]
        with torch.autocast(context_embedding.device.type, enabled=False):
            output, hidden = self.memory(recurrent_input.float(), None if hidden is None else hidden.float())
        h = output[:, 0] if sequence else output[0]
        logits, value = self.score_heads(batch, x, h, compute_value=compute_value)
        return logits, value, hidden


def distribution_stats(logits, indices, *, entropy_grad=True):
    log_probs = logits.log_softmax(-1)
    logp = log_probs.gather(-1, indices[:, None]).squeeze(-1)
    with torch.set_grad_enabled(torch.is_grad_enabled() and entropy_grad):
        ent_log_probs = log_probs if entropy_grad else log_probs.detach()
        probs = ent_log_probs.exp()
        entropy = -(
            probs * ent_log_probs.masked_fill(~torch.isfinite(ent_log_probs), 0)
        ).sum(-1)
        count = torch.isfinite(logits).sum(-1)
        normalized = torch.where(
            count > 1, entropy / count.clamp_min(2).float().log(), 0
        )
    return logp, entropy, normalized
