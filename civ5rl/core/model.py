"""Hex message passing, army memory, and scores over public legal commands."""

from numbers import Integral

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from civ5rl.core.schema import (
    ACTION_DIM,
    ACTION_INPUT_DIM,
    C2_ATTN_HEADS,
    C2_DISTANCE_BUCKETS,
    CONTEXT_INPUT_DIM,
    HEX_DIM,
    HEX_DIRECTIONS,
    HEX_MLP_DIM,
    HIDDEN_SIZE,
    MEMORY_MLP_DIM,
    OPPOSITE_HEX_OFFSET,
    SELF_AND_HEX_DIRECTIONS,
    TILE,
    TILE_DIM,
)
from civ5rl.core.spatial import iter_spatial_blocks
from civ5rl.core.variants import PolicyVariant, variant_spec

GRU_GATES = 3


def batch_inputs(items, device, *, non_blocking=False, spatial_only=False, min_cells=0):
    if not items or any(len(item.cells) == 0 for item in items):
        raise ValueError("Each batch item must contain at least one cell")
    batch_size = len(items)
    num_cells = max(min_cells, max(len(item.cells) for item in items))
    cells = np.zeros((batch_size, num_cells, TILE_DIM), np.float32)
    masks = np.zeros((batch_size, num_cells), bool)
    neighbors = np.full((batch_size, num_cells, HEX_DIRECTIONS), -1, np.int64)
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
    if not spatial_only:
        num_actions = max(len(item.sources) for item in items)
        sources = np.full((batch_size, num_actions), -1, np.int64)
        targets = sources.copy()
        action_mask = np.zeros((batch_size, num_actions), bool)
        features = np.zeros((batch_size, num_actions, ACTION_DIM), np.float32)
        for i, item in enumerate(items):
            action_count = len(item.sources)
            sources[i, :action_count] = item.sources
            targets[i, :action_count] = item.targets
            action_mask[i, :action_count] = True
            features[i, :action_count] = item.action_features
        values.update(
            sources=sources,
            targets=targets,
            action_features=features,
            action_mask=action_mask,
        )
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
    """Small deterministic dense reduction replaces general sorted index backward."""

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
        routing = (indices.unsqueeze(-1) == torch.arange(ctx.num_cells, device=indices.device)).to(
            gradient.dtype
        )
        with torch.autocast(gradient.device.type, enabled=False):
            return torch.bmm(routing.transpose(1, 2), gradient), None


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
        batch_index = torch.arange(b, device=x.device)[:, None, None]
        directions = torch.arange(1, SELF_AND_HEX_DIRECTIONS, device=x.device)[None, None, :]
        messages = (
            HexNeighborGather.apply(projected, neighbors)
            if self.inverse_routing
            else projected[batch_index, neighbors.clamp_min(0), directions]
        )
        message = (
            projected[:, :, 0].float()
            + (messages.float() * (neighbors >= 0).unsqueeze(-1)).sum(2)
            + self.bias
        )
        u = x + F.gelu(self.norm1(message))
        return (u + self.mlp(self.norm2(u))) * mask.unsqueeze(-1)


def hex_distance_buckets_from_neighbors(
    neighbors: torch.Tensor,
    cell_mask: torch.Tensor,
    num_buckets: int = C2_DISTANCE_BUCKETS,
) -> torch.Tensor:
    """Exact pairwise hex distance buckets 0..num_buckets-1 from the neighbors graph."""
    _b, n, _ = neighbors.shape
    row_parity = (neighbors[:, 0, 5] >= 0).long()
    w = torch.where(neighbors[:, 0, 5] >= 0, neighbors[:, 0, 5], neighbors[:, 0, 0]).clamp_min(2)
    idx = torch.arange(n, device=neighbors.device, dtype=torch.long)[None, :]
    x = idx % w[:, None]
    y = idx // w[:, None]
    r = y + row_parity[:, None]
    q = x - ((r - (r & 1)) >> 1)
    dq = q[:, :, None] - q[:, None, :]
    dr = r[:, :, None] - r[:, None, :]
    dist = (dq.abs() + dr.abs() + (dq + dr).abs()) >> 1
    valid_pairs = cell_mask[:, :, None] & cell_mask[:, None, :]
    return torch.where(valid_pairs, dist.clamp(0, num_buckets - 1), 0)


def unit_cell_mask_from_cells(cells: torch.Tensor, cell_mask: torch.Tensor) -> torch.Tensor:
    """Public unit cells: own alive units, visible enemies, or remembered contacts."""
    own_present = (cells[..., TILE["own_type"]] > 0).any(-1)
    enemy_present = (cells[..., TILE["enemy_type"]] > 0).any(-1)
    contact_present = (cells[..., TILE["contact_count"]] > 0).any(-1)
    return cell_mask & (own_present | enemy_present | contact_present)


class BucketBias(torch.autograd.Function):
    """table[:, buckets] with a dense deterministic backward (no sorted index_put)."""

    @staticmethod
    def forward(ctx, table, buckets):
        ctx.save_for_backward(buckets)
        ctx.num_buckets = table.shape[1]
        return table[:, buckets]

    @staticmethod
    def backward(ctx, grad):
        (buckets,) = ctx.saved_tensors
        onehot = F.one_hot(buckets.reshape(-1), ctx.num_buckets).to(grad.dtype)
        with torch.autocast(grad.device.type, enabled=False):
            return grad.reshape(grad.shape[0], -1) @ onehot, None


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

    def forward_with_weights(self, x, cells, neighbors, cell_mask):
        b, n, _ = x.shape
        unit_mask = unit_cell_mask_from_cells(cells, cell_mask)
        buckets = hex_distance_buckets_from_neighbors(neighbors, cell_mask)
        u = self.norm(x)
        q = self.q_proj(u).reshape(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = self.kv_proj(u).reshape(b, n, 2, self.num_heads, self.head_dim).unbind(2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        with torch.autocast(x.device.type, enabled=False):
            bias = BucketBias.apply(self.distance_bias.float(), buckets).permute(1, 0, 2, 3)
            logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * self.scale + bias
            logits = logits.masked_fill(~unit_mask[:, None, None, :], -torch.inf)
            has_units = unit_mask.any(-1, keepdim=True)[:, None, :, None]
            safe_logits = torch.where(has_units, logits, 0.0)
            weights = torch.where(has_units, safe_logits.softmax(-1), 0.0)
            weights = weights * cell_mask[:, None, :, None]
            context = torch.matmul(weights, v.float())
        merged = context.to(u.dtype).transpose(1, 2).reshape(b, n, HEX_DIM)
        out = (x + self.out_proj(merged)) * cell_mask.unsqueeze(-1)
        return out, weights, unit_mask

    def forward(self, x, cells, neighbors, cell_mask):
        out, _, _ = self.forward_with_weights(x, cells, neighbors, cell_mask)
        return out


class _PolicyExecutor:
    """Execution strategy, compilation, and batch padding outside nn.Module state."""

    def __init__(self):
        self.spatial_runner = None
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
        policy.spatial_cuda_graphs = spatial_mode == "reduce-overhead"
        self.spatial_runner = torch.compile(
            policy.spatial,
            dynamic=dynamic,
            mode=spatial_mode or mode,
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

    def encode_spatial(self, policy, batch):
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
            return self._encode_spatial(policy, batch)

    def _encode_spatial(self, policy, batch):
        if policy.coalesced_spatial and torch.is_grad_enabled():
            from civ5rl.core.training_graphs import CoalescedSpatial

            parameters = tuple(
                p
                for name, p in policy.named_parameters()
                if name.startswith(("projection.", "blocks.", "unit_attention.", "context."))
                and p.requires_grad
            )
            return CoalescedSpatial.apply(policy, batch, *parameters)
        if policy.spatial_batch_size:
            # BF16 GEMM output rounding can vary with the row count. Use the
            # same GEMM shapes during action selection, replay and gradients.
            # Padding rows are independent and discarded before memory/loss.
            size = policy.spatial_batch_size
            outputs = []
            runner = policy.spatial_runner
            for part, count in iter_spatial_blocks(batch, size):
                x, context_embedding = runner(part)
                outputs.append((x[:count], context_embedding[:count]))
            return tuple(torch.cat([output[i] for output in outputs], 0) for i in range(2))
        if self.spatial_runner is not None and len(batch["cells"]) == 1:
            padded = _pad_singleton_rows(batch)
            x, context_embedding = self.spatial_runner(padded)
            return x[:1], context_embedding[:1]
        return policy.spatial_runner(batch)

    def score_heads(self, policy, batch, x, h):
        with policy.autocast(x):
            return self._score_heads(policy, batch, x, h)

    def _score_heads(self, policy, batch, x, h):
        if self.heads_runner is None:
            return policy.heads(batch, x, h)
        action_keys = ("sources", "targets", "action_features", "action_mask")
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
        logits, value = self.heads_runner(action_batch, x, h)
        return logits[:num_rows, :num_actions], value[:num_rows]

    def run_recurrent_memory(self, policy, context_embedding, lengths, hidden):
        # CUDA autocast can select FP16 for cuDNN RNN even inside BF16 autocast.
        # Keep both GRU arithmetic and stored army memory explicitly in FP32.
        with torch.autocast(context_embedding.device.type, enabled=False):
            sequences = context_embedding.float().split(lengths)
            padded = nn.utils.rnn.pad_sequence(sequences)
            output, _ = (self.memory_runner or policy.memory)(
                padded, None if hidden is None else hidden.float()
            )
        valid = torch.cat([output[:length, i] for i, length in enumerate(lengths)])
        last = torch.stack([output[length - 1, i] for i, length in enumerate(lengths)])[None]
        return valid, last


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
        self.action_routing = False
        self.maximum_routing = False
        self.spatial_cuda_graphs = False
        self.coalesced_spatial = False
        self.coalesced_accumulation = False
        self._initialize()

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

    def encode_spatial(self, batch):
        return self._executor.encode_spatial(self, batch)

    def score_heads(self, batch, x, h):
        return self._executor.score_heads(self, batch, x, h)

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

    def sequence_memory(self, context_embedding, lengths, hidden=None):
        self._check_sequences(len(context_embedding), lengths, hidden)
        if not self.is_recurrent:
            with self.autocast(context_embedding):
                return self.memory(context_embedding).float(), None
        return self._executor.run_recurrent_memory(self, context_embedding, lengths, hidden)

    def forward_sequences(self, batch, lengths, hidden=None, *, memory_only=False):
        self._check_sequences(len(batch["cells"]), lengths, hidden)
        x, context_embedding = self.encode_spatial(batch)
        h, last = self.sequence_memory(context_embedding, lengths, hidden)
        if memory_only:
            return last
        logits, value = self.score_heads(batch, x, h)
        return logits, value, last

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
            if self.activation_checkpointing and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint

                x = checkpoint(
                    self.unit_attention,
                    x,
                    batch["cells"],
                    batch["neighbors"],
                    mask,
                    use_reentrant=False,
                )
            else:
                x = self.unit_attention(x, batch["cells"], batch["neighbors"], mask)
        mean = x.sum(1) / mask.sum(1, keepdim=True)
        masked = x.masked_fill(~mask.unsqueeze(-1), -torch.inf)
        maximum = MapMaximum.apply(masked) if self.maximum_routing else masked.max(1).values
        with torch.autocast(x.device.type, enabled=False):
            context_embedding = self.context(
                torch.cat((mean, maximum, batch["globals"], batch["previous"]), -1).float()
            )
        return x, context_embedding

    def heads(self, batch, x, h):
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
            logits = (
                self.action(action_input.float()).squeeze(-1).masked_fill(~batch["action_mask"], -torch.inf)
            )
            value = self.value(h.float()).squeeze(-1)
        return logits, value

    def _check_hidden(self, hidden: torch.Tensor | None, batch_size: int) -> None:
        if hidden is not None and tuple(hidden.shape) != (
            1,
            batch_size,
            self.hidden_size,
        ):
            raise ValueError(f"Hidden state must have shape (1, {batch_size}, {self.hidden_size})")

    def forward(self, batch, hidden=None, *, sequence=False):
        if not self.is_recurrent:
            return self.forward_sequences(batch, [len(batch["cells"])], hidden)
        self._check_hidden(hidden, 1 if sequence else len(batch["cells"]))
        x, context_embedding = self.encode_spatial(batch)
        # sequence=True: items are consecutive decisions of one episode.
        recurrent_input = context_embedding[:, None] if sequence else context_embedding[None]
        with torch.autocast(context_embedding.device.type, enabled=False):
            output, hidden = self.memory(recurrent_input.float(), None if hidden is None else hidden.float())
        h = output[:, 0] if sequence else output[0]
        logits, value = self.score_heads(batch, x, h)
        return logits, value, hidden


def distribution_stats(logits, indices):
    log_probs = logits.log_softmax(-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs.masked_fill(~torch.isfinite(log_probs), 0)).sum(-1)
    count = torch.isfinite(logits).sum(-1)
    normalized = torch.where(count > 1, entropy / count.clamp_min(2).float().log(), 0)
    return log_probs.gather(-1, indices[:, None]).squeeze(-1), entropy, normalized
