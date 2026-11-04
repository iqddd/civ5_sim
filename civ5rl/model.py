"""Hex message passing, army memory, and scores over public legal commands."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def batch_inputs(items, device, *, non_blocking=False):
    b, n, a = (
        len(items),
        max(len(x.cells) for x in items),
        max(len(x.sources) for x in items),
    )
    cells = np.zeros((b, n, 73), np.float32)
    masks = np.zeros((b, n), bool)
    neighbors = np.full((b, n, 6), -1, np.int64)
    sources = np.full((b, a), -1, np.int64)
    targets = sources.copy()
    action_mask = np.zeros((b, a), bool)
    features = np.zeros((b, a, 17), np.float32)
    for i, x in enumerate(items):
        nc, na = len(x.cells), len(x.sources)
        cells[i, :nc], neighbors[i, :nc], masks[i, :nc] = x.cells, x.neighbors, True
        sources[i, :na], targets[i, :na], action_mask[i, :na] = (
            x.sources,
            x.targets,
            True,
        )
        features[i, :na] = x.action_features
    values = {
        "cells": cells,
        "cell_mask": masks,
        "neighbors": neighbors,
        "globals": np.stack([x.globals for x in items]),
        "previous": np.stack([x.previous for x in items]),
        "sources": sources,
        "targets": targets,
        "action_features": features,
        "action_mask": action_mask,
    }
    return {
        k: torch.from_numpy(v).to(device, non_blocking=non_blocking)
        for k, v in values.items()
    }


class HexBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(7, 128, 128))
        self.bias = nn.Parameter(torch.zeros(128))
        self.norm1 = nn.LayerNorm(128, eps=1e-5)
        self.norm2 = nn.LayerNorm(128, eps=1e-5)
        self.mlp = nn.Sequential(nn.Linear(128, 256), nn.GELU(), nn.Linear(256, 128))

    def forward(self, x, neighbors, mask):
        # Project all seven directions in one GEMM, then gather by direction.
        # W_d x_neighbor == (W_d x)_neighbor; no model/parameter change.
        b, n, _ = x.shape
        projected = F.linear(x, self.weight.reshape(7 * 128, 128)).reshape(b, n, 7, 128)
        batch_index = torch.arange(b, device=x.device)[:, None, None]
        directions = torch.arange(1, 7, device=x.device)[None, None, :]
        messages = projected[batch_index, neighbors.clamp_min(0), directions]
        message = (
            projected[:, :, 0].float()
            + (messages.float() * (neighbors >= 0).unsqueeze(-1)).sum(2)
            + self.bias
        )
        u = x + F.gelu(self.norm1(message))
        return (u + self.mlp(self.norm2(u))) * mask.unsqueeze(-1)


class Policy(nn.Module):
    def __init__(self, variant="C"):
        super().__init__()
        self.variant = variant
        self.projection = nn.Linear(73, 128)
        self.blocks = nn.ModuleList([HexBlock() for _ in range(4)])
        self.context = nn.Sequential(
            nn.Linear(309, 256), nn.GELU(), nn.Linear(256, 256), nn.GELU()
        )
        self.memory = (
            nn.GRU(256, 256)
            if variant == "C"
            else nn.Sequential(
                nn.Linear(256, 768), nn.GELU(), nn.Linear(768, 256), nn.GELU()
            )
        )
        self.action = nn.Sequential(nn.Linear(529, 128), nn.GELU(), nn.Linear(128, 1))
        self.value = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))
        self.activation_checkpointing = False
        self._spatial_runner = None
        self._heads_runner = None
        self.amp_enabled = False
        self.spatial_batch_size = 0
        self._initialize()

    def enable_amp(self, enabled=True, *, spatial_batch_size=0):
        if enabled and (
            next(self.parameters()).device.type != "cuda"
            or not torch.cuda.is_bf16_supported(including_emulation=False)
        ):
            raise ValueError("BF16 AMP requires a CUDA device with native BF16 support")
        self.amp_enabled = enabled
        if type(spatial_batch_size) is not int or spatial_batch_size < 0:
            raise ValueError("Spatial batch size must be a nonnegative integer")
        self.spatial_batch_size = spatial_batch_size if enabled else 0

    def autocast(self, tensor):
        return torch.autocast(
            tensor.device.type, dtype=torch.bfloat16, enabled=self.amp_enabled
        )

    def enable_compile(self, *, dynamic=True, mode="default", recompile_limit=32):
        """Compile tensor-heavy regions; cuDNN GRU stays eager, weights stay portable."""
        # Inference/no-grad replay/training have separate autograd specializations.
        # Keep a bounded cache, and retain fullgraph=True so fallback is never silent.
        torch._dynamo.config.recompile_limit = recompile_limit
        self._spatial_runner = torch.compile(
            self.spatial, dynamic=dynamic, mode=mode, fullgraph=True
        )
        self._heads_runner = torch.compile(
            self.heads, dynamic=dynamic, mode=mode, fullgraph=True
        )

    def encode_spatial(self, batch):
        with self.autocast(batch["cells"]):
            return self._encode_spatial(batch)

    def _encode_spatial(self, batch):
        if self.spatial_batch_size:
            # BF16 GEMM output rounding can vary with the row count. Use the
            # same GEMM shapes during action selection, replay and gradients.
            # Padding rows are independent and discarded before memory/loss.
            size = self.spatial_batch_size
            keys = ("cells", "cell_mask", "neighbors", "globals", "previous")
            outputs = []
            runner = self._spatial_runner or self.spatial
            for start in range(0, len(batch["cells"]), size):
                part = {k: batch[k][start : start + size] for k in keys}
                count = len(part["cells"])
                if count < size:
                    part = {
                        k: torch.cat((v, v[:1].expand(size - count, *v.shape[1:])), 0)
                        for k, v in part.items()
                    }
                x, ctx = runner(part)
                outputs.append((x[:count], ctx[:count]))
            return tuple(
                torch.cat([output[i] for output in outputs], 0) for i in range(2)
            )
        if self._spatial_runner is not None and len(batch["cells"]) == 1:
            padded = {k: torch.cat((v, v), 0) for k, v in batch.items()}
            x, ctx = self._spatial_runner(padded)
            return x[:1], ctx[:1]
        return (self._spatial_runner or self.spatial)(batch)

    def score_heads(self, batch, x, h):
        with self.autocast(x):
            return self._score_heads(batch, x, h)

    def _score_heads(self, batch, x, h):
        if self._heads_runner is None:
            return self.heads(batch, x, h)
        rows, actions = len(x), batch["sources"].shape[1]
        if rows == 1:
            batch = {k: torch.cat((v, v), 0) for k, v in batch.items()}
            x, h = torch.cat((x, x), 0), torch.cat((h, h), 0)
        if actions == 1:
            batch = dict(batch)
            for key in ("sources", "targets", "action_features", "action_mask"):
                value = batch[key]
                fill = -1 if key in ("sources", "targets") else 0
                batch[key] = torch.cat((value, torch.full_like(value, fill)), 1)
        logits, value = self._heads_runner(batch, x, h)
        return logits[:rows, :actions], value[:rows]

    def sequence_memory(self, ctx, lengths, hidden=None):
        if self.variant != "C":
            with self.autocast(ctx):
                return self.memory(ctx).float(), None
        # CUDA autocast can select FP16 for cuDNN RNN even inside BF16 autocast.
        # Keep both GRU arithmetic and stored army memory explicitly in FP32.
        with torch.autocast(ctx.device.type, enabled=False):
            sequences = ctx.float().split(lengths)
            padded = nn.utils.rnn.pad_sequence(sequences)
            output, _ = self.memory(padded, None if hidden is None else hidden.float())
        valid = torch.cat([output[:length, i] for i, length in enumerate(lengths)])
        last = torch.stack([output[length - 1, i] for i, length in enumerate(lengths)])[
            None
        ]
        return valid, last

    def forward_sequences(self, batch, lengths, hidden=None, *, memory_only=False):
        x, ctx = self.encode_spatial(batch)
        h, last = self.sequence_memory(ctx, lengths, hidden)
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
        if self.variant == "C":
            nn.init.xavier_uniform_(self.memory.weight_ih_l0)
            for w in self.memory.weight_hh_l0.chunk(3):
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
        mean = x.sum(1) / mask.sum(1, keepdim=True)
        maximum = x.masked_fill(~mask.unsqueeze(-1), -torch.inf).max(1).values
        with torch.autocast(x.device.type, enabled=False):
            ctx = self.context(
                torch.cat(
                    (mean, maximum, batch["globals"], batch["previous"]), -1
                ).float()
            )
        return x, ctx

    def heads(self, batch, x, h):
        b, n, _ = x.shape
        offsets = torch.arange(b, device=x.device)[:, None] * n
        flat = x.reshape(b * n, 128)
        embeddings = []
        for key in ("sources", "targets"):
            idx = batch[key]
            embeddings.append(
                flat[(idx.clamp_min(0) + offsets).reshape(-1)].reshape(b, -1, 128)
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
            action_hidden = self.action[1](self.action[0](action_input.float()))
            value_hidden = self.value[1](self.value[0](h.float()))
            logits = (
                self.action[2](action_hidden.float())
                .squeeze(-1)
                .masked_fill(~batch["action_mask"], -torch.inf)
            )
            value = self.value[2](value_hidden.float()).squeeze(-1)
        return logits, value

    def forward(self, batch, hidden=None, *, sequence=False):
        x, ctx = self.encode_spatial(batch)
        if self.variant == "C":
            # sequence=True: items are consecutive decisions of one episode.
            recurrent_input = ctx[:, None] if sequence else ctx[None]
            with torch.autocast(ctx.device.type, enabled=False):
                output, hidden = self.memory(
                    recurrent_input.float(), None if hidden is None else hidden.float()
                )
            h = output[:, 0] if sequence else output[0]
        else:
            with self.autocast(ctx):
                h, hidden = self.memory(ctx).float(), None
        logits, value = self.score_heads(batch, x, h)
        return logits, value, hidden


def distribution_stats(logits, indices):
    log_probs = logits.log_softmax(-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs.masked_fill(~torch.isfinite(log_probs), 0)).sum(-1)
    count = torch.isfinite(logits).sum(-1)
    normalized = torch.where(count > 1, entropy / count.clamp_min(2).float().log(), 0)
    return log_probs.gather(-1, indices[:, None]).squeeze(-1), entropy, normalized
