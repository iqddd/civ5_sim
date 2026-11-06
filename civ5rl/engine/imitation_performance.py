"""Optional resident BC inputs and fewer host synchronizations; same CE objective."""

import math

import torch
from torch.nn import functional as F

from civ5rl.engine.constants import (
    DEFAULT_BC_MICROBATCH_SIZE,
    DEFAULT_GPU_RESERVE_BYTES,
)
from civ5rl.engine.imitation import decision_indices
from civ5rl.engine.training import GradientAccumulator
from civ5rl.engine.ppo_fast import prefixes
from civ5rl.engine.sequence import SequenceBuffer


class ResidentDemonstrations(SequenceBuffer):
    def __init__(self, base, *, deadline=None, reserve_bytes=DEFAULT_GPU_RESERVE_BYTES):
        self.episodes, self.masks = base.episodes, base.masks
        super().__init__(
            base.encoded,
            base.device,
            resident=True,
            deadline=deadline,
            reserve_bytes=reserve_bytes,
        )
        self.labels = {
            "indices": torch.tensor(
                [
                    decision_indices(r)[0]
                    for ep in self.episodes
                    for r in ep["decisions"]
                ],
                device=self.device,
            ),
            "mask": torch.tensor(
                [r["supervised"] for ep in self.episodes for r in ep["decisions"]],
                dtype=torch.bool,
                device=self.device,
            ),
        }

    def batch(self, spans):
        indices = self.indices(spans)
        return (
            self.gather_inputs(spans, indices),
            {k: v.index_select(0, indices) for k, v in self.labels.items()},
            [b - a for _, a, b in spans],
        )


def backward_minibatch_async(
    policy, cache, spans, count, micro_size=DEFAULT_BC_MICROBATCH_SIZE
):
    """Same accumulation; return GPU loss and check finiteness without host waiting."""
    if policy.spatial_cuda_graphs:
        torch.compiler.cudagraph_mark_step_begin()
    hidden = prefixes(policy, cache, spans)
    total = torch.zeros((), device=cache.device)
    accumulator = GradientAccumulator(
        policy.parameters(), coalesced=policy.coalesced_accumulation
    )
    for first in range(0, len(spans), micro_size):
        part = spans[first : first + micro_size]
        if not any(any(cache.masks[ep][a:b]) for ep, a, b in part):
            continue
        batch, labels, lengths = cache.batch(part)
        logits, _, _ = policy.forward_sequences(
            batch, lengths, hidden[:, first : first + len(part)].contiguous()
        )
        losses = F.cross_entropy(logits, labels["indices"], reduction="none")
        loss = (losses * labels["mask"]).sum()
        accumulator.backward(loss / count)
        total.add_(loss.detach())
    result = total / count
    torch._assert_async(torch.isfinite(result), "Non-finite BC loss")
    return result


def backward_minibatch_fast(
    policy, cache, spans, count, micro_size=DEFAULT_BC_MICROBATCH_SIZE
):
    result = float(backward_minibatch_async(policy, cache, spans, count, micro_size))
    if not math.isfinite(result):
        raise FloatingPointError("Non-finite BC loss")
    return result
