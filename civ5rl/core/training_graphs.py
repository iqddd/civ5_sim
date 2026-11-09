"""Opt-in first-order training paths, preserving block sizes and cuDNN GRU math."""

import copy
from collections import Counter, OrderedDict

import torch
from torch.autograd.function import once_differentiable

from civ5rl.core.spatial import SPATIAL_KEYS, iter_spatial_blocks


CUDA_GRAPH_WARMUP_STEPS = 3


class CoalescedSpatial(torch.autograd.Function):
    """Run the usual block graphs; sum their parameter gradients with foreach."""

    @staticmethod
    def forward(ctx, policy, batch, *parameters):
        if any(batch[key].requires_grad for key in SPATIAL_KEYS):
            raise ValueError(
                "Coalesced spatial training expects constant observation inputs"
            )
        size = policy.spatial_batch_size
        outputs, counts = [], []
        runner = policy._spatial_runner or policy.spatial
        with torch.enable_grad():
            for part, count in iter_spatial_blocks(batch, size):
                outputs.append(runner(part))
                counts.append(count)
        ctx.parameter_count = len(parameters)
        ctx.counts = counts
        ctx.save_for_backward(
            *parameters, *(value for pair in outputs for value in pair)
        )
        return tuple(
            torch.cat([pair[j][:count] for pair, count in zip(outputs, counts)], 0)
            for j in range(2)
        )

    @staticmethod
    @once_differentiable
    def backward(ctx, dx, dcontext):
        saved = ctx.saved_tensors
        parameters, outputs = saved[: ctx.parameter_count], saved[ctx.parameter_count :]
        offset = sum(ctx.counts)
        accumulated = None
        for index in reversed(range(len(ctx.counts))):
            count = ctx.counts[index]
            offset -= count
            pair = outputs[2 * index : 2 * index + 2]
            derivatives = []
            for derivative, target in zip((dx, dcontext), pair):
                sliced = derivative[offset : offset + count]
                if count < len(target):
                    sliced = torch.cat((sliced, torch.zeros_like(target[count:])), 0)
                derivatives.append(sliced)
            gradients = torch.autograd.grad(pair, parameters, grad_outputs=derivatives)
            if accumulated is None:
                # Graph replay can reuse its output buffers; own the accumulated sum.
                accumulated = torch._foreach_mul(gradients, 1.0)
            else:
                torch._foreach_add_(accumulated, gradients)
        return None, None, *accumulated


class _GRUEntry:
    def __init__(self, module, x, hidden):
        # Private leaves avoid stale AccumulateGrad streams from the outer graph.
        # Current policy weights are copied before every replay, without Adam state.
        self.module = copy.deepcopy(module)
        self.module.flatten_parameters()
        self.parameters = tuple(self.module.parameters())
        self.x = x.detach().clone().requires_grad_()
        self.hidden = hidden.detach().clone().requires_grad_()
        self.dy = torch.zeros_like(x)
        self.dh = torch.zeros_like(hidden)
        self.active = False
        inputs = (self.x, self.hidden, *self.parameters)
        stream = torch.cuda.Stream(device=x.device)
        stream.wait_stream(torch.cuda.current_stream(x.device))
        with torch.cuda.stream(stream), torch.enable_grad():
            for _ in range(CUDA_GRAPH_WARMUP_STEPS):
                outputs = self.module(self.x, self.hidden)
                torch.autograd.grad(outputs, inputs, (self.dy, self.dh))
        torch.cuda.current_stream(x.device).wait_stream(stream)
        self.forward_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.forward_graph, stream=stream, capture_error_mode="thread_local"), torch.enable_grad():
            self.outputs = self.module(self.x, self.hidden)
        self.backward_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.backward_graph, stream=stream, capture_error_mode="thread_local"):
            self.gradients = torch.autograd.grad(
                self.outputs, inputs, (self.dy, self.dh), retain_graph=True
            )


class _GRUReplay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, entry, x, hidden, *parameters):
        if entry.active:
            raise RuntimeError(
                "GRU graph replay requires backward before reusing the same shape"
            )
        entry.active = True
        entry.x.copy_(x)
        entry.hidden.copy_(hidden)
        torch._foreach_copy_(entry.parameters, parameters)
        entry.forward_graph.replay()
        ctx.entry = entry
        ctx.save_for_backward(*parameters)
        # Subsequent replay must not overwrite outputs retained by the caller.
        return tuple(value.clone() for value in entry.outputs)

    @staticmethod
    @once_differentiable
    def backward(ctx, dy, dh):
        _ = ctx.saved_tensors  # detect parameter mutation between forward/backward
        entry = ctx.entry
        entry.dy.copy_(dy)
        entry.dh.copy_(dh)
        entry.backward_graph.replay()
        gradients = torch._foreach_mul(entry.gradients, 1.0)
        entry.active = False
        return None, *gradients


class RecurrentGraphs:
    """Bounded shape cache for sequential FP32 GRU forward/backward microbatches."""

    def __init__(self, module, *, max_shapes=64):
        if (
            not isinstance(module, torch.nn.GRU)
            or module.num_layers != 1
            or module.bidirectional
            or module.dropout != 0
            or module.batch_first
            or module.input_size != module.hidden_size
            or any(
                p.device.type != "cuda" or p.dtype != torch.float32
                for p in module.parameters()
            )
        ):
            raise ValueError(
                "GRU graphs require the single-layer FP32 CUDA model C GRU"
            )
        if type(max_shapes) is not int or max_shapes < 1:
            raise ValueError("GRU graph max_shapes must be a positive integer")
        self.module, self.max_shapes = module, max_shapes
        self.entries = OrderedDict()
        self._seen = Counter()
        self.captures = 0

    def __call__(self, x, hidden=None):
        if not torch.is_grad_enabled():
            return self.module(x, hidden)
        if x.dtype != torch.float32:
            raise ValueError("GRU graphs require FP32 recurrent inputs")
        if hidden is None:
            hidden = torch.zeros(
                (1, x.shape[1], self.module.hidden_size), device=x.device, dtype=x.dtype
            )
        key = (tuple(x.shape), tuple(hidden.shape), x.device)
        self._seen[key] += 1
        if key not in self.entries:
            if len(self.entries) >= self.max_shapes:
                if self.max_shapes == 1:
                    oldest = next(iter(self.entries))
                    if self.entries[oldest].active:
                        raise RuntimeError("Cannot evict a GRU graph awaiting backward")
                    del self.entries[oldest]
                else:
                    inactive = [k for k, e in self.entries.items() if not e.active]
                    if not inactive:
                        raise RuntimeError("Cannot evict a GRU graph awaiting backward")
                    victim = min(inactive, key=lambda k: self._seen[k])
                    if self._seen[key] <= self._seen[victim]:
                        return self.module(x, hidden)
                    del self.entries[victim]
            elif self.max_shapes > 1 and self._seen[key] < 2:
                return self.module(x, hidden)
            self.entries[key] = _GRUEntry(self.module, x, hidden)
            self.captures += 1
        self.entries.move_to_end(key)
        return _GRUReplay.apply(self.entries[key], x, hidden, *self.module.parameters())
