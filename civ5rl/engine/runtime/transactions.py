"""Optimizer transactions, independent checkpoint snapshots and gradient accumulation."""

import shutil
from collections import defaultdict
from copy import deepcopy

import torch

from civ5rl.provenance.common import atomic_output
from civ5rl.provenance.rng import capture_rng, restore_rng


def atomic_torch_save(value, path, *copies):
    """Publish one independent CPU snapshot and copy the same bytes to each target."""
    value = cpu_snapshot(value)
    with atomic_output(path) as temporary:
        # File-object serialization uses a stable zip archive name.
        with temporary.open("wb") as stream:
            torch.save(value, stream)
    for target in copies:
        with atomic_output(target) as temporary:
            shutil.copyfile(path, temporary)
    return value


class AdamSnapshot:
    def __init__(self, model, optimizer):
        self.model = [(v, v.detach().clone()) for v in model.state_dict().values()]
        self.optimizer = optimizer
        self.states = {
            p: {
                k: (v, v.detach().clone()) if torch.is_tensor(v) else deepcopy(v)
                for k, v in state.items()
            }
            for p, state in optimizer.state.items()
        }
        self.groups = [
            {k: deepcopy(v) for k, v in g.items() if k != "params"}
            for g in optimizer.param_groups
        ]
        self.rng = capture_rng()

    @torch.no_grad()
    def restore(self):
        for current, saved in self.model:
            current.copy_(saved)
        for p in list(self.optimizer.state):
            if p not in self.states:
                del self.optimizer.state[p]
        for p, saved in self.states.items():
            state = self.optimizer.state[p]
            for k in list(state):
                if k not in saved:
                    del state[k]
            for k, value in saved.items():
                if isinstance(value, tuple) and torch.is_tensor(value[0]):
                    original, backup = value
                    original.copy_(backup)
                    state[k] = original
                else:
                    state[k] = deepcopy(value)
        for group, saved in zip(self.optimizer.param_groups, self.groups):
            for k in list(group):
                if k != "params" and k not in saved:
                    del group[k]
            group.update(deepcopy(saved))
        restore_rng(self.rng)
        # A single device reduction also catches accidentally replaced/corrupted buffers.
        if (
            self.model
            and not torch.stack(
                [(current == saved).all() for current, saved in self.model]
            )
            .all()
            .item()
        ):
            raise RuntimeError("Adam rollback failed to restore model exactly")


def finite_state_flags(model, optimizer):
    """Device-resident results let callers combine checks with their metric readback."""
    tensors = list(model.state_dict().values()) + [
        v
        for state in optimizer.state.values()
        for v in state.values()
        if torch.is_tensor(v)
    ]
    # Adam step tensors may be on CPU while parameters live on CUDA.
    by_device = {}
    for tensor in tensors:
        if tensor.is_floating_point():
            by_device.setdefault(tensor.device, []).append(tensor)
    return {
        device: torch.stack([torch.isfinite(v).all() for v in values]).all()
        for device, values in by_device.items()
    }


def require_finite_state(model, optimizer):
    for flag in finite_state_flags(model, optimizer).values():
        if not flag.item():
            raise FloatingPointError("Non-finite model/Adam state after attempt")


def cpu_snapshot(value):
    """Freeze tensor values once, grouping CUDA tensors by device and dtype.

    Each CUDA group is flattened for one D2H transfer, then rebuilt as CPU views.
    CPU tensors and non-tensor metadata are copied independently of live state.
    Checkpoints consequently contain portable CPU tensors instead of CUDA tags.
    """
    groups = defaultdict(list)
    copies = {}

    def collect(item):
        if torch.is_tensor(item):
            if id(item) in copies:
                return
            copies[id(item)] = None
            if item.device.type == "cpu":
                copies[id(item)] = item.detach().clone()
            else:
                groups[(item.device, item.dtype)].append(item)
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                collect(child)

    collect(value)
    for tensors in groups.values():
        flat = torch.cat([t.detach().reshape(-1) for t in tensors]).cpu()
        offset = 0
        for tensor in tensors:
            count = tensor.numel()
            copies[id(tensor)] = flat[offset : offset + count].view(tensor.shape)
            offset += count

    def rebuild(item):
        if torch.is_tensor(item):
            return copies[id(item)]
        if isinstance(item, dict):
            result = type(item)((key, rebuild(child)) for key, child in item.items())
            if hasattr(item, "_metadata"):
                result._metadata = deepcopy(item._metadata)
            return result
        if isinstance(item, list):
            return [rebuild(child) for child in item]
        if isinstance(item, tuple):
            return tuple(rebuild(child) for child in item)
        return deepcopy(item)

    return rebuild(value)


class FrozenParamsCheck:
    def __init__(self, module):
        self.module = module
        self.before = {
            name: value.detach().clone() for name, value in module.state_dict().items()
        }

    def verify(self):
        if any(
            not torch.equal(self.before[name], value)
            for name, value in self.module.state_dict().items()
        ):
            raise RuntimeError("Frozen value head parameters changed")


def bc_optimizer_step(
    policy,
    optimizer,
    backward,
    cache,
    spans,
    count,
    *,
    max_gradient_norm=0.5,
    asynchronous=False,
):
    optimizer.zero_grad(set_to_none=True)
    ce = backward(policy, cache, spans, count)
    norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(), max_gradient_norm, error_if_nonfinite=not asynchronous
    )
    if asynchronous:
        torch._assert_async(torch.isfinite(norm), "Non-finite gradient")
    optimizer.step()
    return ce, norm


class GradientAccumulator:
    """Accumulate scaled losses for one minibatch."""

    def __init__(self, parameters, *, coalesced: bool):
        self.parameters = (
            tuple(p for p in parameters if p.requires_grad) if coalesced else None
        )
        self.accumulated = None

    def backward(self, scaled_loss):
        if self.parameters is None:
            scaled_loss.backward()
            return
        gradients = torch.autograd.grad(scaled_loss, self.parameters)
        if self.accumulated is None:
            self.accumulated = list(gradients)
            for parameter, gradient in zip(self.parameters, self.accumulated):
                parameter.grad = gradient
        else:
            torch._foreach_add_(self.accumulated, gradients)
