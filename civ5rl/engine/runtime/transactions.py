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


def _tensor_groups(tensors):
    groups = defaultdict(list)
    for tensor in tensors:
        groups[tensor.device, tensor.dtype].append(tensor)
    return groups


class AdamSnapshot:
    def __init__(self, model, optimizer):
        self.optimizer = optimizer
        model_tensors = list(model.state_dict().values())
        tensors = model_tensors + [
            value for state in optimizer.state.values() for value in state.values()
            if torch.is_tensor(value)
        ]
        backups = {}
        self._foreach = []
        self._individual = []
        for (device, dtype), values in _tensor_groups(tensors).items():
            group, storages = [], set()
            for value in values:
                if value.layout == torch.strided and value.is_contiguous():
                    storage = value.untyped_storage().data_ptr()
                    if storage not in storages:
                        storages.add(storage)
                        group.append(value)
                        continue
                # Tied weights may overlap another destination during restore.
                backup = value.detach().clone()
                self._individual.append((value, backup))
                backups[id(value)] = backup
            if not group:
                continue
            sources = [v.detach() for v in group]
            sizes = [v.numel() for v in group]
            flat = torch.empty(sum(sizes), device=device, dtype=dtype)
            saved = [part.view(v.shape) for part, v in zip(flat.split(sizes), group)]
            torch._foreach_copy_(saved, sources)
            self._foreach.append((sources, saved))
            backups.update((id(v), backup) for v, backup in zip(group, saved))
        self.model = [(v, backups[id(v)]) for v in model_tensors]
        self.states = {
            p: {
                k: (v, backups[id(v)]) if torch.is_tensor(v) else deepcopy(v)
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
        for current, saved in self._foreach:
            torch._foreach_copy_(current, saved)
        for current, saved in self._individual:
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
                    original, _ = value
                    state[k] = original
                else:
                    state[k] = deepcopy(value)
        for group, saved in zip(self.optimizer.param_groups, self.groups):
            for k in list(group):
                if k != "params" and k not in saved:
                    del group[k]
            group.update(deepcopy(saved))
        restore_rng(self.rng)


def finite_state_flags(model, optimizer):
    """Device-resident results let callers combine checks with their metric readback."""
    tensors = list(model.state_dict().values()) + [
        v
        for state in optimizer.state.values()
        for v in state.values()
        if torch.is_tensor(v)
    ]
    # Infinity norm detects NaN/Inf without overflowing on finite entries.
    # Optimizer counters may be on the CPU and empty tensors are always finite.
    by_device = defaultdict(list)
    for (device, _), group in _tensor_groups(
        v.detach() for v in tensors if v.is_floating_point() and v.numel()
    ).items():
        if device.type == "cuda" and all(v.layout == torch.strided for v in group):
            norms = torch._foreach_norm(group, float("inf"))
            by_device[device].append(torch.isfinite(torch.stack(norms)).all())
        else:
            by_device[device].extend(torch.isfinite(v).all() for v in group)
    return {device: torch.stack(flags).all() for device, flags in by_device.items()}


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
