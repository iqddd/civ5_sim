"""In-place Adam transactions preserve captured CUDA graph tensor identities."""

from copy import deepcopy
import random

import numpy as np
import torch


class AdamSnapshot:
    def __init__(self, model, optimizer):
        self.model = [(v, v.detach().clone()) for v in model.state_dict().values()]
        self.optimizer = optimizer
        self.states = {
            p: {k: (v, v.detach().clone()) if torch.is_tensor(v) else deepcopy(v)
                for k, v in state.items()}
            for p, state in optimizer.state.items()
        }
        self.groups = [{k: deepcopy(v) for k, v in g.items() if k != "params"}
                       for g in optimizer.param_groups]
        self.rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

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
        random.setstate(self.rng[0])
        np.random.set_state(self.rng[1])
        torch.set_rng_state(self.rng[2])
        if self.rng[3]:
            torch.cuda.set_rng_state_all(self.rng[3])
        # A single device reduction also catches accidentally replaced/corrupted buffers.
        if self.model and not torch.stack([
            (current == saved).all() for current, saved in self.model
        ]).all().item():
            raise RuntimeError("Adam rollback failed to restore model exactly")


def require_finite_state(model, optimizer):
    tensors = list(model.state_dict().values()) + [
        v for state in optimizer.state.values() for v in state.values()
        if torch.is_tensor(v)
    ]
    # Adam step tensors may be on CPU while parameters live on CUDA.
    by_device = {}
    for tensor in tensors:
        if tensor.is_floating_point():
            by_device.setdefault(tensor.device, []).append(tensor)
    for values in by_device.values():
        if not torch.stack([torch.isfinite(v).all() for v in values]).all().item():
            raise FloatingPointError("Non-finite model/Adam state after attempt")
