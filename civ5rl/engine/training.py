"""Reusable checks and optimizer operations for behavioral cloning."""

import torch


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
