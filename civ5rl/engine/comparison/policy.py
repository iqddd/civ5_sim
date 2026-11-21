"""Policy setup and validated optimizer steps for pairwise comparison fits."""

from __future__ import annotations

import math
from functools import cache

import torch

from civ5rl.core.agent import RLAgent
from civ5rl.core.model import Policy
from civ5rl.engine.runtime.transactions import clip_and_step_checked

ATTENTION_BACKEND = "compact_fp32"
TRAINABLE_HEADS = ("action", "attack_preview")


def configure_policy(checkpoint, device="cuda", mixed=True):
    policy = RLAgent.from_checkpoint(checkpoint, "cpu", "argmax").policy.to(device).train()
    policy.enable_amp(mixed, spatial_batch_size=32 if mixed else 0)
    policy.enable_inverse_routing()
    policy.enable_action_routing()
    policy.enable_maximum_routing()
    policy.configure_execution(activation_checkpointing=False, attention_backend=ATTENTION_BACKEND)
    if str(device).startswith("cuda"):
        # Fuse eager kernels without introducing private CUDA Graph pools.
        policy.enable_compile(dynamic=True, mode="default", spatial_mode="default")
    for parameter in policy.value.parameters():
        parameter.requires_grad_(False)
    return policy


def _enable_head_training(policy):
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(name.split(".", 1)[0] in TRAINABLE_HEADS)
    return policy


def configure_heads(checkpoint, device="cuda"):
    return _enable_head_training(configure_policy(checkpoint, device=device, mixed=False))


@cache
def head_parameter_count():
    # Rollout comparison fits C2P. Meta avoids touching training RNG or device RAM.
    with torch.device("meta"):
        policy = _enable_head_training(Policy("C2P"))
    return sum(parameter.numel() for parameter in policy.parameters() if parameter.requires_grad)


def step_comparison_policy(policy, optimizer, pair_total, kl_total, *, gradient_error):
    step = clip_and_step_checked(
        policy,
        optimizer,
        (parameter for parameter in policy.parameters() if parameter.requires_grad),
        extra_metrics=(pair_total, kl_total),
        asynchronous=pair_total.is_cuda,
        gradient_error=gradient_error,
    )
    finite = torch.ones((), device=pair_total.device, dtype=torch.bool)
    for device, flag in step.finite_flags.items():
        if device == finite.device:
            finite &= flag
        elif not flag.item():
            raise FloatingPointError("Non-finite model/Adam state after attempt")
    measured = torch.stack((*step.metrics, finite.to(step.gradient_norm.dtype))).detach().cpu().tolist()
    if not measured[-1] or not all(math.isfinite(value) for value in measured[:-1]):
        raise FloatingPointError("Non-finite loss/gradient/model/Adam after attempt")
    return dict(zip(("pair_loss", "anchor_kl", "gradient_norm"), measured[:-1]))
