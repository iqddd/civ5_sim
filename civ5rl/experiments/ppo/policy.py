"""Policy initialization for PPO after own-state imitation."""

from __future__ import annotations

import torch

from civ5rl.experiments.imitation.helpers import load_policy
from civ5rl.provenance.common import ROOT


def create_policy(config, device):
    policy = load_policy(config, ROOT / config["experiment"]["parent"], device)
    for parameter in policy.value.parameters():
        parameter.requires_grad_(True)
    with torch.no_grad():
        policy.value[-1].weight.zero_()
        policy.value[-1].bias.zero_()
    return policy
