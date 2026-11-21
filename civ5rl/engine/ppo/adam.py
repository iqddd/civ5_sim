"""Named Adam groups and inherited counters for PPO branches."""

from __future__ import annotations

import torch

GROUP_ORDER = ("shared_actor", "value", "attack_preview", "action_context")


def adapts_value(config):
    return config.get("experiment", {}).get("initial_value") == "adapt_parent_first_rollout"


def uses_grouped_adam(config):
    return adapts_value(config) or bool(config.get("experiment", {}).get("parent_optimizer_counters"))


def parameter_group(name):
    if name.startswith(("action_attention.", "value_actions.")):
        return "action_context"
    return (
        "attack_preview"
        if name.startswith("attack_preview.")
        else "value"
        if name.startswith("value.")
        else "shared_actor"
    )


def group_names(variant_spec):
    names = ["shared_actor", "value"]
    if variant_spec.attack_preview:
        names.append("attack_preview")
    if variant_spec.action_context:
        names.append("action_context")
    return names


def optimizer_for(policy, config, device):
    optimizer_config = config["optimizer"]
    return torch.optim.Adam(
        policy.parameters(),
        lr=optimizer_config["lr"],
        betas=tuple(optimizer_config["betas"]),
        eps=optimizer_config["eps"],
        weight_decay=optimizer_config["weight_decay"],
        fused=str(device).startswith("cuda"),
    )


def grouped_optimizer(policy, config, device):
    settings = config["optimizer"]
    named = list(policy.named_parameters())
    groups = []
    for name in group_names(policy.variant_spec):
        members = [(n, p) for n, p in named if parameter_group(n) == name]
        groups.append(
            {
                "params": [p for _, p in members],
                "name": name,
                "parameter_names": [n for n, _ in members],
            }
        )
    return torch.optim.Adam(
        groups,
        lr=settings["lr"],
        betas=tuple(settings["betas"]),
        eps=settings["eps"],
        weight_decay=settings["weight_decay"],
        fused=str(device).startswith("cuda"),
    )


def expected_counters(state, extra_ppo_steps=0):
    accepted = int(state.get("optimizer_steps", 0)) + extra_ppo_steps
    if "origin_Adam_counters" in state:
        critic_steps = (
            int(state.get("critic_progress", {}).get("steps", 0))
            if state.get("value_optimizer") == "reset_then_preserve"
            else 0
        )
        return {
            name: count + accepted + (critic_steps if name == "value" else 0)
            for name, count in state["origin_Adam_counters"].items()
        }
    return {
        "shared_actor": int(state.get("origin_Adam_counter", 0)) + accepted,
        "value": int(state.get("critic_progress", {}).get("steps", 0)) + accepted,
    }


def check_counters(optimizer, state, extra_ppo_steps=0):
    expected = expected_counters(state, extra_ppo_steps)
    names = [name for name in GROUP_ORDER if name in expected]
    if set(names) != set(expected):
        raise ValueError("Unknown Adam group in recorded origins")
    if [g.get("name") for g in optimizer.param_groups] != names:
        raise ValueError("Adam groups differ from recorded origins")
    for group in optimizer.param_groups:
        count = expected[group["name"]]
        for parameter in group["params"]:
            item = optimizer.state.get(parameter, {})
            if (not item and count != 0) or (item and int(item["step"]) != count):
                raise ValueError(f"Adam counter differs for {group['name']}: expected {count}")
    return expected
