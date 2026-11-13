"""Named Adam groups and inherited counters for PPO branches."""

import torch


def adapts_value(config):
    return config.get("experiment", {}).get("initial_value") == "adapt_parent_first_rollout"


def uses_grouped_adam(config):
    return adapts_value(config) or bool(config.get("experiment", {}).get("parent_optimizer_counters"))


def grouped_optimizer(policy, config, device):
    settings = config["optimizer"]
    named = list(policy.named_parameters())
    groups = []
    for name, is_value in (("shared_actor", False), ("value", True)):
        members = [(n, p) for n, p in named if n.startswith("value.") == is_value]
        groups.append(dict(params=[p for _, p in members], name=name,
                           parameter_names=[n for n, _ in members]))
    return torch.optim.Adam(groups, lr=settings["lr"], betas=tuple(settings["betas"]),
                            eps=settings["eps"], weight_decay=settings["weight_decay"],
                            fused=str(device).startswith("cuda"))


def expected_counters(state, extra_ppo_steps=0):
    accepted = int(state.get("optimizer_steps", 0)) + extra_ppo_steps
    if "origin_Adam_counters" in state:
        return {name: count + accepted for name, count in state["origin_Adam_counters"].items()}
    return dict(shared_actor=int(state.get("origin_Adam_counter", 0)) + accepted,
                value=int(state.get("critic_progress", {}).get("steps", 0)) + accepted)


def check_counters(optimizer, state, extra_ppo_steps=0):
    expected = expected_counters(state, extra_ppo_steps)
    if [g.get("name") for g in optimizer.param_groups] != ["shared_actor", "value"]:
        raise ValueError("Expected shared_actor/value Adam groups")
    for group in optimizer.param_groups:
        count = expected[group["name"]]
        for parameter in group["params"]:
            item = optimizer.state.get(parameter, {})
            if (not item and count != 0) or (item and int(item["step"]) != count):
                raise ValueError(f"Adam counter differs for {group['name']}: expected {count}")
    return expected
