"""Validate inherited Adam groups and restore or reset optimizer state."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from civ5rl.core.model import Policy
from civ5rl.core.preview import ModelExtension, model_extension
from civ5rl.engine.ppo.adam import (
    adapts_value,
    group_lr,
    grouped_optimizer,
    optimizer_for,
    parameter_group,
)
from civ5rl.experiments.imitation.helpers import assert_equal


def _validate_inherited_groups(
    optimizer: torch.optim.Adam,
    source: dict[str, Any],
    origins: dict[str, int],
    saved_state: dict[str, Any],
    *,
    extension: ModelExtension | None,
) -> None:
    expected_names = {group["name"] for group in optimizer.param_groups}
    if extension:
        expected_names = {g["name"] for g in optimizer.param_groups
                          if not set(g["parameter_names"]) <= extension.new_parameters}
    if set(origins) != expected_names or any(
        type(counter) is not int or counter < 0 for counter in origins.values()
    ):
        raise ValueError("Invalid inherited Adam group counters")
    if saved_state.get("Adam_counters") != origins:
        raise ValueError("Parent ledger and inherited Adam group counters differ")
    expected_groups = optimizer.state_dict()["param_groups"]
    if extension:
        expected_groups = [g for g in expected_groups if g["name"] in expected_names]
    if len(source["param_groups"]) != len(expected_groups):
        raise ValueError("Parent Adam group count differs")
    for actual, expected in zip(source["param_groups"], expected_groups, strict=True):
        if any(actual.get(key) != expected[key] for key in ("name", "parameter_names", "params")):
            raise ValueError("Parent Adam parameter layout differs")


def _validate_adam_state_tensors(
    policy: Policy,
    optimizer: torch.optim.Adam,
    origins: dict[str, int] | None,
    expected: int,
    *,
    extension: ModelExtension | None,
) -> None:
    inherited = {
        p for name, p in policy.named_parameters()
        if extension is None or name not in extension.new_parameters
    }
    if set(optimizer.state) != inherited:
        raise ValueError("Parent Adam must contain all model and critic parameters")
    expected_by_parameter = {
        parameter: origins.get(group["name"], 0) if origins else expected
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    for parameter, moments in optimizer.state.items():
        if float(moments["step"]) != expected_by_parameter[parameter]:
            raise ValueError("Parent Adam counters differ")
        for name in ("exp_avg", "exp_avg_sq"):
            tensor = moments[name]
            if (
                tensor.shape != parameter.shape
                or tensor.dtype != parameter.dtype
                or not torch.isfinite(tensor).all()
            ):
                raise ValueError(f"Invalid parent Adam {name}")


def _reset_value_optimizer(
    policy: Policy, config: dict[str, Any], inherited: torch.optim.Adam
) -> torch.optim.Adam:
    if len(inherited.param_groups) > 1:
        value_group = next(group for group in inherited.param_groups if group["name"] == "value")
        for parameter in value_group["params"]:
            inherited.state.pop(parameter, None)
        return inherited

    optimizer: torch.optim.Adam = grouped_optimizer(policy, config, "cpu")
    options = {key: value for key, value in inherited.param_groups[0].items() if key != "params"}
    for group in optimizer.param_groups:
        group.update(options)
    # Transfer inherited moments in model order and leave the value group empty.
    for name, parameter in policy.named_parameters():
        if parameter_group(name) != "value":
            optimizer.state[parameter] = inherited.state[parameter]
    return optimizer


def load_initial_optimizer(policy: Policy, saved: dict[str, Any], config: dict[str, Any]) -> torch.optim.Adam:
    """Preserve a PPO parent's complete Adam, or start the original BC handoff."""
    experiment = config["experiment"]
    origins = experiment.get("parent_optimizer_counters")
    extension = model_extension(config)
    optimizer: torch.optim.Adam = (
        grouped_optimizer(policy, config, "cpu") if origins else optimizer_for(policy, config, "cpu")
    )
    if experiment.get("parent_optimizer") != "preserve":
        if optimizer.state:
            raise ValueError("New PPO Adam must be empty")
        return optimizer
    if (
        (
            saved.get("variant") != policy.variant
            and not (extension and (saved.get("variant"), policy.variant) == (extension.parent, extension.child))
        )
        or policy.variant != config["model"]["variant"]
        or not saved.get("state", {}).get("critic_complete")
    ):
        raise ValueError("PPO parent architecture/critic boundary is incompatible")
    if saved["state"].get("phase") != "collect":
        raise ValueError("PPO parent must be a completed update boundary")
    source = saved["optimizer"]
    if origins:
        _validate_inherited_groups(optimizer, source, origins, saved["state"], extension=extension)
    if extension:
        source = deepcopy(source)
        extras = optimizer.state_dict()["param_groups"][len(source["param_groups"]):]
        if {name for g in extras for name in g["parameter_names"]} != extension.new_parameters:
            raise ValueError("Unexpected parameters in new Adam group")
        for extra in extras:
            if extension.child == "C2PA":
                extra["fused"] = config["performance"].get("fused_adam", False)
        source["param_groups"].extend(extras)
    optimizer.load_state_dict(source)
    assert_equal(optimizer.state_dict(), source)
    expected = int(experiment["parent_optimizer_steps"])
    if not origins and saved["state"]["optimizer_steps"] != expected:
        raise ValueError("Parent ledger and configured Adam counter differ")
    _validate_adam_state_tensors(policy, optimizer, origins, expected, extension=extension)
    parent_lr = experiment.get("parent_optimizer_lr", config["optimizer"]["lr"])
    for group in optimizer.param_groups:
        is_new = extension and set(group.get("parameter_names", ())) <= extension.new_parameters
        if not is_new and group["lr"] != parent_lr:
            raise ValueError("Parent Adam lr differs from experiment")
        for name in ("eps", "weight_decay"):
            if group[name] != config["optimizer"][name]:
                raise ValueError(f"Parent Adam {name} differs from experiment")
        if tuple(group["betas"]) != tuple(config["optimizer"]["betas"]):
            raise ValueError("Parent Adam betas differ from experiment")
        group["lr"] = group_lr(config, group.get("name", "shared_actor"))
    if adapts_value(config):
        return _reset_value_optimizer(policy, config, optimizer)
    return optimizer
