"""Explicit, function-preserving policy extensions for branching from a checkpoint."""

from dataclasses import dataclass, replace

import torch

from civ5rl.core.variants import PolicyVariant, variant_spec


@dataclass(frozen=True)
class ModelExtension:
    parent: str
    child: str
    adam_group: str
    new_parameters: frozenset[str]
    zero_parameters: tuple[str, ...]

    @property
    def adds_action_context(self):
        return variant_spec(self.child).action_context


_EXTENSIONS = {
    "attack_preview_v1": ModelExtension(
        "C2", "C2P", "attack_preview", frozenset({"attack_preview.weight"}), ("attack_preview.weight",),
    ),
    "action_context_v1": ModelExtension(
        "C2P", "C2PA", "action_context",
        frozenset(
            f"action_attention.{layer}.{field}"
            for layer in ("norm", "q_proj", "k_proj", "v_proj", "out_proj")
            for field in ("weight", "bias")
        ) | {"value_actions.weight"},
        ("action_attention.out_proj.weight", "action_attention.out_proj.bias", "value_actions.weight"),
    ),
}
_EXTENSIONS["action_context_pre_gelu_v1"] = replace(
    _EXTENSIONS["action_context_v1"], child=PolicyVariant.CONTACTS_GRU_C2_PREVIEW_ACTIONS_PRE_GELU,
)


def model_extension(config):
    name = config.get("experiment", {}).get("model_extension")
    if name is None:
        return None
    if name not in _EXTENSIONS:
        raise ValueError(f"Unknown model extension {name!r}")
    return _EXTENSIONS[name]


def extends_parent(config):
    return model_extension(config) is not None


def inherited_model(model, parent, extension=_EXTENSIONS["attack_preview_v1"]):
    if set(model) != set(parent) | extension.new_parameters or set(parent) & extension.new_parameters:
        raise ValueError(f"{extension.child} inheritance changed existing parameter names")
    if any(model[name].dtype != value.dtype or not torch.equal(model[name], value)
           for name, value in parent.items()):
        raise ValueError(f"{extension.child} inherited weights differ")
    if any(torch.count_nonzero(model[name]) for name in extension.zero_parameters):
        raise ValueError(f"{extension.child} residual projections must start at zero")


def load_parent_model(policy, saved, config):
    extension = model_extension(config)
    if extension and saved["variant"] != policy.variant:
        if (saved["variant"], policy.variant) != (extension.parent, extension.child):
            raise ValueError(f"Model extension only supports {extension.parent} -> {extension.child}")
        initialized = policy.state_dict()
        model = {**saved["model"], **{name: initialized[name] for name in extension.new_parameters}}
        policy.load_state_dict(model, strict=True)
        inherited_model(policy.state_dict(), saved["model"], extension)
    else:
        policy.load_state_dict(saved["model"], strict=True)
