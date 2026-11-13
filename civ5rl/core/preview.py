"""Explicit, zero-initialized C2 -> C2P model inheritance."""

import torch


def extends_parent(config):
    return config.get("experiment", {}).get("model_extension") == "attack_preview_v1"


def inherited_model(model, parent):
    if set(model) != {*parent, "attack_preview.weight"}:
        raise ValueError("C2P inheritance changed existing parameter names")
    if any(not torch.equal(model[name], value) for name, value in parent.items()):
        raise ValueError("C2P inherited weights differ")
    if model["attack_preview.weight"].shape != (128, 5) or torch.count_nonzero(model["attack_preview.weight"]):
        raise ValueError("C2P preview projection must start at zero")


def load_parent_model(policy, saved, config):
    if extends_parent(config) and saved["variant"] != policy.variant:
        if saved["variant"] != "C2" or policy.variant != "C2P":
            raise ValueError("attack_preview_v1 only supports C2 -> C2P")
        model = {**saved["model"], "attack_preview.weight": torch.zeros_like(policy.attack_preview.weight)}
        policy.load_state_dict(model, strict=True)
        inherited_model(policy.state_dict(), saved["model"])
    else:
        policy.load_state_dict(saved["model"], strict=True)
