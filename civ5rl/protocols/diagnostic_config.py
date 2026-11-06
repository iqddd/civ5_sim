"""Validate fixed diagnostic protocol settings before workers run."""

from civ5rl.protocols.common import require_config


def _validate_step_ppo(config, *, multishot):
    ppo = config["ppo"]
    if ppo.get("valid_tokens_per_minibatch") != 1024:
        message = (
            "Unsupported D0-multishot PPO configuration"
            if multishot
            else "Unsupported configuration: ppo.valid_tokens_per_minibatch"
        )
        raise ValueError(message)
    if ppo["kl_measurement"] != "whole_rollout_after_optimizer_step":
        message = (
            "Unsupported D0-multishot PPO configuration"
            if multishot
            else "Unsupported configuration: ppo.kl_measurement"
        )
        raise ValueError(message)


def validate_diagnostic_config(config):
    mode = config["diagnostic"].get("mode", "D0-D1")
    if mode not in ("D0-D1", "D0-one-decision", "D1-transfer", "D0-multishot"):
        raise ValueError("Unsupported configuration: diagnostic.mode")
    expected = {
        "train_layouts": 128,
        "dev_layouts": 64,
        "seeds": [17, 42],
        "D0_updates": 4,
        "D1_updates": 8,
        "training_seconds": 1800,
        "protocol": "docs/rl-diagnostic-protocol.md",
    }
    if mode == "D0-one-decision":
        expected.update(
            seeds=[17],
            D0_updates=8,
            D1_updates=0,
            protocol="docs/rl-d0-one-decision-protocol.md",
            learner_decision_limit=1,
            initial_checkpoint="artifacts/rl/diagnostic-results/seed-17/checkpoints/initial.pt",
            source_dataset="artifacts/rl/diagnostic-results/manifests/manifest.json",
        )
    if mode == "D1-transfer":
        expected.update(
            seeds=[17],
            D0_updates=0,
            D1_updates=8,
            protocol="docs/rl-d1-transfer-protocol.md",
            initial_checkpoint="artifacts/rl/diagnostic-one-decision/seed-17/checkpoints/update-000006.pt",
            random_checkpoint="artifacts/rl/diagnostic-results/seed-17/checkpoints/initial.pt",
            source_dataset="artifacts/rl/diagnostic-results/manifests/manifest.json",
        )
        _validate_step_ppo(config, multishot=False)
    if mode == "D0-multishot":
        expected.update(
            seeds=[17],
            D0_updates=8,
            D1_updates=0,
            protocol="docs/rl-multishot-protocol.md",
            initial_checkpoint="artifacts/rl/diagnostic-one-decision/seed-17/checkpoints/update-000006.pt",
            random_checkpoint="artifacts/rl/diagnostic-results/seed-17/checkpoints/initial.pt",
            source_dataset="artifacts/rl/diagnostic-results/manifests/manifest.json",
            specification="configs/rl-diagnostic-multishot.spec.json",
        )
        _validate_step_ppo(config, multishot=True)
    require_config(
        config["diagnostic"], expected, prefix="Unsupported configuration: diagnostic."
    )
