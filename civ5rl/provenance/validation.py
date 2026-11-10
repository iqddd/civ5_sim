"""Reject config claims unsupported by the fixed RL-01 implementation."""

import math

from civ5rl.core.schema import (
    ACTION_INPUT_DIM,
    AGE_CAP,
    CONTEXT_INPUT_DIM,
    GLOBAL_DIM,
    HEX_BLOCKS,
    HEX_DIM,
    HEX_DIRECTIONS,
    HEX_MLP_DIM,
    HIDDEN_SIZE,
    PREV_ACTION_DIM,
    ROSTER_SCALE,
    TILE_DIM,
)

SUPPORTED = {
    "observation": {
        "schema": 1,
        "tile_features": TILE_DIM,
        "global_features": GLOBAL_DIM,
        "previous_action_features": PREV_ACTION_DIM,
        "history": True,
        "age_cap_enemy_turns": AGE_CAP,
        "roster_count_scale": ROSTER_SCALE,
        "enemy_private_fields": False,
        "privileged_features": False,
        "raw_reward_input": False,
        "step_result_input": False,
        "seed_input": False,
    },
    "model": {
        "type": "hex_gru_action_scorer",
        "tile_projection": [TILE_DIM, HEX_DIM],
        "hex_blocks": HEX_BLOCKS,
        "hex_width": HEX_DIM,
        "hex_directions": HEX_DIRECTIONS,
        "hex_mlp": [HEX_DIM, HEX_MLP_DIM, HEX_DIM],
        "context_mlp": [CONTEXT_INPUT_DIM, HIDDEN_SIZE, HIDDEN_SIZE],
        "recurrent": {
            "type": "GRU",
            "input_size": HIDDEN_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": 1,
            "bidirectional": False,
            "initial_state": "zero",
            "prefix_recompute": "full_current_weights_no_grad",
        },
        "action_mlp": [ACTION_INPUT_DIM, HEX_DIM, 1],
        "value_mlp": [HIDDEN_SIZE, HEX_DIM, 1],
        "activation": "gelu",
        "layer_norm_eps": 1e-05,
        "dropout": 0,
        "expected_parameter_count": 1375362,
        "shared_actor_critic_encoder": True,
        "critic_privileged": False,
        "normalization": "fixed_no_running_statistics",
    },
    "reward": {
        "win": 1,
        "loss": -1,
        "draw": 0,
        "nonterminal": 0,
        "shaping": False,
        "damage_bonus": 0,
        "exploration_bonus": 0,
        "command_penalty": 0,
    },
    "collection": {
        "num_envs": 32,
        "cpu_workers": 4,
        "envs_per_worker": 8,
        "worker_math_threads": 1,
        "min_learner_decisions_per_update": 8192,
        "complete_episodes_only": True,
        "deterministic_episode_waves": 32,
        "freeze_policy_per_update": True,
        "carry_episodes_between_updates": False,
        "retain_previous_rollouts_for_ppo": False,
    },
    "ppo": {
        "epochs": 4,
        "gamma": 1,
        "gae_lambda": 1,
        "return_estimator": "complete_episode_monte_carlo",
        "advantage_normalization": "rollout_valid_tokens",
        "advantage_epsilon": 1e-08,
        "clip_ratio": 0.2,
        "value_loss": "mean_squared_error_without_half",
        "value_coefficient": 0.5,
        "value_clip": False,
        "entropy": "categorical_divided_by_log_legal_action_count",
        "entropy_schedule": {
            "initial": 0.01,
            "final": 0.002,
            "linear_decay_decisions": 2000000,
            "after_decay": 0.002,
        },
        "target_kl": 0.01,
        "stop_kl": 0.015,
        "kl_measurement": "whole_rollout_after_epoch",
        "sequence_length": 128,
        "sequences_per_minibatch": 8,
        "sequences_per_microbatch": 8,
        "max_gradient_norm": 0.5,
        "reward_normalization": False,
        "return_normalization": False,
    },
    "optimizer": {
        "type": "Adam",
        "lr_schedule": {
            "initial": 0.0003,
            "final": 0.0001,
            "linear_decay_decisions": 2000000,
            "after_decay": 0.0001,
        },
        "betas": [0.9, 0.999],
        "eps": 1e-05,
        "weight_decay": 0,
    },
    "numerics": {
        "precision": "mixed_bfloat16",
        "amp": True,
        "amp_dtype": "bfloat16",
        "parameter_precision": "float32",
        "input_projection_precision": "float32",
        "hex_dense_precision": "bfloat16",
        "hex_message_sum_precision": "float32",
        "context_precision": "float32",
        "heads_precision": "float32",
        "gru_precision": "float32",
        "policy_value_outputs_precision": "float32",
        "loss_precision": "float32",
        "grad_scaler": False,
        "fp32_replay_max_logp_error": 0.001,
        "bf16_replay_max_logp_error": 0.005,
        "replay_max_kl": 1e-06,
        "require_full_behavior_distribution": True,
        "replay_max_total_variation": 0.005,
        "tf32": False,
        "cudnn_benchmark": False,
        "deterministic_reference": True,
    },
}


def validate_config(config):
    if config.get("kind") == "rl01.ppo_after_target_bc":
        from civ5rl.protocols.ppo_after_target_bc import validate_profile

        return validate_profile(config)
    if config.get("kind") == "rl01.ppo_target_scale":
        from civ5rl.protocols.ppo_target_scale import validate_profile

        return validate_profile(config)
    if config.get("kind") == "rl01.ppo_scale_up":
        from civ5rl.protocols.ppo_scale_up import validate_profile

        return validate_profile(config)
    reference = SUPPORTED
    fixed = {
        "model": tuple(reference["model"]),
        "observation": tuple(reference["observation"]),
        "reward": tuple(reference["reward"]),
        "numerics": (
            "precision",
            "amp",
            "amp_dtype",
            "parameter_precision",
            "input_projection_precision",
            "hex_dense_precision",
            "hex_message_sum_precision",
            "context_precision",
            "heads_precision",
            "gru_precision",
            "policy_value_outputs_precision",
            "loss_precision",
            "grad_scaler",
            "tf32",
            "cudnn_benchmark",
            "deterministic_reference",
        ),
        "ppo": (
            "gamma",
            "gae_lambda",
            "return_estimator",
            "advantage_normalization",
            "advantage_epsilon",
            "value_loss",
            "value_clip",
            "entropy",
            "reward_normalization",
            "return_normalization",
            "target_kl",
        ),
        "optimizer": ("type",),
        "collection": (
            "worker_math_threads",
            "complete_episodes_only",
            "freeze_policy_per_update",
            "carry_episodes_between_updates",
            "retain_previous_rollouts_for_ppo",
        ),
    }
    for section, keys in fixed.items():
        for key in keys:
            expected = reference[section][key]
            if (
                config.get("kind") in ("rl01.ppo_after_imitation", "rl01.ppo_controlled_updates")
                and section == "reward"
                and key == "loss"
            ):
                expected = 0
            if config.get(section, {}).get(key) != expected:
                raise ValueError(f"Unsupported configuration: {section}.{key}")
    for section, key, low, high in (
        ("ppo", "clip_ratio", 0, 1),
        ("ppo", "value_coefficient", 0, float("inf")),
        ("optimizer", "weight_decay", 0, float("inf")),
    ):
        value = config[section][key]
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < low
            or value >= high
            or (key == "clip_ratio" and value == 0)
        ):
            raise ValueError(f"Invalid configuration: {section}.{key}")
    acceptance = config["ppo"].get("step_acceptance")
    if acceptance is not None and config.get("kind") != "rl01.ppo_controlled_updates":
        raise ValueError("Unsupported configuration: ppo.step_acceptance")
    mode = config["ppo"].get("kl_measurement")
    if mode not in ("whole_rollout_after_epoch", "whole_rollout_after_optimizer_step"):
        raise ValueError("Unsupported configuration: ppo.kl_measurement")
    if acceptance == "full_categorical_kl_rollback" and mode != "whole_rollout_after_optimizer_step":
        raise ValueError(
            "Unsupported configuration: full_categorical_kl_rollback requires "
            "whole_rollout_after_optimizer_step"
        )
    tokens = config["ppo"].get("valid_tokens_per_minibatch")
    if tokens is not None and (type(tokens) is not int or tokens < 1):
        raise ValueError("Invalid configuration: ppo.valid_tokens_per_minibatch")
    c = config["collection"]
    if c["num_envs"] != c["cpu_workers"] * c["envs_per_worker"]:
        raise ValueError("Invalid configuration: collection.num_envs")
    if config.get("kind") in (
        "rl01.ppo_after_imitation",
        "rl01.ppo_controlled_updates",
    ):
        from civ5rl.protocols.ppo_after_imitation import validate_profile

        validate_profile(config)
    for section, keys in (
        (
            "collection",
            (
                "num_envs",
                "cpu_workers",
                "envs_per_worker",
                "min_learner_decisions_per_update",
            ),
        ),
        (
            "ppo",
            (
                "epochs",
                "sequence_length",
                "sequences_per_minibatch",
                "sequences_per_microbatch",
            ),
        ),
    ):
        for key in keys:
            if type(config[section][key]) is not int or config[section][key] < 1:
                raise ValueError(f"Invalid configuration: {section}.{key}")
    if config["ppo"]["sequences_per_microbatch"] > config["ppo"]["sequences_per_minibatch"]:
        raise ValueError("Invalid configuration: ppo.sequences_per_microbatch")
    if "diagnostic" in config:
        from civ5rl.protocols.diagnostic_config import validate_diagnostic_config

        validate_diagnostic_config(config)
