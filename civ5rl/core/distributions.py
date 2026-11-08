"""One FP32 action-policy boundary shared by deployment, collection and PPO."""

import math


def validate_temperature(temperature):
    if (
        isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("Policy temperature must be finite and positive")
    return float(temperature)


def policy_logits(logits, temperature=1.0):
    """Scale current raw logits once; never apply to saved behavior log-probs."""
    logits = logits.float()
    return logits if temperature == 1 else logits / temperature


def require_rollout_temperature(episodes, temperature):
    validate_temperature(temperature)
    if any(
        episode.get("policy_temperature", 1.0) != temperature for episode in episodes
    ):
        raise ValueError("Rollout policy temperature differs from current PPO policy")
