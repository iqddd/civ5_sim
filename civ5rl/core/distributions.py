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


def masked_kl_terms(reference_logp, current_logp, mask=None, *, probabilities=None):
    """KL contributions after the caller's chosen normalization and casts."""
    delta = reference_logp - current_logp
    if mask is not None:
        delta = delta.masked_fill(~mask, 0.0)
    if probabilities is None:
        probabilities = reference_logp.exp()
    return probabilities * delta


def masked_kl_rows(reference_logp, current_logp, mask=None, *, probabilities=None):
    return masked_kl_terms(
        reference_logp, current_logp, mask, probabilities=probabilities
    ).sum(-1)


def require_rollout_temperature(episodes, temperature):
    validate_temperature(temperature)
    if any(
        episode.get("policy_temperature", 1.0) != temperature for episode in episodes
    ):
        raise ValueError("Rollout policy temperature differs from current PPO policy")
