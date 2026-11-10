"""One fixed local PPO experiment after target-army BC epoch ten."""

from civ5rl.config import load_config as resolve_config
from civ5rl.engine.counters import TrainingCounters
from civ5rl.provenance.common import ROOT

KIND = "rl01.ppo_after_target_bc"


def validate_profile(config):
    profile = (
        "rl-ppo-target-bc-t05.json"
        if config.get("policy_temperature", 1) == 0.5
        else "rl-ppo-after-target-bc.json"
    )
    expected = resolve_config(ROOT / "configs" / profile)
    if config != expected:
        raise ValueError(
            "Configuration differs from the fixed PPO-after-target-BC profile"
        )
    if (
        config["kind"] != KIND
        or config["collection"]["num_envs"] != 16
        or config["collection"]["cpu_workers"] != 4
        or config["ppo"]["sequences_per_microbatch"] != 8
        or config["performance"]["worker_reply_size"] != 4
        or config["optimizer"]["lr"] != 1e-5
        or config["reward"]["loss"] != 0
        or config["experiment"]["parent_optimizer_steps"] != 0
    ):
        raise ValueError("Unsupported local PPO-after-target-BC conditions")


def load_config(path):
    config = resolve_config(path)
    validate_profile(config)
    return config


def record_collection(state, summary, config):
    """Collection evidence counts even when the following PPO is skipped."""
    counters = TrainingCounters.from_dict(state, protocol=KIND)
    counters = counters.replace(rollouts_collected=counters.rollouts_collected + 1)
    state.update(counters.to_state_dict(protocol=KIND))
    state["decisions"] += int(summary["decisions"])
    if "episodes" in state and "episodes" in summary:
        state["episodes"] += int(summary["episodes"])
    state["winless_streak"] = state["winless_streak"] + 1 if summary["W"] == 0 else 0
    if state["winless_streak"] >= config["experiment"]["max_winless_rollouts"]:
        state["stop_reason"] = "no_sampling_wins"
        return False
    return True


def record_optimization(state, metrics, config):
    counters = TrainingCounters.from_dict(state, protocol=KIND)
    accepted = int(metrics.get("optimizer_steps", 0))
    attempted = int(metrics.get("attempted_optimizer_steps", accepted))
    counters = counters.replace(
        rollouts_with_accepted_steps=counters.rollouts_with_accepted_steps
        + (1 if accepted > 0 else 0),
        adam_steps_attempted=counters.adam_steps_attempted + attempted,
        adam_steps_accepted=counters.adam_steps_accepted + accepted,
    )
    state.update(counters.to_state_dict(protocol=KIND))
    state["updates"] = int(state.get("updates", 0)) + 1
    state["rejected_steps"] = (
        counters.adam_steps_attempted - counters.adam_steps_accepted
    )
    state["zero_accepted_streak"] = (
        0 if accepted else state["zero_accepted_streak"] + 1
    )
    if (
        state["zero_accepted_streak"]
        >= config["experiment"]["max_zero_accepted_rollouts"]
    ):
        state["stop_reason"] = "no_accepted_updates"


def check_development_gain(state, config):
    boundary = config["experiment"]["early_gain_decisions"]
    required = config["experiment"]["milestones"][:2]
    if all(str(point) in state["development_points"] for point in required) and (
        state["decisions"] >= boundary
        and state["best"]["wins"] <= state["baseline_wins"]
    ):
        state["stop_reason"] = "no_development_gain"
