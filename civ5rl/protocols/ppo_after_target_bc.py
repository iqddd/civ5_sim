"""One fixed local PPO experiment after target-army BC epoch ten."""

from civ5rl.config import load_config as resolve_config
from civ5rl.provenance.common import ROOT

KIND = "rl01.ppo_after_target_bc"


def validate_profile(config):
    expected = resolve_config(ROOT / "configs/rl-ppo-after-target-bc.json")
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
    state["rollouts"] += 1
    state["decisions"] += summary["decisions"]
    state["winless_streak"] = state["winless_streak"] + 1 if summary["W"] == 0 else 0
    if state["winless_streak"] >= config["experiment"]["max_winless_rollouts"]:
        state["stop_reason"] = "no_sampling_wins"
        return False
    return True


def record_optimization(state, metrics, config):
    state["updates"] += 1
    state["optimizer_steps"] += metrics["optimizer_steps"]
    state["attempted_steps"] += metrics["attempted_optimizer_steps"]
    state["rejected_steps"] += metrics["rejected_optimizer_steps"]
    state["zero_accepted_streak"] = (
        0 if metrics["optimizer_steps"] else state["zero_accepted_streak"] + 1
    )
    if (
        state["zero_accepted_streak"]
        >= config["experiment"]["max_zero_accepted_rollouts"]
    ):
        state["stop_reason"] = "no_accepted_updates"


def check_development_gain(state, config):
    boundary = config["experiment"]["early_gain_decisions"]
    required = config["experiment"]["milestones"][:2]
    if all(str(point) in state["development_points"] for point in required):
        if (
            state["decisions"] >= boundary
            and state["best"]["wins"] <= state["baseline_wins"]
        ):
            state["stop_reason"] = "no_development_gain"
