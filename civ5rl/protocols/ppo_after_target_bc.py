"""Fixed target-army PPO profiles, including a preserved-Adam continuation."""

import math

from civ5rl.config import load_config as resolve_config
from civ5rl.engine.counters import TrainingCounters
from civ5rl.provenance.common import ROOT

KIND = "rl01.ppo_after_target_bc"


def validate_profile(config):
    if config.get("output") == "artifacts/rl/ppo-c2-h60-gae-gamma1":
        profile = "rl-ppo-c2-h60-gae-gamma1.json"
    elif config.get("output") == "artifacts/rl/ppo-c2-h60-gae-continuation":
        profile = "rl-ppo-c2-h60-gae-continuation.json"
    elif config.get("output") == "artifacts/rl/ppo-c2-h60-gae":
        profile = "rl-ppo-c2-h60-gae.json"
    elif config.get("policy_temperature", 1) == 0.5:
        profile = "rl-ppo-target-bc-t05.json"
    else:
        profile = "rl-ppo-after-target-bc.json"
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
        or (config["experiment"]["parent_optimizer_steps"] != 0
            and config["experiment"].get("parent_optimizer") != "preserve")
    ):
        raise ValueError("Unsupported local PPO-after-target-BC conditions")
    if config["experiment"].get("time_budget_check"):
        compute_time_budget_check(config)


def compute_time_budget_check(config):
    """Separate measured expected work from pessimistic, latched-KL replay bounds."""
    experiment = config["experiment"]
    spec = experiment.get("time_budget_check")
    if not spec:
        historical = experiment.get("historical_active_time_estimate")
        if historical:
            low, high = historical["decisions_per_second_range"]
            adaptation = (float(config["critic"]["active_seconds"])
                          if experiment.get("initial_value") == "adapt_parent_first_rollout" else 0.0)
            seconds = [experiment["decisions"] / rate + adaptation for rate in (high, low)]
            return {"basis": "historical_full_cycle_active_time", "source": historical["source"],
                    "expected_active_seconds_range": seconds,
                    "critic_adaptation_allowance_seconds": adaptation,
                    "active_seconds_limit": experiment["active_seconds"],
                    "reserve_seconds_range": [experiment["active_seconds"] - s for s in reversed(seconds)],
                    "fits_historical_range": seconds[-1] <= experiment["active_seconds"],
                    "warnings": ["Historical late-window throughput is not a guarantee for a new machine; evaluations pause active time."]
                    + (["Mandatory critic refresh may exceed the preparation allowance and still consumes active time."] if adaptation else [])}
        return None
    decisions_budget = int(experiment["decisions"])
    active_limit = float(experiment["active_seconds"])
    min_rollout = int(config["collection"]["min_learner_decisions_per_update"])
    tokens_per_mb = int(config["ppo"]["valid_tokens_per_minibatch"])
    epochs = int(config["ppo"]["epochs"])
    kl_mode = config["ppo"].get(
        "kl_measurement", "whole_rollout_after_optimizer_step"
    )

    max_rollouts = math.ceil(decisions_budget / min_rollout)
    terminal_overshoot = int(spec.get("terminal_episode_overshoot_decisions", 4096))
    nominal_steps = math.ceil(decisions_budget / tokens_per_mb) * epochs
    partial_minibatch_steps = max_rollouts * epochs
    terminal_overshoot_steps = math.ceil(terminal_overshoot / tokens_per_mb) * epochs
    planned_steps_upper_bound = (
        nominal_steps + partial_minibatch_steps + terminal_overshoot_steps
    )

    # A spike latches the per-step safeguard until the END OF THE UPDATE.
    # One extra spike check is not an upper bound on subsequent replay work.
    # Include one possible rollback verification per planned rollout.
    planned_kl_checks_upper_bound = 2 * max_rollouts + planned_steps_upper_bound
    no_spike_checks = max_rollouts * (1 + epochs)
    expected_checks_per_rollout = spec.get("expected_kl_checks_per_rollout")
    warnings = []
    if kl_mode == "whole_rollout_after_epoch_or_minibatch_spike":
        if expected_checks_per_rollout is None:
            planned_kl_checks = None
            warnings.append("No measured expected_kl_checks_per_rollout: expected fit is unknown; a single-spike allowance cannot model the latched safeguard.")
        else:
            expected_checks_per_rollout = float(expected_checks_per_rollout)
            if not math.isfinite(expected_checks_per_rollout) or expected_checks_per_rollout < 1:
                raise ValueError("Expected KL checks per rollout must be finite and >= 1")
            planned_kl_checks = math.ceil(max_rollouts * expected_checks_per_rollout)
    elif kl_mode == "whole_rollout_after_optimizer_step":
        planned_kl_checks = max_rollouts + planned_steps_upper_bound
    else:
        planned_kl_checks = no_spike_checks
        planned_kl_checks_upper_bound = no_spike_checks

    step_seconds = float(spec["target_gpu_step_seconds"])
    kl_check_seconds = float(spec["target_gpu_kl_check_seconds"])
    collection_rate = float(spec["target_collection_decisions_per_second"])
    compilation_seconds = float(
        spec.get("runtime_compilation_allowance_seconds", 0.0)
    )
    critic_seconds = float(config.get("critic", {}).get("active_seconds", 0.0))

    expected_ppo_step_seconds = planned_steps_upper_bound * step_seconds
    expected_collection_seconds = (decisions_budget + terminal_overshoot) / collection_rate
    non_kl_seconds = expected_collection_seconds + critic_seconds + compilation_seconds + expected_ppo_step_seconds
    expected_kl_replay_seconds = planned_kl_checks * kl_check_seconds if planned_kl_checks is not None else None
    expected_active_seconds = non_kl_seconds + expected_kl_replay_seconds if expected_kl_replay_seconds is not None else None
    worst_case_seconds = non_kl_seconds + planned_kl_checks_upper_bound * kl_check_seconds
    if expected_active_seconds is not None and expected_active_seconds > active_limit:
        warnings.append(f"Expected volume exceeds active budget: {expected_active_seconds:.1f}s > {active_limit:.1f}s")
    if worst_case_seconds > active_limit:
        warnings.append(f"Pessimistic KL scenario exceeds active budget: {worst_case_seconds:.1f}s > {active_limit:.1f}s; this is a warning, not an automatic launch ban.")
    return {
        "decisions_budget": decisions_budget,
        "terminal_episode_overshoot_decisions": terminal_overshoot,
        "min_rollout_decisions": min_rollout,
        "max_rollouts": max_rollouts,
        "epochs_per_rollout": epochs,
        "nominal_optimizer_steps": nominal_steps,
        "partial_minibatch_steps_allowance": partial_minibatch_steps,
        "terminal_overshoot_steps_allowance": terminal_overshoot_steps,
        "planned_optimizer_steps_upper_bound": planned_steps_upper_bound,
        "kl_measurement": kl_mode,
        "expected_kl_checks_per_rollout": expected_checks_per_rollout,
        "planned_kl_checks_expected": planned_kl_checks,
        "planned_kl_checks_upper_bound": planned_kl_checks_upper_bound,
        "no_spike_kl_checks_for_planned_rollouts": no_spike_checks,
        "target_gpu": spec.get("target_gpu"),
        "target_gpu_step_seconds": step_seconds,
        "target_gpu_kl_check_seconds": kl_check_seconds,
        "target_collection_decisions_per_second": collection_rate,
        "expected_ppo_step_seconds": round(expected_ppo_step_seconds, 2),
        "expected_kl_replay_seconds": round(expected_kl_replay_seconds, 2) if expected_kl_replay_seconds is not None else None,
        "expected_collection_seconds": round(expected_collection_seconds, 2),
        "expected_critic_warmup_seconds": round(critic_seconds, 2),
        "expected_compilation_seconds": round(compilation_seconds, 2),
        "expected_active_seconds": round(expected_active_seconds, 2) if expected_active_seconds is not None else None,
        "no_spike_scenario_active_seconds": round(non_kl_seconds + no_spike_checks * kl_check_seconds, 2),
        "worst_case_active_seconds": round(worst_case_seconds, 2),
        "active_seconds_limit": active_limit,
        "reserve_seconds": round(active_limit - expected_active_seconds, 2) if expected_active_seconds is not None else None,
        "fits_within_active_seconds": expected_active_seconds <= active_limit if expected_active_seconds is not None else None,
        "worst_case_fits_within_active_seconds": worst_case_seconds <= active_limit,
        "warnings": warnings,
    }


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
    required = [
        point for point in config["experiment"]["milestones"] if point <= boundary
    ]
    min_delta = float(
        config["experiment"].get("early_gain_min_win_rate_delta", 0.0)
    )
    eval_purpose = config.get("ppo", {}).get("evaluation_purpose", "development")
    eval_matches = 512 if eval_purpose == "development_512" else 128
    min_wins_gain = config["experiment"].get("early_gain_min_wins") or (
        math.ceil(min_delta * eval_matches - 1e-12) if min_delta > 0 else 1
    )
    if state.get("baseline_wins") is None:
        return
    threshold = state.get("early_gain_required_wins", state["baseline_wins"] + min_wins_gain)
    if all(str(point) in state["development_points"] for point in required) and (
        state["decisions"] >= boundary
        and state["best"]["wins"] < threshold
    ):
        state["stop_reason"] = "no_development_gain"
