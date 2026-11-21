"""Configuration checks shared by bounded PPO protocols."""

from __future__ import annotations

import math


def validate_bounded_training(config, *, kinds):
    if config["kind"] not in kinds:
        raise ValueError(f"Unsupported PPO protocol: {config['kind']}")
    experiment = config["experiment"]
    if type(experiment["decisions"]) is not int or experiment["decisions"] < 1:
        raise ValueError("experiment.decisions must be a positive integer")
    seconds = experiment["active_seconds"]
    if isinstance(seconds, bool) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("experiment.active_seconds must be positive and finite")
    milestones = experiment["milestones"]
    if any(type(point) is not int or point < 1 for point in milestones) or (
        milestones != sorted(set(milestones))
    ):
        raise ValueError("experiment.milestones must be increasing positive decision counts")

    collection = config["collection"]
    ppo = config["ppo"]
    for section, fields in (
        (collection, ("num_envs", "cpu_workers", "envs_per_worker", "min_learner_decisions_per_update")),
        (ppo, ("epochs", "sequence_length", "sequences_per_minibatch", "sequences_per_microbatch")),
    ):
        for field in fields:
            if type(section[field]) is not int or section[field] < 1:
                raise ValueError(f"{field} must be a positive integer")
    performance = config["performance"]
    envs, workers = collection["num_envs"], collection["cpu_workers"]
    scheduler = performance.get("collection_scheduler", "barrier")
    if scheduler not in ("barrier", "ready", "streaming"):
        raise ValueError(f"Invalid collection scheduler: {scheduler}")
    per_worker = (envs + workers - 1) // workers
    if (
        workers > envs
        or collection["envs_per_worker"] != per_worker
        or (scheduler != "streaming" and envs % workers)
    ):
        raise ValueError("Collection environments do not fit the worker assignment")
    reply_size = performance.get("worker_reply_size", 0)
    if type(reply_size) is not int or not 0 <= reply_size <= per_worker:
        raise ValueError("worker_reply_size must fit one worker's environment slots")
    if ppo["sequences_per_microbatch"] > ppo["sequences_per_minibatch"]:
        raise ValueError("sequences_per_microbatch exceeds sequences_per_minibatch")
    if config["optimizer"]["type"] != "Adam":
        raise ValueError("Bounded PPO requires Adam")
    if any(
        config["reward"][key] != value
        for key, value in {
            "win": 1,
            "draw": 0,
            "loss": 0,
            "nonterminal": 0,
        }.items()
    ):
        raise ValueError("This PPO protocol requires the P(win) reward")
