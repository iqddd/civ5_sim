"""Narrow, recorded migration: numerical replay audit only, never training rules."""

ALLOWED_SOURCES = {
    "civ5rl/collector.py",
    "civ5rl/ppo_fast.py",
    "civ5rl/migration.py",
    "tools/migrate_rl_audit.py",
    "tools/control_rl.py",
    "tools/tune_rl.py",
    "configs/rl-v1.json",
    "docs/rl-v1-spec.md",
}
ALLOWED_CONFIG = {
    ("spec_version",),
    ("numerics", "require_full_behavior_distribution"),
    ("numerics", "replay_max_total_variation"),
}


def differences(a, b, prefix=()):
    if isinstance(a, dict) and isinstance(b, dict):
        return [
            item
            for key in sorted(a.keys() | b.keys())
            for item in differences(a.get(key), b.get(key), (*prefix, key))
        ]
    return [] if a == b else [prefix]


def validate_audit_migration(old_config, new_config, old_identity, new_identity):
    config_changes = differences(old_config, new_config)
    if any(path not in ALLOWED_CONFIG for path in config_changes):
        raise ValueError(f"Migration changes training configuration: {config_changes}")
    old_environment = {
        k: v
        for k, v in old_identity.items()
        if k not in ("source_hashes", "git_commit")
    }
    new_environment = {
        k: v
        for k, v in new_identity.items()
        if k not in ("source_hashes", "git_commit")
    }
    if old_environment != new_environment:
        raise ValueError("Migration changes corpus/environment identity")
    before, after = old_identity["source_hashes"], new_identity["source_hashes"]
    sources = sorted(
        k for k in before.keys() | after.keys() if before.get(k) != after.get(k)
    )
    if set(sources) - ALLOWED_SOURCES:
        raise ValueError(
            f"Migration changes model/simulator/training implementation: {sources}"
        )
    return {
        "config_changes": [".".join(p) for p in config_changes],
        "source_changes": sources,
    }


PIPELINE_SOURCES = ALLOWED_SOURCES | {
    "civ5rl/shared.py",
    "civ5rl/ppo.py",
    "civ5rl/env.py",
    "civ5sim/greedy_agent.py",
    "tools/benchmark_rl_pipeline.py",
    "civ5rl/model.py",
    "civ5rl/agent.py",
    "tools/train_rl.py",
}
PIPELINE_CONFIG = {
    ("spec_version",),
    ("name",),
    *(
        ("collection", key)
        for key in (
            "num_envs",
            "cpu_workers",
            "envs_per_worker",
            "deterministic_episode_waves",
        )
    ),
    *(
        ("performance", key)
        for key in (
            "reuse_encoded",
            "worker_encoding",
            "collection_scheduler",
            "shared_action_capacity",
            "spatial_batch_size",
        )
    ),
}


def validate_pipeline_migration(old_config, new_config, old_identity, new_identity):
    """Explicit throughput revision, preserving learnable state, not trajectories.

    New wave sizes change update grouping and ready scheduling changes BF16
    batching. This is a recorded continuation under revised collection conditions.
    Model, objective, optimizer, precision, corpus and simulation must stay fixed.
    """
    config_changes = differences(old_config, new_config)
    if any(path not in PIPELINE_CONFIG for path in config_changes):
        raise ValueError(f"Pipeline migration changes learning rules: {config_changes}")
    collection = new_config["collection"]
    if (
        collection["num_envs"]
        != collection["cpu_workers"] * collection["envs_per_worker"]
        or collection["num_envs"] != collection["deterministic_episode_waves"]
    ):
        raise ValueError("Pipeline worker/wave counts differ")
    before_env = {
        k: v
        for k, v in old_identity.items()
        if k not in ("source_hashes", "git_commit")
    }
    after_env = {
        k: v
        for k, v in new_identity.items()
        if k not in ("source_hashes", "git_commit")
    }
    if before_env != after_env:
        raise ValueError("Pipeline migration changes corpus/environment identity")
    before, after = old_identity["source_hashes"], new_identity["source_hashes"]
    sources = sorted(
        k for k in before.keys() | after.keys() if before.get(k) != after.get(k)
    )
    if set(sources) - PIPELINE_SOURCES:
        raise ValueError(f"Pipeline migration changes model/simulator: {sources}")
    return {
        "config_changes": [".".join(p) for p in config_changes],
        "source_changes": sources,
    }
