"""Narrow, recorded migration: numerical replay audit only, never training rules."""

import json

from civ5rl.provenance.common import ROOT


def _load_policies():
    policy = json.loads(
        (ROOT / "configs/rl-migration-policies.json").read_text(encoding="utf-8")
    )
    if policy.get("schema") != 1 or set(policy.get("policies", {})) != {
        "audit",
        "pipeline",
    }:
        raise ValueError("Unsupported migration policy data")
    return policy["policies"]


_POLICIES = _load_policies()
ALLOWED_SOURCES = set(_POLICIES["audit"]["sources"])
ALLOWED_CONFIG = {tuple(path) for path in _POLICIES["audit"]["config_fields"]}
PIPELINE_SOURCES = set(_POLICIES["pipeline"]["sources"])
PIPELINE_CONFIG = {tuple(path) for path in _POLICIES["pipeline"]["config_fields"]}


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
