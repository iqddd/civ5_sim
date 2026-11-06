"""Narrow, recorded migration: numerical replay audit only, never training rules."""

import json
from functools import cache

from civ5rl.provenance.common import ROOT
from civ5rl.provenance.identity import execution_identity


@cache
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


def differences(a, b, prefix=()):
    if isinstance(a, dict) and isinstance(b, dict):
        return [
            item
            for key in sorted(a.keys() | b.keys())
            for item in differences(a.get(key), b.get(key), (*prefix, key))
        ]
    return [] if a == b else [prefix]


def _validate_migration(
    old_config,
    new_config,
    old_identity,
    new_identity,
    *,
    allowed_config,
    allowed_sources,
    config_error,
    identity_error,
    source_error,
    check_collection=False,
):
    config_changes = differences(old_config, new_config)
    if any(path not in allowed_config for path in config_changes):
        raise ValueError(f"{config_error}: {config_changes}")
    if check_collection:
        collection = new_config["collection"]
        if (
            collection["num_envs"]
            != collection["cpu_workers"] * collection["envs_per_worker"]
            or collection["num_envs"] != collection["deterministic_episode_waves"]
        ):
            raise ValueError("Pipeline worker/wave counts differ")
    old_fields = execution_identity(old_identity, include_markdown=True)
    new_fields = execution_identity(new_identity, include_markdown=True)
    old_environment = {k: v for k, v in old_fields.items() if k != "source_hashes"}
    new_environment = {k: v for k, v in new_fields.items() if k != "source_hashes"}
    if old_environment != new_environment:
        raise ValueError(identity_error)
    before, after = old_fields["source_hashes"], new_fields["source_hashes"]
    sources = sorted(
        k for k in before.keys() | after.keys() if before.get(k) != after.get(k)
    )
    if set(sources) - allowed_sources:
        raise ValueError(f"{source_error}: {sources}")
    return {
        "config_changes": [".".join(p) for p in config_changes],
        "source_changes": sources,
    }


def validate_audit_migration(old_config, new_config, old_identity, new_identity):
    policy = _load_policies()["audit"]
    return _validate_migration(
        old_config,
        new_config,
        old_identity,
        new_identity,
        allowed_config={tuple(path) for path in policy["config_fields"]},
        allowed_sources=set(policy["sources"]),
        config_error="Migration changes training configuration",
        identity_error="Migration changes corpus/environment identity",
        source_error="Migration changes model/simulator/training implementation",
    )


def validate_pipeline_migration(old_config, new_config, old_identity, new_identity):
    """Explicit throughput revision, preserving learnable state, not trajectories.

    New wave sizes change update grouping and ready scheduling changes BF16
    batching. This is a recorded continuation under revised collection conditions.
    Model, objective, optimizer, precision, corpus and simulation must stay fixed.
    """
    policy = _load_policies()["pipeline"]
    return _validate_migration(
        old_config,
        new_config,
        old_identity,
        new_identity,
        allowed_config={tuple(path) for path in policy["config_fields"]},
        allowed_sources=set(policy["sources"]),
        config_error="Pipeline migration changes learning rules",
        identity_error="Pipeline migration changes corpus/environment identity",
        source_error="Pipeline migration changes model/simulator",
        check_collection=True,
    )
