"""Integrity checks for inputs reused by experiment protocols."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Any

from civ5rl.provenance.common import file_sha256
from civ5rl.provenance.identity import policy_hash
from civ5sim.engine import digest


def verify_file_hashes(
    root: str | PathLike[str],
    mapping: Mapping[str, str],
    *,
    label: str,
) -> None:
    """Verify the given file hashes in the mapping's order."""
    root = Path(root)
    for name, expected in mapping.items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"{label}: {name}")


def verify_checkpoint_identity(
    path: str | PathLike[str],
    *,
    expected_sha256: str,
    expected_policy_hash: str,
    sha_error: str = "Parent checkpoint SHA256 differs",
    policy_error: str = "Parent policy differs",
) -> dict[str, Any]:
    """Verify a parent checkpoint's bytes and its actor parameters."""
    if file_sha256(path) != expected_sha256:
        raise ValueError(sha_error)

    # File and manifest checks also run in Torch-free workers.
    import torch

    saved: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
    if policy_hash(saved["model"]) != expected_policy_hash:
        raise ValueError(policy_error)
    return saved


def verify_manifest_digest(plan: Mapping[str, Any], *, error_msg: str = "Manifest changed") -> None:
    """Verify the manifest's hash field and preserve missing-key errors."""
    if digest({key: value for key, value in plan.items() if key != "hash"}) != plan["hash"]:
        raise ValueError(error_msg)
