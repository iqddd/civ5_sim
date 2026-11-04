"""Atomic completed-update checkpoints and immutable environment passports."""

import hashlib
import importlib.metadata
import json
import os
import platform
import random
import subprocess
import zipfile
from pathlib import Path

import numpy as np

from civ5rl.provenance.common import ROOT, write_json


def same_execution_identity(left, right):
    """Git HEAD records provenance; exact source hashes define executed content."""
    return {k: v for k, v in left.items() if k != "git_commit"} == {
        k: v for k, v in right.items() if k != "git_commit"
    }


DEFAULT_ARTIFACT_GLOBS = (
    "civ5rl/**/*.py",
    "civ5sim/**/*.py",
    "civ5map/**/*.py",
    "native/**/*.cpp",
    "native/**/*.c",
    "native/**/*.h",
    "native/**/*.hpp",
    "tools/**/*.py",
    "build/combat.dll",
    "build/combat_rules.json",
    "configs/**/rl*.json",
    "docs/**/rl*.md",
    "requirements-rl*.txt",
)


def artifact_sources(config):
    """Discover the default snapshot and optional workspace-relative artifacts."""
    extra = config.get("passport_artifacts", [])
    if not isinstance(extra, list) or not all(isinstance(p, str) for p in extra):
        raise ValueError("passport_artifacts must be a list of relative glob patterns")
    sources = set()
    for pattern in (*DEFAULT_ARTIFACT_GLOBS, *extra):
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise ValueError(
                "Passport artifact patterns must stay within the workspace"
            )
        for path in ROOT.glob(pattern):
            if path.is_file():
                if not path.resolve().is_relative_to(ROOT.resolve()):
                    raise ValueError("Passport artifact resolves outside the workspace")
                sources.add(path)
    return sorted(sources, key=lambda p: p.relative_to(ROOT).as_posix())


def passport(directory, config, manifest):
    import torch

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sources = artifact_sources(config)
    hashes = {
        str(p.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(
            p.read_bytes()
        ).hexdigest()
        for p in sources
        if p.exists()
    }
    versions = {
        d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
    }
    value = {
        "schema": 1,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": versions,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_hashes": hashes,
        "corpus_hash": hashlib.sha256(
            json.dumps(manifest, sort_keys=True).encode()
        ).hexdigest(),
    }
    write_json(directory / "passport.json", value)
    write_json(directory / "resolved-config.json", config)
    write_json(directory / "corpus-manifest.json", manifest)
    (directory / "dirty.diff").write_bytes(
        subprocess.check_output(["git", "diff", "--binary"], cwd=ROOT)
    )
    # Include untracked implementation as well as tracked sources in the frozen snapshot.
    with zipfile.ZipFile(
        directory / "sources.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for p in sources:
            if p.exists():
                archive.write(p, p.relative_to(ROOT).as_posix())
    (directory / "environment-lock.txt").write_text(
        "\n".join(f"{k}=={v}" for k, v in sorted(versions.items())) + "\n",
        encoding="utf-8",
    )
    return value


def save(path, policy, optimizer, state, shuffle_rng, config, identity):
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = {
        "schema": 1,
        "kind": "rl01.checkpoint",
        "variant": policy.variant,
        "model": policy.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": state,
        "config": config,
        "identity": identity,
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "shuffle_rng": shuffle_rng.getstate(),
    }
    tmp = path.with_suffix(".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def restore(path, policy, optimizer, shuffle_rng, config, identity):
    import torch

    value = torch.load(path, map_location="cpu", weights_only=False)
    if value["config"] != config or not same_execution_identity(
        value["identity"], identity
    ):
        raise ValueError("Resume config/source/corpus/environment identity mismatch")
    policy.load_state_dict(value["model"])
    optimizer.load_state_dict(value["optimizer"])
    random.setstate(value["python_rng"])
    np.random.set_state(value["numpy_rng"])
    torch.set_rng_state(value["torch_rng"])
    if value["cuda_rng"]:
        torch.cuda.set_rng_state_all(value["cuda_rng"])
    shuffle_rng.setstate(value["shuffle_rng"])
    return value["state"]
