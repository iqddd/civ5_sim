"""Atomic completed-update checkpoints and immutable environment passports."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import zipfile
from pathlib import Path

from civ5rl.provenance.common import ROOT, write_json
from civ5rl.provenance.identity import execution_identity, is_readonly_source
from civ5rl.provenance.rng import capture_rng, restore_rng


def same_execution_identity(left, right):
    """Compare execution identity, ignoring documentation and read-only tooling."""

    left = {**left, "source_hashes": left.get("source_hashes", {})}
    right = {**right, "source_hashes": right.get("source_hashes", {})}
    return execution_identity(left, include_markdown=False) == execution_identity(
        right, include_markdown=False
    )


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
    """Discover execution sources; read-only tooling never pins a training run."""
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
                if not is_readonly_source(path.relative_to(ROOT).as_posix()):
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
        "config_hash": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
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


def verify_frozen_passport(directory, *, config=None, manifest=None):
    """Verify saved provenance without importing or hashing the working tree.

    All archived hashes are checked, including reporting files in old passports.
    Historical passports/archives are never rewritten to match today's policy.
    This establishes artifact integrity, not permission to resume on new code.
    """
    directory = Path(directory)
    saved_config = json.loads(
        (directory / "resolved-config.json").read_text(encoding="utf-8")
    )
    corpus_path = directory / "corpus-manifest.json"
    if not corpus_path.exists():
        # Some historical runs retained only manifest.json. It is acceptable
        # only if its complete content matches the passport's corpus digest.
        corpus_path = directory / "manifest.json"
    saved_manifest = json.loads(corpus_path.read_text(encoding="utf-8"))
    identity = json.loads((directory / "passport.json").read_text(encoding="utf-8"))
    if (
        "config_hash" in identity
        and identity["config_hash"]
        != hashlib.sha256(json.dumps(saved_config, sort_keys=True).encode()).hexdigest()
    ):
        raise ValueError("Saved configuration hash differs")
    if config is not None and config != saved_config:
        raise ValueError("Saved experiment configuration differs")
    if manifest is not None and manifest != saved_manifest:
        raise ValueError("Saved experiment manifest differs")
    corpus_hash = hashlib.sha256(
        json.dumps(saved_manifest, sort_keys=True).encode()
    ).hexdigest()
    if identity["corpus_hash"] != corpus_hash:
        raise ValueError("Saved corpus manifest hash differs")
    expected = identity["source_hashes"]
    with zipfile.ZipFile(directory / "sources.zip") as archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        if len(names) != len(set(names)) or set(names) != set(expected):
            raise ValueError("Saved source archive membership differs")
        for name in names:
            if "\\" in name or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Unsafe path in saved source archive")
            with archive.open(name) as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            if checksum != expected[name]:
                raise ValueError(f"Saved source archive corrupted: {name}")
    return identity


def verify_execution_passport(directory, config, manifest):
    """Check frozen evidence first, then require the current execution identity."""
    import tempfile

    frozen = verify_frozen_passport(directory, config=config, manifest=manifest)
    with tempfile.TemporaryDirectory(prefix="civ5-identity-") as temporary:
        current = passport(temporary, config, manifest)
    if not same_execution_identity(frozen, current):
        raise ValueError("Training source/corpus/environment identity changed")
    return frozen


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
        **capture_rng(shuffle_rng),
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
    restore_rng(value, shuffle_rng)
    return value["state"]
