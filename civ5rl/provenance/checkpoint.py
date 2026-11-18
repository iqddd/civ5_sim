"""Atomic training state and compact, descriptive run metadata."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import zipfile
from pathlib import Path

from civ5rl.provenance.common import ROOT, write_json
from civ5rl.provenance.rng import capture_rng, restore_rng


def same_execution_identity(left, right):
    """Compare run/data identity, not producer code or installed packages.

    The name remains a checkpoint API; Git resume enforcement is at the shared
    CLI execution boundary. Config and tensor compatibility are checked by the
    caller/load_state_dict. Historical metadata is descriptive, not ancestry law.
    """
    descriptive={'source_hashes','git_commit','config_hash','python','platform','packages','cuda','gpu','schema'}
    return ({k:v for k,v in left.items() if k not in descriptive} ==
            {k:v for k,v in right.items() if k not in descriptive})


def passport(directory, config, manifest):
    """Record a run's config/data and descriptive environment; code lives in Git.

    Source archives from completed historical runs remain readable by the audit
    helper. New runs do not copy or hash the entire repository on preparation.
    """
    import torch
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    versions={name:importlib.metadata.version(name) for name in ('torch','numpy')}
    value=dict(schema=2,python=platform.python_version(),platform=platform.platform(),
        packages=versions,cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_hashes={},
        config_hash=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
        corpus_hash=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest())
    write_json(directory/'passport.json',value)
    write_json(directory/'resolved-config.json',config)
    write_json(directory/'corpus-manifest.json',manifest)
    return value


def verify_frozen_passport(directory, *, config=None, manifest=None, sources=True):
    """Verify config/data metadata and, for explicit audits, historical sources.

    This never compares the producer with today's workspace or packages.
    sources=False is sufficient for consuming a completed parent artifact.
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
    expected = identity.get("source_hashes", {})
    if not sources or not expected:return identity
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
    """Validate the saved run/data contract; CLI execution_session guards Git.

    Source history and environment are not compatibility requirements. Full
    verification of historical source archives is an explicit audit operation.
    """
    return verify_frozen_passport(directory,config=config,manifest=manifest,sources=False)


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
        raise ValueError("Resume config/run/data identity mismatch")
    policy.load_state_dict(value["model"])
    optimizer.load_state_dict(value["optimizer"])
    restore_rng(value, shuffle_rng)
    return value["state"]
