"""Explicit host-only resume revisions; never rewrite frozen scientific evidence."""

import hashlib
import json
import re
import zipfile
from pathlib import Path

from civ5rl.provenance.identity import execution_identity

# The first migration introduces only this verifier and its checkpoint call site.
HOST_SUPPORT_SOURCES = {"civ5rl/provenance/host_resume.py", "civ5rl/provenance/checkpoint.py"}


def identity_digest(identity):
    value = execution_identity(identity, include_markdown=False)
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def host_delta(before, after):
    """Allow a Linux kernel change, not different packages, GPU, config or math."""
    if before.get("config_hash") != after.get("config_hash"):
        raise ValueError("Host resume cannot change the scientific config")
    left = execution_identity(before, include_markdown=False)
    right = execution_identity(after, include_markdown=False)
    pattern = r"Linux-.+-(x86_64|aarch64)-with-(glibc\d+\.\d+)"
    platforms = [re.fullmatch(pattern, x.get("platform", "")) for x in (left, right)]
    if not all(platforms) or platforms[0].groups() != platforms[1].groups():
        raise ValueError("Host resume requires the same Linux architecture and glibc")
    changes = {}
    for name in left["source_hashes"].keys() | right["source_hashes"].keys():
        old, new = left["source_hashes"].get(name), right["source_hashes"].get(name)
        if old != new:
            if name not in HOST_SUPPORT_SOURCES:
                raise ValueError(f"Host resume cannot change execution source: {name}")
            changes[name] = {"before": old, "after": new}
    for value in (left, right):
        value.pop("platform", None)
        for name in HOST_SUPPORT_SOURCES:
            value["source_hashes"].pop(name, None)
    if left != right:
        raise ValueError("Host resume cannot change Python/packages/CUDA/GPU/corpus identity")
    return changes


def verify_revision(path, before, after):
    path = Path(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if (record.get("kind") != "rl01.host_resume" or not record.get("reason")
            or record.get("before_identity") != identity_digest(before)
            or record.get("after_identity") != identity_digest(after)
            or record.get("source_changes") != host_delta(before, after)):
        raise ValueError("Host resume revision does not match this execution identity")
    for name in ("checkpoint.pt", "state-before.json", "ledger-before.jsonl", "identity-after.json", "sources-after.zip"):
        file = path.parent / name
        with file.open("rb") as stream:
            checksum = hashlib.file_digest(stream, "sha256").hexdigest()
        if record.get("files", {}).get(name) != checksum:
            raise ValueError(f"Host resume evidence damaged: {name}")
    with zipfile.ZipFile(path.parent / "sources-after.zip") as archive:
        if set(archive.namelist()) != set(record["source_changes"]):
            raise ValueError("Host resume source archive membership differs")
        for name, change in record["source_changes"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != change["after"]:
                raise ValueError(f"Host resume source archive differs: {name}")
    return record


def require_host_revision(directory, before, after):
    # A successful match is explicit, local to this run and bound to exact hashes.
    for path in sorted((Path(directory) / "source-revisions").glob("host-*/revision.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("after_identity") == identity_digest(after):
            return verify_revision(path, before, after)
    raise ValueError("Training source/corpus/environment identity changed; no verified host-resume revision exists")
