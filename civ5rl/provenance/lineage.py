"""Fail-closed validation of parent source archives and execution identity."""

import hashlib
import tempfile
import zipfile

from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import read_json
from civ5rl.provenance.identity import execution_identity


def verify_parent_lineage(parent, previous, current, allowed, *, same_corpus=False):
    with zipfile.ZipFile(parent / "sources.zip") as archive:
        for name, expected in previous["source_hashes"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Corrupted parent source archive: {name}")
    old_identity = execution_identity(
        previous, include_markdown=False, include_corpus=same_corpus
    )
    new_identity = execution_identity(
        current, include_markdown=False, include_corpus=same_corpus
    )
    old, new = old_identity["source_hashes"], new_identity["source_hashes"]
    changes = sorted(
        name
        for name in old.keys() | new.keys()
        if not name.lower().endswith(".md") and old.get(name) != new.get(name)
    )
    if set(changes) - allowed:
        raise ValueError(f"Unapproved source changes from parent: {changes}")
    old_fields = {
        key: value for key, value in old_identity.items() if key != "source_hashes"
    }
    new_fields = {
        key: value for key, value in new_identity.items() if key != "source_hashes"
    }
    if old_fields.keys() != new_fields.keys():
        raise ValueError("Parent execution identity fields differ")
    for key, value in old_fields.items():
        if value != new_fields[key]:
            raise ValueError(f"Parent execution identity differs: {key}")
    return changes


def current_parent_lineage(parent, config, manifest, allowed, *, same_corpus=False):
    previous = read_json(parent / "passport.json")
    with tempfile.TemporaryDirectory() as directory:
        current = checkpoint.passport(directory, config, manifest)
    changes = verify_parent_lineage(
        parent, previous, current, allowed, same_corpus=same_corpus
    )
    return current, changes


def verify_existing_run(out, config, current):
    identity = read_json(out / "passport.json")
    if read_json(
        out / "resolved-config.json"
    ) != config or not checkpoint.same_execution_identity(identity, current):
        raise ValueError("Execution identity mismatch")
    return identity
