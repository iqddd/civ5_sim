"""Verified storage of inactive comparison data trees."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

from civ5rl.engine.runtime.archive import member_name, read_manifest, seven_zip
from civ5rl.engine.runtime.storage import ArchivedTree
from civ5rl.provenance.common import file_sha256


def _tree(out, name):
    if name not in ("sources", "continuations"):
        raise ValueError("Unknown comparison tree")
    return ArchivedTree(
        out,
        name,
        token="comparison-" + name,
        label="Comparison",
        temporary_prefix="verify-comparison-",
        changed_message="Comparison data changed during verification",
    )


def comparison_tree_paths(out, name):
    tree = _tree(out, name)
    return tree.root, tree.source, tree.packet


def archive_tree(out, name):
    _tree(out, name).archive()


def restore_tree(out, name):
    _tree(out, name).restore()


@contextmanager
def readable_tree(out, name):
    """Read an immutable packet; discard verified extraction without recompression.

    Caller owns the run lock. Retain modified files on failure, rather than
    silently deleting evidence or rewriting the original PPMd archive.
    """
    with _tree(out, name).readable(cleanup_existing=True) as source:
        yield source


@contextmanager
def readable_members(packet, names):
    """Extract only requested packet members and verify their original bytes.

    Solid archives still require decoding preceding data, but selecting members
    avoids materializing unrelated multi-gigabyte replay trees. This is a
    disposable read cache, never a resume dependency or a rewritten input packet.
    """
    packet = Path(packet).resolve()
    manifest = read_manifest(packet)
    names = sorted({member_name(name) for name in names})
    if not set(names).issubset(manifest["files"]):
        raise ValueError("Requested comparison input is absent from packet")
    archived = set(manifest["archived"])
    with tempfile.TemporaryDirectory(prefix="comparison-members-") as temporary:
        root = Path(temporary)
        selected = [n for n in names if n in archived]
        if selected:
            archive = packet / "json.7z"
            if file_sha256(archive) != manifest["archive_sha256"]:
                raise ValueError("Comparison archive SHA-256 mismatch")
            listing = root / "members.txt"
            listing.write_text("\n".join(selected) + "\n", encoding="utf-8")
            subprocess.run(
                [
                    seven_zip(),
                    "x",
                    "-y",
                    "-bso0",
                    "-bsp0",
                    "-scsUTF-8",
                    "-spd",
                    "-o" + str(root),
                    str(archive),
                    "@" + str(listing),
                ],
                check=True,
            )
            listing.unlink()
        for name in names:
            path = root / name
            if name not in archived:
                path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(packet / "files" / name, path)
            if (
                path.stat().st_size != manifest["files"][name]["size"]
                or file_sha256(path) != manifest["files"][name]["sha256"]
            ):
                raise ValueError(f"Comparison member changed: {name}")
        yield root
