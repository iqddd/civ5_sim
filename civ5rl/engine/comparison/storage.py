"""Verified storage of inactive comparison data trees."""

import shutil
from contextlib import contextmanager
from pathlib import Path

from civ5rl.deploy.archive import unpack
from civ5rl.engine.runtime.storage import pack_verified


def comparison_tree_paths(out, name):
    if name not in ("sources", "continuations"):
        raise ValueError("Unknown comparison tree")
    out = Path(out).resolve()
    return out, out / name, out / (name + ".storage")


def archive_tree(out, name):
    out, source, packet = comparison_tree_paths(out, name)
    if not source.exists():
        return
    pack_verified(
        source,
        packet,
        token="comparison-" + name,
        temporary_prefix="verify-comparison-",
        changed_message="Comparison data changed during verification",
    )
    if source.resolve() != out / name or source.is_symlink():
        raise ValueError("Unsafe comparison storage path")
    shutil.rmtree(source)


def restore_tree(out, name):
    _, source, packet = comparison_tree_paths(out, name)
    if not source.exists() and packet.exists():
        unpack(packet, source, token="comparison-" + name, direction="download")


@contextmanager
def readable_tree(out, name):
    """Read an immutable packet; discard verified extraction without recompression.

    Caller owns the run lock. Retain modified files on failure, rather than
    silently deleting evidence or rewriting the original PPMd archive.
    """
    from civ5rl.deploy.archive import _inventory
    from civ5rl.provenance.common import read_json

    out, source, packet = comparison_tree_paths(out, name)
    restored = packet.exists()
    restore_tree(out, name)
    try:
        yield source
    finally:
        if restored:
            manifest = read_json(packet / "manifest.json")
            if _inventory(source) != (manifest["files"], manifest["directories"]):
                raise RuntimeError("Comparison reader changed files; retaining extracted data")
            if source.resolve() != out / name or source.is_symlink():
                raise ValueError("Unsafe comparison tree")
            shutil.rmtree(source)


@contextmanager
def readable_members(packet, names):
    """Extract only requested packet members and verify their original bytes.

    Solid archives still require decoding preceding data, but selecting members
    avoids materializing unrelated multi-gigabyte replay trees. This is a
    disposable read cache, never a resume dependency or a rewritten input packet.
    """
    import subprocess
    import tempfile

    from civ5rl.deploy.archive import _name, seven_zip
    from civ5rl.provenance.common import file_sha256, read_json

    packet = Path(packet).resolve()
    manifest = read_json(packet / "manifest.json")
    names = sorted(set(_name(n) for n in names))
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
