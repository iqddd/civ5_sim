"""Shared verified packaging for inactive reporting data trees."""

import tempfile
from pathlib import Path

from civ5rl.deploy.archive import _extract, _inventory, _read, pack


def pack_verified(source, packet, *, token, temporary_prefix, temporary_dir=None, changed_message):
    """Package and verify a tree; callers retain lock and deletion ownership."""
    pack(source, packet, token=token, direction="download", force=True)
    manifest = _read(Path(packet) / "manifest.json")
    with tempfile.TemporaryDirectory(prefix=temporary_prefix, dir=temporary_dir) as temporary:
        _extract(packet, Path(temporary), manifest)
    if _inventory(source) != (manifest["files"], manifest["directories"]):
        raise RuntimeError(changed_message)
    return manifest
