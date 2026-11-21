"""Shared verified packaging for inactive data trees."""

from __future__ import annotations

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from civ5rl.engine.runtime.archive import extract, inventory, member_name, pack, read_manifest, unpack


def pack_verified(source, packet, *, token, temporary_prefix, temporary_dir=None, changed_message):
    """Package and verify a tree; callers retain lock and deletion ownership."""
    pack(source, packet, token=token, direction="download", force=True)
    manifest = read_manifest(packet)
    with tempfile.TemporaryDirectory(prefix=temporary_prefix, dir=temporary_dir) as temporary:
        extract(packet, Path(temporary), manifest)
    if inventory(source) != (manifest["files"], manifest["directories"]):
        raise RuntimeError(changed_message)
    return manifest


class ArchivedTree:
    """Manage a tree and its sibling .storage packet under the caller's lock."""

    def __init__(self, root, name, *, token, label, temporary_prefix, changed_message, verify_in_root=False):
        if Path(member_name(name)).name != name:
            raise ValueError("Archived tree must be a direct child of its root")
        self.root = Path(root).resolve()
        self.source = self.root / name
        self.packet = self.root / (name + ".storage")
        self.token = token
        self.label = label
        self.temporary_prefix = temporary_prefix
        self.temporary_dir = self.root if verify_in_root else None
        self.changed_message = changed_message

    def _remove_source(self):
        if self.source.resolve() != self.source or self.source.is_symlink():
            raise ValueError(f"Unsafe {self.label.lower()} storage path")
        shutil.rmtree(self.source)

    def archive(self):
        """Verify the stored bytes before removing the active tree."""
        if not self.source.exists():
            return None
        manifest = pack_verified(
            self.source,
            self.packet,
            token=self.token,
            temporary_prefix=self.temporary_prefix,
            temporary_dir=self.temporary_dir,
            changed_message=self.changed_message,
        )
        self._remove_source()
        return manifest

    def restore(self):
        if self.source.exists():
            return {"status": "already_restored"}
        if not self.packet.exists():
            return {"status": "absent"}
        return unpack(self.packet, self.source, token=self.token, direction="download")

    @contextmanager
    def readable(self, *, cleanup_existing=False):
        """Discard a read copy, including an existing copy when explicitly requested.

        Inventory comparison protects edits made inside the context and any
        preexisting copy from being silently deleted.
        """
        temporary = self.packet.exists() and (cleanup_existing or not self.source.exists())
        self.restore()
        try:
            yield self.source
        finally:
            if temporary:
                manifest = read_manifest(self.packet)
                if inventory(self.source) != (manifest["files"], manifest["directories"]):
                    raise RuntimeError(f"{self.label} reader changed files; retaining extracted data")
                self._remove_source()
