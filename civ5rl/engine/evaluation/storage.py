"""Verified PPMd storage of inactive evaluation trees."""

from __future__ import annotations

from contextlib import contextmanager

from civ5rl.engine.runtime.control import RunLock
from civ5rl.engine.runtime.storage import ArchivedTree

TOKEN = "evaluation-storage-v1"


def _tree(run):
    return ArchivedTree(
        run,
        "evaluation",
        token=TOKEN,
        label="Evaluation",
        temporary_prefix=".verify-evaluation-",
        changed_message="Evaluation changed during archive verification",
        verify_in_root=True,
    )


def archive_evaluations(run):
    tree = _tree(run)
    with RunLock(tree.root):
        manifest = tree.archive()
        if manifest is None:
            return {"status": "already_archived" if tree.packet.exists() else "absent"}
        return {
            "status": "archived",
            "files": len(manifest["files"]),
            "original_bytes": sum(f["size"] for f in manifest["files"].values()),
            "archive_bytes": sum(path.stat().st_size for path in tree.packet.rglob("*") if path.is_file()),
        }


def restore_evaluations(run):
    tree = _tree(run)
    with RunLock(tree.root):
        return tree.restore()


@contextmanager
def readable_evaluations(run):
    """Caller owns the run lock; temporary reads never change the stored archive."""
    with _tree(run).readable() as source:
        yield source
