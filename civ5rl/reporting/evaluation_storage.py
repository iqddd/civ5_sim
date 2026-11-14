"""Verified PPMd storage of inactive evaluation trees; scientific bytes stay intact."""

import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from civ5rl.deploy.archive import _extract, _inventory, _read, pack, unpack
from civ5rl.engine.control import RunLock

TOKEN = "evaluation-storage-v1"


def _paths(run):
    run = Path(run).resolve()
    return run, run / "evaluation", run / "evaluation.storage"


def archive_evaluations(run):
    run, source, packet = _paths(run)
    with RunLock(run):
        if not source.exists():
            return {"status": "already_archived" if packet.exists() else "absent"}
        pack(source, packet, token=TOKEN, direction="download", force=True)
        manifest = _read(packet / "manifest.json")
        with tempfile.TemporaryDirectory(prefix=".verify-evaluation-", dir=run) as temp:
            _extract(packet, Path(temp), manifest)
        if _inventory(source) != (manifest["files"], manifest["directories"]):
            raise RuntimeError("Evaluation changed during archive verification")
        # Both paths are resolved fixed children of run; never delete an arbitrary target.
        if source.resolve() != run / "evaluation" or source.is_symlink():
            raise ValueError("Unsafe evaluation directory")
        shutil.rmtree(source)
        return {"status": "archived", "files": len(manifest["files"]),
                "original_bytes": sum(f["size"] for f in manifest["files"].values()),
                "archive_bytes": sum(path.stat().st_size for path in packet.rglob("*") if path.is_file())}


def restore_evaluations(run):
    run, source, packet = _paths(run)
    with RunLock(run):
        if source.exists():
            return {"status": "already_restored"}
        if not packet.exists():
            return {"status": "absent"}
        return unpack(packet, source, token=TOKEN, direction="download")


@contextmanager
def readable_evaluations(run):
    """Caller owns the run lock; temporary reads never change the stored archive."""
    run, source, packet = _paths(run)
    restored = packet.exists() and not source.exists()
    if restored:
        unpack(packet, source, token=TOKEN, direction="download")
    try:
        yield
    finally:
        if restored:
            manifest = _read(packet / "manifest.json")
            if _inventory(source) != (manifest["files"], manifest["directories"]):
                raise RuntimeError("Evaluation reader changed files; retaining extracted data")
            shutil.rmtree(source)
