"""Verified inactive comparison data storage, outside execution passports."""

from pathlib import Path
from contextlib import contextmanager
import shutil

from civ5rl.deploy.archive import unpack
from civ5rl.reporting.storage import pack_verified


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
def readable_tree(out,name):
    """Read an immutable packet; discard verified extraction without recompression.

    Caller owns the run lock. Retain modified files on failure, rather than
    silently deleting evidence or rewriting the original PPMd archive.
    """
    from civ5rl.deploy.archive import _inventory
    from civ5rl.provenance.common import read_json
    out,source,packet=comparison_tree_paths(out,name)
    restored=packet.exists()
    restore_tree(out,name)
    try:yield source
    finally:
        if restored:
            manifest=read_json(packet/'manifest.json')
            if _inventory(source)!=(manifest['files'],manifest['directories']):
                raise RuntimeError('Comparison reader changed files; retaining extracted data')
            if source.resolve()!=out/name or source.is_symlink():raise ValueError('Unsafe comparison tree')
            shutil.rmtree(source)
