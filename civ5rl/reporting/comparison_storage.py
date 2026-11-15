"""Verified inactive comparison data storage, outside execution passports."""

from pathlib import Path
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
