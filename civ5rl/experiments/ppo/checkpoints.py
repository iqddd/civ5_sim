"""Checkpoint retention for PPO experiment directories."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

from civ5rl.provenance.common import read_json, write_json


def prune_checkpoints(out: str | PathLike[str], state: dict[str, Any]) -> None:
    """Keep initial, evaluated, selected, and the latest two saved boundaries."""
    out = Path(out).resolve()
    directory = out / "checkpoints"
    paths = sorted(directory.glob("update-*.pt"))
    keep = {"initial.pt", Path(state["latest"]).name, Path(state["best"]["path"]).name}
    keep.update(p.name for p in paths[-2:])
    keep.update(Path(e["checkpoint"]).name for e in state["evaluated"].values())
    keep.update(Path(e["path"]).name for e in state.get("self_play_archive", ()))
    journal = out / "checkpoint-retention.json"
    removed = read_json(journal) if journal.exists() else []
    for path in paths:
        if path.name in keep:
            continue
        if path.resolve().parent != directory:
            raise ValueError("Checkpoint retention target outside run directory")
        removed.append({"path": path.relative_to(out).as_posix(), "bytes": path.stat().st_size})
        # Persist the intention before deletion; no model or metrics are rewritten.
        write_json(journal, removed)
        path.unlink()
