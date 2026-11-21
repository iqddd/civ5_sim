"""Artifact I/O shared by experiment protocols."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import torch

from civ5rl.engine.runtime.transactions import atomic_torch_save
from civ5rl.provenance.common import atomic_write_jsonl

__all__ = [
    "append_jsonl",
    "atomic_legacy_torch_save",
    "atomic_torch_save",
    "atomic_write_jsonl",
    "read_jsonl",
    "read_jsonl_lines",
    "recover_jsonl_prefix",
]


def read_jsonl_lines(
    path: str | PathLike[str],
    *,
    skip_blank: bool = True,
    encoding: str | None = "utf-8",
) -> list[str]:
    """Read ledger lines without decoding JSON or changing retained text."""
    lines = Path(path).read_text(encoding=encoding).splitlines()
    return [line for line in lines if line.strip()] if skip_blank else lines


def read_jsonl(
    path: str | PathLike[str],
    *,
    missing_ok: bool = False,
    skip_blank: bool = True,
    streaming: bool = False,
    encoding: str | None = "utf-8",
) -> list[dict[str, Any]]:
    """Read JSONL records without discarding malformed or truncated lines.

    Streaming only splits on universal newlines, so raw Unicode line separators
    in JSON strings remain intact. The default preserves ledger splitlines().
    """
    path = Path(path)
    try:
        if streaming:
            with path.open(encoding=encoding) as stream:
                return [json.loads(line) for line in stream if not skip_blank or line.strip()]
        return [json.loads(line) for line in read_jsonl_lines(path, skip_blank=skip_blank, encoding=encoding)]
    except FileNotFoundError:
        if not missing_ok:
            raise
        return []


def recover_jsonl_prefix(
    path: Path, lines: list[str], committed: int, *, missing_error: str
) -> list[dict[str, Any]]:
    """Drop orphaned lines before decoding and keep the original committed text."""
    if len(lines) < committed:
        raise ValueError(missing_error)
    if len(lines) > committed:
        # A process can stop mid-append. Retained records may use custom spacing.
        lines = lines[:committed]
        path.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
    return [json.loads(line) for line in lines]


def append_jsonl(path: str | PathLike[str], row: dict[str, Any]) -> None:
    """Append a finite JSON object using the ledger's existing encoding."""
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


def atomic_legacy_torch_save(value: object, path: str | PathLike[str]) -> None:
    """Publish the path-based Torch archive used by rollouts and critic capsules."""
    path = Path(path)
    # Torch uses the temporary file's stem as the ZIP entry prefix. A private
    # directory lets us keep that stem while isolating concurrent writers.
    with TemporaryDirectory(dir=path.parent, prefix=".write-") as directory:
        temporary = Path(directory) / path.with_suffix(".tmp").name
        torch.save(value, temporary)
        temporary.replace(path)
