"""Strict JSONL I/O for artifact ledgers."""

from __future__ import annotations

import json
from os import PathLike
from pathlib import Path
from typing import Any


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
