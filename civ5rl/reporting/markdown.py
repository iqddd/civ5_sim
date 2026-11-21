"""Markdown primitives for artifact reports."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

Alignment = Literal["left", "right", "center", "---", ":---", "---:", ":---:"]

_ALIGNMENT_MARKER: dict[Alignment, str] = {
    "left": "---",
    "right": "---:",
    "center": ":---:",
    "---": "---",
    ":---": ":---",
    "---:": "---:",
    ":---:": ":---:",
}


def render_table(
    headers: Sequence[str],
    alignments: Sequence[Alignment],
    rows: Iterable[Sequence[object]] = (),
) -> list[str]:
    """Render the compact divider and spaced cells used in saved reports."""
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(_ALIGNMENT_MARKER[alignment] for alignment in alignments) + "|",
    ]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    return lines


def format_wdl(summary: Mapping[str, object]) -> str:
    return f"{summary['W']}/{summary['D']}/{summary['L']}"


class MarkdownReport(list[str]):
    """Report lines with the same ordering and newline handling as existing reports."""

    def table(
        self,
        headers: Sequence[str],
        alignments: Sequence[Alignment],
        rows: Iterable[Sequence[object]] = (),
    ) -> None:
        self.extend(render_table(headers, alignments, rows))

    def render(self, *, trailing_newline: bool = True) -> str:
        return "\n".join(self) + ("\n" if trailing_newline else "")

    def write(self, path: Path, *, trailing_newline: bool = True) -> None:
        path.write_text(self.render(trailing_newline=trailing_newline), encoding="utf-8")
