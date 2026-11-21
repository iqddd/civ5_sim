"""Validate derived output locations without depending on reporting workflows."""

from __future__ import annotations

from pathlib import Path


def derived_output(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if directory == output or directory.is_relative_to(output):
        raise ValueError("Derived output must be separate from the scientific run root")
    if output.is_relative_to(directory) and output.relative_to(directory).parts[0] not in (
        "derived",
        "snapshots",
        "interim-snapshot",
    ):
        raise ValueError("In-run derived output must use derived/, snapshots/ or interim-snapshot/")
    return output
