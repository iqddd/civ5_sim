"""Storage access for artifact reports."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from civ5rl.engine.evaluation.storage import (
    TOKEN,
    archive_evaluations,
    readable_evaluations,
    restore_evaluations,
)
from civ5rl.engine.runtime.archive import extract, inventory, member_name, read_manifest
from civ5rl.engine.runtime.control import RunLock
from civ5rl.reporting.output_paths import derived_output

__all__ = ["TOKEN", "archive_evaluations", "extract_evaluation_match", "readable_evaluations", "restore_evaluations"]


def extract_evaluation_match(run, match, output):
    """Export one saved match without restoring the evaluation tree."""
    run = Path(run).resolve()
    match = member_name(match)
    output = derived_output(run, output)
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Match output must be an empty directory")
    with RunLock(run):
        if (run / "evaluation").is_dir():
            active = run / "evaluation" / match
            if active.resolve() != active or not (active / "replay.json").is_file():
                raise ValueError("Select one saved match directory containing replay.json")
            inventory(active, hash_files=False)
            shutil.copytree(active, output, dirs_exist_ok=True)
        else:
            storage = run / "evaluation.storage"
            manifest = read_manifest(storage)
            if manifest.get("schema") != 1 or manifest.get("token") != TOKEN or manifest.get("direction") != "download":
                raise ValueError("Expected an evaluation storage archive")
            if match + "/replay.json" not in manifest["files"]:
                raise ValueError("Select one archived match directory containing replay.json")
            files = [name for name in manifest["files"] if name.startswith(match + "/")]
            with tempfile.TemporaryDirectory(prefix="evaluation-match-") as temporary:
                staging = Path(temporary)
                extract(storage, staging, manifest, selection=files)
                shutil.copytree(staging / match, output, dirs_exist_ok=True)
    return {"status": "extracted", "match": match, "out": str(output)}
