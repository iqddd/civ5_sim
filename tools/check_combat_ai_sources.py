"""Verify the pinned CP source inventory without modifying either checkout."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(cp: Path):
    manifest = json.loads(
        (ROOT / "docs/combat-ai-sources.json").read_text(encoding="utf-8")
    )
    commit = subprocess.check_output(
        ["git", "-C", str(cp), "rev-parse", "HEAD"], text=True
    ).strip()
    if commit != manifest["commit"]:
        raise ValueError(f"CP commit mismatch: {commit}")
    for name, expected in manifest["sources"].items():
        actual = hashlib.sha256((cp / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"CP source SHA-256 mismatch: {name}")
    print(f"Verified {len(manifest['sources'])} CP sources at {commit}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cp", type=Path, default=Path("E:/civ5/Community-Patch-DLL"))
    verify(parser.parse_args().cp.resolve())
