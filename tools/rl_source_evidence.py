"""Verify or restore historical source variants from Git blobs and small patches."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def restore_source(record, evidence_dir, *, root=ROOT):
    """Reconstruct exact bytes, including the original line endings."""
    base = record.get("git_base")
    if base:
        data = subprocess.check_output(
            ["git", "show", f"{base['commit']}:{base['path']}"], cwd=root
        )
        if sha256(data) != base["sha256"]:
            raise ValueError("Git source base differs")
        data = data.replace(b"\r\n", b"\n")
    else:
        data = b""
    if record.get("patch"):
        directory = Path(evidence_dir).resolve()
        path = (directory / record["patch"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError("Patch is outside evidence directory")
        patch = path.read_bytes()
        if sha256(patch) != record["patch_sha256"]:
            raise ValueError("Source patch differs")
        # Every patch targets this one temporary file; no checkout is modified.
        headers = [line for line in patch.splitlines() if line.startswith((b"--- ", b"+++ "))]
        if headers != [b"--- a/source.py", b"+++ b/source.py"]:
            raise ValueError("Unexpected source patch paths")
        with tempfile.TemporaryDirectory(prefix="civ5-source-") as temp:
            target = Path(temp) / "source.py"
            target.write_bytes(data)
            subprocess.run(
                ["git", "-c", "core.autocrlf=false", "-c", "core.eol=lf",
                 "apply", "--whitespace=nowarn", "-"], cwd=temp,
                input=patch, check=True, capture_output=True,
            )
            data = target.read_bytes().replace(b"\r\n", b"\n")
    if record["line_endings"] == "crlf":
        data = data.replace(b"\n", b"\r\n")
    elif record["line_endings"] != "lf":
        raise ValueError("Unsupported source line endings")
    if len(data) != record["bytes"] or sha256(data) != record["sha256"]:
        raise ValueError(f"Source reconstruction differs: {record['original_path']}")
    return data


def load_historical_source(manifest, logical_source, expected_sha256):
    """Validate a recorded input without comparing it to today's changed sources."""
    manifest = Path(manifest)
    entries = json.loads(manifest.read_text(encoding="utf-8"))["sources"]
    matches = [row for row in entries if row["logical_source"] == logical_source
               and row["sha256"] == expected_sha256]
    if not matches:
        raise ValueError(f"Historical source not recorded: {logical_source}: {expected_sha256}")
    return restore_source(matches[0], manifest.parent)


def verify_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for entry in manifest["sources"]:
        restore_source(entry, path.parent)
    # Results and telemetry were not edited during migration.
    for name, expected in manifest.get("preserved_files", {}).items():
        target = (path.parent / name).resolve()
        if not target.is_relative_to(path.parent.resolve()):
            raise ValueError("Preserved artifact outside evidence directory")
        if sha256(target.read_bytes()) != expected:
            raise ValueError(f"Historical result changed: {name}")
    historical = manifest.get("historical_manifest")
    if historical:
        target = (path.parent / historical).resolve()
        if not target.is_relative_to(path.parent.resolve()):
            raise ValueError("Historical manifest outside evidence directory")
        if sha256(target.read_bytes()) != manifest["historical_manifest_sha256"]:
            raise ValueError("Historical hash manifest changed")
    return {"sources_verified": len(manifest["sources"]),
            "preserved_files_verified": len(manifest.get("preserved_files", {})),
            "retired_helpers": len(manifest.get("retired_helpers", []))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("manifest", type=Path)
    restore = sub.add_parser("restore")
    restore.add_argument("manifest", type=Path)
    restore.add_argument("original_path", help="Path recorded in the evidence manifest")
    restore.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify_manifest(args.manifest)
    else:
        entries = json.loads(args.manifest.read_text(encoding="utf-8"))["sources"]
        rows = [row for row in entries if row["original_path"] == args.original_path]
        if len(rows) != 1:
            parser.error("Select exactly one recorded original_path")
        data = restore_source(rows[0], args.manifest.parent)
        if args.output.exists():
            parser.error("Restore output already exists")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(data)
        result = {"output": str(args.output), "sha256": sha256(data)}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
