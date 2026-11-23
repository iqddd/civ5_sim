"""Publish and install the final, already-archived output of a remote job."""

import os
import shutil
import tempfile
from pathlib import Path

from civ5rl.deploy.remote import _read, _write
from civ5rl.engine.evaluation.storage import TOKEN as EVALUATION_TOKEN
from civ5rl.engine.runtime.archive import inventory, pack, read_manifest
from civ5rl.engine.runtime.control import RunLock

FORMAT = "final-run-v1"
COMPLETE = ("complete", "completed", "finished", "closed")


def _link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def publish_final_run(run, mirror, token):
    """Prepare an immutable post-sync bundle after the runner has exited."""
    from civ5rl.reporting.operations import audit_run

    run, mirror = Path(run).resolve(), Path(mirror)
    status_path = run / "status.json"
    if not status_path.is_file() or _read(status_path).get("status") not in COMPLETE:
        return

    bundle = mirror / "final"
    with tempfile.TemporaryDirectory(prefix=".final-", dir=mirror.parent) as temporary:
        staged = Path(temporary) / "bundle"
        print("Auditing the completed run on the server...", flush=True)
        audit = audit_run(run, staged / "audit")
        with RunLock(run):
            # Keep the live run intact. A later resume on the same host may need it.
            evaluations = run / "evaluation"
            archive_evaluations = evaluations.is_dir()
            excluded = {"run.lock"}
            if archive_evaluations:
                excluded.update(("evaluation", "evaluation.storage"))
            shutil.copytree(
                run, staged / "run", symlinks=True,
                ignore=lambda directory, names: excluded.intersection(names) if Path(directory) == run else (),
            )
            storage = staged / "run/evaluation.storage"
            if archive_evaluations:
                print("Packing the final evaluation archive on the server...", flush=True)
                archive = pack(evaluations, storage, token=EVALUATION_TOKEN,
                               direction="download", force=True, source_locked=True)
            else:
                archive = read_manifest(storage) if storage.is_dir() else {"archived": []}
            _write(staged / "receipt.json", {
                "format": FORMAT,
                "token": token,
                "scientific_audit": audit["status"],
                "archived_files": len(archive["archived"]),
            })
            staged.replace(bundle)


def _install_tree(source, destination, *, keep=()):
    """Apply a skill-verified tree without reading payload bytes for a second hash."""
    files, directories = inventory(source, hash_files=False)
    existing, old_dirs = inventory(destination, hash_files=False)
    keep = set(keep)
    with tempfile.TemporaryDirectory(prefix=".install-", dir=destination) as temporary:
        staged = Path(temporary) / "file"
        for name in existing.keys() - files.keys() - keep:
            (destination / name).unlink()
        for name in sorted(set(old_dirs) - set(directories), key=lambda name: name.count("/"), reverse=True):
            (destination / name).rmdir()
        for name in directories:
            (destination / name).mkdir(parents=True, exist_ok=True)
        for name in sorted(files.keys() - keep, key=lambda item: (item == "status.json", item)):
            _link_or_copy(source / name, staged)
            staged.replace(destination / name)


def install_final_run(transfer, post_sync, audit_dir):
    """Check job binding and publish the already verified, still compressed output."""
    mirror = Path(transfer["local"]) / "upload"
    bundle = mirror / "final"
    receipt = _read(bundle / "receipt.json")
    audit = _read(bundle / "audit/audit.json")
    if (
        receipt["format"] != FORMAT
        or receipt["token"] != transfer["token"]
        or receipt["scientific_audit"] != audit["status"]
        or Path(post_sync["local_dir"]).resolve() != mirror.resolve()
        or post_sync["remote_sync_dir"] != transfer["remote_upload"]
        or _read(bundle / "run/status.json").get("status") not in COMPLETE
        or (bundle / "run/evaluation").exists()
    ):
        raise ValueError("Final output does not match the verified job mirror")
    files, _ = inventory(bundle, hash_files=False)
    if len(files) > post_sync["verified_files"] + post_sync["unchanged_files"]:
        raise ValueError("Final output exceeds the skill's verified file count")
    destination = Path(transfer["local_destination"])
    with RunLock(destination):
        _install_tree(bundle / "run", destination, keep=("run.lock",))
    audit_dir.mkdir(parents=True, exist_ok=True)
    _install_tree(bundle / "audit", audit_dir)
    return {
        "format": FORMAT,
        "verified_files": len(files),
        "archived_files": receipt["archived_files"],
        "bytes": sum(item["size"] for item in files.values()),
    }
