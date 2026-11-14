"""Record an explicit host transfer at an unchanged paused PPO boundary."""

import shutil
import tempfile
import zipfile
from pathlib import Path

from civ5rl.engine.control import RunLock
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import ROOT, file_sha256, read_json, write_json
from civ5rl.provenance.host_resume import host_delta, identity_digest, verify_revision


def record_host_resume(run, host_id, reason):
    run = Path(run).resolve()
    if not reason.strip() or not host_id.isdigit():
        raise ValueError("A numeric host ID and reason are required")
    with RunLock(run):
        state = read_json(run / "state.json")
        if read_json(run / "status.json").get("status") != "paused" or state["phase"] != "collect":
            raise ValueError("Host transfer requires a paused completed-update boundary")
        config = read_json(run / "resolved-config.json")
        manifest = read_json(run / "manifest.json")
        before = checkpoint.verify_frozen_passport(run, config=config, manifest=manifest)
        with tempfile.TemporaryDirectory(prefix="host-identity-") as temp:
            after = checkpoint.passport(temp, config, manifest)
        changes = host_delta(before, after)
        target = run / "source-revisions" / ("host-" + host_id)
        if target.exists():
            return verify_revision(target / "revision.json", before, after)
        saved_checkpoint = (run / state["latest"]).resolve()
        if not saved_checkpoint.is_relative_to(run):
            raise ValueError("Checkpoint path escapes the run")
        target.mkdir(parents=True)
        shutil.copyfile(saved_checkpoint, target / "checkpoint.pt")
        shutil.copyfile(run / "state.json", target / "state-before.json")
        shutil.copyfile(run / "updates.jsonl", target / "ledger-before.jsonl")
        write_json(target / "identity-after.json", after)
        with zipfile.ZipFile(target / "sources-after.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            for name in changes:
                archive.write(ROOT / name, name)
        files = {name: file_sha256(target / name) for name in (
            "checkpoint.pt", "state-before.json", "ledger-before.jsonl", "identity-after.json", "sources-after.zip")}
        record = dict(schema=1, kind="rl01.host_resume", reason=reason, host_id=host_id,
                      before_identity=identity_digest(before), after_identity=identity_digest(after),
                      platform_before=before["platform"], platform_after=after["platform"],
                      source_changes=changes, checkpoint=state["latest"], updates=state["updates"],
                      decisions=state["decisions"], Adam_counters=state.get("Adam_counters"), files=files)
        write_json(target / "revision.json", record)
        return verify_revision(target / "revision.json", before, after)
