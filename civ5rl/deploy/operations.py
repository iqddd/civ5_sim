"""Declarative remote deployments using prepare.json and split input/output/scratch."""

import hashlib
import json
import re
import shutil
import subprocess
import sys
import shlex
import tempfile
from uuid import uuid4
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from civ5rl.deploy.remote import (
    ROOT,
    _read,
    _relative_file,
    _resolve_config,
    _sha,
    _write,
    inspect_remote,
    stage_inputs,
)
from civ5rl.engine.control import RunLock

CAPABILITIES = {
    "rl01.ppo_after_target_bc": {"run", "tune", "resume"},
    "rl01.ppo_target_scale": {"run", "resume"},
    "rl01.ppo_scale_up": {"run"},
    "rl01.cp_imitation_target_more_data": {"run", "resume"},
    "rl01.cp_imitation_target_c2": {"run", "resume"},
    "rl01.cp_imitation_target_scale": {"run"},
}

SYNC_MAX_ENTRIES = 100000
SYNC_MAX_MIB = 16384


def _relative(name):
    path = PurePosixPath(str(name).replace("\\", "/"))
    if (
        path.is_absolute()
        or ":" in str(path)
        or ".." in path.parts
        or str(path) in ("", ".")
    ):
        raise ValueError(f"Expected project-relative path: {name}")
    return str(path)


def _instance(path):
    raw = _read(path)
    host = raw.get("host") or raw.get("SSH_host")
    port = int(raw.get("port") or raw.get("SSH_port") or 22)
    identifier = raw.get("instance_id") or raw.get("id")
    if not identifier or not host or not 0 < port < 65536:
        raise ValueError(
            "Instance JSON requires instance_id/id, host/SSH_host and port"
        )
    return {"instance_id": int(identifier), "host": str(host), "port": port}


def _receipt(deployment, operation, **values):
    deployment.setdefault("operations", []).append(
        {"operation": operation, "at": datetime.now(UTC).isoformat(), **values}
    )


def _github_repo(root, explicit=None):
    raw = (
        explicit
        or subprocess.check_output(
            ["git", "remote", "get-url", "origin"], cwd=root, text=True
        ).strip()
    )
    match = re.match(
        r"^(?:https://github\.com/|git@github\.com:)?([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?)(?:\.git)?/?$",
        raw.strip(),
    )
    if not match:
        raise ValueError(f"Unsupported GitHub repository specification: {raw!r}")
    repo_spec = match.group(1)
    repo_name = repo_spec.split("/", 1)[1]
    return repo_spec, repo_name


def _verify_git_state(root, *, verify_origin=True):
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if not verify_origin:
        return commit
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).strip()
    if dirty:
        raise ValueError(
            "Working tree has uncommitted changes; commit and push before packaging:\n"
            + dirty
        )
    branches = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=root, text=True
    ).strip()
    if not branches:
        raise ValueError(
            f"Commit {commit[:12]} is not pushed to origin; run git push before packaging"
        )
    return commit


def package(
    config,
    output,
    *,
    instance=None,
    extra_inputs=(),
    root=ROOT,
    repo=None,
    verify_origin=True,
):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Deployment output must be empty; frozen packages are immutable"
        )
    name = _relative_file(root, config)
    resolved_config, _ = _resolve_config(root, name)
    if resolved_config.get("kind") not in CAPABILITIES:
        raise ValueError("Unsupported remote experiment kind")
    scientific_output = _relative(resolved_config["output"])
    local_output = (root / scientific_output).resolve()
    if output.is_relative_to(local_output) or local_output.is_relative_to(output):
        raise ValueError(
            "Deployment directory must stay outside the experiment output mirror: "
            f"{local_output}"
        )

    target = _instance(instance) if instance else None
    extras = [_relative_file(root, path) for path in extra_inputs]
    commit = _verify_git_state(root, verify_origin=verify_origin)
    repo_spec, repo_name = _github_repo(root, repo)

    output.mkdir(parents=True, exist_ok=True)
    inputs_dir = output / "inputs"
    staged = stage_inputs(name, inputs_dir, root=root, extra_inputs=extras)
    inputs_manifest_path = output / "inputs-manifest.json"
    _write(inputs_manifest_path, staged)

    fingerprint = hashlib.sha256(
        json.dumps(
            {
                "commit": commit,
                "repo": repo_spec,
                "config": name,
                "staged_files": staged["staged_files"],
                "tracked_inputs": staged["tracked_inputs"],
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    from civ5rl.deploy.archive import needs_archive, pack as pack_transfer

    input_packets = []
    input_sync = []
    for index, group in enumerate(staged["groups"]):
        source = inputs_dir / group
        if needs_archive(source):
            packet = output / "input-packets" / str(index)
            pack_transfer(source, packet, token=fingerprint, direction="upload")
            remote = f"transport/inputs/{index}"
            input_sync.append({"local": str(packet), "remote": remote})
            input_packets.append(dict(packet=remote, destination=group, token=fingerprint))
        else:
            input_sync.append({"local": str(source.resolve()), "remote": f"{repo_name}/{group}"})
    input_packet_files = {
        path.relative_to(output).as_posix(): _sha(path)
        for path in (output / "input-packets").rglob("*") if path.is_file()
    }
    prepare_spec = {
        "execution_subdir": repo_name,
        "revision": fingerprint,
        "python": "3.13",
        "repositories": [{"repo": repo_spec, "commit": commit}],
        "requirements": [f"{repo_name}/requirements-rl.txt"],
        "sync": input_sync,
    }
    prepare_path = output / "prepare.json"
    _write(prepare_path, prepare_spec)

    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(scientific_output).name).strip("-")
    remote_root = f"/workspace/civ5-runs/{slug}-{fingerprint[:12]}"
    parent_checkpoint = resolved_config.get("parent") or resolved_config.get(
        "experiment", {}
    ).get("parent")
    carriers = [
        item
        for item in extras
        if item.endswith(".pt") and item != parent_checkpoint
    ]
    deployment = {
        "schema": 2,
        "archive_transport": 1,
        "input_packets": input_packets,
        "input_packet_files": input_packet_files,
        "kind": resolved_config["kind"],
        "config": name,
        "resolved_config": resolved_config,
        "commit": commit,
        "repo": repo_spec,
        "repo_name": repo_name,
        "fingerprint": fingerprint,
        "prepare_sha256": _sha(prepare_path),
        "inputs_manifest_sha256": _sha(inputs_manifest_path),
        "output": scientific_output,
        "local_output": str(local_output),
        "remote_root": remote_root,
        "remote_project": f"{remote_root}/{repo_name}",
        "remote_scratch": f"{remote_root}/scratch",
        "instance": target,
        "tuning_rollout": carriers[0] if len(carriers) == 1 else None,
        "job": None,
        "runtime": None,
        "operations": [],
    }
    _receipt(
        deployment,
        "package",
        status="complete",
        staged_groups=staged["groups"],
        staged_files=len(staged["staged_files"]),
    )
    _write(output / "deployment.json", deployment)
    return deployment


def verify_package(directory, deployment):
    prepare_path = directory / "prepare.json"
    manifest_path = directory / "inputs-manifest.json"
    if _sha(prepare_path) != deployment["prepare_sha256"]:
        raise ValueError("Frozen prepare.json changed")
    if _sha(manifest_path) != deployment["inputs_manifest_sha256"]:
        raise ValueError("Frozen inputs-manifest.json changed")
    manifest = _read(manifest_path)
    inputs_dir = directory / "inputs"
    actual = {}
    if inputs_dir.exists():
        for path in sorted(inputs_dir.rglob("*")):
            if path.is_file():
                rel = path.relative_to(inputs_dir).as_posix()
                actual[rel] = _sha(path)
    if actual != manifest["staged_files"]:
        raise ValueError("Staged input files differ from inputs-manifest.json")
    for name, digest in deployment.get("input_packet_files", {}).items():
        if _sha(directory / name) != digest:
            raise ValueError(f"Frozen input packet changed: {name}")


class Transport:
    """Subprocess boundary around remote-python-ssh and vastai-instance-manager."""

    def __init__(self):
        self.script = (
            Path.home()
            / ".codex/skills/remote-python-ssh/scripts/assistant_remote_python.py"
        )

    def invoke(self, instance, args):
        command = [
            sys.executable,
            str(self.script),
            "--host",
            instance["host"],
            "--port",
            str(instance["port"]),
            *args,
        ]
        return subprocess.run(
            command, stdin=subprocess.DEVNULL, check=False
        ).returncode

    def validate_attach(self, deployment):
        job = deployment["job"]
        current = Path.home() / ".cache/remote-python-ssh/current"
        for path in current.glob("*.json"):
            record = _read(path)
            serialized = json.dumps(record)
            if deployment["remote_root"] in serialized:
                if job.get("id") and json.dumps(job["id"]) not in serialized:
                    continue
                identifier = record.get("job_id")
                if identifier:
                    job["id"] = identifier
                    return
        raise ValueError(
            "Matching durable registry unavailable; refusing attach to an unrelated job"
        )

    def delete(self, instance):
        script = (
            Path.home()
            / ".codex/skills/vastai-instance-manager/scripts/vastai_instance.py"
        )
        subprocess.run(
            [sys.executable, str(script), "delete", str(instance["instance_id"])],
            check=True,
        )

    def download_packet(self, deployment, transfer):
        from civ5rl.deploy.archive import unpack
        from civ5rl.deploy.remote import _connection, _remote_path, _ssh_json

        connection = _connection(deployment["instance"])
        code = """
import json, sys
from pathlib import Path
p=json.load(sys.stdin)
sys.path.insert(0,p['root'])
from civ5rl.deploy.archive import pack
from civ5rl.deploy.remote import _sha
from civ5rl.deploy.remote import _read
from civ5rl.engine.control import RunLock
if not Path(p['source']).is_dir() or _read(Path(p['upload'])/'applied.json').get('token') != p['token']:
    raise ValueError('Scientific upload was not applied; refusing to replace local results')
with RunLock(p['source']):
    m=pack(p['source'],p['packet'],token=p['token'],direction='download')
print(json.dumps({'files':len(m['files']),'archived_files':len(m['archived']),
                 'manifest_sha256':_sha(Path(p['packet'])/'manifest.json')}))
"""
        print("Packaging stopped remote run for verified transfer...", flush=True)
        packed = _ssh_json(connection, code, dict(root=_remote_path(deployment["remote_project"]),
                  source=_remote_path(transfer["destination"]), packet=_remote_path(transfer["remote_download"]),
                  upload=_remote_path(transfer["remote_upload"]), token=transfer["token"]), timeout=7200,
                  python_bin=deployment["remote_root"] + "/.venv/bin/python")
        host, port, user, key = connection
        # One SCP session for the packet (one JSON archive plus uncompressed non-JSON files).
        # -O makes the explicitly shell-quoted remote pathname unambiguous across SCP versions.
        local_parent = Path(transfer["local"])
        local_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="download-", dir=local_parent) as temporary:
            subprocess.run(["scp", "-O", "-r", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
                            "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=3",
                            "-P", str(port), "-i", key,
                            f"{user}@{host}:" + shlex.quote(transfer["remote_download"]),
                            temporary], stdin=subprocess.DEVNULL, check=True)
            packet = Path(temporary) / PurePosixPath(transfer["remote_download"]).name
            if _sha(packet / "manifest.json") != packed["manifest_sha256"]:
                raise ValueError("Downloaded manifest SHA-256 mismatch; instance retained")
            verified = unpack(packet, transfer["local_destination"],
                              token=transfer["token"], direction="download")
            shutil.copyfile(packet / "manifest.json", local_parent / "download-manifest.json")
            return {**verified, "manifest_sha256": packed["manifest_sha256"]}


def _job_result(directory, deployment):
    path = directory / deployment["job"]["result"]
    return _read(path) if path.exists() else {}


def _attach(directory, deployment, transport):
    transport.validate_attach(deployment)
    code = transport.invoke(
        deployment["instance"],
        [
            "--attach",
            "--sync-max-entries",
            str(SYNC_MAX_ENTRIES),
            "--sync-max-mib",
            str(SYNC_MAX_MIB),
            "--sync-max-destination-entries",
            str(SYNC_MAX_ENTRIES),
        ],
    )
    result = _job_result(directory, deployment)
    if result.get("job_id") != deployment["job"].get("id"):
        raise ValueError("Attached job identity differs")
    _receipt(
        deployment,
        "attach",
        returncode=code,
        status=result.get("status"),
        phase=result.get("lifecycle", {}).get("phase"),
    )
    _write(directory / "deployment.json", deployment)
    return result


def launch(directory, mode, *, instance=None, transport=None, evaluation_runtime=None, host_resume_reason=None):
    directory = Path(directory).resolve()
    transport = transport or Transport()
    with RunLock(directory):
        deployment = _read(directory / "deployment.json")
        if deployment.get("instance_deleted"):
            raise ValueError("The bound instance has been deleted")
        if mode not in CAPABILITIES.get(deployment["kind"], set()):
            raise ValueError(f"Unsupported {mode} for {deployment['kind']}")
        if evaluation_runtime and mode == "tune":
            raise ValueError("Evaluation runtime options do not apply to collection tuning")
        if evaluation_runtime and deployment["kind"] not in {
            "rl01.ppo_after_target_bc", "rl01.ppo_target_scale", "rl01.cp_imitation_target_scale",
            "rl01.cp_imitation_target_more_data", "rl01.cp_imitation_target_c2",
        }:
            raise ValueError("This runner does not support evaluation runtime options")
        target = _instance(instance) if instance else deployment.get("instance")
        if target is None:
            raise ValueError("Specify --instance at package time or first launch")
        if deployment.get("instance") not in (None, target):
            raise ValueError("Cannot change the instance of a bound deployment")
        deployment["instance"] = target
        verify_package(directory, deployment)
        if deployment.get("job"):
            previous = _job_result(directory, deployment)
            phase = previous.get("lifecycle", {}).get("phase")
            cancelled_before_launch = (
                phase == "cancelled_before_launch"
                and previous.get("job_id") == deployment["job"].get("id")
                and previous.get("execution", {}).get("status") == "not_started"
                and previous.get("pre_sync", {}).get("status") == "not_started"
            )
            if phase == "prelaunch_interrupted" or cancelled_before_launch:
                _receipt(deployment, "prelaunch_retry", job_id=deployment["job"].get("id"),
                         previous_status=previous.get("status"))
                deployment["job"] = None
            elif phase != "finalized":
                if evaluation_runtime and evaluation_runtime != deployment.get("evaluation_runtime", {}):
                    raise ValueError("Cannot change evaluation runtime while attaching to an active job")
                return _attach(directory, deployment, transport)
            elif not deployment["job"].get("pulled"):
                raise ValueError(
                    "Pull and verify the previous job before launching another operation"
                )
            elif mode == "run" and deployment.get("run_started"):
                raise ValueError("Run already exists; use supported resume")
        if mode == "tune" and not deployment.get("tuning_rollout"):
            raise ValueError(
                "Tune requires exactly one explicit case-carrier .pt via --extra-input"
            )
        local_output_path = Path(deployment["local_output"])
        local_has_progress = any(
            (local_output_path / name).exists()
            for name in ("state.json", "collection.json", "run.json")
        )
        if mode == "resume" and not (deployment.get("run_started") or local_has_progress):
            raise ValueError("Resume requires an existing remote or local run")
        if (
            mode in ("run", "resume")
            and deployment["kind"] == "rl01.ppo_after_target_bc"
        ):
            tune = deployment.get("tune", {})
            if (
                not tune.get("confirmed")
                or tune.get("fingerprint") != deployment["fingerprint"]
            ):
                raise ValueError(
                    "Confirmed comparable tuning of this exact package/config is required"
                )

        if host_resume_reason and mode != "resume":
            raise ValueError("--host-resume-reason applies only to resume")
        sequence = len(deployment["operations"])
        job = {
            "mode": mode,
            "result": f"job-{sequence}.json",
            "id": None,
            "pulled": False,
            "operation": sequence,
        }
        deployment["job"] = job

        if mode == "tune":
            local_sync_dir = directory / "tuning"
            local_sync_dir.mkdir(parents=True, exist_ok=True)
            remote_sync_dir = deployment["remote_root"] + "/tuning"
            worker_args = [
                "--mode",
                "tune",
                "--config",
                deployment["config"],
                "--tuning-rollout",
                deployment["tuning_rollout"],
                "--tuning-dir",
                remote_sync_dir,
            ]
        else:
            local_sync_dir = Path(deployment["local_output"])
            local_sync_dir.mkdir(parents=True, exist_ok=True)
            remote_sync_dir = deployment["remote_project"] + "/" + deployment["output"]
            worker_args = [
                "--mode",
                mode,
                "--config",
                deployment["config"],
            ]
            if deployment["kind"] == "rl01.ppo_after_target_bc":
                worker_args += [
                    "--runtime-config",
                    deployment["remote_root"] + "/tuning/recommended-runtime.json",
                ]
            if host_resume_reason:
                worker_args += ["--host-resume-id", str(deployment["instance"]["instance_id"]),
                                "--host-resume-reason", host_resume_reason]
            for key, value in (evaluation_runtime or {}).items():
                if key not in {"device", "workers", "max_batch", "wait_seconds"}:
                    raise ValueError(f"Unknown evaluation runtime key {key!r}")
                worker_args += ["--evaluation-" + key.replace("_", "-"), str(value)]
            deployment["evaluation_runtime"] = evaluation_runtime or {}

        resolved_cfg = deployment["resolved_config"]
        if deployment.get("archive_transport") == 1:
            from civ5rl.deploy.archive import pack as pack_transfer

            scientific_destination = local_sync_dir
            remote_destination = remote_sync_dir
            token = uuid4().hex
            transfer_dir = directory / "transfers" / f"job-{sequence}"
            local_sync_dir = transfer_dir / "upload"
            remote_sync_dir = deployment["remote_root"] + f"/transport/job-{sequence}/upload"
            from civ5rl.engine.control import is_active

            if is_active(scientific_destination):
                raise ValueError("Cannot transfer an actively written local training run")
            # Windows byte-range locks prevent even reading run.lock while held.
            # pack independently checks that the entire source stayed unchanged.
            pack_transfer(scientific_destination, local_sync_dir, token=token, direction="upload")
            job["transfer"] = dict(local=str(transfer_dir), remote_upload=remote_sync_dir,
                                   remote_download=remote_sync_dir.rsplit("/", 1)[0] + "/download",
                                   destination=remote_destination, local_destination=str(scientific_destination), token=token)
            worker_args += ["--transfer-dir", remote_sync_dir, "--transfer-token", token,
                            "--transfer-destination", remote_destination]
            for item in deployment.get("input_packets", []):
                worker_args += ["--input-transfer", json.dumps(dict(
                    packet=deployment["remote_root"] + "/" + item["packet"],
                    destination=deployment["remote_project"] + "/" + item["destination"], token=item["token"]))]
        budget_seconds = int(
            resolved_cfg.get("experiment", {}).get("active_seconds", 0)
            or (
                int(resolved_cfg.get("collection_seconds", 0))
                + int(resolved_cfg.get("training_seconds", 0))
            )
            or 7200
        )
        args = [
            "--prepare",
            str(directory / "prepare.json"),
            "--venv",
            "auto",
            "--local-sync-dir",
            str(local_sync_dir),
            "--remote-sync-dir",
            remote_sync_dir,
            "--remote-dir",
            deployment["remote_root"],
            "--sync-max-entries",
            str(SYNC_MAX_ENTRIES),
            "--sync-max-mib",
            str(SYNC_MAX_MIB),
            "--sync-max-destination-entries",
            str(SYNC_MAX_ENTRIES),
            "--timeout-sec",
            str(max(86400, budget_seconds * 10)),
            "--env",
            f"CIV5RL_SCRATCH_DIR={deployment['remote_scratch']}",
            "--env",
            f"MPLCONFIGDIR={deployment['remote_scratch']}/matplotlib",
            "--remote-script",
            "civ5rl/deploy/worker.py",
            *[f"--script-arg={item}" for item in worker_args],
            "--local-output-file",
            str(directory / f"job-{sequence}.log"),
            "--result-json",
            str(directory / job["result"]),
            "--detach",
        ]
        _receipt(deployment, "launch_intent", mode=mode)
        _write(directory / "deployment.json", deployment)
        code = transport.invoke(target, args)
        result = _job_result(directory, deployment)
        job["id"] = result.get("job_id")
        _receipt(deployment, "launch", mode=mode, returncode=code, job_id=job["id"])
        _write(directory / "deployment.json", deployment)
        if code or not job["id"]:
            raise RuntimeError(
                "Durable launch not confirmed; inspect/attach the saved operation, do not restart"
            )
        return {"job_id": job["id"], "mode": mode, "status": "launched"}


def status(directory):
    directory = Path(directory).resolve()
    deployment = _read(directory / "deployment.json")
    if deployment.get("instance_deleted"):
        return {"status": "instance_deleted", "job": deployment.get("job")}
    if not deployment.get("instance"):
        return {"status": "not_bound", "job": deployment.get("job")}
    return {
        "job": deployment.get("job"),
        "remote": inspect_remote(
            deployment["instance"],
            deployment["remote_project"] + "/" + deployment["output"],
        ),
    }


def extend_remote_budget(directory, additional_seconds, reason, *, request_id=None):
    from uuid import uuid4
    from civ5rl.deploy.remote import _connection, _remote_path, _ssh_json

    deployment = _read(Path(directory) / "deployment.json")
    if deployment.get("instance_deleted") or not deployment.get("instance"):
        raise ValueError("Deployment has no live instance")
    request_id = request_id or uuid4().hex
    # Print the ID before SSH so an interrupted caller can retry exactly once.
    print(f"Budget request ID: {request_id}", file=sys.stderr, flush=True)
    code = """
import json, sys
from pathlib import Path
p=json.load(sys.stdin)
sys.path.insert(0,p['root'])
from civ5rl.reporting.operations import extend_budget
result=extend_budget(Path(p['root'])/p['output'],p['seconds'],p['reason'],request_id=p['request_id'])
print(json.dumps(result,allow_nan=False))
"""
    return _ssh_json(_connection(deployment["instance"]), code, dict(
        root=_remote_path(deployment["remote_project"]), output=deployment["output"],
        seconds=additional_seconds, reason=reason, request_id=request_id))


def pull(directory, *, delete_instance=False, transport=None):
    directory = Path(directory).resolve()
    transport = transport or Transport()
    with RunLock(directory):
        deployment = _read(directory / "deployment.json")
        if not deployment.get("job"):
            raise ValueError("No linked durable job")
        result = _job_result(directory, deployment)
        if result.get("lifecycle", {}).get("phase") != "finalized":
            result = _attach(directory, deployment, transport)
        post_sync = result.get("post_sync", {})
        execution = result.get("execution", {})
        if (
            post_sync.get("status") != "ok"
            or post_sync.get("file_errors")
            or execution.get("status") != "exited"
        ):
            raise ValueError("Execution/transport not finalized; instance retained")
        if result.get("job_id") != deployment["job"]["id"]:
            raise ValueError("Final transport metadata belongs to a different job")

        transferred = None
        if deployment["job"].get("transfer"):
            # The durable job mirrors only its upload packet. Fetch the authoritative output
            # after exit, including after hard failures, before auditing or enabling deletion.
            transferred = deployment["job"].get("transfer_verified")
            if transferred is None:
                transferred = transport.download_packet(deployment, deployment["job"]["transfer"])
                deployment["job"]["transfer_verified"] = transferred

        exit_code = execution.get("exit_code")
        mode = deployment["job"]["mode"]
        if mode == "tune":
            destination = directory / "tuning"
            summary = (
                _read(destination / "summary.json")
                if (destination / "summary.json").exists()
                else {}
            )
            confirmed = (
                exit_code == 0
                and summary.get("preliminary") is False
                and summary.get("protected_inputs_unchanged") is True
            )
            deployment["tune"] = {
                "confirmed": confirmed,
                "fingerprint": deployment["fingerprint"],
            }
            deployment["runtime"] = (
                _read(destination / "recommended-runtime.json")
                if confirmed and (destination / "recommended-runtime.json").exists()
                else None
            )
            audit = None
        else:
            from civ5rl.reporting.operations import audit_run

            destination = Path(deployment["local_output"])
            audit = audit_run(
                destination, directory / "audits" / deployment["job"]["id"]
            )
            if audit["status"] == "passed" and _read(destination / "status.json").get("status") in ("complete", "completed", "finished", "closed"):
                from civ5rl.reporting.evaluation_storage import archive_evaluations
                archive_evaluations(destination)
            deployment["run_started"] = any(
                (destination / name).is_file()
                for name in ("state.json", "collection.json", "run.json")
            )

        deployment["job"]["pulled"] = True
        receipt = {
            "status": "verified",
            "destination": str(destination),
            "copied_files": post_sync.get("copied_files", 0),
            "verified_files": post_sync.get("verified_files", 0),
            "instance_deleted": bool(deployment.get("instance_deleted")),
            "runner_exit_code": exit_code,
        }
        if transferred is not None:
            receipt.update(transferred)
            receipt["copied_files"] = None  # SCP packet counts differ from scientific file counts.
        if audit is not None:
            receipt["scientific_audit"] = audit["status"]
        _receipt(deployment, "pull", **receipt)
        _write(directory / "deployment.json", deployment)

        if delete_instance:
            if deployment.get("instance_deleted"):
                return {**receipt, "instance_deleted": True}
            if mode == "tune":
                mandatory_ok = deployment["tune"]["confirmed"]
            else:
                mandatory_ok = (
                    exit_code == 0
                    and all(
                        audit["sections"].get(k, {}).get("status") == "passed"
                        for k in ("passport", "ppo_ledger", "checkpoints_adam_rng")
                    )
                    and not any(
                        section["status"] == "failed"
                        for section in audit["sections"].values()
                    )
                )
            if not mandatory_ok:
                raise ValueError(
                    "Downloaded artifacts saved, but mandatory checks failed; instance retained"
                )
            transport.delete(deployment["instance"])
            receipt["instance_deleted"] = True
            deployment["instance_deleted"] = True
            _receipt(
                deployment,
                "delete_instance",
                instance_id=deployment["instance"]["instance_id"],
            )
            _write(directory / "deployment.json", deployment)
        return receipt
