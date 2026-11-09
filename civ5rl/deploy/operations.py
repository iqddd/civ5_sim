"""Declarative remote deployments using prepare.json and split input/output/scratch."""

import hashlib
import json
import re
import subprocess
import sys
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

    prepare_spec = {
        "execution_subdir": repo_name,
        "revision": fingerprint,
        "python": "3.13",
        "repositories": [{"repo": repo_spec, "commit": commit}],
        "requirements": [f"{repo_name}/requirements-rl.txt"],
        "sync": [
            {
                "local": str((inputs_dir / group).resolve()),
                "remote": f"{repo_name}/{group}",
            }
            for group in staged["groups"]
        ],
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
        return subprocess.run(command, check=False).returncode

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


def launch(directory, mode, *, instance=None, transport=None):
    directory = Path(directory).resolve()
    transport = transport or Transport()
    with RunLock(directory):
        deployment = _read(directory / "deployment.json")
        if deployment.get("instance_deleted"):
            raise ValueError("The bound instance has been deleted")
        if mode not in CAPABILITIES.get(deployment["kind"], set()):
            raise ValueError(f"Unsupported {mode} for {deployment['kind']}")
        target = _instance(instance) if instance else deployment.get("instance")
        if target is None:
            raise ValueError("Specify --instance at package time or first launch")
        if deployment.get("instance") not in (None, target):
            raise ValueError("Cannot change the instance of a bound deployment")
        deployment["instance"] = target
        verify_package(directory, deployment)
        if deployment.get("job"):
            previous = _job_result(directory, deployment)
            if previous.get("lifecycle", {}).get("phase") != "finalized":
                return _attach(directory, deployment, transport)
            if not deployment["job"].get("pulled"):
                raise ValueError(
                    "Pull and verify the previous job before launching another operation"
                )
            if mode == "run" and deployment.get("run_started"):
                raise ValueError("Run already exists; use supported resume")
        if mode == "tune" and not deployment.get("tuning_rollout"):
            raise ValueError(
                "Tune requires exactly one explicit case-carrier .pt via --extra-input"
            )
        if mode == "resume" and not deployment.get("run_started"):
            raise ValueError("Resume requires an existing remote run on this instance")
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

        resolved_cfg = deployment["resolved_config"]
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
