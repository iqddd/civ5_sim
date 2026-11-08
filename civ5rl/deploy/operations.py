"""Declarative deployments using the installed SSH lifecycle and Vast adapters."""

import json
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
    _validate_zip,
    _write,
    inspect_remote,
    package_capsule,
    safe_extract,
)
from civ5rl.engine.control import RunLock

CAPABILITIES = {
    "rl01.ppo_after_target_bc": {"run", "tune", "resume"},
    "rl01.ppo_target_scale": {"run", "resume"},
    "rl01.ppo_scale_up": {"run"},
}


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
    # Deliberately whitelist non-secret fields; credentials are environment-only.
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


def package(config, output, *, instance=None, extra_inputs=(), root=ROOT):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Deployment output must be empty; frozen packages are immutable"
        )
    name = _relative_file(root, config)
    config, _ = _resolve_config(root, name)
    if config.get("kind") not in CAPABILITIES:
        raise ValueError("Unsupported remote experiment kind")
    scientific_output = _relative(config["output"])
    target = _instance(instance) if instance else None
    extras = [_relative_file(root, path) for path in extra_inputs]
    output.mkdir(parents=True, exist_ok=True)
    payload = output / "payload"
    package_capsule(name, payload, root=root, extra_inputs=extras)
    subprocess.run(
        ["git", "bundle", "create", str(payload / "base.bundle"), "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    transfer = _read(payload / "transfer.json")
    fingerprints = {
        n: _sha(payload / n)
        for n in ("source.zip", "inputs.zip", "transfer.json", "base.bundle")
    }
    import hashlib

    fingerprint = hashlib.sha256(
        json.dumps(fingerprints, sort_keys=True).encode()
    ).hexdigest()
    remote_root = f"/workspace/civ5-runs/{fingerprint[:20]}"
    carriers = [
        name
        for name in extras
        if name.endswith(".pt") and name != config["experiment"]["parent"]
    ]
    deployment = {
        "schema": 1,
        "kind": config["kind"],
        "config": name,
        "resolved_config": config,
        "fingerprints": fingerprints,
        "fingerprint": fingerprint,
        "commit": transfer["commit"],
        "output": scientific_output,
        "remote_root": remote_root,
        "remote_project": remote_root + "/project",
        "instance": target,
        "tuning_rollout": carriers[0] if len(carriers) == 1 else None,
        "job": None,
        "runtime": None,
        "operations": [],
    }
    _receipt(deployment, "package", status="complete")
    _write(output / "deployment.json", deployment)
    return deployment


def verify_package(directory, deployment):
    payload = directory / "payload"
    for name, digest in deployment["fingerprints"].items():
        if _sha(payload / name) != digest:
            raise ValueError(f"Frozen package changed: {name}")
    transfer = _read(payload / "transfer.json")
    for name, meta in transfer["archives"].items():
        if (payload / name).stat().st_size != meta["bytes"]:
            raise ValueError(f"Archive size differs: {name}")
        _validate_zip(payload / name, meta["members"])


class Transport:
    """Subprocess boundary, replaceable in offline checks; no supervisor of its own."""

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
        # Attach has no --job-id flag: ensure its current registry belongs to us.
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
    code = transport.invoke(deployment["instance"], ["--attach"])
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
        }
        deployment["job"] = job
        payload = directory / "payload"
        # Only execution metadata enters the bounded mirror. No Python source copies.
        request = {
            k: deployment[k]
            for k in (
                "kind",
                "config",
                "resolved_config",
                "output",
                "remote_root",
                "remote_project",
                "fingerprints",
                "fingerprint",
                "commit",
                "tuning_rollout",
            )
        }
        request.update(mode=mode, operation=sequence, runtime=deployment.get("runtime"))
        job["operation"] = sequence
        _write(payload / "request.json", request)
        _write(directory / "prepare.json", {"python": "3.13"})
        # Upload only the frozen worker bytes, even if local tooling was edited meanwhile.
        # The small bootstrap reads it from the verified package in memory.
        bootstrap = (
            "import hashlib,pathlib,zipfile; p=pathlib.Path('payload/source.zip'); "
            f"assert hashlib.sha256(p.read_bytes()).hexdigest()=={deployment['fingerprints']['source.zip']!r}; "
            "exec(compile(zipfile.ZipFile(p).read('civ5rl/deploy/worker.py'), '<frozen-rl-worker>', 'exec'))"
        )
        args = [
            "--local-sync-dir",
            str(payload),
            "--remote-sync-dir",
            deployment["remote_root"] + "/payload",
            "--remote-dir",
            deployment["remote_root"],
            "--venv",
            deployment["remote_root"] + "/.venv",
            "--sync-max-entries",
            "20000",
            "--sync-max-mib",
            "16384",
            "--timeout-sec",
            str(
                max(
                    86400,
                    int(
                        deployment["resolved_config"]
                        .get("experiment", {})
                        .get("active_seconds", 7200)
                    )
                    * 10,
                )
            ),
            "--remote-script=-c",
            "--script-arg",
            bootstrap,
            "--local-output-file",
            str(directory / f"job-{sequence}.log"),
            "--result-json",
            str(directory / job["result"]),
            "--detach",
        ]
        if not deployment.get("prepared"):
            args += ["--prepare", str(directory / "prepare.json")]
        _receipt(deployment, "launch_intent", mode=mode)
        _write(directory / "deployment.json", deployment)
        code = transport.invoke(target, args)
        result = _job_result(directory, deployment)
        job["id"] = result.get("job_id")
        _receipt(deployment, "launch", mode=mode, returncode=code, job_id=job["id"])
        # Keep uncertain intents. Never retry a start automatically after contact loss.
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
        if (
            result.get("post_sync", {}).get("status") != "ok"
            or result.get("execution", {}).get("status") != "exited"
        ):
            raise ValueError("Execution/transport not finalized; instance retained")
        if result.get("job_id") != deployment["job"]["id"]:
            raise ValueError("Final transport metadata belongs to a different job")
        payload = directory / "payload"
        descriptor = _read(payload / "result.json")
        if (
            descriptor["fingerprint"] != deployment["fingerprint"]
            or descriptor["mode"] != deployment["job"]["mode"]
            or descriptor["operation"] != deployment["job"]["operation"]
            or descriptor["exit_code"] != result.get("execution", {}).get("exit_code")
        ):
            raise ValueError("Result belongs to a different package/operation")
        archive = payload / "result.zip"
        if (
            archive.stat().st_size != descriptor["bytes"]
            or _sha(archive) != descriptor["sha256"]
        ):
            raise ValueError("Result size/SHA256 differs; instance retained")
        destination = directory / "results" / deployment["job"]["id"]
        safe_extract(archive, destination, expected=descriptor["members"])
        if deployment["job"]["mode"] == "tune":
            summary = (
                _read(destination / "tuning/summary.json")
                if (destination / "tuning/summary.json").exists()
                else {}
            )
            confirmed = (
                descriptor["exit_code"] == 0
                and summary.get("preliminary") is False
                and summary.get("protected_inputs_unchanged") is True
            )
            deployment["tune"] = {
                "confirmed": confirmed,
                "fingerprint": deployment["fingerprint"],
            }
            deployment["runtime"] = (
                _read(destination / "tuning/recommended-runtime.json")
                if confirmed
                else None
            )
        else:
            from civ5rl.reporting.operations import audit_run

            audit = audit_run(
                destination / "run", directory / "audits" / deployment["job"]["id"]
            )
            deployment["run_started"] = (destination / "run/state.json").is_file()
        deployment["prepared"] = descriptor["prepared"]
        deployment["job"]["pulled"] = True
        receipt = {
            "status": "verified",
            "destination": str(destination),
            "sha256": descriptor["sha256"],
            "files": len(descriptor["members"]),
            "instance_deleted": bool(deployment.get("instance_deleted")),
            "runner_exit_code": descriptor["exit_code"],
        }
        if deployment["job"]["mode"] != "tune":
            receipt["scientific_audit"] = audit["status"]
        _receipt(deployment, "pull", **receipt)
        _write(directory / "deployment.json", deployment)
        if delete_instance:
            if deployment.get("instance_deleted"):
                return {**receipt, "instance_deleted": True}
            if deployment["job"]["mode"] == "tune":
                mandatory_ok = deployment["tune"]["confirmed"]
            else:
                mandatory_ok = all(
                    audit["sections"].get(k, {}).get("status") == "passed"
                    for k in ("passport", "ppo_ledger", "checkpoints_adam_rng")
                ) and not any(
                    section["status"] == "failed"
                    for section in audit["sections"].values()
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
