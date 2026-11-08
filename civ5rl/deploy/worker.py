"""Frozen remote entrypoint. The SSH skill owns durability and process lifecycle."""

import json
import subprocess
import sys
import traceback
import zipfile
from pathlib import Path


def main():
    base = Path.cwd()
    payload = base / "payload"
    request = json.loads((payload / "request.json").read_text())
    sys.path.insert(0, str(payload / "source.zip"))
    from civ5rl.deploy.remote import _read, _sha, _validate_zip, _write, safe_extract

    project = Path(request["remote_project"])
    mode = request["mode"]
    prepared = False
    code = 1
    transfer = None
    try:
        for name, expected in request["fingerprints"].items():
            if _sha(payload / name) != expected:
                raise ValueError(f"Package differs: {name}")
        transfer = _read(payload / "transfer.json")
        for name, meta in transfer["archives"].items():
            _validate_zip(payload / name, meta["members"])
        marker = base / "prepared.json"
        if marker.exists():
            if _read(marker)["fingerprint"] != request["fingerprint"]:
                raise ValueError(
                    "Prepared project belongs to a different frozen package"
                )
            # Never overlay source files or scientific state on a resumed experiment.
            for meta in transfer["archives"].values():
                for name, expected in meta["members"].items():
                    if _sha(project / name) != expected:
                        raise ValueError(f"Prepared source/input differs: {name}")
        else:
            if mode == "resume":
                raise ValueError("Resume cannot bootstrap or migrate a run")
            for name, meta in transfer["archives"].items():
                safe_extract(payload / name, project, expected=meta["members"])
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    sys.executable,
                    "-r",
                    "requirements-rl.txt",
                ],
                cwd=project,
                check=True,
            )
            sys.path.insert(0, str(project))
            from civ5rl.deploy.remote import bootstrap_linux_node

            hardware = bootstrap_linux_node(payload, project)
            for args in (
                ["init", "-q"],
                ["fetch", str(payload / "base.bundle"), "HEAD"],
                ["reset", "--mixed", "FETCH_HEAD"],
            ):
                subprocess.run(["git", *args], cwd=project, check=True)
            _write(marker, {"fingerprint": request["fingerprint"], **hardware})
        prepared = True
        (payload / "worker-error.txt").unlink(missing_ok=True)
        output = project / request["output"]
        python = sys.executable
        if mode == "tune":
            command = [
                python,
                "tools/tune_rl_collection.py",
                "--config",
                request["config"],
                "--checkpoint",
                request["resolved_config"]["experiment"]["parent"],
                "--rollout",
                request["tuning_rollout"],
                "--out",
                str(base / "tuning"),
                "--budget-seconds",
                "600",
            ]
            if (base / "run-started.json").exists():
                raise ValueError("Cannot retune an existing scientific run")
        else:
            started = base / "run-started.json"
            if mode == "run":
                if started.exists() or (output / "state.json").exists():
                    raise ValueError("Run already exists; use resume")
            elif not started.exists() or not (output / "state.json").exists():
                raise ValueError(
                    "Resume requires the original remote state/model/Adam/RNG"
                )
            command = [
                python,
                "tools/run_ppo_scale_up.py",
                mode,
                "--config",
                request["config"],
            ]
            if request["kind"] == "rl01.ppo_after_target_bc":
                summary = _read(base / "tuning/summary.json")
                tune = _read(base / "tuning/identity.json")
                if (
                    summary.get("preliminary") is not False
                    or summary.get("protected_inputs_unchanged") is not True
                    or tune["fingerprint"] != request["fingerprint"]
                ):
                    raise ValueError(
                        "No confirmed comparable tuning for this frozen experiment"
                    )
                runtime = base / "tuning/recommended-runtime.json"
                if _read(runtime) != request["runtime"]:
                    raise ValueError("Runtime differs from confirmed tuning")
                command += ["--runtime-config", str(runtime)]
            if mode == "run":
                _write(started, {"fingerprint": request["fingerprint"]})
        code = subprocess.run(command, cwd=project, check=False).returncode
        if mode == "tune" and code == 0:
            _write(
                base / "tuning/identity.json", {"fingerprint": request["fingerprint"]}
            )
    except BaseException:
        code = 1
        (payload / "worker-error.txt").write_text(traceback.format_exc())
        raise
    finally:
        # A final archive is produced even for runner errors; killed workers have no
        # matching operation token and cannot be mistaken for a successful pull.
        archive = payload / "result.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as stream:
            if transfer and (project / request["output"] / "run.json").exists():
                from hashlib import sha256

                saved_run = _read(project / request["output"] / "run.json")
                config_hash = sha256(
                    json.dumps(saved_run["config"], sort_keys=True).encode()
                ).hexdigest()
                provenance = {
                    "schema": 1,
                    "scope": "frozen_deployment",
                    "config_hash": config_hash,
                    "fingerprint": request["fingerprint"],
                    "source_hashes": transfer["archives"]["source.zip"]["members"],
                }
                stream.writestr("run/deployment-passport.json", json.dumps(provenance))
                stream.write(payload / "source.zip", "run/deployment-sources.zip")
            for prefix, directory in (
                ("run", project / request["output"]),
                ("tuning", base / "tuning"),
            ):
                if directory.exists():
                    for path in sorted(directory.rglob("*")):
                        if path.is_file() and not path.is_symlink():
                            stream.write(
                                path,
                                prefix + "/" + path.relative_to(directory).as_posix(),
                            )
            if (payload / "worker-error.txt").exists():
                stream.write(payload / "worker-error.txt", "worker-error.txt")
            if (base / "prepared.json").exists():
                stream.write(base / "prepared.json", "machine.json")
        _write(
            payload / "result.json",
            {
                "fingerprint": request["fingerprint"],
                "mode": mode,
                "operation": request["operation"],
                "prepared": prepared,
                "exit_code": code,
                "bytes": archive.stat().st_size,
                "sha256": _sha(archive),
                "members": _validate_zip(archive),
            },
        )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
