"""Prevent duplicate scientific jobs and false success on incomplete artifacts."""

import contextlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from civ5rl.deploy import operations as remote
from civ5rl.deploy import worker
from civ5rl.deploy.remote import _sha, _validate_zip
from civ5rl.engine.control import RunLock
from civ5rl.reporting.operations import audit_run
from tools.ctl_rl import main as ctl
from tools.remote_rl import main as remote_cli


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class FakeTransport:
    def __init__(self, directory):
        self.directory = directory
        self.starts = 0
        self.attaches = 0
        self.deletes = 0

    def validate_attach(self, deployment):
        if deployment["job"]["id"] != "fixture-job":
            raise ValueError("unrelated job")

    def invoke(self, instance, args):
        if "--attach" in args:
            self.attaches += 1
        else:
            self.starts += 1
            result = args[args.index("--result-json") + 1]
            write(
                Path(result),
                {
                    "job_id": "fixture-job",
                    "execution": {"status": "running"},
                    "lifecycle": {"phase": "running"},
                },
            )
        return 0

    def delete(self, instance):
        self.deletes += 1


class ManagementTests(unittest.TestCase):
    def test_resume_worker_preserves_saved_boundary_and_archives_runner_error(self):
        # A source overlay or fresh initialization on resume could silently reset
        # model/Adam/RNG/budget; test the real worker with only its runner replaced.
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)
            payload = base / "payload"
            payload.mkdir()
            project = base / "project"
            run = project / "artifacts/rl/fixed"
            run.mkdir(parents=True)
            frozen = {
                "state.json": b'{"decisions":123,"active_seconds":42}',
                "checkpoints/latest.pt": b"saved-model-adam-rng",
            }
            for name, value in frozen.items():
                (run / name).parent.mkdir(parents=True, exist_ok=True)
                (run / name).write_bytes(value)
            write(run / "run.json", {"config": {"kind": "rl01.ppo_target_scale"}})
            with zipfile.ZipFile(payload / "source.zip", "w") as archive:
                archive.writestr("frozen.txt", "unchanged")
            (project / "frozen.txt").write_text("unchanged")
            transfer = {
                "archives": {
                    "source.zip": {"members": _validate_zip(payload / "source.zip")}
                }
            }
            write(payload / "transfer.json", transfer)
            write(base / "prepared.json", {"fingerprint": "fixture"})
            write(base / "run-started.json", {"fingerprint": "fixture"})
            request = {
                "remote_project": str(project),
                "output": "artifacts/rl/fixed",
                "mode": "resume",
                "kind": "rl01.ppo_target_scale",
                "config": "configs/fixed.json",
                "fingerprint": "fixture",
                "operation": 1,
                "fingerprints": {"source.zip": _sha(payload / "source.zip")},
            }
            write(payload / "request.json", request)
            with (
                contextlib.chdir(base),
                patch.object(worker.sys, "path", worker.sys.path.copy()),
                patch.object(
                    worker.subprocess, "run", return_value=Mock(returncode=7)
                ) as execute,
            ):
                self.assertEqual(worker.main(), 7)
            command = execute.call_args.args[0]
            self.assertEqual(command[1:3], ["tools/run_ppo.py", "resume"])
            self.assertEqual(execute.call_count, 1)
            for name, value in frozen.items():
                self.assertEqual((run / name).read_bytes(), value)
            descriptor = json.loads((payload / "result.json").read_text())
            self.assertEqual(descriptor["exit_code"], 7)
            self.assertEqual(descriptor["operation"], 1)
            self.assertEqual(descriptor["sha256"], _sha(payload / "result.zip"))
            with zipfile.ZipFile(payload / "result.zip") as archive:
                self.assertEqual(
                    archive.read("run/checkpoints/latest.pt"),
                    frozen["checkpoints/latest.pt"],
                )

    def deployment(self, root, kind="rl01.ppo_after_target_bc"):
        payload = root / "payload"
        payload.mkdir()
        write(payload / "transfer.json", {"archives": {}})
        record = {
            "kind": kind,
            "config": "configs/fixed.json",
            "output": "artifacts/rl/fixed",
            "resolved_config": {},
            "remote_root": "/workspace/civ5-runs/fixture",
            "remote_project": "/workspace/civ5-runs/fixture/project",
            "instance": {"instance_id": 123, "host": "example.invalid", "port": 22},
            "fingerprints": {
                "transfer.json": _sha(payload / "transfer.json"),
                "source.zip": "frozen",
            },
            "fingerprint": "fixture",
            "commit": "fixture",
            "runtime": {"num_envs": 16},
            "tune": {"confirmed": True, "fingerprint": "fixture"},
            "tuning_rollout": "cases.pt",
            "operations": [],
            "job": None,
        }
        # The test exercises orchestration, not archive generation (covered separately).
        write(root / "deployment.json", record)
        return FakeTransport(root)

    def test_repeat_launch_attaches_and_resume_uses_frozen_payload(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(remote, "verify_package"),
        ):
            root = Path(temp)
            transport = self.deployment(root)
            remote.launch(root, "run", transport=transport)
            remote.launch(root, "run", transport=transport)
            self.assertEqual((transport.starts, transport.attaches), (1, 1))
            saved = json.loads((root / "deployment.json").read_text())
            job = saved["job"]
            write(
                root / job["result"],
                {"job_id": job["id"], "lifecycle": {"phase": "finalized"}},
            )
            saved["run_started"] = True
            saved["job"]["pulled"] = True
            write(root / "deployment.json", saved)
            remote.launch(root, "resume", transport=transport)
            request = json.loads((root / "payload/request.json").read_text())
            self.assertEqual(request["mode"], "resume")
            self.assertEqual(request["runtime"], {"num_envs": 16})
            self.assertEqual(transport.starts, 2)

    def test_unsupported_and_failed_tuning_never_start_training(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(remote, "verify_package"),
        ):
            root = Path(temp)
            transport = self.deployment(root, "rl01.ppo_target_scale")
            with self.assertRaises(ValueError):
                remote.launch(root, "tune", transport=transport)
            saved = json.loads((root / "deployment.json").read_text())
            saved["kind"] = "rl01.ppo_after_target_bc"
            saved["tune"]["confirmed"] = False
            write(root / "deployment.json", saved)
            with self.assertRaises(ValueError):
                remote.launch(root, "run", transport=transport)
            self.assertEqual(transport.starts, 0)

    def test_pull_requires_final_transfer_hashes_and_checks_before_delete(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(remote, "verify_package"),
        ):
            root = Path(temp)
            transport = self.deployment(root)
            remote.launch(root, "tune", transport=transport)
            with self.assertRaises(ValueError):
                remote.pull(root, delete_instance=True, transport=transport)
            saved = json.loads((root / "deployment.json").read_text())
            write(
                root / saved["job"]["result"],
                {
                    "job_id": "fixture-job",
                    "execution": {"status": "exited", "exit_code": 1},
                    "lifecycle": {"phase": "finalized"},
                    "post_sync": {"status": "failed"},
                },
            )
            with self.assertRaises(ValueError):
                remote.pull(root, delete_instance=True, transport=transport)
            result = json.loads((root / saved["job"]["result"]).read_text())
            result["post_sync"]["status"] = "ok"
            write(root / saved["job"]["result"], result)
            archive = root / "payload/result.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr(
                    "tuning/summary.json",
                    json.dumps(
                        {"preliminary": True, "protected_inputs_unchanged": True}
                    ),
                )
            descriptor = {
                "fingerprint": "fixture",
                "mode": "tune",
                "operation": saved["job"]["operation"],
                "prepared": True,
                "exit_code": 1,
                "bytes": archive.stat().st_size,
                "sha256": "wrong",
                "members": _validate_zip(archive),
            }
            write(root / "payload/result.json", descriptor)
            with self.assertRaises(ValueError):
                remote.pull(root, delete_instance=True, transport=transport)
            descriptor["sha256"] = _sha(archive)
            write(root / "payload/result.json", descriptor)
            with self.assertRaises(ValueError):
                remote.pull(root, delete_instance=True, transport=transport)
            self.assertTrue(
                (root / "results/fixture-job/tuning/summary.json").is_file()
            )
            self.assertEqual(transport.deletes, 0)

    def test_active_and_historical_audits_never_claim_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"
            root.mkdir()
            (root / "run.lock").write_bytes(b"0")
            write(root / "run.json", {"kind": "unsupported-historical"})
            with RunLock(root):
                result = audit_run(root, Path(temp) / "active-audit")
            self.assertEqual(result["status"], "incomplete")
            result = audit_run(root, Path(temp) / "old-audit")
            self.assertTrue(
                all(v["status"] == "unavailable" for v in result["sections"].values())
            )

    def test_verified_pull_keeps_instance_until_explicit_delete(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(remote, "verify_package"),
        ):
            root = Path(temp)
            transport = self.deployment(root)
            remote.launch(root, "tune", transport=transport)
            saved = json.loads((root / "deployment.json").read_text())
            write(
                root / saved["job"]["result"],
                {
                    "job_id": "fixture-job",
                    "execution": {"status": "exited", "exit_code": 0},
                    "lifecycle": {"phase": "finalized"},
                    "post_sync": {"status": "ok"},
                },
            )
            archive = root / "payload/result.zip"
            with zipfile.ZipFile(archive, "w") as stream:
                stream.writestr(
                    "tuning/summary.json",
                    json.dumps(
                        {"preliminary": False, "protected_inputs_unchanged": True}
                    ),
                )
                stream.writestr("tuning/recommended-runtime.json", '{"num_envs":16}')
            write(
                root / "payload/result.json",
                {
                    "fingerprint": "fixture",
                    "mode": "tune",
                    "operation": saved["job"]["operation"],
                    "prepared": True,
                    "exit_code": 0,
                    "bytes": archive.stat().st_size,
                    "sha256": _sha(archive),
                    "members": _validate_zip(archive),
                },
            )
            remote.pull(root, transport=transport)
            self.assertEqual(transport.deletes, 0)
            remote.pull(root, delete_instance=True, transport=transport)
            self.assertEqual(transport.deletes, 1)
            remote.pull(root, delete_instance=True, transport=transport)
            self.assertEqual(transport.deletes, 1)

    def test_cli_routing_keeps_bootstrap_settings_and_explicit_delete(self):
        with (
            tempfile.TemporaryDirectory() as temp,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            with patch("tools.ctl_rl.compare", return_value={}) as compare:
                ctl(
                    [
                        "compare",
                        "--before",
                        "before",
                        "--after",
                        "after",
                        "--expected-n",
                        "128",
                        "--out",
                        temp,
                        "--seed",
                        "42",
                        "--samples",
                        "100",
                    ]
                )
                self.assertEqual(compare.call_args.kwargs["seed"], 42)
                self.assertEqual(compare.call_args.kwargs["samples"], 100)
            with patch("tools.remote_rl.pull", return_value={}) as pull:
                remote_cli(["pull", "--deployment", temp])
                self.assertFalse(pull.call_args.kwargs["delete_instance"])
                remote_cli(["pull", "--deployment", temp, "--delete-instance"])
                self.assertTrue(pull.call_args.kwargs["delete_instance"])


if __name__ == "__main__":
    unittest.main()
