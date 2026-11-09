"""Prevent duplicate scientific jobs and false success on incomplete artifacts."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from civ5rl.deploy import operations as remote
from civ5rl.deploy import worker
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
        self.last_args = None

    def validate_attach(self, deployment):
        if deployment["job"]["id"] != "fixture-job":
            raise ValueError("unrelated job")

    def invoke(self, instance, args):
        self.last_args = args
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
    def test_resume_worker_preserves_saved_boundary_and_refuses_duplicate_run(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write(
                root / "configs/fixed.json",
                {"kind": "rl01.ppo_target_scale", "output": "artifacts/rl/fixed"},
            )
            run = root / "artifacts/rl/fixed"
            run.mkdir(parents=True)
            frozen = {
                "state.json": b'{"decisions":123,"active_seconds":42}',
                "checkpoints/latest.pt": b"saved-model-adam-rng",
            }
            for name, value in frozen.items():
                (run / name).parent.mkdir(parents=True, exist_ok=True)
                (run / name).write_bytes(value)

            with self.assertRaises(ValueError):
                worker.main(["--mode", "run", "--config", "configs/fixed.json"], root=root)

            with (
                patch.object(worker, "ensure_native_libraries"),
                patch.object(worker, "hardware", return_value={"cpu": "test"}),
                patch.object(
                    worker.subprocess, "run", return_value=Mock(returncode=7)
                ) as execute,
            ):
                code = worker.main(
                    ["--mode", "resume", "--config", "configs/fixed.json"], root=root
                )
            self.assertEqual(code, 7)
            command = execute.call_args.args[0]
            self.assertEqual(command[1:3], ["tools/run_ppo.py", "resume"])
            for name, value in frozen.items():
                self.assertEqual((run / name).read_bytes(), value)

    def deployment(self, root, kind="rl01.ppo_after_target_bc"):
        local_out = root / "local-run"
        deploy = root / "deploy"
        deploy.mkdir(parents=True, exist_ok=True)
        write(deploy / "prepare.json", {"execution_subdir": "civ5_sim"})
        write(
            deploy / "inputs-manifest.json",
            {"schema": 1, "output": "artifacts/rl/fixed", "groups": [], "staged_files": {}, "tracked_inputs": {}},
        )
        record = {
            "schema": 2,
            "kind": kind,
            "config": "configs/fixed.json",
            "output": "artifacts/rl/fixed",
            "local_output": str(local_out),
            "resolved_config": {},
            "remote_root": "/workspace/civ5-runs/fixture",
            "remote_project": "/workspace/civ5-runs/fixture/civ5_sim",
            "remote_scratch": "/workspace/civ5-runs/fixture/scratch",
            "instance": {"instance_id": 123, "host": "example.invalid", "port": 22},
            "prepare_sha256": remote._sha(deploy / "prepare.json"),
            "inputs_manifest_sha256": remote._sha(deploy / "inputs-manifest.json"),
            "fingerprint": "fixture",
            "commit": "fixture",
            "repo": "iqddd/civ5_sim",
            "repo_name": "civ5_sim",
            "runtime": {"num_envs": 16},
            "tune": {"confirmed": True, "fingerprint": "fixture"},
            "tuning_rollout": "cases.pt",
            "operations": [],
            "job": None,
        }
        write(deploy / "deployment.json", record)
        return deploy, FakeTransport(deploy)

    def test_repeat_launch_attaches_and_resume_uses_split_mirror_and_scratch(self):
        with tempfile.TemporaryDirectory() as temp:
            deploy, transport = self.deployment(Path(temp))
            remote.launch(deploy, "run", transport=transport)
            self.assertIn("--prepare", transport.last_args)
            self.assertIn(
                "--env",
                transport.last_args,
            )
            self.assertIn(
                "CIV5RL_SCRATCH_DIR=/workspace/civ5-runs/fixture/scratch",
                transport.last_args,
            )
            remote.launch(deploy, "run", transport=transport)
            self.assertEqual((transport.starts, transport.attaches), (1, 1))
            saved = json.loads((deploy / "deployment.json").read_text())
            job = saved["job"]
            write(
                deploy / job["result"],
                {"job_id": job["id"], "lifecycle": {"phase": "finalized"}},
            )
            saved["run_started"] = True
            saved["job"]["pulled"] = True
            write(deploy / "deployment.json", saved)
            remote.launch(deploy, "resume", transport=transport)
            self.assertIn("--script-arg=resume", transport.last_args)
            self.assertEqual(transport.starts, 2)

    def test_unsupported_and_failed_tuning_never_start_training(self):
        with tempfile.TemporaryDirectory() as temp:
            deploy, transport = self.deployment(Path(temp), "rl01.ppo_target_scale")
            with self.assertRaises(ValueError):
                remote.launch(deploy, "tune", transport=transport)
            saved = json.loads((deploy / "deployment.json").read_text())
            saved["kind"] = "rl01.ppo_after_target_bc"
            saved["tune"]["confirmed"] = False
            write(deploy / "deployment.json", saved)
            with self.assertRaises(ValueError):
                remote.launch(deploy, "run", transport=transport)
            self.assertEqual(transport.starts, 0)

    def test_pull_requires_verified_post_sync_and_checks_before_delete(self):
        with tempfile.TemporaryDirectory() as temp:
            deploy, transport = self.deployment(Path(temp))
            remote.launch(deploy, "tune", transport=transport)
            saved = json.loads((deploy / "deployment.json").read_text())
            write(
                deploy / saved["job"]["result"],
                {
                    "job_id": "fixture-job",
                    "execution": {"status": "exited", "exit_code": 1},
                    "lifecycle": {"phase": "finalized"},
                    "post_sync": {"status": "failed"},
                },
            )
            with self.assertRaises(ValueError):
                remote.pull(deploy, delete_instance=True, transport=transport)

            write(
                deploy / "tuning/summary.json",
                {"preliminary": True, "protected_inputs_unchanged": True},
            )
            write(
                deploy / saved["job"]["result"],
                {
                    "job_id": "fixture-job",
                    "execution": {"status": "exited", "exit_code": 1},
                    "lifecycle": {"phase": "finalized"},
                    "post_sync": {"status": "ok", "file_errors": []},
                },
            )
            with self.assertRaises(ValueError):
                remote.pull(deploy, delete_instance=True, transport=transport)
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
        with tempfile.TemporaryDirectory() as temp:
            deploy, transport = self.deployment(Path(temp))
            remote.launch(deploy, "tune", transport=transport)
            saved = json.loads((deploy / "deployment.json").read_text())
            write(
                deploy / "tuning/summary.json",
                {"preliminary": False, "protected_inputs_unchanged": True},
            )
            write(deploy / "tuning/recommended-runtime.json", {"num_envs": 16})
            write(
                deploy / saved["job"]["result"],
                {
                    "job_id": "fixture-job",
                    "execution": {"status": "exited", "exit_code": 0},
                    "lifecycle": {"phase": "finalized"},
                    "post_sync": {
                        "status": "ok",
                        "file_errors": [],
                        "copied_files": 2,
                        "verified_files": 2,
                    },
                },
            )
            remote.pull(deploy, transport=transport)
            self.assertEqual(transport.deletes, 0)
            remote.pull(deploy, delete_instance=True, transport=transport)
            self.assertEqual(transport.deletes, 1)
            remote.pull(deploy, delete_instance=True, transport=transport)
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
