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
from civ5rl.engine.runtime.control import RunLock
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
        self.runtime = {"num_envs": 16}
        self.attach_result = None

    def validate_attach(self, deployment):
        if deployment["job"]["id"] != "fixture-job":
            raise ValueError("unrelated job")

    def invoke(self, instance, args):
        self.last_args = args
        if "--attach" in args:
            self.attaches += 1
            if self.attach_result is not None:
                record = json.loads((self.directory / "deployment.json").read_text())
                write(self.directory / record["job"]["result"], self.attach_result)
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

    def tuning_runtime(self, deployment, transfer=None):
        return self.runtime


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
            transport.runtime = None
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

    def test_tuning_confirms_remote_runtime_without_transferring_probe_files(self):
        with tempfile.TemporaryDirectory() as temp:
            deploy, transport = self.deployment(Path(temp))
            record = deploy / "deployment.json"
            saved = json.loads(record.read_text())
            write(record, {**saved, "archive_transport": 1})
            write(deploy / "tuning/old-probe/result.json", {"obsolete": True})
            remote.launch(deploy, "tune", transport=transport)
            job = json.loads(record.read_text())["job"]
            packet = Path(job["transfer"]["local"]) / "upload/manifest.json"
            self.assertEqual(json.loads(packet.read_text())["files"], {})
            write(deploy / job["result"], {
                "job_id": job["id"],
                "execution": {"status": "exited", "exit_code": 0},
                "lifecycle": {"phase": "finalized"},
                "post_sync": {"status": "ok", "file_errors": []},
            })
            runtime = {"num_envs": 16, "cpu_workers": 8, "worker_reply_size": 1}
            transport.tuning_runtime = Mock(return_value=runtime)
            transport.download_packet = Mock(side_effect=AssertionError("tuning must stay on the instance"))
            receipt = remote.pull(deploy, transport=transport)
            self.assertTrue(receipt["tuning_confirmed"])
            self.assertEqual(receipt["runtime"], runtime)
            self.assertEqual(receipt["destination"], job["transfer"]["destination"] + "/measurement-0")
            self.assertFalse((deploy / "tuning/recommended-runtime.json").exists())
            transport.tuning_runtime.assert_called_once()
            after = json.loads(record.read_text())
            self.assertFalse(after["job"]["pulled"])
            self.assertNotIn("tune", after)
            self.assertNotIn("runtime", after)
            remote.launch(deploy, "run", transport=transport)
            self.assertIn("--script-arg=/workspace/civ5-runs/fixture/tuning/measurement-0/recommended-runtime.json",
                          transport.last_args)

    def test_run_waits_for_tuning_without_a_separate_pull(self):
        with tempfile.TemporaryDirectory() as temp:
            deploy, transport = self.deployment(Path(temp))
            remote.launch(deploy, "tune", transport=transport)
            transport.attach_result = {
                "job_id": "fixture-job",
                "execution": {"status": "exited", "exit_code": 0},
                "lifecycle": {"phase": "finalized"},
                "post_sync": {"status": "ok", "file_errors": []},
            }
            remote.launch(deploy, "run", transport=transport)
            self.assertEqual((transport.starts, transport.attaches), (2, 1))

    def test_finished_pull_preserves_the_server_archive_and_audit(self):
        from civ5rl.deploy.final_run import publish_final_run
        from civ5rl.engine.runtime.archive import seven_zip

        try:
            seven_zip()
        except RuntimeError as exc:
            self.skipTest(str(exc))

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            deploy, transport = self.deployment(root)
            record = deploy / "deployment.json"
            saved = json.loads(record.read_text())
            write(record, {**saved, "archive_transport": 1, "final_run_bundle": 1})
            remote.launch(deploy, "run", transport=transport)
            self.assertIn("--script-arg=--final-run-bundle", transport.last_args)
            saved = json.loads(record.read_text())
            job = saved["job"]
            transfer = job["transfer"]
            mirror = Path(transfer["local"]) / "upload"
            server = root / "server-run"
            write(server / "status.json", {"status": "complete"})
            write(server / "state.json", {"decisions": 999})
            write(server / "evaluation/set/case-a/replay.json", {"events": [1]})
            write(server / "evaluation/set/case-b/replay.json", {"events": [2]})
            checkpoint = server / "checkpoints/final.pt"
            checkpoint.parent.mkdir()
            checkpoint.write_bytes(b"model-adam-rng")
            audit = {"status": "passed", "sections": {
                k: {"status": "passed"} for k in ("passport", "ppo_ledger", "checkpoints_adam_rng")
            }}

            def server_audit(_run, destination):
                write(destination / "audit.json", audit)
                (destination / "audit.md").write_text("Saved-run audit: passed", encoding="utf-8")
                return audit

            destination = Path(saved["local_output"])
            write(destination / "evaluation/old.json", {"stale": True})
            result = {
                "job_id": job["id"],
                "execution": {"status": "exited", "exit_code": 0},
                "lifecycle": {"phase": "finalized"},
                "post_sync": {
                    "status": "failed", "file_errors": [], "verified_files": 30, "unchanged_files": 0,
                    "local_dir": str(mirror), "remote_sync_dir": transfer["remote_upload"],
                },
            }
            with patch("civ5rl.reporting.operations.audit_run", side_effect=server_audit) as run_audit:
                publish_final_run(server, mirror, transfer["token"])
                archive = (mirror / "final/run/evaluation.storage/json.7z").read_bytes()
                write(deploy / job["result"], result)
                with self.assertRaises(ValueError):
                    remote.pull(deploy, delete_instance=True, transport=transport)
                self.assertEqual(transport.deletes, 0)
                self.assertTrue((destination / "evaluation/old.json").is_file())
                result["post_sync"]["status"] = "ok"
                write(deploy / job["result"], result)
                receipt = remote.pull(deploy, transport=transport)
                self.assertEqual(receipt["scientific_audit"], "passed")
                self.assertFalse((destination / "evaluation").exists())
                self.assertFalse((mirror / "final").exists())
                self.assertEqual((destination / "evaluation.storage/json.7z").read_bytes(), archive)
                self.assertEqual((destination / "checkpoints/final.pt").read_bytes(), checkpoint.read_bytes())
                remote.pull(deploy, delete_instance=True, transport=transport)
                self.assertEqual(transport.deletes, 1)
                run_audit.assert_called_once()

            match = root / "match"
            with contextlib.redirect_stdout(io.StringIO()):
                ctl(["restore-evaluations", "--run", str(destination), "--match", "set/case-a", "--out", str(match)])
            self.assertEqual({p.name for p in match.iterdir()}, {"replay.json"})
            self.assertEqual(json.loads((match / "replay.json").read_text()), {"events": [1]})
            self.assertFalse((destination / "evaluation").exists())

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
            with patch("civ5rl.deploy.operations.pull", return_value={}) as pull:
                remote_cli(["pull", "--deployment", temp])
                self.assertFalse(pull.call_args.kwargs["delete_instance"])
                remote_cli(["pull", "--deployment", temp, "--delete-instance"])
                self.assertTrue(pull.call_args.kwargs["delete_instance"])


if __name__ == "__main__":
    unittest.main()
