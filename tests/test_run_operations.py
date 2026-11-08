import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from civ5rl.engine.control import (
    RunLock,
    close_run,
    pause_due,
    request_pause,
    target_bc_evaluation_due,
)
from civ5rl.reporting.snapshot import build_snapshot


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class OperationsTest(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        (root / "run.lock").write_bytes(b"0")
        (root / "checkpoints").mkdir()
        (root / "checkpoints/update-0001.pt").write_bytes(b"checkpoint")
        write(root / "run.json", {"kind": "rl01.ppo_after_target_bc",
                                  "run_id": "fixture", "shuffle_seed": 1,
                                  "config": {"experiment": {"milestones": [10]}}})
        write(root / "status.json", {"status": "paused"})
        state = {"phase": "collect", "rollouts": 1, "updates": 1, "decisions": 5,
                 "optimizer_steps": 1, "attempted_steps": 1, "rejected_steps": 0,
                 "latest": "checkpoints/update-0001.pt", "stop_reason": None,
                 "development_points": {},
                 "evaluated": {"baseline": {"checkpoint": "checkpoints/initial.pt"}}}
        write(root / "state.json", state)
        (root / "updates.jsonl").write_text(json.dumps({"rollout": 1, "total_decisions": 5,
            "W": 1, "D": 1, "L": 3, "ppo": {"optimizer_steps": 1,
            "attempted_optimizer_steps": 1, "rejected_optimizer_steps": 0}}) + "\n")
        return root, state

    def test_close_preserves_boundary_and_refuses_active_or_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, state = self.fixture(tmp)
            old = (root / "state.json").read_bytes()
            checkpoint = (root / state["latest"]).read_bytes()
            with RunLock(root), self.assertRaises(RuntimeError):
                close_run(root, "done")
            state["phase"] = "ppo"
            write(root / "state.json", state)
            with self.assertRaises(ValueError):
                close_run(root, "done")
            state["phase"] = "collect"
            state["pending_evaluation"] = True
            write(root / "state.json", state)
            with self.assertRaises(ValueError):
                close_run(root, "done")
            (root / "state.json").write_bytes(old)
            close_run(root, "done")
            self.assertEqual((root / "state.json").read_bytes(), old)
            self.assertEqual((root / state["latest"]).read_bytes(), checkpoint)
            self.assertEqual(json.loads((root / "status.json").read_text())["status"], "closed_by_user")

    def test_pause_threshold_only_at_completed_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.fixture(tmp)
            request_pause(root, after_update=2)
            self.assertFalse(pause_due(root, "update", 1))
            self.assertFalse(pause_due(root, "update", 2, at_boundary=False))
            self.assertTrue(pause_due(root, "update", 2))
            with self.assertRaises(ValueError):
                request_pause(root, after_epoch=1)

    def test_unevaluated_milestone_is_pending_and_blocks_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, state = self.fixture(tmp)
            run = json.loads((root / "run.json").read_text())
            run["config"]["experiment"]["milestones"] = [5]
            write(root / "run.json", run)
            self.assertTrue(target_bc_evaluation_due(run["config"], state, current_policy_hash="new"))
            with self.assertRaisesRegex(ValueError, "development evaluation"):
                close_run(root, "done")
            from civ5rl.experiments import ppo_after_target_bc as runner

            order = []
            policy = MagicMock()
            with (patch.object(runner, "configure"),
                  patch.object(runner, "make_policy", return_value=policy),
                  patch.object(runner, "optimizer_for"),
                  patch.object(runner.checkpoint, "restore", return_value=state),
                  patch.object(runner, "policy_hash", return_value="new"),
                  patch.object(runner, "ActiveClock"),
                  patch.object(runner, "development", side_effect=lambda *args: order.append("evaluation")),
                  patch.object(runner, "pause_due", side_effect=lambda *args, **kwargs: order.append("pause") or True)):
                runner.train(run["config"], {}, root, "cpu")
            self.assertEqual(order, ["evaluation", "pause"])
            state["development_points"]["5"] = "new"
            state["evaluated"]["new"] = {"checkpoint": state["latest"]}
            self.assertFalse(target_bc_evaluation_due(run["config"], state, current_policy_hash="new"))

    def test_snapshot_pools_episodes_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, state = self.fixture(tmp)
            point = {"checkpoint": "checkpoints/initial.pt", "purpose": "development",
                     "runs": {"argmax": {"cp": {"summary": {"W": 2, "D": 1, "L": 1}}}}}
            state["evaluated"] = {"hash": point}
            write(root / "state.json", state)
            write(root / "evaluations.json", {"development:hash": point})
            with (root / "updates.jsonl").open("a") as stream:
                stream.write(json.dumps({"rollout": 2, "total_decisions": 10, "W": 9, "D": 0, "L": 1}) + "\n")
            report = build_snapshot(root, block_size=2)
            self.assertEqual(len(report["development_argmax_cp"]), 1)
            self.assertAlmostEqual(report["training_blocks"][0]["win_rate"], 10 / 15)

    def test_snapshot_ignores_only_trailing_partial_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, _ = self.fixture(tmp)
            with (root / "updates.jsonl").open("a", encoding="utf-8") as stream:
                stream.write('{"rollout":')
            self.assertEqual(len(build_snapshot(root)["training_blocks"]), 1)
            with (root / "updates.jsonl").open("a", encoding="utf-8") as stream:
                stream.write('\n{"rollout": 2}\n')
            with self.assertRaises(json.JSONDecodeError):
                build_snapshot(root)
