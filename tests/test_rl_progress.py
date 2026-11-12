"""Catch plausible but incorrect ETA after resume, budget pauses and partial logs.

These errors silently misstate resource needs; a successful training run would
not expose wrong scope, double-counted evaluations or poll-dependent EMA.
"""

import json
from pathlib import Path
import tempfile
import unittest

from civ5rl.reporting.progress import Observation, Sample, Stage, estimate
from civ5rl.reporting.progress_adapters import ADAPTERS
from civ5rl.reporting.progress_process import ProcessProgress, collect_progress
from civ5rl.reporting.progress_source import ArtifactSource


class ProgressTests(unittest.TestCase):
    def write(self, root, name, data):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")

    def log(self, root, name, rows):
        (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def test_generic_work_units_and_separate_budgets(self):
        observation = Observation("training", "search", "nodes", 100, 210,
                                  samples=[Sample(100, 10), Sample(100, 5)],
                                  stages=[Stage("dev", 2, 10), Stage("extra", 1, 30, conditional=True)],
                                  work_budget="active", budgets={"active": dict(limit_seconds=115, spent_seconds=100)})
        result = estimate(observation)
        self.assertEqual(result["ema_per_second"], 11)
        self.assertEqual(result["eta_work_seconds"], 10)
        self.assertEqual(result["eta_total_seconds"], 30)
        self.assertEqual(result["eta_conditional_seconds"], 30)
        self.assertTrue(result["budgets"]["active"]["fits"])
        observation.stages.append(Stage("unknown", 1, None))
        self.assertIsNone(estimate(observation)["eta_total_seconds"])

    def test_ppo_committed_work_resume_scope_and_poll_independence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = dict(kind="rl01.ppo_after_target_bc", experiment=dict(decisions=400, active_seconds=1000, milestones=[200, 400]))
            self.write(root, "resolved-config.json", cfg)
            self.write(root, "status.json", dict(status="paused"))
            self.write(root, "state.json", dict(decisions=300, active_seconds=115, development_points={"200": "h"}, evaluated={"h": {}}))
            self.write(root, "derived/progress/session.json", dict(id="new", ppo_rollout=1, evaluation_keys=["old"]))
            self.write(root, "evaluations.json", dict(old=dict(purpose="development", seconds=999),
                                                      cpu=dict(purpose="development", seconds=500, runtime=dict(device="cpu")),
                                                      new=dict(purpose="development", seconds=20, runtime=dict(device="cuda"),
                                                               runs=dict(argmax=dict(cp=dict(summary=dict(matches=512)))))))
            rows = [dict(rollout=1, total_decisions=100, active_seconds=100),
                    dict(rollout=2, total_decisions=200, active_seconds=110, ppo=dict(runtime_compilation_seconds=5))]
            self.log(root, "updates.jsonl", rows)
            with (root / "updates.jsonl").open("a") as stream:
                stream.write('{"rollout":3')
            source = ArtifactSource(root)
            result = collect_progress(source)
            self.assertEqual(result["completed"], 200)
            self.assertEqual(result["details"]["collected"], 300)
            self.assertEqual(result["ema_per_second"], 10)
            self.assertEqual(result["rate_samples"], 1)
            self.assertEqual(result["eta_total_seconds"], 40)
            self.assertIsNone(result["estimated_finish_at"])
            self.assertEqual(collect_progress(source)["ema_per_second"], 10)
            self.assertEqual(collect_progress(source, exclude_compilation=True)["ema_per_second"], 20)
            with (root / "updates.jsonl").open("a") as stream:
                stream.write(',"total_decisions":300,"active_seconds":115}\n')
            self.assertEqual(collect_progress(source)["completed"], 300)
            # Pending rollout 4 was collected before resume. Its cheap PPO-only
            # completion must not inflate the new process's full-cycle rate.
            self.write(root, "run.json", dict(config=cfg))
            self.write(root, "state.json", dict(rollouts=4, phase="ppo", active_seconds=116))
            with ProcessProgress(root, cfg):
                with (root / "updates.jsonl").open("a") as stream:
                    for row in [dict(rollout=4, total_decisions=400, active_seconds=118),
                                dict(rollout=5, total_decisions=500, active_seconds=128)]:
                        stream.write(json.dumps(row) + "\n")
            result = collect_progress(ArtifactSource(root))
            self.assertEqual(result["rate_samples"], 1)
            self.assertEqual(result["ema_per_second"], 10)

    def test_bc_active_time_after_old_format_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = dict(kind="rl01.cp_imitation_target_scale", epochs=10, training_seconds=200, evaluate_epochs=[])
            self.write(root, "resolved-config.json", cfg)
            self.write(root, "status.json", dict(status="training"))
            self.write(root, "collection.json", dict(labels=10))
            self.write(root, "state.json", dict(active_seconds=100))
            self.write(root, "derived/progress/session.json", dict(bc_step=1, active_seconds=100))
            self.log(root, "steps.jsonl", [dict(step=1, labels=10, seconds=1), dict(step=2, labels=10, seconds=1, active_seconds=110)])
            result = collect_progress(ArtifactSource(root))
            self.assertEqual(result["ema_per_second"], 1)
            self.assertEqual(result["eta_work_seconds"], 80)
            self.assertEqual(result["budgets"]["active"]["spent_seconds"], 110)
            self.assertTrue(result["budgets"]["active"]["fits"])
            (root / "derived/progress/session.json").unlink()
            result = collect_progress(ArtifactSource(root))
            self.assertIsNone(result["budgets"]["active"]["fits"])

    def test_process_final_snapshot_on_failure_and_future_adapter(self):
        class FutureAdapter:
            def observe(self, source):
                return Observation("training", "render", "frames", 5, 10, samples=[Sample(5, 1)])

        ADAPTERS["future.render"] = FutureAdapter
        self.addCleanup(ADAPTERS.pop, "future.render")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = dict(kind="future.render")
            self.write(root, "run.json", dict(config=cfg))
            original = (root / "run.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "scientific failure"):
                with ProcessProgress(root, cfg):
                    raise ValueError("scientific failure")
            result = json.loads((root / "derived/progress/latest.json").read_text())
            self.assertEqual(result["unit"], "frames")
            self.assertEqual(result["status"], "technical_error")
            self.assertFalse(result["process"]["running"])
            self.assertIsNone(result["estimated_finish_at"])
            self.assertEqual((root / "run.json").read_bytes(), original)
            refreshed = collect_progress(ArtifactSource(root))
            self.assertEqual(refreshed["status"], "technical_error")
            self.assertEqual(refreshed["process_state"], "stopped")


if __name__ == "__main__":
    unittest.main()
