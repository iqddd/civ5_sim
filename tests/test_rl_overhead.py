"""Keep nested PPO timers from silently inflating active-time accounting."""

import json
import tempfile
import unittest
from pathlib import Path

from civ5rl.reporting.overhead import summarize_overhead


class OverheadAccountingTests(unittest.TestCase):
    def test_additive_balance_and_reference_excess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "telemetry").mkdir()
            (root / "telemetry/phases.jsonl").write_text(
                json.dumps({"session_id": "one", "rollout": 1}) + "\n"
            )
            rows = [
                {"rollout": i, "decisions": 10, "active_seconds": active,
                 "collection": {"collection_seconds": 2},
                 "ppo": {"operation_seconds": operation,
                         "runtime_compilation_seconds": comp, "kl_seconds": kl,
                         "attempted_optimizer_steps": 1, "optimizer_step_records": [{}]}}
                for i, active, operation, comp, kl in [(1, 5, 2, .5, 1), (2, 12, 5, 1, 4)]
            ]
            (root / "updates.jsonl").write_text("\n".join(map(json.dumps, rows)))
            (root / "state.json").write_text(json.dumps({"active_seconds": 12}))
            (root / "status.json").write_text(json.dumps({"status": "complete"}))
            report = summarize_overhead(root, reference_start=1, reference_stop=1)
            total = report["additive_active_accounting"]
            self.assertEqual(total["unattributed_active_seconds"], 1)
            self.assertEqual(total["ppo_without_callbacks_seconds"], 5.5)
            self.assertEqual(report["reference_comparison"]["excess_seconds"], 2)
            self.assertEqual(report["nested_ppo_timers_not_additive"]["kl_seconds"], 5)
            self.assertEqual(sum(total[k] for k in (
                "collection_seconds", "ppo_dynamo_callback_seconds",
                "ppo_without_callbacks_seconds", "unattributed_active_seconds")), 12)
