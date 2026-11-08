"""Catch misleading GPU averages from phase boundaries, missing data and resume.

A normal training run prints plausible numbers even when intervals are assigned
to the wrong phase or an incomplete rollout leaks into the last-N window.
"""

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from civ5rl.engine.performance import summarize_performance


class PerformanceTests(unittest.TestCase):
    def write(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def row(self, seconds, phase, rollout, value=None, session="s"):
        return {
            "timestamp_utc": (
                datetime(2026, 1, 1, tzinfo=UTC) + timedelta(seconds=seconds)
            ).isoformat(),
            "elapsed_seconds": seconds,
            "session_id": session,
            "phase": phase,
            "rollout": rollout,
            "gpu_core_load_percent": value,
        }

    def update(self, rollout):
        return {
            "rollout": rollout,
            "ppo": {"ppo_seconds": 7, "runtime_compilation_seconds": 2},
            "collection": {"collection_seconds": 3},
        }

    def test_exact_boundaries_split_samples_and_select_only_completed_cycles(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            self.write(out / "updates.jsonl", [self.update(1), self.update(2)])
            self.write(
                out / "telemetry/phases.jsonl",
                [
                    self.row(0, "ppo", 1),
                    self.row(1, "ppo", 2),
                    self.row(2.5, "compilation", 2),
                    self.row(3, "ppo", 2),
                    self.row(5, "collection", 3),
                    self.row(6, "stopped", 3),
                ],
            )
            # 0% interval [2,3] must contribute only .5s to PPO; compile gets .5s.
            self.write(
                out / "telemetry/resources.jsonl",
                [
                    self.row(0, "ppo", 1, 100),
                    self.row(1, "ppo", 2, 80),
                    self.row(2, "ppo", 2, 0),
                    self.row(3, "ppo", 2, 100),
                    self.row(5, "collection", 3, 0),
                    self.row(6, "stopped", 3, 0),
                ],
            )
            result = summarize_performance(out, last=1)
            self.assertEqual(result["rollout_ids"], [2])
            self.assertEqual(result["phase_boundaries"], "events")
            ppo = result["phases"]["ppo"]
            self.assertAlmostEqual(ppo["seconds"], 3.5)
            self.assertAlmostEqual(ppo["gpu_core_load_percent_mean"], 280 / 3.5)
            self.assertEqual(ppo["nvml_coverage_fraction"], 1)
            self.assertAlmostEqual(result["phases"]["compilation"]["seconds"], 0.5)
            self.assertNotIn("collection", result["phases"])
            self.assertIsNone(result["timings"]["ppo_operation_without_compilation"])

    def test_legacy_resume_missing_nvml_partial_tail_and_time_weighting(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            self.write(out / "updates.jsonl", [self.update(1), self.update(2)])
            with (out / "updates.jsonl").open("a") as stream:
                stream.write('{"rollout":3')
            rows = [
                self.row(0, "ppo", 2, 0),
                self.row(1, "ppo", 2, 100),
                self.row(4, "ppo", 2, None),
                self.row(5, "ppo", 2, 100),
                self.row(30, "ppo", 2, 0),
                self.row(31, "collection", 3, 0),
            ]
            for row in rows:
                row.pop("session_id")
            # Resume: UTC increases but monotonic elapsed resets; do not bridge it.
            rows[4]["elapsed_seconds"] = 0
            rows[5]["elapsed_seconds"] = 1
            self.write(out / "telemetry/resources.jsonl", rows)
            result = summarize_performance(out, last=1)
            self.assertEqual(result["selected_cycles"], 1)
            self.assertEqual(result["phase_boundaries"], "sampled_or_mixed")
            ppo = result["phases"]["ppo"]
            self.assertEqual(ppo["seconds"], 6)
            self.assertEqual(ppo["nvml_seconds"], 5)
            self.assertEqual(ppo["gpu_core_load_percent_mean"], 60)
            self.assertAlmostEqual(ppo["nvml_coverage_fraction"], 5 / 6)
            self.assertEqual(ppo["gpu_60s_windows"]["count"], 0)

    def test_sixty_second_window_does_not_join_across_compilation(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            self.write(out / "updates.jsonl", [self.update(1)])
            self.write(
                out / "telemetry/resources.jsonl",
                [
                    self.row(
                        t,
                        "compilation" if 61 <= t < 63 else "ppo",
                        1,
                        100 if t < 30 else 50,
                    )
                    for t in range(100)
                ],
            )
            ppo = summarize_performance(out)["phases"]["ppo"]
            windows = ppo["gpu_60s_windows"]
            self.assertEqual(windows["count"], 2)
            self.assertAlmostEqual(windows["min_percent"], (29 * 100 + 31 * 50) / 60)
            self.assertEqual(windows["below_80_count"], 2)
            self.assertEqual(ppo["longest_contiguous_nvml_seconds"], 61)


if __name__ == "__main__":
    unittest.main()
