import unittest

from civ5rl.engine.resources import integrate_gpu_interval


class GpuIntervalTest(unittest.TestCase):
    def test_splits_phase_and_excludes_gaps_and_resumes(self):
        def sample(t, value, session="a"):
            return {"session_id": session, "elapsed_seconds": t,
                    "gpu_core_load_percent": value, "phase": "train"}

        samples = [sample(0, 20), sample(2, 80), sample(4, None),
                   sample(6, 100), sample(20, 100), sample(0, 100, "b")]
        events = [{"session_id": "a", "elapsed_seconds": 0, "phase": "train"},
                  {"session_id": "a", "elapsed_seconds": 1, "phase": "pause"},
                  {"session_id": "a", "elapsed_seconds": 3, "phase": "train"}]
        result = integrate_gpu_interval(samples, 0.5, 20, session_id="a",
                                        sample_interval=1, phase_events=events,
                                        active_phase="train")
        self.assertAlmostEqual(result["covered_seconds"], 1.5)
        self.assertAlmostEqual(result["estimated_busy_seconds"], 0.9)
        self.assertAlmostEqual(result["gpu_core_load_percent_mean"], 60)
        self.assertAlmostEqual(result["coverage_fraction"], 1.5 / 19.5)

    def test_unavailable_gpu_is_not_zero_load(self):
        samples = [{"session_id": "x", "elapsed_seconds": t,
                    "gpu_core_load_percent": None, "phase": "train"} for t in (0, 1)]
        result = integrate_gpu_interval(samples, 0, 1, session_id="x")
        self.assertEqual(result["coverage_fraction"], 0)
        self.assertIsNone(result["gpu_core_load_percent_mean"])
        self.assertIsNone(result["estimated_busy_seconds"])
