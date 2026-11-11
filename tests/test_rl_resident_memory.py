"""Regression for repeated rollout failures caused by allocator cache checks."""

import unittest
from contextlib import nullcontext
from unittest.mock import patch

from civ5rl.engine.sequence import resident_memory_preflight


class ResidentMemoryTests(unittest.TestCase):
    def test_cached_memory_can_satisfy_resident_buffer_after_reclaim(self):
        # The old guard rejected this even though no live tensor had to be freed.
        with (
            patch("civ5rl.engine.sequence.cuda_memory_snapshot", side_effect=[
                {"driver_free_bytes": 30, "allocated_bytes": 20, "reserved_bytes": 70},
                {"driver_free_bytes": 80, "allocated_bytes": 20, "reserved_bytes": 20},
            ]),
            patch("torch.cuda.device", return_value=nullcontext()),
            patch("torch.cuda.empty_cache"),
        ):
            result = resident_memory_preflight("cuda", 40, 10)
        self.assertTrue(result["reclaimed"])
        self.assertEqual(result["after"]["allocated_bytes"], 20)
        self.assertGreaterEqual(result["after"]["driver_free_bytes"], 50)

    def test_unreclaimable_reservations_do_not_hide_real_shortage(self):
        # Reserved-minus-allocated is not guaranteed to be reclaimable capacity.
        memory = {"driver_free_bytes": 30, "allocated_bytes": 20, "reserved_bytes": 70}
        with (
            patch("civ5rl.engine.sequence.cuda_memory_snapshot", return_value=memory),
            patch("torch.cuda.device", return_value=nullcontext()),
            patch("torch.cuda.empty_cache"),
        ):
            with self.assertRaisesRegex(MemoryError, "allocated_bytes.*reserved_bytes"):
                resident_memory_preflight("cuda", 40, 10)


if __name__ == "__main__":
    unittest.main()
