"""Regressions for allocator admission and safe optional replay fallback."""

from __future__ import annotations

import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from civ5rl.engine.ppo.cache import RolloutCache
from civ5rl.engine.ppo.replay import ReplayPart
from civ5rl.engine.runtime.sequence import SequenceBuffer, allocation_with_reserve


MEMORY = {"driver_free_bytes": 0, "allocated_bytes": 20, "reserved_bytes": 70}


class ResidentMemoryTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA pinned index transfers")
    def test_outstanding_index_uploads_keep_distinct_contents(self):
        buffer = SequenceBuffer(None, np.asarray([0, 64, 128]), None, device="cuda")
        buffer.enable_async_transfers()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(20_000_000)
            outputs = [buffer.indices([(1, i, i + 3), (0, i, i + 2)]) for i in range(16)]
        stream.synchronize()
        for i, result in enumerate(outputs):
            self.assertEqual(result.tolist(), [64 + i, 65 + i, 66 + i, i, i + 1])

    def test_low_driver_free_does_not_reject_allocator_success(self):
        # Recorded low driver free is telemetry, not an allocation oracle.
        empty = torch.empty
        probes = []
        def allocate(size, **kwargs):
            tensor = empty(1)
            probes.append(weakref.ref(tensor))
            return tensor
        with (
            patch("civ5rl.engine.runtime.sequence.cuda_memory_snapshot", return_value=MEMORY),
            patch("torch.empty", side_effect=allocate),
            patch("torch.cuda.empty_cache", side_effect=AssertionError("cache flushed")),
        ):
            with allocation_with_reserve("cuda", 32) as result:
                data = empty(4)
        self.assertEqual(result["status"], "admitted")
        self.assertEqual(data.numel(), 4)
        self.assertIsNone(probes[0]())

    def test_real_oom_type_and_partial_resident_cleanup(self):
        template = {"cells": torch.zeros(1, 1, 1)}
        encoded = [[SimpleNamespace(cells=[0], sources=[0])]]
        allocated = []
        empty = torch.empty
        def allocate(size, **kwargs):
            if isinstance(size, int):
                raise torch.cuda.OutOfMemoryError("reserve really unavailable")
            tensor = empty(size, dtype=kwargs["dtype"])
            allocated.append(weakref.ref(tensor))
            return tensor
        with (
            patch("civ5rl.engine.runtime.sequence.batch_inputs", return_value=template),
            patch("civ5rl.engine.runtime.sequence.cuda_memory_snapshot", return_value=MEMORY),
            patch("torch.empty", side_effect=allocate),
        ):
            try:
                SequenceBuffer.from_encoded(encoded, "cuda", resident=True, reserve_bytes=32)
            except torch.cuda.OutOfMemoryError as error:
                self.assertIn("reserve_probe", " ".join(error.__notes__))
            else:
                self.fail("A genuine reserve OOM must propagate")
        self.assertTrue(allocated)
        self.assertTrue(all(ref() is None for ref in allocated))

    def cache(self):
        cache = RolloutCache.__new__(RolloutCache)
        cache.device = "cuda"
        cache.replay_inputs = True
        cache._replay_plan = None
        cache._pinned_replay = None
        cache.replay_memory_admission = None
        return cache

    def test_optional_replay_oom_falls_back_and_drops_partial_tensors(self):
        cache = self.cache()
        refs = []
        def parts(*args):
            tensor = torch.zeros(4)
            refs.append(weakref.ref(tensor))
            yield ReplayPart({"cells": tensor}, {}, [1], None, 1)
            raise torch.cuda.OutOfMemoryError("next replay part does not fit")
        with (
            patch("civ5rl.engine.ppo.replay.replay_group_specs", return_value=[(1, [])]),
            patch("civ5rl.engine.ppo.replay.replay_parts", side_effect=parts),
            patch("civ5rl.engine.runtime.sequence.cuda_memory_snapshot", return_value=MEMORY),
        ):
            plan = cache.prepare_replay_plan(prefer_device=True)
        self.assertEqual(plan.mode, "on_demand")
        self.assertFalse(cache.replay_inputs)
        self.assertTrue(all(ref() is None for ref in refs))

    def test_optional_replay_does_not_hide_non_memory_errors(self):
        cache = self.cache()
        with (
            patch("civ5rl.engine.ppo.replay.replay_group_specs", return_value=[(1, [])]),
            patch("civ5rl.engine.ppo.replay.replay_parts", side_effect=RuntimeError("invalid data")),
            patch("civ5rl.engine.runtime.sequence.cuda_memory_snapshot", return_value=MEMORY),
        ):
            with self.assertRaisesRegex(RuntimeError, "invalid data"):
                cache.prepare_replay_plan(prefer_device=True)

    def test_host_fallback_reuses_already_pinned_parts(self):
        cache = self.cache()
        data, labels, lengths = {"cells": torch.zeros(4)}, {"returns": torch.zeros(1)}, [1]
        key = ((0, 0, 1),)
        cache._pinned_replay = {key: (data, labels, lengths)}
        cache.pinned_replay_bytes = 64
        cache._pinned_replay_limit = 128
        groups = [(1, [(key, data, labels, lengths, None, 1, 64)])]
        with patch("torch.Tensor.pin_memory", side_effect=AssertionError("pinned twice")):
            result = cache._build_host_groups(8, cpu_groups=groups)
        self.assertEqual(cache.pinned_replay_bytes, 64)
        self.assertIs(result[0][1][0].data, data)


if __name__ == "__main__":
    unittest.main()
