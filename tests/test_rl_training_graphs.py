"""Catch stale GRU replay inputs/weights and wrong shared-weight gradient sums."""

import copy
import unittest

import torch

from civ5rl.core.model import Policy
from civ5rl.core.schema import GLOBAL_DIM, PREV_ACTION_DIM, TILE_DIM
from civ5rl.core.spatial import SPATIAL_KEYS, StaticSpatialGraph
from civ5rl.core.training_graphs import RecurrentGraphs


class TrainingGraphTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA graph replay requires CUDA")
    @torch.no_grad()
    def test_spatial_replay_reads_live_weights_and_preserves_previous_outputs(self):
        weight = torch.full((), 2.0, device="cuda")
        first = {"cells": torch.ones((32, 256), device="cuda")}
        second = {"cells": torch.full_like(first["cells"], 3.0)}

        def spatial(part):
            cells = part["cells"] * weight
            return cells, cells.sum(dim=1)

        runner = StaticSpatialGraph(spatial, 32, 256)
        expected_first = spatial(first)
        actual_first = runner(first)
        weight.fill_(5.0)
        actual_second = runner(second)
        for actual, expected in zip(actual_second, spatial(second)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for actual, expected in zip(actual_first, expected_first):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

        policy = Policy()
        policy.spatial_batch_size = 32
        cells = torch.arange(37, device="cuda", dtype=torch.float32).view(-1, 1, 1)
        cells = cells.expand(-1, 256, 1).contiguous()
        batch = dict.fromkeys(SPATIAL_KEYS, cells)
        block_runner = StaticSpatialGraph(spatial, 32, 256)
        expected_batch = spatial(batch)
        previous = policy.encode_spatial(batch, spatial_runner=block_runner)
        weight.fill_(7.0)
        maps, context = policy.encode_spatial(batch, spatial_runner=block_runner, return_cells=False)
        self.assertIsNone(maps)
        torch.testing.assert_close(context, spatial(batch)[1], rtol=0, atol=0)
        for actual, expected in zip(previous, expected_batch):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_coalesced_spatial_preserves_padded_block_gradients(self):
        torch.manual_seed(17)
        reference = Policy()
        reference.spatial_batch_size = 2
        candidate = copy.deepcopy(reference)
        candidate.enable_coalesced_spatial()
        batch = {
            "cells": torch.randn(5, 3, TILE_DIM),
            "cell_mask": torch.ones(5, 3, dtype=torch.bool),
            "neighbors": torch.full((5, 3, 6), -1, dtype=torch.long),
            "globals": torch.randn(5, GLOBAL_DIM),
            "previous": torch.randn(5, PREV_ACTION_DIM),
        }
        expected = reference.encode_spatial(batch)
        actual = candidate.encode_spatial(batch)
        for a, b in zip(expected, actual):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        sum(value.square().mean() for value in expected).backward()
        sum(value.square().mean() for value in actual).backward()
        for a, b in zip(reference.parameters(), candidate.parameters()):
            if a.grad is None:
                self.assertIsNone(b.grad)
            else:
                torch.testing.assert_close(a.grad, b.grad, rtol=2e-5, atol=2e-6)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA graph replay requires CUDA")
    def test_gru_replay_uses_current_inputs_hidden_and_updated_weights(self):
        torch.manual_seed(17)
        reference = torch.nn.GRU(256, 256).cuda()
        candidate = copy.deepcopy(reference)
        runner = RecurrentGraphs(candidate, max_shapes=1)
        optimizers = [torch.optim.Adam(module.parameters(), lr=1e-4, eps=1e-5) for module in (reference, candidate)]
        # Reset hidden after a nonzero history, then evict/recreate a shape.
        for step, length in enumerate((5, 5, 5, 7, 5)):
            value = torch.randn(length, 2, 256, device="cuda")
            hidden = torch.randn(1, 2, 256, device="cuda") if step == 1 else torch.zeros(1, 2, 256, device="cuda")
            results = []
            for function, module, optimizer in zip((reference, runner), (reference, candidate), optimizers):
                optimizer.zero_grad(set_to_none=True)
                x, h = value.clone().requires_grad_(), hidden.clone().requires_grad_()
                y, last = function(x, h)
                (y.square().mean() + last.square().mean()).backward()
                results.append((y.detach().clone(), x.grad.clone(), h.grad.clone(), [p.grad.clone() for p in module.parameters()]))
                optimizer.step()
            for a, b in zip(results[0][:3], results[1][:3]):
                torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-7)
            for a, b in zip(results[0][3], results[1][3]):
                torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-7)
            for a, b in zip(reference.parameters(), candidate.parameters()):
                torch.testing.assert_close(a, b, rtol=3e-5, atol=2e-7)
        self.assertEqual(runner.captures, 3)
        self.assertEqual(len(runner.entries), 1)


if __name__ == "__main__":
    unittest.main()
