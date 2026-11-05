"""Wrong hex inverse directions silently corrupt training gradients at map edges."""

import unittest

import torch

from civ5map.maps import adjacent
from civ5rl.core.model import ActionEmbeddingGather, HexNeighborGather, MapMaximum


class InverseRoutingTests(unittest.TestCase):
    def test_repeated_action_targets_and_maximum_ties_preserve_gradients(self):
        generator = torch.Generator().manual_seed(17)
        for operation in ("actions", "maximum"):
            reference = torch.randn(2, 7, 5, generator=generator, dtype=torch.float64)
            if operation == "maximum":
                reference[:, :2] = 10  # first winner must receive tied gradients
            reference.requires_grad_()
            candidate = reference.detach().clone().requires_grad_()
            if operation == "actions":
                indices = torch.tensor([[0, 0, 3, -1, 6], [2, 2, 2, 6, -1]])
                expected = reference[torch.arange(2)[:, None], indices.clamp_min(0)] * (
                    indices >= 0
                ).unsqueeze(-1)
                actual = ActionEmbeddingGather.apply(candidate, indices)
            else:
                expected, actual = reference.max(1).values, MapMaximum.apply(candidate)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            gradient = torch.randn(
                expected.shape, generator=generator, dtype=torch.float64
            )
            expected.backward(gradient)
            actual.backward(gradient)
            torch.testing.assert_close(
                candidate.grad, reference.grad, rtol=1e-14, atol=1e-14
            )

    def test_neighbor_values_and_gradients_match_at_boundaries_and_both_parities(self):
        generator = torch.Generator().manual_seed(17)
        neighbors = []
        for parity in (0, 1):
            m = {"width": 5, "height": 7, "row_parity": parity, "wrap_x": False}
            rows = []
            for i in range(35):
                row = []
                for d in range(6):
                    xy = adjacent(m, i % 5, i // 5, d)
                    row.append(xy[1] * 5 + xy[0] if xy else -1)
                rows.append(row)
            neighbors.append(rows)
        indices = torch.tensor(neighbors)
        reference = torch.randn(
            2, 35, 7, 5, dtype=torch.float64, generator=generator, requires_grad=True
        )
        candidate = reference.detach().clone().requires_grad_()
        expected = reference[
            torch.arange(2)[:, None, None],
            indices.clamp_min(0),
            torch.arange(1, 7)[None, None, :],
        ] * (indices >= 0).unsqueeze(-1)
        actual = HexNeighborGather.apply(candidate, indices)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        gradient = torch.randn(expected.shape, dtype=torch.float64, generator=generator)
        expected.backward(gradient)
        actual.backward(gradient)
        torch.testing.assert_close(candidate.grad, reference.grad, rtol=0, atol=0)
