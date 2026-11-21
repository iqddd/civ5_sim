import unittest

import torch

from civ5rl.core.model import Policy
from civ5rl.core.preview import load_parent_model
from civ5rl.core.schema import ACTION_DIM, ATTACK_PREVIEW_DIM, HEX_DIM, HIDDEN_SIZE
from civ5rl.provenance.identity import policy_hash


class ActionContextTests(unittest.TestCase):
    def test_zero_start_gradients_and_masked_context(self):
        torch.manual_seed(17)
        parent, policy = Policy("C2P"), Policy("C2PA")
        load_parent_model(
            policy,
            {"variant": parent.variant, "model": parent.state_dict()},
            {"experiment": {"model_extension": "action_context_v1"}},
        )
        mask = torch.tensor([[True, True, False], [True, False, False], [False, False, False]])
        batch = {
            "sources": torch.tensor([[0, 1, -1], [1, -1, -1], [-1, -1, -1]]),
            "targets": torch.tensor([[1, -1, -1], [0, -1, -1], [-1, -1, -1]]),
            "action_features": torch.randn(3, 3, ACTION_DIM),
            "attack_preview": torch.randn(3, 3, ATTACK_PREVIEW_DIM),
            "action_mask": mask,
        }
        cells = torch.randn(3, 2, HEX_DIM)
        hidden = torch.randn(3, HIDDEN_SIZE)
        with torch.no_grad():
            for baseline, extended in zip(parent.heads(batch, cells, hidden), policy.heads(batch, cells, hidden)):
                self.assertTrue(torch.equal(baseline.view(torch.int32), extended.view(torch.int32)))

        _, values = policy.heads(batch, cells, hidden)
        values.sum().backward()
        self.assertGreater(torch.count_nonzero(policy.value_actions.weight.grad).item(), 0)
        for module in (policy.action_attention, policy.action, policy.attack_preview):
            self.assertTrue(all(p.grad is None for p in module.parameters()))
        policy.zero_grad(set_to_none=True)
        logits, _ = policy.heads(batch, cells, hidden, compute_value=False)
        logits[mask].sum().backward()
        self.assertGreater(torch.count_nonzero(policy.action_attention.out_proj.weight.grad).item(), 0)
        self.assertEqual(torch.count_nonzero(policy.action_attention.q_proj.weight.grad).item(), 0)

        with torch.no_grad():
            policy.action_attention.out_proj.weight.normal_(std=0.02)
            actor_hash = policy_hash(policy.state_dict())
            policy.value_actions.weight.normal_(std=0.02)
            self.assertEqual(policy_hash(policy.state_dict()), actor_hash)
            baseline_logits, baseline_values = policy.heads(batch, cells, hidden)
            for key in ("action_features", "attack_preview"):
                batch[key][~mask] = 100
            logits, values = policy.heads(batch, cells, hidden)
            self.assertTrue(torch.equal(logits, baseline_logits))
            self.assertTrue(torch.equal(values, baseline_values))
            self.assertTrue(torch.isfinite(values).all())
