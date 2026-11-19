"""Silent scientific errors: null outcomes and fixed hierarchical normalization."""
import unittest
import torch
from civ5rl.engine.comparison.turn_training import preference_loss


class TurnFitMathTests(unittest.TestCase):
    def test_training_kl_averages_anchors_across_map_sizes(self):
        from civ5rl.core.model import Policy, batch_inputs
        from civ5rl.core.observation import ObservationEncoder, ordered_actions
        from civ5rl.engine.comparison.inputs import History
        from civ5rl.engine.comparison.training import DeviceReferences, categorical_kl, target_logits
        from civ5rl.engine.comparison.turn_training import logical_step, pack_batch
        from civ5rl.protocols.memory import flat_map
        from civ5rl.provenance.common import configure
        from civ5sim import Battle
        from tests.test_combat import scenario, unit
        from tests.test_rl import context

        configure(17, 'cpu')
        encoded = {}
        for episode, size in enumerate((2, 3, 2)):
            spec = scenario(unit(x=0, y=0), unit(side=1, x=1, y=1), terrain=flat_map(size))
            with Battle(spec) as battle:
                actions, previews = battle.actions_with_previews()
                observation = battle.observe(0)
                observation['attack_previews'] = previews
                item = ObservationEncoder(context(spec), 'C2P').encode(observation, ordered_actions(actions))
                encoded[episode] = [item]

        class Histories:
            def get(self, episode):
                return History(self, episode, 0, len(encoded[episode])), None

            def item(self, episode, index):
                return encoded[episode][index]

            def spatial_inputs(self, sequences, device):
                # The prepared-history path concatenates packed tensors without cell padding.
                parts = [batch_inputs(list(seq), device, spatial_only=True, compact_units=True)
                         for seq in sequences]
                return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}

        histories = Histories()
        policy = Policy('C2P')
        anchors = [(0, 0), (1, 0), (2, 0)]
        references = {}
        with torch.no_grad():
            for episode, index in anchors:
                logits = target_logits(policy, encoded[episode], index)
                references[episode, index] = (logits / .5 + torch.arange(len(logits)) * .3).log_softmax(-1)
            expected = torch.stack([
                categorical_kl(references[episode, index], target_logits(policy, encoded[episode], index))
                for episode, index in anchors
            ]).mean()
        optimizer = torch.optim.Adam(policy.parameters(), lr=0.)
        batch = pack_batch([[]], 'cpu')
        metrics = logical_step(policy, optimizer, histories, batch, anchors, torch.empty(0),
                               DeviceReferences(references, 'cpu'), microbatch=2)
        self.assertGreater(float(expected), 0.)
        self.assertAlmostEqual(metrics['anchor_kl'], float(expected), places=6)

    def test_zero_outcomes_and_label_exchange(self):
        current=torch.tensor([.3,-.2],requires_grad=True)
        parent=torch.zeros(2)
        indices=torch.tensor([[[0],[1]]])
        mask=torch.ones_like(indices,dtype=torch.float32)
        zero=preference_loss(current,parent,indices,mask,torch.tensor([0.]),torch.tensor([1.]))
        zero.backward()
        self.assertEqual(float(zero.detach()),0.)
        self.assertTrue(torch.equal(current.grad,torch.zeros(2)))
        a=preference_loss(current,parent,indices,mask,torch.tensor([1.]),torch.tensor([1.]))
        b=preference_loss(current,parent,indices.flip(1),mask,torch.tensor([-1.]),torch.tensor([1.]))
        self.assertEqual(float(a.detach()),float(b.detach()))

    def test_zero_comparisons_stay_in_fixed_denominators_and_vjp_is_exact(self):
        x=torch.tensor([.1,.2,-.1],requires_grad=True)
        parent=torch.zeros_like(x)
        ids=torch.tensor([[[0,1],[2,2]],[[1,1],[2,2]]])
        mask=torch.tensor([[[1.,1.],[1.,0.]],[[1.,0.],[1.,0.]]])
        signs=torch.tensor([1.,0.])
        loss=preference_loss(x,parent,ids,mask,signs,torch.tensor([1/(64*6*8),1/(64*6*4)]))
        expected=torch.nn.functional.softplus(-((x[0]+x[1])/2-x[2]))/(64*6*8)
        self.assertTrue(torch.allclose(loss,expected,rtol=0,atol=1e-10))
        coefficients,=torch.autograd.grad(loss,x)
        y=x.detach().clone().requires_grad_(True)
        (coefficients*y).sum().backward()
        self.assertTrue(torch.equal(coefficients,y.grad))

    def test_trie_and_selection_matrix_match_direct_bptt_and_collect_prefixes(self):
        from copy import deepcopy
        from civ5rl.core.model import Policy, batch_inputs
        from civ5rl.core.observation import ObservationEncoder, ordered_actions
        from civ5rl.engine.comparison.inputs import History
        from civ5rl.engine.comparison.training import DeviceReferences, target_logits
        from civ5rl.engine.comparison.turn_training import (
            AnchorSubset, TurnPreference, control_kl, current_chosen, hidden_states_at, logical_step, pack_batch,
        )
        from civ5rl.protocols.memory import flat_map
        from civ5rl.provenance.common import configure
        from civ5sim import Battle
        from tests.test_combat import scenario, unit
        from tests.test_rl import context

        configure(17, 'cpu')
        spec = scenario(unit(x=0, y=0), unit(side=1, x=1, y=1), terrain=flat_map(3))
        with Battle(spec) as battle:
            actions, previews = battle.actions_with_previews()
            observation = battle.observe(0)
            observation['attack_previews'] = previews
            item = ObservationEncoder(context(spec), 'C2P').encode(observation, ordered_actions(actions))
        spec2 = scenario(unit(x=1, y=0), unit(side=1, x=2, y=2), terrain=flat_map(3))
        with Battle(spec2) as battle2:
            actions2, previews2 = battle2.actions_with_previews()
            obs2 = battle2.observe(0)
            obs2['attack_previews'] = previews2
            item2 = ObservationEncoder(context(spec2), 'C2P').encode(obs2, ordered_actions(actions2))

        encoded = {
            0: [item] * 131,
            10: [item, item, item, item],
            11: [item, item, item, item2],
        }
        branches = {
            10: dict(episode=0, prefix=2),
            11: dict(episode=0, prefix=2),
        }
        branch_steps = {
            10: [('shared_t2', 0), ('tail_a', 0)],
            11: [('shared_t2', 1), ('tail_b', 0)],
        }
        targets = {(10, 2): 0, (10, 3): 0, (11, 2): 1, (11, 3): 0}

        class MockHistories:
            def __init__(self):
                self.branches = branches
                self.targets = targets

            def get(self, episode):
                return History(self, episode, 0, len(encoded[episode])), None

            def item(self, episode, index):
                return encoded[episode][index]

            def spatial_inputs(self, sequences, device):
                parts = [batch_inputs(list(seq), device, spatial_only=True, compact_units=True)
                         for seq in sequences]
                return {key: torch.cat([part[key] for part in parts]) for key in parts[0]}

        histories = MockHistories()
        policy_trie = Policy('C2P')
        policy_ref = deepcopy(policy_trie)
        anchors = [(0, 130)]
        with torch.no_grad():
            ref_logits = target_logits(policy_ref, encoded[0], 130)
            ref_dist = {(0, 130): (ref_logits / .5).log_softmax(-1)}
            parent_chosen = torch.tensor([
                float((target_logits(policy_ref, encoded[ep], idx) / .5).log_softmax(-1)[targets[ep, idx]])
                for ep, idx in sorted(targets)
            ])
        refs = DeviceReferences(ref_dist, 'cpu')
        ctrl_refs = DeviceReferences(ref_dist, 'cpu', control=True)

        # Verify current_chosen trie parity
        trie_chosen = current_chosen(policy_trie, histories, sorted(targets), branch_steps=branch_steps)
        self.assertTrue(torch.allclose(trie_chosen, parent_chosen, rtol=0, atol=1e-6))

        # Verify control_kl prefix collection matches hidden_states_at
        collected = {}
        kl, _ = control_kl(
            policy_trie, histories, anchors, ctrl_refs,
            anchor_subset=AnchorSubset(frozenset()),
            collect_prefixes=anchors, collected_prefixes=collected,
        )
        self.assertAlmostEqual(kl, 0., places=6)
        direct_h = hidden_states_at(policy_trie, histories, [(0, 2)])[(0, 2)]
        self.assertTrue(torch.allclose(collected[(0, 130)], direct_h, rtol=0, atol=1e-5))

        # Verify logical_step gradient parity vs direct target_logits(..., gradient=True)
        prefs = [[TurnPreference([(10, 2), (10, 3)], [(11, 2), (11, 3)], 1, 4)]]
        batch = pack_batch(prefs, 'cpu', histories=histories, branch_steps=branch_steps)
        opt_trie = torch.optim.SGD(policy_trie.parameters(), lr=0.)
        logical_step(policy_trie, opt_trie, histories, batch, anchors, parent_chosen, refs)

        policy_ref.zero_grad(set_to_none=True)
        direct_vals = torch.stack([
            (target_logits(policy_ref, encoded[ep], idx, gradient=True) / .5).log_softmax(-1)[targets[ep, idx]]
            for ep, idx in sorted(targets)
        ])
        loss_ref = preference_loss(direct_vals, parent_chosen, batch.indices, batch.mask, batch.signs, batch.weights)
        kl_logits = target_logits(policy_ref, encoded[0], 130, gradient=True)
        loss_ref = loss_ref + (refs[(0, 130)].exp() * (refs[(0, 130)] - (kl_logits / .5).log_softmax(-1))).sum()
        loss_ref.backward()
        for p1, p2 in zip(policy_trie.parameters(), policy_ref.parameters()):
            if p1.grad is None and p2.grad is None:
                continue
            self.assertTrue(torch.allclose(p1.grad, p2.grad, rtol=1e-5, atol=1e-6))


if __name__=='__main__':unittest.main()
