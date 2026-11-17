"""Silent scientific failures: weighting, held-out utility and adaptive futures."""
import unittest
import numpy as np
import torch
from civ5rl.engine.action_value import outcome_loss,split_worlds,world_scores,utility,bootstrap,fork_targets,public_action_vector


class ActionValueTests(unittest.TestCase):
    def test_extra_futures_affect_training_but_not_confirmation(self):
        fork=dict(executed_repetitions=8,side=0,
            winners={f'{r}-{a}':None for r in range(8) for a in range(2)})
        before=fork_targets(fork)
        for r in range(4,8):fork['winners'][f'{r}-1']=0
        after=fork_targets(fork)
        self.assertEqual(before[2],after[2])
        self.assertEqual(after[0],[0,4])
        self.assertEqual(after[1],[8,8])

    def test_public_vector_uses_current_hidden_and_zero_missing_cell(self):
        record=dict(offsets=[0,1],h=torch.ones(1,256),x=torch.full((1,2,128),2.),
            actions=dict(sources=torch.tensor([-1]),targets=torch.tensor([1]),
                action_features=torch.full((1,17),3.),attack_preview=torch.full((1,5),4.)))
        vector=public_action_vector(record,0,0)
        self.assertEqual(vector.shape,(534,))
        torch.testing.assert_close(vector,torch.cat([torch.ones(256),torch.zeros(128),
            torch.full((128,),2.),torch.full((17,),3.),torch.full((5,),4.)]))

    def test_count_weighted_bce_matches_individual_outcomes_and_gradients(self):
        z=torch.tensor([.3,-.7],requires_grad=True,dtype=torch.float64)
        w=torch.tensor([1.,5.],dtype=torch.float64);n=torch.tensor([4.,8.],dtype=torch.float64)
        loss=outcome_loss(z,w,n)
        expanded=torch.nn.functional.binary_cross_entropy_with_logits(
            torch.cat((z[0].expand(4),z[1].expand(8))),torch.tensor([1.,0.,0.,0.,1.,1.,1.,1.,1.,0.,0.,0.],dtype=torch.float64))
        torch.testing.assert_close(loss,expanded)
        torch.testing.assert_close(torch.autograd.grad(loss,z,retain_graph=True)[0],torch.autograd.grad(expanded,z)[0])

    def test_world_split_is_disjoint_and_independent_of_input_order(self):
        worlds=list(range(10000,10057));split=split_worlds(worlds)
        self.assertEqual(split,split_worlds(worlds[::-1]))
        self.assertEqual([len(x) for x in split.values()],[33,12,12])
        self.assertEqual(len(set(sum(split.values(),[]))),57)

    def test_ties_keep_parent_and_worlds_have_equal_weight(self):
        scores=world_scores([[0,0],[0,1],[1,0]],[[0,4],[0,4],[4,0]],[1,2,2])
        np.testing.assert_array_equal(scores,[[0,0,1],[1,.5,.5]])
        self.assertEqual(utility(scores),-.25)

    def test_bootstrap_recomputes_best_constant_for_every_sample(self):
        scores=np.array([[1,1,0],[1,0,1]])
        result=bootstrap(scores,samples=10000)
        # Resampling a single world twice makes a constant rule optimal: G=0.
        self.assertEqual(result['G'],.5)
        self.assertEqual(result['ci95'],[0.,.5])
        self.assertFalse(result['supported'])
        self.assertEqual(result,bootstrap(scores,samples=10000))


if __name__=='__main__':unittest.main()
