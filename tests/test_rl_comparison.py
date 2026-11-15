"""Silent math/selection failures would invalidate counterfactual labels."""
import unittest
import torch
from civ5rl.experiments.comparison_collection import preference,select_forks
from civ5rl.engine.comparison_training import pair_loss,categorical_kl

class ComparisonTests(unittest.TestCase):
    def test_collection_recovers_dataset_publication(self):
        import tempfile
        from pathlib import Path
        from civ5rl.provenance.common import write_json,read_json
        from civ5rl.experiments.rollout_comparison import _collect
        class Budget:
            def elapsed(self):return 7205.
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)
            folder=out/'sources/1'
            write_json(folder/'case.json',{'learner_side':0})
            write_json(folder/'history.json',[dict(actions=[{},{}])])
            # Legacy crash boundary: complete committed but dataset absent.
            write_json(out/'collection.json',dict(complete=True,sources=[1],forks=[],seconds=7200))
            cfg=dict(namespace='regression',collection_seconds=7200)
            _collect(cfg,out,budget=Budget())
            self.assertEqual(read_json(out/'dataset.json')['accepted_pairs'],0)
            self.assertTrue(read_json(out/'collection.json')['finalized'])
            self.assertEqual(read_json(out/'collection.json')['overshoot_seconds'],5.)
            before=(out/'dataset.json').read_bytes()
            _collect(cfg,out,budget=Budget())
            self.assertEqual((out/'dataset.json').read_bytes(),before)

    def test_strict_two_halves_and_reward(self):
        # First pair +1/+2; changing draws to losses cannot affect label.
        a=[0,None,1,1,0,None,1,1]
        b=[0,0,1,1,0,0,0,1]
        p=preference([a,b],0)
        self.assertTrue(p['accepted']);self.assertEqual(p['weight'],3/8)
        self.assertEqual(p['deltas'],[1,2])
        self.assertEqual(preference([[1 if x is None else x for x in a],b],0)['weight'],3/8)
        self.assertEqual(preference([a,a],0)['reason'],'both_zero')
        self.assertEqual(preference([a,b[:4]+a[4:]],0)['reason'],'one_zero')
        self.assertEqual(preference([a,b[:4]+[1]*4],0)['reason'],'opposite_signs')
    def test_pair_gradient_and_kl_direction(self):
        z=torch.tensor([1.,-.5,0.],requires_grad=True)
        loss=pair_loss(z,1,0,.5);loss.backward()
        self.assertLess(float(z.grad[1]),0);self.assertGreater(float(z.grad[0]),0);self.assertEqual(float(z.grad[2]),0)
        z2=z.detach().requires_grad_();half=pair_loss(z2,1,0,.25);half.backward()
        torch.testing.assert_close(z2.grad,z.grad/2)
        ref=(z.detach()/.5).log_softmax(-1)
        self.assertAlmostEqual(float(categorical_kl(ref,z.detach(),double=True)),0.,places=12)
        changed=torch.tensor([0.,1.,2.]);expected=(ref.double().exp()*(ref.double()-(changed/.5).log_softmax(-1).double())).sum()
        torch.testing.assert_close(categorical_kl(ref,changed,double=True),expected)
    def test_random_selection_balances_rounds_not_commands(self):
        rows=[dict(chosen=0,alternative=1,observation=dict(turn=t)) for t in range(6) for _ in range(t+1)]
        indices=select_forks(rows,123,'test')
        self.assertEqual(len(indices),4);self.assertEqual(len({rows[i]['observation']['turn'] for i in indices}),4)
        self.assertEqual(indices,select_forks(rows,123,'test'))
        rows[0]['chosen']=rows[0]['alternative']
        self.assertNotIn(0,select_forks(rows,123,'test',maximum=8))
        # Passing quota before sampling must leave late rounds reachable.
        rounds={rows[i]['observation']['turn'] for episode in range(128) for i in select_forks(rows,episode,'quota',maximum=2)}
        self.assertEqual(rounds,set(range(1,6)))

if __name__=='__main__':unittest.main()
