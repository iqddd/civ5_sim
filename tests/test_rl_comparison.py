"""Silent math/selection failures would invalidate counterfactual labels."""
import unittest
import torch
from civ5rl.experiments.comparison_collection import preference,select_forks
from civ5rl.engine.comparison_training import pair_loss,categorical_kl

class ComparisonTests(unittest.TestCase):
    def test_batched_histories_keep_prefix_boundaries_and_preference_weights(self):
        # Wrong GRU grouping, padding, or loss multiplicity can silently train on
        # another episode's memory. Compare the actual accumulated gradients.
        from copy import deepcopy
        from dataclasses import replace
        from civ5rl.core.model import Policy
        from civ5rl.core.observation import ObservationEncoder,ordered_actions
        from civ5rl.engine.comparison_training import target_logits,logical_step,reference_distributions
        from civ5rl.provenance.common import configure
        from civ5rl.protocols.memory import flat_map
        from civ5sim import Battle
        from tests.test_combat import scenario,unit
        from tests.test_rl import context
        configure(17,'cpu')
        s=scenario(unit(x=0,y=0),unit(side=1,x=1,y=1),terrain=flat_map(2))
        with Battle(s) as battle:
            legal,previews=battle.actions_with_previews()
            obs=battle.observe(0);obs['attack_previews']=previews
            item=ObservationEncoder(context(s),'C2P').encode(obs,ordered_actions(legal))
        first=[replace(item,previous=item.previous+i*.001) for i in range(130)]
        second=[replace(item,previous=item.previous-i*.003) for i in range(3)]
        class Histories:
            def get(self,e):return ({1:first,2:second}[e],None)
        histories=Histories()
        a=Policy('C2P');a.configure_execution(inverse_routing=True,action_routing=True,maximum_routing=True,attention_backend='compact_fp32')
        b=deepcopy(a)
        pairs=[dict(episode=1,index=129,better=1,worse=0,weight=.5),dict(episode=2,index=2,better=0,worse=1,weight=.25)]
        anchors=[(1,129),(2,0),(2,2),(2,2)]
        refs=reference_distributions(a,histories,anchors)
        with torch.no_grad():
            for key in set(anchors):
                h,_=histories.get(key[0]);z=target_logits(a,h,key[1])
                torch.testing.assert_close(refs[key],(z/.5).log_softmax(-1),atol=2e-6,rtol=2e-5)
        # Nonzero KL as well as preference gradients; one position appears three times.
        with torch.no_grad():a.action[-1].weight[0,0].add_(.01);b.load_state_dict(a.state_dict())
        for row in pairs:
            h,_=histories.get(row['episode'])
            z=target_logits(a,h,row['index'],gradient=True)
            (pair_loss(z,row['better'],row['worse'],row['weight'])/len(pairs)).backward()
        for key in anchors:
            h,_=histories.get(key[0]);z=target_logits(a,h,key[1],gradient=True)
            (categorical_kl(refs[key],z)/len(anchors)).backward()
        torch.nn.utils.clip_grad_norm_([p for p in a.parameters() if p.requires_grad],.5)
        optimizer=torch.optim.Adam(b.parameters(),lr=0.)
        logical_step(b,optimizer,histories,pairs,anchors,refs)
        for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            if p.grad is not None:torch.testing.assert_close(q.grad,p.grad,atol=2e-6,rtol=3e-4,msg=name)
        if torch.cuda.is_available():
            # Calling CoalescedSpatial directly must retain the executor's BF16
            # autocast boundary. A missing boundary silently changes precision.
            from civ5rl.engine.comparison_training import _encode_histories
            from civ5rl.core.model import batch_inputs
            p=deepcopy(b).cuda();q=deepcopy(b).cuda()
            for model in (p,q):
                model.zero_grad(set_to_none=True);model.enable_amp(True,spatial_batch_size=32)
                model.spatial_cell_padding=16
            sequences=[first[:16],second]
            inputs=batch_inputs([item for seq in sequences for item in seq],'cuda',spatial_only=True,compact_units=True)
            expected=p.encode_spatial(inputs);actual=_encode_histories(q,sequences)
            for x,y in zip(expected,actual):torch.testing.assert_close(y,x,atol=2e-6,rtol=2e-5)
            sum(x.square().mean() for x in expected).backward()
            sum(x.square().mean() for x in actual).backward()
            for (name,x),(_,y) in zip(p.named_parameters(),q.named_parameters()):
                if x.grad is not None:torch.testing.assert_close(y.grad,x.grad,atol=2e-6,rtol=3e-4,msg=name)

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
