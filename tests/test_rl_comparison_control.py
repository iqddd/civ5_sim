"""Rejected updates must not leak Adam/RNG changes into later accepted work."""
from copy import deepcopy
import random
import unittest

import numpy as np
import torch

from civ5rl.engine.comparison_control import initial_state, run_updates
from civ5rl.engine.tensor_snapshot import cpu_snapshot
from civ5rl.provenance.rng import capture_rng, restore_rng


class Budget:
    ended = False

    def expired(self):
        return self.ended


class ControlTests(unittest.TestCase):
    def test_cyclic_anchors_preserve_inherited_adam_and_retry_boundary(self):
        # A wraparound must not skip anchors or reset inherited optimizer/RNG.
        model,opt,state,anchors,options=self.fixture(max_accepted_steps=12,cycle_anchors=True)
        for _ in range(10):
            opt.zero_grad();(-model.weight.sum()).backward();opt.step()
        state.update(steps=10,attempts=10,anchor_cursor=5)
        order=state['anchor_order'][:];seen=[];counters=[]
        def update(selected):
            seen.append((selected[:],float(torch.rand(()))))
            opt.zero_grad();(-model.weight.sum()).backward();opt.step()
            counters.append(int(next(iter(opt.state.values()))['step']))
            return {}
        checks=iter([.02,0.,0.])
        run_updates(model,opt,state,anchors,options,Budget(),lambda:None,update,lambda:next(checks))
        self.assertEqual(seen[0][0],[anchors[order[5]],anchors[order[0]]])
        self.assertEqual(seen[0],seen[1])
        self.assertEqual(seen[2][0],[anchors[order[1]],anchors[order[2]]])
        self.assertTrue(all(len(set(batch))==2 for batch,_ in seen))
        self.assertEqual(counters,[11,11,12])
        self.assertEqual(state['anchor_cursor'],9)
        self.assertEqual(state['anchor_order'],order)
        self.assertEqual(state['steps'],12)
        self.assertEqual(state['attempts'],13)
        self.assertEqual(state['stop_reason'],'accepted_step_limit')
        run_updates(model,opt,state,anchors,options,Budget(),lambda:None,
                    lambda selected:self.fail('Terminal fit restarted'),lambda:0.)

    def test_budget_keeps_inherited_time_after_transient_publication_denial(self):
        # A Windows reader can briefly deny replacement. Publication retries
        # must retain the full inherited cost when the phase is resumed.
        import tempfile
        from unittest.mock import patch
        from civ5rl.engine.phase_budget import PhaseBudget
        from civ5rl.provenance.common import read_json,write_json
        with tempfile.TemporaryDirectory() as tmp:
            calls=[]
            def denied_once(path,value):
                calls.append(value['seconds'])
                if len(calls)==1:raise PermissionError('temporary Windows reader')
                write_json(path,value)
            with patch('civ5rl.engine.phase_budget.write_json',side_effect=denied_once):
                with PhaseBudget(tmp,'training',1800,initial_seconds=482.2195) as budget:
                    self.assertGreaterEqual(budget.elapsed(),482.2195)
                    self.assertLess(1800-budget.elapsed(),1317.781)
                    self.assertFalse(budget.expired())
            saved=read_json(budget.path)
            restored=PhaseBudget(tmp,'training',1800,initial_seconds=482.2195)
            self.assertEqual(restored.base,saved['seconds'])
            self.assertGreaterEqual(saved['seconds'],482.2195)

    def test_cached_head_update_matches_full_pair_and_anchor_means(self):
        # The cache path must retain independent loss denominators and all legal
        # logits; a plausible but misnormalized loss can silently spoil a run.
        from civ5rl.core.model import Policy
        from civ5rl.engine.comparison_heads import HeadFeatures
        from civ5rl.engine.comparison_training import categorical_kl,pair_loss
        torch.manual_seed(11)
        policy=Policy('C2P')
        for name,p in policy.named_parameters():p.requires_grad_(name.startswith(('action.','attack_preview.')))
        manual=deepcopy(policy)
        cache=object.__new__(HeadFeatures)
        cache.policy=policy;cache.device=torch.device('cpu')
        cache.locations={(1,0):(1,0),(1,1):(1,1)}
        cache.records={1:dict(x=torch.randn(2,4,128),h=torch.randn(2,256),offsets=[0,3,5],
            actions=dict(sources=torch.tensor([0,1,-1,2,-1]),targets=torch.tensor([1,0,-1,1,-1]),
                         action_features=torch.randn(5,17),attack_preview=torch.randn(5,5)))}
        keys=[(1,0),(1,1)]
        batch,x,h=cache.inputs(keys)
        with torch.no_grad():
            initial=policy.heads(batch,x,h)[0]
            cache.references={k:((initial[i,:cache.count(k)]+torch.arange(cache.count(k))*.1)/.5).log_softmax(-1)
                              for i,k in enumerate(keys)}
        pairs=[dict(episode=1,index=0,better=1,worse=0,weight=.5),
               dict(episode=1,index=1,better=0,worse=1,weight=.25)]
        optimizer=torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=1e-5,eps=1e-5)
        expected_optimizer=torch.optim.Adam([p for p in manual.parameters() if p.requires_grad],lr=1e-5,eps=1e-5)
        logits=manual.heads(batch,x,h)[0]
        pair=sum(pair_loss(logits[i],p['better'],p['worse'],p['weight']) for i,p in enumerate(pairs))/2
        kl=sum(categorical_kl(cache.references[k],logits[i,:cache.count(k)]) for i,k in enumerate(keys))/2
        (pair+kl).backward()
        torch.nn.utils.clip_grad_norm_([p for p in manual.parameters() if p.requires_grad],.5)
        expected_optimizer.step()
        metrics=cache.update(optimizer,pairs,keys)
        self.assertAlmostEqual(metrics['pair_loss'],float(pair.detach()),places=6)
        self.assertAlmostEqual(metrics['anchor_kl'],float(kl.detach()),places=6)
        for name,value in policy.state_dict().items():
            torch.testing.assert_close(value,manual.state_dict()[name],atol=2e-7,rtol=1e-6)

    def test_rejected_learning_does_not_replace_last_accepted_diagnostics(self):
        # A rejected model must not appear as learned corrections in reports.
        model,opt,state,anchors,options=self.fixture(max_accepted_steps=2,max_retries=0)
        def update(selected):
            opt.zero_grad();(-model.weight.sum()).backward();opt.step();return {}
        def measure():
            return (0. if state['steps']==0 else .02), {'preferred':state['attempts']}
        run_updates(model,opt,state,anchors,options,Budget(),lambda:None,update,measure)
        self.assertEqual(state['learning'],{'preferred':1})
        self.assertEqual(state['history'][0]['learning'],{'preferred':1})
        self.assertNotIn('learning',state['history'][1])

    def test_learning_distinguishes_pair_preference_argmax_and_parent_confirmation(self):
        # Better than the paired action need not mean best among all legal ones.
        from civ5rl.engine.comparison_heads import learning_metrics
        pairs=[dict(episode=1,index=i,better=1 if i!=1 else 0) for i in range(3)]
        original={(1,i):0 for i in range(3)}
        baseline=dict(pairs=[(0,-.5,.7),(0,.2,.3),(0,0.,.4)],
                      choices={(1,0):0,(2,0):0})
        current=dict(pairs=[(2,.5,.2),(1,-.2,.5),(0,0.,.4)],
                     choices={(1,0):2,(2,0):1})
        result=learning_metrics(pairs,original,baseline,current,[(1,0),(2,0)])
        alternative=result['pairs']['alternative']
        self.assertEqual(alternative['count'],2)
        self.assertEqual(alternative['preferred'],1)
        self.assertEqual(alternative['ties'],1)
        self.assertEqual(alternative['better_argmax'],0)
        self.assertEqual(alternative['preference_gained'],1)
        self.assertEqual(result['pairs']['parent_confirmation']['preference_lost'],1)
        self.assertEqual(result['pairs']['parent_confirmation']['argmax_lost'],1)
        self.assertEqual(result['anchors']['outside_pairs'],dict(count=1,changed=1,fraction=1.))

    def fixture(self, **overrides):
        random.seed(4);np.random.seed(4);torch.manual_seed(4)
        options = dict(initial_lr=.1, max_accepted_steps=2, anchors_per_step=2,
                       kl_limit=.001, max_retries=2)
        options.update(overrides)
        anchors = [(i,0) for i in range(6)]
        model = torch.nn.Linear(1,1,bias=False)
        with torch.no_grad():model.weight.zero_()
        optimizer = torch.optim.Adam(model.parameters(),lr=options['initial_lr'])
        state = initial_state(anchors,options,19)
        return model,optimizer,state,anchors,options

    def test_halving_restores_adam_rng_and_uses_distinct_accepted_anchor_blocks(self):
        model,opt,state,anchors,options = self.fixture()
        observations=[];image=None
        def save():
            nonlocal image
            image=cpu_snapshot(dict(state=state,model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng()))
        def update(selected):
            observations.append((deepcopy(selected),random.random(),float(np.random.random()),float(torch.rand(()))))
            opt.zero_grad();(-model.weight.sum()).backward();opt.step()
            return {'pair_loss':float(-model.weight.detach().sum())}
        # This is the exact KL of N(0,1) against N(weight,1), not a scripted gate.
        def kl():return float(model.weight.detach().square().sum()/2)
        run_updates(model,opt,state,anchors,options,Budget(),save,update,kl)
        self.assertEqual(state['steps'],2)
        self.assertEqual(state['attempts'],5)
        self.assertEqual(state['rejected_attempts'],3)
        self.assertEqual(state['anchor_cursor'],4)
        self.assertEqual(state['lr'],.0125)
        self.assertEqual(int(next(iter(opt.state.values()))['step']),2)
        self.assertEqual(observations[0],observations[1]);self.assertEqual(observations[1],observations[2])
        self.assertEqual(observations[3],observations[4])
        self.assertFalse(set(observations[0][0]) & set(observations[3][0]))
        # Terminal resume may publish again but must never enter another update.
        model.load_state_dict(image['model']);opt.load_state_dict(image['optimizer']);restore_rng(image['rng'])
        before=model.weight.detach().clone()
        run_updates(model,opt,state,anchors,options,Budget(),save,update,kl)
        torch.testing.assert_close(model.weight,before,rtol=0,atol=0)
        self.assertEqual(len(observations),5)

    def test_deadline_finishes_entered_attempt_and_exact_boundary_is_accepted(self):
        for accepted in (True,False):
            with self.subTest(accepted=accepted):
                model,opt,state,anchors,options=self.fixture(kl_limit=.015)
                budget=Budget();before=model.weight.detach().clone()
                def update(selected):
                    opt.zero_grad();(-model.weight.sum()).backward();opt.step();budget.ended=True
                    return {}
                run_updates(model,opt,state,anchors,options,budget,lambda:None,update,
                            lambda:.015 if accepted else .015001)
                self.assertEqual(state['attempts'],1)
                self.assertEqual(state['steps'],int(accepted))
                self.assertEqual(state['anchor_cursor'],2*int(accepted))
                self.assertEqual(state['stop_reason'],'time_limit')
                self.assertIsNone(state['pending'])
                if not accepted:
                    torch.testing.assert_close(model.weight,before,rtol=0,atol=0)
                    self.assertFalse(opt.state)

    def test_interrupted_attempt_replays_its_id_without_reapplying_an_accept(self):
        model,opt,state,anchors,options=self.fixture(max_accepted_steps=1)
        saved=None;calls=0;randomness=[]
        def save():
            nonlocal saved
            saved=cpu_snapshot(dict(state=state,model=model.state_dict(),optimizer=opt.state_dict(),rng=capture_rng()))
        def interrupted(selected):
            randomness.append(float(torch.rand(())))
            opt.zero_grad();(-model.weight.sum()).backward();opt.step()
            raise RuntimeError('lost process before KL')
        with self.assertRaisesRegex(RuntimeError,'lost process'):
            run_updates(model,opt,state,anchors,options,Budget(),save,interrupted,lambda:0.)
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
        state=saved['state']
        def resumed(selected):
            nonlocal calls
            calls+=1;randomness.append(float(torch.rand(())))
            opt.zero_grad();(-model.weight.sum()).backward();opt.step()
            return {}
        run_updates(model,opt,state,anchors,options,Budget(),save,resumed,lambda:0.)
        self.assertEqual(randomness[0],randomness[1])
        self.assertEqual(state['attempts'],1);self.assertEqual(state['steps'],1)
        self.assertEqual(int(next(iter(opt.state.values()))['step']),1)
        self.assertEqual(calls,1)

    def test_three_rejections_leave_last_verified_model_and_adam(self):
        model,opt,state,anchors,options=self.fixture()
        before=model.weight.detach().clone()
        def update(selected):
            opt.zero_grad();(-model.weight.sum()).backward();opt.step()
            return {}
        run_updates(model,opt,state,anchors,options,Budget(),lambda:None,update,lambda:.02)
        self.assertEqual(state['attempts'],3);self.assertEqual(state['steps'],0)
        self.assertEqual(state['rejected_attempts'],3)
        self.assertEqual(state['stop_reason'],'retry_limit')
        self.assertEqual(state['anchor_cursor'],0)
        self.assertEqual(state['attempted_final_kl'],.02)
        torch.testing.assert_close(model.weight,before,rtol=0,atol=0)
        self.assertFalse(opt.state)


if __name__=='__main__':unittest.main()
