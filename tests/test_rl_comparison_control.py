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
