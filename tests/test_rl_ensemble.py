"""Ensemble histories must follow executed commands, never component proposals."""
import unittest
from copy import deepcopy
import torch
from civ5rl.core.agent import RLAgent
from civ5rl.core.model import Policy
from civ5rl.reporting.ensemble import EnsembleInference, mean_probabilities
from civ5rl.provenance.common import configure
from civ5sim import Battle
from tests.test_combat import scenario, unit
from tests.test_rl import context


class EnsembleTests(unittest.TestCase):
    def test_probability_mixture_is_not_logit_average(self):
        a=torch.tensor([[8.,0.,-2.]])
        b=torch.tensor([[-1.,2.,3.]])
        actual=mean_probabilities([a,b])
        torch.testing.assert_close(actual,.5*(a/.5).softmax(-1)+.5*(b/.5).softmax(-1))
        self.assertFalse(torch.allclose(actual,((a+b)/2/.5).softmax(-1)))

    def test_history_forced_and_isolated_battles(self):
        self.check_history('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_batched_history(self):
        self.check_history('cuda')

    def check_history(self, device):
        configure(17, device)
        policy=Policy('C2P').to(device).eval()
        initial=scenario(unit(),unit(side=1,x=3))
        with Battle(initial) as battle:
            actions,previews=battle.actions_with_previews()
            obs=battle.observe(0); obs['attack_previews']=previews
        def agent():
            a=RLAgent(policy,device,'argmax'); a.reset(context(initial))
            other=RLAgent(policy,device,'argmax');other.reset(context(initial))
            a.members=[a,other]
            return a
        a,b=agent(),agent()
        single=RLAgent(policy,device,'argmax');single.reset(context(initial))
        backend=EnsembleInference([policy,policy],device)
        chosen=backend.infer([(a,'act',obs,actions,None),(b,'act',obs,actions,None)])
        self.assertEqual(chosen[0][0],single.act(obs,actions))
        torch.testing.assert_close(a.hidden,single.hidden,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(a.hidden,b.hidden)
        saved=b.hidden.clone()
        forced=actions[-1]
        backend.infer([(a,'observe',obs,actions,forced)])
        single.observe_step(obs,actions,forced)
        torch.testing.assert_close(a.hidden,single.hidden,atol=2e-6,rtol=2e-5)
        torch.testing.assert_close(a.members[1].hidden,a.hidden)
        torch.testing.assert_close(b.hidden,saved,atol=0,rtol=0)
        following=backend.infer([(a,'act',obs,actions,None)])[0][0]
        self.assertEqual(following,single.act(obs,actions))
        torch.testing.assert_close(a.hidden,single.hidden,atol=2e-6,rtol=2e-5)
        fresh=agent()
        self.assertIsNone(fresh.hidden)
        self.assertIsNone(fresh.members[1].hidden)
        self.assertIsNot(fresh.encoder,fresh.members[1].encoder)

if __name__=='__main__': unittest.main()
