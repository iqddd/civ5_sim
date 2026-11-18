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
    def test_encoded_ensemble_shares_only_public_inputs_not_hidden_states(self):
        # One worker encoder serves two networks, but their recurrent states
        # must still follow each network's weights and every forced decision.
        from types import SimpleNamespace
        from civ5rl.core.observation import ordered_actions
        configure(17,'cpu')
        policies=[Policy('C2P').eval(),Policy('C2P').eval()]
        initial=scenario(unit(),unit(side=1,x=3))
        with Battle(initial) as battle:
            actions,previews=battle.actions_with_previews()
            actions=ordered_actions(actions);obs=battle.observe(0);obs['attack_previews']=previews
        agents=[RLAgent(p,'cpu','argmax') for p in policies]
        for agent in agents:agent.reset(context(initial))
        agents[0].members=agents
        worker=RLAgent(policies[0],'cpu','argmax');worker.reset(context(initial))
        server=SimpleNamespace(members=[SimpleNamespace(hidden=None) for p in policies])
        raw=EnsembleInference(policies,'cpu');encoded=EnsembleInference(policies,'cpu')
        for step in range(3):
            command='observe' if step==1 else 'act';forced=actions[-1]
            public=worker.encoder.encode(obs,actions)
            expected=raw.infer([(agents[0],command,obs,actions,forced)])[0]
            actual=encoded.infer_encoded([(server,command,public)])[0]
            for a,b in zip(server.members,agents):torch.testing.assert_close(a.hidden,b.hidden,atol=2e-6,rtol=2e-5)
            if command=='observe':
                self.assertIsNone(actual);worker.encoder.remember_action(forced,obs)
            else:self.assertEqual(worker.action_from_index(actual['index'],obs,actions),expected[0])
        self.assertFalse(torch.equal(server.members[0].hidden,server.members[1].hidden))

    def test_batched_greedy_ties_sample_rng_and_selection_metrics(self):
        from civ5rl.engine.evaluation.executor import BatchedInference
        configure(17,'cpu')
        initial=scenario(unit(),unit(side=1,x=3))
        with Battle(initial) as battle:
            legal,previews=battle.actions_with_previews()
            obs=battle.observe(0);obs['attack_previews']=previews
        for device in ('cpu',*(['cuda'] if torch.cuda.is_available() else [])):
            policy=Policy('C2P').to(device).eval()
            with torch.no_grad():policy.action[-1].weight.zero_();policy.action[-1].bias.zero_()
            modes=[('argmax',False),('argmax',True),('sample',False)]
            def agents():
                result=[RLAgent(policy,device,mode,collect_selection_metrics=metrics) for mode,metrics in modes]
                for agent in result:agent.reset(context(initial))
                return result
            shared,single=agents(),agents()
            actual=BatchedInference(policy,device).infer([(a,'act',obs,legal,None) for a in shared])
            for a,b,(choice,_) in zip(shared,single,actual):
                self.assertEqual(choice,b.act(obs,legal))
                self.assertEqual(a.selection_metrics,b.selection_metrics)
                self.assertEqual(a.rng.getstate(),b.rng.getstate())
                torch.testing.assert_close(a.hidden,b.hidden,atol=2e-6,rtol=2e-5)

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
