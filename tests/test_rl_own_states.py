"""Silent risks: advice replacing history, mutated real RNG, or biased mixtures."""

import random
import unittest
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import torch

from civ5rl.core.observation import ObservationEncoder, ordered_actions, public_map
from civ5rl.engine.imitation import DemonstrationCache, decision_indices
from civ5rl.engine.imitation_own_states import LabelStream, eta_decision
from civ5rl.protocols.imitation import make_case, prepare_manifest, search_config
from civ5rl.protocols.imitation_own_states import AdvisedPolicy, ReplanningCPAdvisor
from civ5rl.provenance.common import configure, read_json
from civ5sim import Action, Battle
from civ5sim.combat_ai import CallPlanningServices, ContactMemory, PlanningState


class OwnStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(17)
        cls.config = read_json("configs/rl-cp-imitation-own-states.json")
        cls.base = prepare_manifest(cls.config)
        cls.case = make_case(cls.config, cls.base, "training", 0)
        cls.context = {
            "map": public_map(cls.case["scenario"]["map"]), "side":0,
            "armies":cls.case["armies"], "max_rounds":30, "fog":True,"random_damage":True,"seed":17,
        }

    def test_teacher_labels_do_not_replace_executed_history(self):
        rows = []
        with Battle(self.case["scenario"]) as battle:
            for _ in range(2):
                legal = ordered_actions(battle.available_actions())
                executed = next(a for a in legal if a.kind == "end_turn")
                teacher = next(a for a in legal if a != executed)
                rows.append({"observation":battle.observe(0),"actions":[asdict(a) for a in legal],
                             "teacher_action":legal.index(teacher),"executed_action":legal.index(executed),"supervised":True})
                battle.step(executed)
                battle.step(Action("end_turn"))
        episode = {"context":self.context,"decisions":rows}
        cache = DemonstrationCache([episode],"cpu")
        _,labels,_ = cache.batch([(0,0,2)])
        self.assertEqual(labels["indices"].tolist(),[r["teacher_action"] for r in rows])
        expected, wrong = ObservationEncoder(self.context), ObservationEncoder(self.context)
        for index,row in enumerate(rows):
            legal = tuple(Action(**a) for a in row["actions"])
            good = expected.encode(row["observation"],legal)
            bad = wrong.encode(row["observation"],legal)
            np.testing.assert_array_equal(good.previous,cache.encoded[0][index].previous)
            if index:
                self.assertFalse(np.array_equal(good.previous,bad.previous))
            expected.remember_action(legal[row["executed_action"]],row["observation"])
            wrong.remember_action(legal[row["teacher_action"]],row["observation"])
        legacy = {"chosen":1,"actions":[{},{}],"supervised":True}
        self.assertEqual(decision_indices(legacy),(1,1))
        expected = ObservationEncoder(self.context)
        np.testing.assert_array_equal(expected.encode(rows[0]["observation"],tuple(Action(**a) for a in rows[0]["actions"])).previous,
                                      DemonstrationCache([episode,episode],"cpu").encoded[1][0].previous)

    def test_advice_preserves_real_battle_rng_and_only_actual_public_memory(self):
        advisor = ReplanningCPAdvisor(search_config(self.config))
        advisor.reset(self.context)
        try:
            with Battle(self.case["scenario"]) as battle, battle.clone() as reference:
                observation = battle.observe(0)
                before = battle.snapshot()
                legal = tuple(battle.available_actions())
                service = CallPlanningServices(advisor.planner,PlanningState.from_battle(battle,0),0,observation["turn"])
                teacher = advisor.advise(observation,legal,service)
                service.expire()
                self.assertEqual(before,battle.snapshot())
                self.assertIsNone(advisor._pending)
                self.assertFalse(advisor._queue)
                self.assertEqual(advisor.memory,ContactMemory().updated(observation))
                actual = next(a for a in legal if a != teacher and a.kind == "move")
                result = battle.step(actual)
                self.assertEqual(asdict(result),asdict(reference.step(actual)))
                self.assertEqual(battle.snapshot(),reference.snapshot())
                after = battle.observe(0)
                memory = advisor.memory.updated(after)
                advisor.observe_actual_result(actual,result,after)
                self.assertEqual(advisor.memory,memory)
                self.assertIsNone(advisor._pending)
                self.assertFalse(advisor._objectives)
                advisor.reset(self.context)
                self.assertEqual(advisor.memory,ContactMemory())
                self.assertTrue(all(v == -1 for v in advisor._seen_at))
        finally:
            advisor.close()

    def test_wrapper_passes_only_public_inputs_to_policy(self):
        class Learner:
            def reset(self,context):
                self.context = deepcopy(context)
            def act(self,observation,actions):
                self.inputs = deepcopy(observation),tuple(actions)
                return actions[-1]
            def drain_diagnostics(self):
                return []
        learner = Learner()
        wrapper = AdvisedPolicy(learner,self.config)
        wrapper.reset(self.context)
        try:
            with Battle(self.case["scenario"]) as battle:
                obs,actions = battle.observe(0),tuple(battle.available_actions())
                service = CallPlanningServices(wrapper.planner,PlanningState.from_battle(battle,0),0,obs["turn"])
                executed = wrapper.act_with_planning(obs,actions,service)
                service.expire()
                self.assertEqual(learner.inputs,(obs,ordered_actions(actions)))
                row = wrapper.decisions[0]
                self.assertEqual(row["actions"][row["executed_action"]],asdict(executed))
                self.assertEqual(set(row),{"observation","actions","teacher_action","executed_action","supervised"})
        finally:
            wrapper.close()

    def test_mixture_counts_normalization_and_stable_ETA_stop(self):
        masks = [[i%3 != 0 for i in range(260)] for _ in range(4)]
        chunks = [(e,a,min(a+128,260)) for e in range(4) for a in range(0,260,128)]
        streams = [LabelStream([c for c in chunks if (c[0]<2) == (source==0)],masks,random.Random(17+source)) for source in (0,1)]
        for _ in range(4):
            for source,stream in enumerate(streams):
                batch = stream.take(512)
                self.assertEqual(sum(sum(masks[e][a:b]) for e,a,b in batch),512)
                self.assertTrue(all(b-a<=128 and (e<2)==(source==0) for e,a,b in batch))
        x = torch.randn(1024,3,requires_grad=True)
        target = torch.arange(1024)%3
        whole = torch.nn.functional.cross_entropy(x,target)
        parts = sum(torch.nn.functional.cross_entropy(x[a:a+128],target[a:a+128],reduction="sum")/1024 for a in range(0,1024,128))
        torch.testing.assert_close(torch.autograd.grad(whole,x,retain_graph=True)[0],torch.autograd.grad(parts,x)[0])
        settings = self.config["own_states"]
        waves = [{"matches":4,"labels":40,"seconds":10.} for _ in range(16)]
        streak = 0
        for _ in range(3):
            r = eta_decision(waves,25000,160.,settings,streak)
            streak = r["streak"]
        self.assertTrue(r["stop"])
        self.assertFalse(eta_decision(waves,500,160.,settings,streak)["stop"])
        fast = [{"matches":4,"labels":1000,"seconds":1.} for _ in range(16)]
        self.assertEqual(eta_decision(fast,25000,60.,settings,streak)["streak"],0)


if __name__ == "__main__":
    unittest.main()
