"""Attack-first baseline: real movement rules and observation-only probes."""

import unittest
from copy import deepcopy

from test_cp_agent import scenario, unit

from civ5sim import Action, Battle
from civ5sim.greedy_agent import GreedyAgent


class GreedyApproachTests(unittest.TestCase):
    def agent(self, battle, side):
        agent = GreedyAgent()
        agent.reset({"map": battle._map, "side": side})
        return agent

    def assert_move_then_attack(self, battle, side, kind):
        agent = self.agent(battle, side)
        obs, actions = battle.observe(side), tuple(battle.available_actions())
        self.assertFalse([a for a in actions if a.kind in ("melee", "ranged")])
        before, original_obs = battle.snapshot(), deepcopy(obs)
        move = agent.act(obs, actions)
        self.assertEqual(before, battle.snapshot())
        self.assertEqual(original_obs, obs)
        self.assertIn(move, actions)
        self.assertEqual(move.kind, "move")
        self.assertTrue(battle.step(move).ok)
        attack = agent.act(battle.observe(side), tuple(battle.available_actions()))
        self.assertEqual(attack.kind, kind)
        self.assertEqual(attack.unit, move.unit)
        self.assertTrue(battle.step(attack).ok)
        return move, attack

    def test_melee_approach_avoids_spending_last_movement_in_zone_of_control(self):
        with Battle(scenario([unit("warrior", 0, 53), unit("chariot_archer", 1, 39)],
                             width=16, height=16)) as battle:
            self.assert_move_then_attack(battle, 0, "melee")

    def test_ranged_approach_on_both_sides(self):
        for side in (0, 1):
            with self.subTest(side=side):
                value = scenario([unit("archer", side, 17), unit("warrior", 1-side, 20)],
                                 width=8, height=5)
                value["active_side"] = side
                with Battle(value) as battle:
                    self.assert_move_then_attack(battle, side, "ranged")

    def test_immediate_attack_precedes_move_attack_without_trade_evaluation(self):
        with Battle(scenario([unit("warrior", 0, 17, hp=1), unit("horseman", 1, 18),
                              unit("horseman", 0, 28)])) as battle:
            agent = self.agent(battle, 0)
            actions = tuple(battle.available_actions())
            self.assertIn(Action("melee", 0, 18), actions)
            before = battle.snapshot()
            attack = agent.act(battle.observe(0), actions)
            self.assertEqual(before, battle.snapshot())
            self.assertEqual(attack, Action("melee", 0, 18))
            self.assertTrue(battle.step(attack).ok)

    def test_hidden_enemy_changes_do_not_affect_sparse_id_approach(self):
        choices = []
        observations = []
        for hidden_tile in (240, 255):
            value = scenario([unit("warrior", 0, hidden_tile),
                              unit("archer", 1, 17),
                              unit("warrior", 0, 255-hidden_tile, hp=0),
                              unit("horseman", 1, 35),
                              unit("warrior", 0, 20)], width=16, height=16, fog=True)
            value["active_side"] = 1
            with Battle(value) as battle:
                obs = battle.observe(1)
                self.assertNotIn(0, [u["id"] for u in obs["units"]])
                self.assertNotIn(2, [u["id"] for u in obs["units"]])
                self.assertIn(4, [u["id"] for u in obs["units"]])
                observations.append(obs)
                choices.append(self.assert_move_then_attack(battle, 1, "ranged"))
        self.assertEqual(observations[0], observations[1])
        self.assertEqual(choices[0], choices[1])


if __name__ == "__main__":
    unittest.main()
