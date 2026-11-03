"""Reinforcements choose local targets without paying extra route distance."""

import unittest

from test_cp_agent import SMALL, scenario, unit

from civ5sim import Battle
from civ5sim.combat_ai import LIBRARY_PATH
from civ5sim.cp_agent import CPAgent, hex_distance


@unittest.skipUnless(LIBRARY_PATH.exists(), "Build combat and AI libraries first")
class ApproachSelectionTests(unittest.TestCase):
    def setup_scenario(self, own_tiles, enemy_tiles, *, width=16, height=4, barrier=()):
        value = scenario([unit("warrior", 0, t) for t in own_tiles]
                         + [unit("warrior", 1, t) for t in enemy_tiles],
                         width=width, height=height)
        for tile in barrier:
            value["map"]["tiles"][tile]["elevation"] = "mountain"
        self.battle = self.enterContext(Battle(value))
        self.agent = CPAgent(SMALL)
        self.agent.reset({"side": 0, "map": self.battle._map})
        self.addCleanup(self.agent.close)
        self.agent._phase = 3
        self.obs = self.battle.observe(0)
        self.own = [u for u in self.obs["units"] if u["side"] == 0]
        self.enemies = [u for u in self.obs["units"] if u["side"] == 1]

    def group(self):
        before = self.battle.snapshot()
        result = self.agent._approach_group(self.own, self.enemies, self.obs)
        self.assertEqual(self.battle.snapshot(), before, "Planning changed battle or RNG")
        return result

    def test_separate_fronts_keep_their_own_reinforcements(self):
        self.setup_scenario([1, 2, 13], [0, 15])
        first = self.agent._next_task(self.own, self.enemies, self.obs)
        self.assertEqual((first["target"], first["selected"]), (0, (0, 1)))
        second = self.agent._next_task(self.own, self.enemies, self.obs)
        self.assertEqual((second["target"], second["selected"]), (15, (2,)))
        self.assertIsNone(self.agent._next_task(self.own, self.enemies, self.obs))

    def test_equal_nearest_targets_join_the_larger_group(self):
        self.setup_scenario([1, 4], [8, 0])
        self.assertEqual(self.group(), (0, (0, 1)))
        event = self.agent.drain_diagnostics()[-1]
        self.assertEqual(event["policy"], "nearest_per_unit")
        self.assertEqual(event["units"][1]["nearest_enemies"], [2, 3])
        self.assertEqual(event["candidates"][0]["distance_sum"], 5)
        # Input iteration order cannot resolve a tie differently.
        self.own.reverse()
        self.enemies.reverse()
        self.assertEqual(self.group(), (0, (0, 1)))

    def test_group_ties_use_distance_sum_then_enemy_id(self):
        self.setup_scenario([3, 14], [0, 15])
        self.assertEqual(self.group(), (15, (1,)))
        self.own[0]["tile"] = 1
        self.assertEqual(self.group(), (0, (0,)))

    def test_barrier_detour_overrides_hex_proximity(self):
        self.setup_scenario([2], [4, 16], width=8, height=5,
                            barrier=[y * 8 + 3 for y in range(4)])
        self.assertLess(hex_distance(self.agent.map, 2, 4),
                        hex_distance(self.agent.map, 2, 16))
        self.assertGreater(self.agent._distance(2, 4), self.agent._distance(2, 16))
        self.assertEqual(self.group(), (16, (0,)))

    def test_unreachable_target_cannot_recruit_across_a_barrier(self):
        self.setup_scenario([2], [4, 16], width=8, height=5,
                            barrier=[y * 8 + 3 for y in range(5)])
        self.assertEqual(self.agent._distance(2, 4), 1_000_000)
        self.assertEqual(self.group(), (16, (0,)))
        self.agent._approach_attempts.add((0, 16))
        self.assertIsNone(self.group())

    def test_only_allowed_targets_enter_the_per_unit_minimum(self):
        self.setup_scenario([2], [0, 15])
        self.agent._target_blocked[0, 1] = 0
        self.assertEqual(self.group(), (15, (0,)))
        self.agent._approach_attempts.add((0, 15))
        self.assertIsNone(self.group())
        self.assertEqual(self.agent.drain_diagnostics()[-1]["reason"], "no_allowed_target")

    def test_thirteen_unit_limit_leaves_the_rest_for_another_task(self):
        self.setup_scenario(list(range(1, 16)), [0], width=32, height=2)
        first = self.agent._next_task(self.own, self.enemies, self.obs)
        self.assertEqual(first["selected"], tuple(range(13)))
        second = self.agent._next_task(self.own, self.enemies, self.obs)
        self.assertEqual(second["selected"], (13, 14))

    def test_last_seen_positions_use_the_same_local_grouping(self):
        self.setup_scenario([1, 2, 13], [0, 15])
        self.agent._last_seen = {u["id"]: u["tile"] for u in self.enemies}
        self.enemies = []
        self.assertEqual(self.group(), (0, (0, 1)))
        self.assertEqual(self.agent.drain_diagnostics()[-1]["source"], "last_seen")
