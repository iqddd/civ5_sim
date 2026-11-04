"""Behavioral checks for forced history and scenario sampling."""

import unittest

import torch
from civ5rl.agent import RLAgent
from civ5rl.context import context_for
from civ5rl.memory import flat_map
from civ5rl.model import Policy

from civ5sim import Action, Battle


def fixture():
    scenario = {
        "schema": 1,
        "kind": "civ5sim.scenario",
        "map": flat_map(10),
        "fog": True,
        "random_damage": True,
        "seed": 17,
        "active_side": 0,
        "units": [
            {"type": "warrior", "side": 0, "tile": 11},
            {"type": "archer", "side": 0, "tile": 21},
            {"type": "horseman", "side": 1, "tile": 88},
            {"type": "archer", "side": 1, "tile": 98},
        ],
    }
    return scenario, context_for(
        scenario, [["warrior", "archer"], ["horseman", "archer"]], 0, 5, 42
    )


class CoreCleanupTests(unittest.TestCase):
    def test_forced_step_updates_history_without_sampling_or_events(self):
        scenario, context = fixture()
        agent = RLAgent(Policy("C"))
        agent.reset(context)
        with Battle(scenario) as battle:
            observation, actions = battle.observe(0), battle.available_actions()
            chosen = Action("end_turn")
            rng = agent.rng.getstate()
            agent.observe_step(observation, actions, chosen)
            self.assertEqual(agent.rng.getstate(), rng)
            self.assertEqual(agent.drain_diagnostics(), [])
            self.assertIsNotNone(agent.hidden)
            before = agent.hidden.clone()
            with self.assertRaisesRegex(ValueError, "not legal"):
                agent.observe_step(observation, actions, Action("move", 999, 999))
            self.assertTrue(torch.equal(agent.hidden, before))
            self.assertEqual(agent.rng.getstate(), rng)
