"""Behavioral checks for forced history and scenario sampling."""

import unittest
from unittest.mock import patch

import torch

from civ5rl.agent import RLAgent
from civ5rl.context import context_for
from civ5rl.corpus import ScenarioFactory
from civ5rl.diagnostic import DecisionEnvContext
from civ5rl.env import DecisionEnv
from civ5rl.memory import ForcedPrefix, flat_map
from civ5rl.model import Policy
from civ5sim import Action, Battle
from civ5sim.engine import digest


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

    def test_forced_prefix_rejects_illegal_without_advancing(self):
        scenario, context = fixture()
        prefix = ForcedPrefix(Policy("C"), "cpu", [Action("move", 999, 999)], True)
        prefix.reset(context)
        with (
            Battle(scenario) as battle,
            self.assertRaisesRegex(ValueError, "not legal"),
        ):
            prefix.act(battle.observe(0), battle.available_actions())
        self.assertEqual(prefix.index, 0)
        self.assertIsNone(prefix.agent.hidden)

    def test_context_manager_alias_closes_on_exception(self):
        self.assertIs(DecisionEnvContext, DecisionEnv)
        env = object.__new__(DecisionEnv)
        with (
            patch.object(env, "close") as close,
            self.assertRaisesRegex(RuntimeError, "stop"),
            env,
        ):
            raise RuntimeError("stop")
        close.assert_called_once_with()

    def test_seeded_scenarios_preserve_legacy_hashes(self):
        from civ5rl.observation import distance

        board = flat_map(10)
        board["spawn_candidates"] = [
            [[x, y] for x in range(3) for y in range(3)],
            [[x, y] for x in range(7, 10) for y in range(7, 10)],
        ]
        manifest = {
            "maps": [{"size": 10, "split": "train", "parent_seed": 7, "path": "dummy"}]
        }
        config = {
            "seed": 17,
            "corpus": {"placement_rejection_limit": 50},
            "environment": {"max_calls_per_turn": 100},
        }

        class FakeBattle:
            def __init__(self, scenario):
                self.scenario = scenario

            def __enter__(self):
                return self

            def __exit__(self, *_):
                pass

            def observe(self, side):
                units = self.scenario["units"]
                near = any(
                    distance(board, a["tile"], b["tile"]) <= 3
                    for a in units
                    if a["side"] == 0
                    for b in units
                    if b["side"] == 1
                )
                return {"units": [{"side": 1 - side}] if near else []}

        hashes = []
        with patch("civ5rl.corpus.Battle", FakeBattle):
            for contact in (0, 1):
                for seed in range(50):
                    stage = {
                        "map_size": [10],
                        "units_per_side": [2, 4],
                        "contact_probability": contact,
                        "max_rounds": 5,
                        "opponents": {"random": 0.2, "greedy": 0.8},
                    }
                    factory = ScenarioFactory(config, manifest)
                    factory.get_map = lambda _: board
                    hashes.append(digest(factory.make(seed, stage)))
        self.assertEqual(
            digest(hashes),
            "46c91505ff313d6177ff2548a3151001c63931a3a2afa4413ee6368633a24e44",
        )
