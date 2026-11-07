"""Silent experimental risks: altered RNG/history, masks, and temperature cache collisions."""

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from civ5rl.core.agent import RLAgent, action_probabilities
from civ5rl.core.context import context_for
from civ5rl.core.model import Policy
from civ5rl.core.observation import ObservationEncoder
from civ5rl.experiments.target_bc_sampling import completed_case, mode_key
from civ5rl.protocols.common import flat_map
from civ5rl.provenance.common import choose, configure, write_json
from civ5sim import Action, Battle


class SamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(17, "cpu")

    def test_temperature_one_seeded_choices_mask_and_concentration(self):
        # A normal battle could conceal a different RNG mapping behind a similar W/D/L.
        logits = torch.tensor([[2.0, -float("inf"), 0.3, -1.2, 2.0]])
        before, after = random.Random(42), random.Random(42)
        old = logits.softmax(-1)[0].numpy()
        new = action_probabilities(logits, 1)[0].numpy()
        np.testing.assert_array_equal(old, new)
        self.assertEqual(
            [choose(old, before) for _ in range(256)],
            [choose(new, after) for _ in range(256)],
        )
        self.assertEqual(before.getstate(), after.getstate())
        cold = action_probabilities(logits, 0.5)[0]
        self.assertEqual(float(cold[1]), 0)
        self.assertEqual(int(cold.argmax()), int(logits[0].argmax()))
        self.assertGreater(float(cold.max()), float(new.max()))
        torch.testing.assert_close(cold, (logits / 0.5).softmax(-1)[0], rtol=0, atol=0)

    def test_metrics_preserve_rng_executed_history_and_reset(self):
        # These bugs do not throw; they contaminate later decisions and episodes.
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
                {"type": "archer", "side": 1, "tile": 88},
            ],
        }
        context = context_for(scenario, [["warrior"], ["archer"]], 0, 30, 42)
        policy = Policy("C")
        plain = RLAgent(policy)
        measured = RLAgent(policy, collect_selection_metrics=True)
        for agent in (plain, measured):
            agent.reset(context)
        with Battle(scenario) as battle:
            obs, actions = battle.observe(0), battle.available_actions()
            first = plain.act(obs, actions)
            self.assertEqual(first, measured.act(obs, actions))
            self.assertEqual(plain.rng.getstate(), measured.rng.getstate())
            torch.testing.assert_close(plain.hidden, measured.hidden, rtol=0, atol=0)
            expected = ObservationEncoder(context)
            expected.remember_action(first, obs)
            np.testing.assert_array_equal(measured.encoder.previous, expected.previous)
            self.assertEqual(len(measured.selection_metrics), 1)
            forced = Action("end_turn")
            for agent in (plain, measured):
                rng_before = agent.rng.getstate()
                agent.act(obs, [forced])
                expected_rng = random.Random()
                expected_rng.setstate(rng_before)
                expected_rng.random()
                self.assertEqual(agent.rng.getstate(), expected_rng.getstate())
            self.assertEqual(len(measured.selection_metrics), 1)
            expected.remember_action(forced, obs)
            np.testing.assert_array_equal(measured.encoder.previous, expected.previous)
            rng_before = measured.rng.getstate()
            measured.observe_step(obs, actions, forced)
            self.assertEqual(measured.rng.getstate(), rng_before)
            self.assertEqual(len(measured.selection_metrics), 1)
            measured.reset(context)
            self.assertIsNone(measured.hidden)
            self.assertEqual(measured.selection_metrics, [])
            self.assertEqual(first, measured.act(obs, actions))
            argmax = RLAgent(
                policy, mode="argmax", temperature=0.5, collect_selection_metrics=True
            )
            argmax.reset(context)
            rng_before = argmax.rng.getstate()
            argmax.act(obs, actions)
            self.assertEqual(argmax.rng.getstate(), rng_before)
            self.assertTrue(argmax.selection_metrics[0]["argmax_selected"])

    def test_cached_match_cannot_cross_temperature_or_accept_partial_artifacts(self):
        # Weight-only caching would silently reuse the wrong policy behavior.
        identity = {"checkpoint_sha256": "parent", "ordered_case_hashes": ["case"]}
        one = mode_key(identity, {"mode": "sample", "temperature": 1.0})
        half = mode_key(identity, {"mode": "sample", "temperature": 0.5})
        self.assertNotEqual(one, half)
        self.assertNotEqual(
            one, mode_key(identity, {"mode": "argmax", "temperature": 1.0})
        )
        case = {"case_hash": "case"}
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            write_json(out / "case.json", case)
            write_json(
                out / "metrics.json",
                {"status": "completed", "evaluation_key": one, "case_hash": "case"},
            )
            for name in (
                "result.json",
                "replay.json",
                "final.json",
                "selection-metrics.json",
            ):
                write_json(out / name, {"status": "completed"})
            self.assertTrue(completed_case(out, case, one))
            self.assertFalse(completed_case(out, case, half))
            (out / "replay.json").unlink()
            self.assertFalse(completed_case(out, case, one))


if __name__ == "__main__":
    unittest.main()
