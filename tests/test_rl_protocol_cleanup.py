import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from civ5rl.protocols.diagnostic import _minimum_shot_damage
from civ5rl.protocols.evaluation import _wdl_stats
from civ5rl.provenance.migration import validate_pipeline_migration


class ProtocolCleanupTests(unittest.TestCase):
    def test_damage_checks_sentinel_and_every_roll(self):
        rolls = []

        def damage(attacker, defender, roll):
            self.assertEqual((attacker, defender), (17, 23))
            rolls.append(roll)
            return -1 if roll == 1200 else 2 + roll % 7

        with patch(
            "civ5rl.protocols.diagnostic.native.library",
            return_value=SimpleNamespace(sim_damage=damage),
        ):
            self.assertEqual(
                _minimum_shot_damage(
                    SimpleNamespace(attacker_strength=17, defender_strength=23),
                    "changed",
                ),
                2,
            )
        self.assertEqual(rolls, [1200, *range(1200)])
        with (
            patch(
                "civ5rl.protocols.diagnostic.native.library",
                return_value=SimpleNamespace(sim_damage=lambda *args: 0),
            ),
            self.assertRaisesRegex(ValueError, "changed"),
        ):
            _minimum_shot_damage(
                SimpleNamespace(attacker_strength=17, defender_strength=23),
                "changed",
            )

    def test_wdl_counts_and_empty_side(self):
        self.assertEqual(
            _wdl_stats(np.asarray([1, 0, -1, 1])),
            {"games": 4, "wins": 2, "draws": 1, "losses": 1, "score": 0.625},
        )
        self.assertEqual(
            _wdl_stats(np.asarray([])),
            {"games": 0, "wins": 0, "draws": 0, "losses": 0, "score": None},
        )

    def test_pipeline_validation_order(self):
        old = {
            "collection": {
                "num_envs": 4,
                "cpu_workers": 2,
                "envs_per_worker": 2,
                "deterministic_episode_waves": 4,
            },
            "fixed": 1,
        }
        bad = {**old, "fixed": 2, "collection": {**old["collection"], "num_envs": 3}}
        identity = {"source_hashes": {}, "environment": "a"}
        other = {"source_hashes": {"forbidden": "x"}, "environment": "b"}
        with self.assertRaisesRegex(ValueError, "learning rules"):
            validate_pipeline_migration(old, bad, identity, other)
        bad["fixed"] = 1
        with self.assertRaisesRegex(ValueError, "worker/wave"):
            validate_pipeline_migration(old, bad, identity, other)
        with self.assertRaisesRegex(ValueError, "corpus/environment"):
            validate_pipeline_migration(old, old, identity, other)
        other["environment"] = "a"
        with self.assertRaisesRegex(ValueError, "model/simulator"):
            validate_pipeline_migration(old, old, identity, other)


if __name__ == "__main__":
    unittest.main()
