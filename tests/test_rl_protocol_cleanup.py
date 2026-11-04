import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from civ5rl.protocols.diagnostic import _minimum_shot_damage
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
        self.assertEqual(set(rolls), set(range(1201)))
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

    def test_pipeline_migration_rejects_each_unsupported_change(self):
        old = {
            "collection": {
                "num_envs": 4,
                "cpu_workers": 2,
                "envs_per_worker": 2,
                "deterministic_episode_waves": 4,
            },
            "fixed": 1,
        }
        identity = {"source_hashes": {}, "environment": "a"}
        changed_config = deepcopy(old)
        changed_config["fixed"] = 2
        changed_workers = deepcopy(old)
        changed_workers["collection"]["num_envs"] = 3
        cases = (
            (changed_config, identity),
            (changed_workers, identity),
            (old, {**identity, "environment": "b"}),
            (old, {**identity, "source_hashes": {"forbidden": "x"}}),
        )
        for config, changed_identity in cases:
            with (
                self.subTest(config=config, identity=changed_identity),
                self.assertRaises(ValueError),
            ):
                validate_pipeline_migration(old, config, identity, changed_identity)


if __name__ == "__main__":
    unittest.main()
