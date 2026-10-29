"""Integration checks against the locally installed original scripts and SDK."""

import unittest
from pathlib import Path

from civ5map.maps import content_hash, select_crops, validate
from civ5map.native import Native
from civ5map.runtime import World, generate

GAME = Path("E:/civ5/Sid Meier's Civilization V")
AVAILABLE = GAME.exists() and Path("build/fractal.dll").exists()


@unittest.skipUnless(AVAILABLE, "Requires local Civ V assets and built SDK fractal")
class GenerationTests(unittest.TestCase):
    def test_rng_matches_sdk_unsigned_32bit_sequence(self):
        n = Native(42)
        try:
            seed = 42
            for limit in (0, 1, 2, 256, 65535) * 30:
                seed = (1103515245 * seed + 12345) & 0xFFFFFFFF
                expected = ((seed >> 16) * limit) // 65536
                self.assertEqual(n.rand(limit), expected)
        finally:
            n.close()

    def test_same_seed_repeats_and_other_seeds_differ(self):
        first = validate(generate(GAME, 42))
        self.assertEqual(
            content_hash(first), content_hash(validate(generate(GAME, 42)))
        )
        self.assertNotEqual(
            content_hash(first), content_hash(validate(generate(GAME, 43)))
        )
        self.assertEqual(len(select_crops(first, 10, 42)), 10)
        self.assertTrue(any(any(f >= 0 for f in t["rivers"]) for t in first["tiles"]))
        self.assertTrue(any(t["feature"] == "forest" for t in first["tiles"]))

    def test_options_and_small_world(self):
        a = validate(generate(GAME, 7, size=0, options=(1, 3, 3, 1, 2)))
        b = validate(generate(GAME, 7, size=0, options=(3, 1, 1, 3, 2)))
        self.assertEqual((a["width"], a["height"]), (40, 24))
        self.assertNotEqual(content_hash(a), content_hash(b))

    def test_river_incidence_and_feature_requirements(self):
        w = World(GAME)
        try:
            p = w.get(10, 10)
            n = w.direction(10, 10, 1)
            p.SetPlotType(2, False)
            n.SetPlotType(2, False)
            p.SetWOfRiver(True, 0)
            self.assertTrue(n.IsRiver())
            p.SetTerrainType(w.types["TERRAIN_DESERT"])
            self.assertTrue(p.CanHaveFeature(w.types["FEATURE_FLOOD_PLAINS"]))
            p.SetPlotType(1, False)
            self.assertFalse(p.CanHaveFeature(w.types["FEATURE_FLOOD_PLAINS"]))
        finally:
            w.native.close()


if __name__ == "__main__":
    unittest.main()
