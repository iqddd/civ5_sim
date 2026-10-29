import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from civ5map.maps import (
    adjacent,
    components,
    content_hash,
    crop,
    load,
    save,
    select_crops,
    validate,
)


def field(w=24, h=24):
    return {
        "schema": 1,
        "width": w,
        "height": h,
        "row_parity": 0,
        "wrap_x": False,
        "metadata": {},
        "tiles": [
            {
                "terrain": "grass",
                "elevation": "flat",
                "feature": None,
                "rivers": [-1, -1, -1],
            }
            for _ in range(w * h)
        ],
    }


class MapTests(unittest.TestCase):
    def test_crop_preserves_all_internal_neighbors_at_both_parities(self):
        parent = field()
        for sy in (2, 3):
            m = crop(parent, 3, sy)
            for y in range(16):
                for x in range(16):
                    for d in range(6):
                        n = adjacent(m, x, y, d)
                        if n:
                            self.assertEqual(
                                adjacent(parent, x + 3, y + sy, d),
                                (n[0] + 3, n[1] + sy),
                            )
                            self.assertEqual(adjacent(m, *n, (d + 3) % 6), (x, y))

    def test_crop_is_independent_and_non_wrapping(self):
        m = field()
        m["wrap_x"] = True
        c = crop(m, 2, 3)
        c["tiles"][0]["rivers"][0] = 0
        self.assertEqual(m["tiles"][3 * 24 + 2]["rivers"][0], -1)
        self.assertIsNone(adjacent(c, 0, 0, 4))

    def test_river_from_outside_crop_and_nested_crop(self):
        m = field()
        m["tiles"][3 * 24 + 1]["rivers"][0] = 0
        c = crop(m, 2, 3)
        edge = {"x": 0, "y": 0, "direction": 4, "flow": 0}
        self.assertIn(edge, c["boundary_rivers"])
        self.assertIn(edge, crop(c, 0, 0, 8, 8)["boundary_rivers"])

    def test_barrier_splits_passable_land(self):
        m = field(8, 8)
        for y in range(8):
            m["tiles"][y * 8 + 4]["elevation"] = "mountain"
        self.assertEqual(sorted(map(len, components(m))), [24, 32])

    def test_selection_has_connected_two_sided_spawn_space(self):
        cuts = select_crops(field(48, 48), 10, 12)
        self.assertEqual(
            [content_hash(c) for c in cuts],
            [content_hash(c) for c in select_crops(field(48, 48), 10, 12)],
        )
        for c in cuts:
            largest = components(c)[0]
            for side in c["spawn_candidates"]:
                self.assertGreaterEqual(len(side), 10)
                self.assertTrue(all(y * 16 + x in largest for x, y in side))

    def test_impossible_selection_fails_explicitly(self):
        m = field()
        for t in m["tiles"]:
            t["terrain"], t["elevation"] = "ocean", "water"
        with self.assertRaisesRegex(ValueError, "Only 0"):
            select_crops(m)

    def test_roundtrip_and_invalid_data(self):
        m = crop(field(), 1, 1)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "m.json"
            save(m, path)
            self.assertEqual(load(path), m)
        bad = deepcopy(m)
        bad["tiles"][0]["rivers"] = [1, -1, -1]
        with self.assertRaises(ValueError):
            validate(bad)
        with self.assertRaises(ValueError):
            crop(m, 15, 15)


if __name__ == "__main__":
    unittest.main()
