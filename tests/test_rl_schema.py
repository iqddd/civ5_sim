import unittest

import numpy as np
from civ5rl.observation import ObservationEncoder, ordered_actions
from civ5rl.schema import (
    ACTION,
    ACTION_DIM,
    ACTION_INPUT_DIM,
    CONTEXT_INPUT_DIM,
    GLOBAL,
    GLOBAL_DIM,
    PREV_ACTION_DIM,
    PREVIOUS,
    TILE,
    TILE_DIM,
)
from test_rl import context, scenario

from civ5sim import Action, Battle


class SchemaTests(unittest.TestCase):
    def test_layouts_cover_all_columns_without_overlap(self):
        for layout, dim in (
            (TILE, TILE_DIM),
            (GLOBAL, GLOBAL_DIM),
            (PREVIOUS, PREV_ACTION_DIM),
            (ACTION, ACTION_DIM),
        ):
            columns = []
            for item in layout.values():
                columns.extend(
                    range(item.start, item.stop) if isinstance(item, slice) else [item]
                )
            self.assertEqual(columns, list(range(dim)))
        self.assertEqual(
            (TILE_DIM, GLOBAL_DIM, PREV_ACTION_DIM, ACTION_DIM), (73, 33, 20, 17)
        )
        self.assertEqual((CONTEXT_INPUT_DIM, ACTION_INPUT_DIM), (309, 529))

    def test_encoding_feature_semantics(self):
        s = scenario()
        with Battle(s) as battle:
            obs, actions = (
                battle.observe(0),
                ordered_actions(battle.available_actions()),
            )
        encoder = ObservationEncoder(context(s))
        encoder.remember_action(Action("skip", 0), obs)
        encoded = encoder.encode(obs, actions)
        self.assertEqual(encoded.cells.dtype, np.float32)
        self.assertEqual(encoded.neighbors.dtype, np.int64)
        np.testing.assert_array_equal(encoded.previous[[4, 8, 18]], [1, 1, 1])
        self.assertEqual(encoded.cells[11, TILE["own_type"]].sum(), 1)
        self.assertEqual(encoded.cells[11, TILE["own_status"]][0], 1)
        self.assertEqual(encoded.globals[GLOBAL["own_roster"]].sum(), 0.25)
