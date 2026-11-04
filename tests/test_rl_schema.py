import unittest

import numpy as np
import torch
from test_rl import context, scenario

from civ5rl.model import Policy
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
from civ5rl.shared import EncodedSlots
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

    def test_encoding_and_shared_round_trip(self):
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
        slots = EncodedSlots(1, action_capacity=len(actions))
        try:
            result = slots.read(0, slots.write(0, encoded))
            for name in encoded.__dataclass_fields__:
                np.testing.assert_array_equal(
                    getattr(encoded, name), getattr(result, name)
                )
        finally:
            slots.close()

    def test_checkpoint_shapes_and_keys(self):
        model = Policy()
        shapes = {key: tuple(value.shape) for key, value in model.state_dict().items()}
        expected = {
            "projection.weight": (128, 73),
            "projection.bias": (128,),
            "context.0.weight": (256, 309),
            "memory.weight_ih_l0": (768, 256),
            "memory.weight_hh_l0": (768, 256),
            "action.0.weight": (128, 529),
            "value.0.weight": (128, 256),
        }
        for key, shape in expected.items():
            self.assertEqual(shapes[key], shape)
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1375362)
        self.assertEqual((model.hidden_size, model.hex_dim), (256, 128))
        model.load_state_dict(
            {k: torch.zeros_like(v) for k, v in model.state_dict().items()}, strict=True
        )
