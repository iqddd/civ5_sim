"""CPU-safe description of the arrays in one encoded decision."""

from typing import NamedTuple

import numpy as np

from civ5rl.core.schema import (
    ACTION_DIM,
    GLOBAL_DIM,
    HEX_DIRECTIONS,
    PREV_ACTION_DIM,
    TILE_DIM,
)


class EncodedField(NamedTuple):
    name: str
    dtype: np.dtype
    axis: str
    trailing_shape: tuple[int, ...]


ENCODED_FIELDS = (
    EncodedField("cells", np.dtype(np.float32), "cells", (TILE_DIM,)),
    EncodedField("globals", np.dtype(np.float32), "fixed", (GLOBAL_DIM,)),
    EncodedField("previous", np.dtype(np.float32), "fixed", (PREV_ACTION_DIM,)),
    EncodedField("neighbors", np.dtype(np.int64), "cells", (HEX_DIRECTIONS,)),
    EncodedField("sources", np.dtype(np.int64), "actions", ()),
    EncodedField("targets", np.dtype(np.int64), "actions", ()),
    EncodedField("action_features", np.dtype(np.float32), "actions", (ACTION_DIM,)),
)
