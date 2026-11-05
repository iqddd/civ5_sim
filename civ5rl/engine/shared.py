"""Single-producer encoded slots; ownership transfers at each Pipe reply/command.

Only player observations enter these buffers. The parent copies a slot before
dispatching its next command, so the worker cannot overwrite retained PPO input.
No torch import or CUDA context is needed in workers.
"""

from multiprocessing.shared_memory import SharedMemory

import numpy as np

from civ5rl.core.observation import Encoded
from civ5rl.core.schema import (
    ACTION_DIM,
    GLOBAL_DIM,
    HEX_DIRECTIONS,
    MAX_MAP_CELLS,
    PREV_ACTION_DIM,
    TILE_DIM,
)


class EncodedSlots:
    def __init__(self, slots=None, action_capacity=4096, *, descriptor=None):
        self.owner = descriptor is None
        if self.owner:
            descriptor = {"slots": slots, "action_capacity": action_capacity}
        slots, capacity = descriptor["slots"], descriptor["action_capacity"]
        layout = {
            "cells": ((slots, MAX_MAP_CELLS, TILE_DIM), np.float32, "cell"),
            "globals": ((slots, GLOBAL_DIM), np.float32, None),
            "previous": ((slots, PREV_ACTION_DIM), np.float32, None),
            "neighbors": ((slots, MAX_MAP_CELLS, HEX_DIRECTIONS), np.int64, "cell"),
            "sources": ((slots, capacity), np.int64, "action"),
            "targets": ((slots, capacity), np.int64, "action"),
            "action_features": ((slots, capacity, ACTION_DIM), np.float32, "action"),
        }
        self.axis_kind = {key: axis for key, (_, _, axis) in layout.items()}
        size = sum(
            int(np.prod(shape)) * np.dtype(dtype).itemsize
            for shape, dtype, _ in layout.values()
        )
        self.memory = (
            SharedMemory(create=True, size=size)
            if self.owner
            else SharedMemory(name=descriptor["name"])
        )
        self.arrays = {}
        try:
            self.descriptor = {**descriptor, "name": self.memory.name}
            offset = 0
            for key, (shape, dtype, _) in layout.items():
                self.arrays[key] = np.ndarray(
                    shape, dtype, buffer=self.memory.buf, offset=offset
                )
                offset += self.arrays[key].nbytes
        except BaseException:
            self.arrays.clear()
            self.memory.close()
            if self.owner:
                self.memory.unlink()
            raise

    def write(self, slot, encoded):
        nc, na = len(encoded.cells), len(encoded.sources)
        if nc > MAX_MAP_CELLS or na > self.descriptor["action_capacity"]:
            # Unusually large legal lists remain supported without truncation.
            return {"encoded_inline": encoded}
        for key, array in self.arrays.items():
            value = getattr(encoded, key)
            target = array[slot]
            if self.axis_kind[key] is not None:
                target = target[: len(value)]
            target[...] = value
        return {"encoded_shape": (nc, na)}

    def read(self, slot, state):
        if "encoded_inline" in state:
            return state["encoded_inline"]
        nc, na = state["encoded_shape"]
        values = {}
        for key, array in self.arrays.items():
            value = array[slot]
            axis = self.axis_kind[key]
            if axis is not None:
                value = value[: nc if axis == "cell" else na]
            values[key] = value.copy()
        return Encoded(**values)

    def close(self):
        self.arrays.clear()
        self.memory.close()
        if self.owner:
            self.memory.unlink()
