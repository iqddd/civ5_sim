"""Single-producer encoded slots; ownership transfers at each Pipe reply/command.

Only player observations enter these buffers. The parent copies a slot before
dispatching its next command, so the worker cannot overwrite retained PPO input.
No torch import or CUDA context is needed in workers.
"""

from multiprocessing.shared_memory import SharedMemory

import numpy as np

from .observation import Encoded


class EncodedSlots:
    def __init__(self, slots=None, action_capacity=4096, *, descriptor=None):
        self.owner = descriptor is None
        if self.owner:
            descriptor = {"slots": slots, "action_capacity": action_capacity}
        slots, capacity = descriptor["slots"], descriptor["action_capacity"]
        layout = {
            "cells": ((slots, 256, 73), np.float32),
            "globals": ((slots, 33), np.float32),
            "previous": ((slots, 20), np.float32),
            "neighbors": ((slots, 256, 6), np.int64),
            "sources": ((slots, capacity), np.int64),
            "targets": ((slots, capacity), np.int64),
            "action_features": ((slots, capacity, 17), np.float32),
        }
        size = sum(
            int(np.prod(shape)) * np.dtype(dtype).itemsize
            for shape, dtype in layout.values()
        )
        self.memory = (
            SharedMemory(create=True, size=size)
            if self.owner
            else SharedMemory(name=descriptor["name"])
        )
        self.descriptor = {**descriptor, "name": self.memory.name}
        self.arrays = {}
        offset = 0
        for key, (shape, dtype) in layout.items():
            self.arrays[key] = np.ndarray(
                shape, dtype, buffer=self.memory.buf, offset=offset
            )
            offset += self.arrays[key].nbytes

    def write(self, slot, encoded):
        nc, na = len(encoded.cells), len(encoded.sources)
        if nc > 256 or na > self.descriptor["action_capacity"]:
            # Unusually large legal lists remain supported without truncation.
            return {"encoded_inline": encoded}
        for key, array in self.arrays.items():
            value = getattr(encoded, key)
            target = array[slot]
            if key not in ("globals", "previous"):
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
            if key in ("cells", "neighbors"):
                value = value[:nc]
            elif key not in ("globals", "previous"):
                value = value[:na]
            values[key] = value.copy()
        return Encoded(**values)

    def close(self):
        self.arrays.clear()
        self.memory.close()
        if self.owner:
            self.memory.unlink()
