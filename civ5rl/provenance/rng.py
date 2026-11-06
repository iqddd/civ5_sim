"""Capture and restore random streams without importing torch at module import."""

import random

import numpy as np


def capture_rng(shuffle_rng=None):
    import torch

    state = {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if shuffle_rng is not None:
        state["shuffle_rng"] = shuffle_rng.getstate()
    return state


def restore_rng(state, shuffle_rng=None):
    import torch

    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda_rng"]])
    if shuffle_rng is not None:
        shuffle_rng.setstate(state["shuffle_rng"])
