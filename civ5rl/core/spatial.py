"""Shared partitioning for fixed-size spatial inference and training."""

import torch

SPATIAL_KEYS = ("cells", "cell_mask", "neighbors", "globals", "previous")


def iter_spatial_blocks(batch, size):
    for start in range(0, len(batch["cells"]), size):
        part = {key: batch[key][start : start + size] for key in SPATIAL_KEYS}
        count = len(part["cells"])
        if count < size:
            part = {
                key: torch.cat(
                    (value, value[:1].expand(size - count, *value.shape[1:])), 0
                )
                for key, value in part.items()
            }
        yield part, count
