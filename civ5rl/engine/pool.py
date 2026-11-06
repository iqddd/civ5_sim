"""Spawn workers with the same environment as imitation collection."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from civ5rl.protocols.imitation import worker_setup


def pool(workers):
    return ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=worker_setup,
    )
