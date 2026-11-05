"""Shared protocol statistics and checkpointed stage transitions."""

from copy import deepcopy

import numpy as np


def bootstrap_ci95(values, *, samples=10000, seed=20261103):
    values = np.asarray(values)
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(len(values), size=(samples, len(values)))].mean(1)
    return np.quantile(draws, [0.025, 0.975])


def paired_layout_deltas(rows, baseline, *, layouts=64):
    def paired(values):
        result = {(r["layout_id"], r["side"]): int(r["win"]) for r in values}
        if len(values) != 2 * layouts or len(result) != 2 * layouts:
            raise ValueError(f"Bootstrap requires {2 * layouts} unique paired games")
        return result

    a, b = paired(rows), paired(baseline)
    if a.keys() != b.keys():
        raise ValueError("Bootstrap cases differ")
    layout_ids = sorted({k[0] for k in a})
    if len(layout_ids) != layouts or any(
        (i, side) not in a for i in layout_ids for side in (0, 1)
    ):
        raise ValueError(f"Bootstrap requires {layouts} complete layouts")
    return np.asarray(
        [sum(a[i, side] - b[i, side] for side in (0, 1)) / 2 for i in layout_ids]
    )


def advance_stage_state(
    state, passed, *, update_field, update_limit, success, failure, time_limit=None
):
    value = deepcopy(state)
    value["consecutive"] = value.get("consecutive", 0) + 1 if passed else 0
    if value["consecutive"] >= 2:
        value.update(success)
    elif value[update_field] >= update_limit:
        value.update(failure)
    elif time_limit is not None and value["training_seconds"] >= time_limit[0]:
        value.update(time_limit[1])
    return value
