"""Shared protocol statistics and checkpointed stage transitions."""

import json
from collections import Counter
from copy import deepcopy

import numpy as np

from civ5rl.core.env import learner_outcome
from civ5sim.engine import digest

__all__ = [
    "CoverageTracker",
    "advance_stage_state",
    "bootstrap_ci95",
    "case_content_hash",
    "flat_map",
    "learner_outcome",
    "own_state_physical_hash",
    "paired_layout_deltas",
    "physical_hash",
    "require_config",
    "self_digest",
]


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
    if len(layout_ids) != layouts or any((i, side) not in a for i in layout_ids for side in (0, 1)):
        raise ValueError(f"Bootstrap requires {layouts} complete layouts")
    return np.asarray([sum(a[i, side] - b[i, side] for side in (0, 1)) / 2 for i in layout_ids])


def advance_stage_state(state, passed, *, update_field, update_limit, success, failure, time_limit=None):
    value = deepcopy(state)
    value["consecutive"] = value.get("consecutive", 0) + 1 if passed else 0
    if value["consecutive"] >= 2:
        value.update(success)
    elif value[update_field] >= update_limit:
        value.update(failure)
    elif time_limit is not None and value["training_seconds"] >= time_limit[0]:
        value.update(time_limit[1])
    return value


def flat_map(size=16, parity=0):
    return {
        "schema": 1,
        "width": size,
        "height": size,
        "row_parity": parity,
        "wrap_x": False,
        "tiles": [
            {
                "terrain": "grass",
                "elevation": "flat",
                "feature": None,
                "rivers": [-1, -1, -1],
            }
            for _ in range(size * size)
        ],
    }


def physical_hash(scenario):
    """Ignore RNG/unit ordering and identify side-permuted copies together."""
    variants = []
    for swap in (0, 1):
        units = [{**u, "side": u["side"] ^ swap} for u in scenario["units"]]
        units.sort(key=lambda u: json.dumps(u, sort_keys=True))
        variants.append(
            digest(
                {
                    "map": scenario["map"],
                    "units": units,
                    "fog": True,
                    "random_damage": True,
                }
            )
        )
    return min(variants)


def own_state_physical_hash(scenario):
    # Unlike physical_hash, hash only map digest and selected unit fields; keep its wire format.
    # Seeds, side-to-play and unit list order do not create a new physical layout.
    units = [(u["side"], u["type"], u["tile"], u.get("hp", 100)) for u in scenario["units"]]
    flipped = [(1 - s, k, t, hp) for s, k, t, hp in units]
    return digest({"map": digest(scenario["map"]), "units": min(sorted(units), sorted(flipped))})


def require_config(config, expected, prefix="Unsupported configuration: "):
    """Compare dotted fields using the protocol's missing-as-None semantics."""
    for field, expected_value in expected.items():
        value = config
        for key in field.split("."):
            value = value.get(key) if isinstance(value, dict) else None
        if value != expected_value:
            raise ValueError(f"{prefix}{field}")


def case_content_hash(case, key="case_hash"):
    return digest({name: value for name, value in case.items() if name != key})


self_digest = case_content_hash


class CoverageTracker:
    """Track issued physical layouts and insertion-ordered coverage counts."""

    def __init__(self):
        self.physical = set()
        self.duplicates = 0
        self.conditions = Counter()
        self.worlds = Counter()
        self.crops = Counter()

    def record(self, physical_hash, condition, world, crop):
        self.duplicates += physical_hash in self.physical
        self.physical.add(physical_hash)
        self.conditions[condition] += 1
        self.worlds[world] += 1
        self.crops[crop] += 1

    def summary(self):
        return {
            "episodes": sum(self.conditions.values()),
            "physical_unique": len(self.physical),
            "physical_repeats": self.duplicates,
            "conditions": dict(self.conditions),
            "worlds": dict(self.worlds),
            "crops": dict(self.crops),
            "historical_physical_overlap": "not_measured",
        }
