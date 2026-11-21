"""Whole-turn branch identities shared by screening, confirmation and fitting."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from civ5rl.provenance.common import seed_for

NAMESPACE = "c2p_turn_comparison_v1"
SCREENING_VARIANTS = range(4)
SCREENING_REPETITIONS = range(4)
CONFIRMATION_REPETITIONS = range(4, 12)


def branch_spec(
    position: Mapping[str, Any], variant: int, repetition: int, source: str | Path
) -> dict[str, Any]:
    key = position["key"]
    return {
        "source": str(source),
        "command": position["command"],
        "turn_variant": variant,
        "repetition": repetition,
        "position_key": key,
        "seed": seed_for(17, key, repetition, namespace=NAMESPACE + "_damage") or 1,
        "sampling_seed": seed_for(17, key, variant, namespace=NAMESPACE + "_sampling"),
    }
