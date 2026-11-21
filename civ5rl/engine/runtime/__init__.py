"""Shared execution utilities for RL engines."""

from __future__ import annotations


def is_int_at_least(value: object, minimum: int = 1) -> bool:
    """Validate integer options without accepting booleans."""
    return type(value) is int and value >= minimum
