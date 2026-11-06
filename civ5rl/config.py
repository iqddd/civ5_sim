"""Resolve experiment configuration inheritance without changing JSON artifact I/O."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    """Load a config and recursively merge mappings; other values replace the base."""
    def load(path, stack):
        path = Path(path).resolve()
        if path in stack:
            raise ValueError(f"Config inheritance cycle: {path}")
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError(f"Config must be an object: {path}")
        base = value.pop("base_config", None)
        if base is None:
            return value
        if not isinstance(base, str):
            raise ValueError(f"base_config must be a path string: {path}")
        result = load(ROOT / base, (*stack, path))
        merge(result, value)
        return result

    def merge(base, override):
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                merge(base[key], value)
            else:
                base[key] = value

    return load(path, ())
