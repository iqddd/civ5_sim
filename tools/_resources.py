"""Analysis of saved resource samples for CLI measurements."""

import json
from itertools import pairwise

def collection_resources(path, phase="collection"):
    if not path.exists():
        return {}
    samples = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    selected = [s for s in samples if s["phase"] == phase]
    result = {"samples": len(selected)}
    for name in ("gpu_core_load_percent", "cpu_tree_percent"):
        weighted = duration = 0.0
        for before, after in pairwise(samples):
            if (
                before["phase"] == after["phase"] == phase
                and before.get(name) is not None
            ):
                seconds = after["elapsed_seconds"] - before["elapsed_seconds"]
                weighted += before[name] * seconds
                duration += seconds
        result[f"{name}_mean"] = weighted / duration if duration else None
    for name in ("ram_tree_rss_bytes", "vram_used_bytes"):
        result[f"{name}_peak"] = max(
            (s[name] for s in selected if s.get(name) is not None), default=None
        )
    return result
