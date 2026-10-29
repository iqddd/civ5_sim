"""Versioned terrain files, parity-preserving crops and scenario selection."""

import hashlib
import json
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path

from .runtime import neighbor


def validate(m):
    if m.get("schema") != 1:
        raise ValueError("Unsupported map schema")
    w, h = m["width"], m["height"]
    if type(w) is not int or type(h) is not int or min(w, h) < 1:
        raise ValueError("Invalid dimensions")
    if len(m["tiles"]) != w * h or m["row_parity"] not in (0, 1):
        raise ValueError("Invalid tile count or row parity")
    if type(m["wrap_x"]) is not bool:
        raise ValueError("wrap_x must be boolean")
    for t in m["tiles"]:
        if t["terrain"] not in (
            "grass",
            "plains",
            "desert",
            "tundra",
            "snow",
            "coast",
            "ocean",
        ):
            raise ValueError("Unknown terrain")
        if t["elevation"] not in ("flat", "hills", "mountain", "water"):
            raise ValueError("Unknown elevation")
        if (t["elevation"] == "water") != (t["terrain"] in ("coast", "ocean")):
            raise ValueError("Inconsistent terrain and elevation")
        if t["feature"] is not None and not isinstance(t["feature"], str):
            raise ValueError("Invalid feature")
        if len(t["rivers"]) != 3 or any(
            type(f) is not int or f not in allowed
            for f, allowed in zip(t["rivers"], ((-1, 0, 3), (-1, 1, 4), (-1, 2, 5)))
        ):
            raise ValueError("Invalid river flow for E/SE/SW edge")
    for edge in m.get("boundary_rivers", []):
        if not (
            0 <= edge["x"] < w and 0 <= edge["y"] < h and edge["direction"] in (0, 4, 5)
        ):
            raise ValueError("Invalid boundary river")
        if adjacent(m, edge["x"], edge["y"], edge["direction"]) is not None:
            raise ValueError("Boundary river is not on the boundary")
        allowed = {0: (2, 5), 4: (0, 3), 5: (1, 4)}[edge["direction"]]
        if edge["flow"] not in allowed:
            raise ValueError("Invalid boundary river flow")
    return m


def save(m, path):
    validate(m)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(m, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load(path):
    return validate(json.loads(Path(path).read_text(encoding="utf-8")))


def content_hash(m):
    payload = {
        k: m[k] for k in ("schema", "width", "height", "row_parity", "wrap_x", "tiles")
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def adjacent(m, x, y, d):
    parity = m["row_parity"]
    nx, ny = neighbor(x, y + parity, d)
    ny -= parity
    if m["wrap_x"]:
        nx %= m["width"]
    return (nx, ny) if 0 <= nx < m["width"] and 0 <= ny < m["height"] else None


def crop(m, x, y, width=16, height=16):
    validate(m)
    if not (
        width > 0
        and height > 0
        and 0 <= x <= m["width"] - width
        and 0 <= y <= m["height"] - height
    ):
        raise ValueError("Crop must fit inside the parent map (no seam crossing)")
    result = {
        "schema": 1,
        "width": width,
        "height": height,
        "row_parity": (m["row_parity"] + y) % 2,
        "wrap_x": False,
        "metadata": deepcopy(m.get("metadata", {})),
        "tiles": [
            deepcopy(m["tiles"][(y + j) * m["width"] + x + i])
            for j in range(height)
            for i in range(width)
        ],
    }
    result["metadata"]["crop"] = {"x": x, "y": y, "parent_hash": content_hash(m)}
    # Rivers remain on owned E/SE/SW edges, including the outer boundary.
    # Keep opposite boundary edges separately so no visible river is lost.
    boundary = []
    for j in range(height):
        for i in range(width):
            for d, edge in ((0, 2), (4, 0), (5, 1)):
                if adjacent(result, i, j, d) is not None:
                    continue
                source = adjacent(m, x + i, y + j, d)
                if source is not None:
                    sx, sy = source
                    flow = m["tiles"][sy * m["width"] + sx]["rivers"][edge]
                    if flow >= 0:
                        boundary.append({"x": i, "y": j, "direction": d, "flow": flow})
                else:
                    for edge in m.get("boundary_rivers", []):
                        if (edge["x"], edge["y"], edge["direction"]) == (
                            x + i,
                            y + j,
                            d,
                        ):
                            boundary.append(
                                {"x": i, "y": j, "direction": d, "flow": edge["flow"]}
                            )
    result["boundary_rivers"] = boundary
    return validate(result)


def passable(tile):
    return tile["elevation"] in ("flat", "hills") and tile["feature"] != "ice"


def components(m):
    remaining = {i for i, t in enumerate(m["tiles"]) if passable(t)}
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        found, stack = {start}, [start]
        while stack:
            i = stack.pop()
            for d in range(6):
                n = adjacent(m, i % m["width"], i // m["width"], d)
                if n is None:
                    continue
                ni = n[1] * m["width"] + n[0]
                if ni in remaining:
                    remaining.remove(ni)
                    found.add(ni)
                    stack.append(ni)
        result.append(found)
    return sorted(result, key=lambda c: (-len(c), min(c)))


def summary(m):
    c = components(m)
    return {
        "terrain": dict(Counter(t["terrain"] for t in m["tiles"])),
        "features": dict(Counter(t["feature"] or "none" for t in m["tiles"])),
        "elevation": dict(Counter(t["elevation"] for t in m["tiles"])),
        "river_edges": sum(sum(f >= 0 for f in t["rivers"]) for t in m["tiles"]),
        "passable": sum(map(len, c)),
        "largest_component": len(c[0]) if c else 0,
    }


def select_crops(m, count=10, seed=42, size=16, min_connected=0.65, army_size=10):
    if not (count > 0 and size >= 4 and 0 < min_connected <= 1 and army_size > 0):
        raise ValueError("Invalid crop selection settings")
    candidates = [
        (x, y)
        for y in range(m["height"] - size + 1)
        for x in range(m["width"] - size + 1)
    ]
    random.Random(seed).shuffle(candidates)
    selected, origins = [], []
    for x, y in candidates:
        # Spread the samples, but permit overlap; split datasets by parent map.
        if any(abs(x - ox) + abs(y - oy) < size // 2 for ox, oy in origins):
            continue
        c = crop(m, x, y, size, size)
        groups = components(c)
        if not groups or len(groups[0]) < size * size * min_connected:
            continue
        group = groups[0]
        sides = [
            [i for i in sorted(group) if i % size < 3],
            [i for i in sorted(group) if i % size >= size - 3],
        ]
        if min(map(len, sides)) < army_size:
            continue
        c["metadata"]["selection"] = {
            "seed": seed,
            "min_connected": min_connected,
            "army_size": army_size,
        }
        c["spawn_candidates"] = [
            [[i % size, i // size] for i in side] for side in sides
        ]
        c["metadata"]["summary"] = summary(c)
        selected.append(c)
        origins.append((x, y))
        if len(selected) == count:
            return selected
    raise ValueError(
        f"Only {len(selected)} suitable crops found; request fewer crops or use another world seed/size"
    )
