"""Mutual two-vanguard contact placement with compact hidden main bodies."""

from __future__ import annotations

from collections import deque
from typing import Any

from civ5map.maps import adjacent, passable
from civ5sim import Battle

CONTACT_V1 = "contact_v1"
CONTACT_TWO_VISIBLE = "two_visible_compact_v1"
CONTACT_MODES = (CONTACT_V1, CONTACT_TWO_VISIBLE)


def contact_mode(config: dict[str, Any]) -> str:
    value = config.get("environment", {}).get("contact_placement", CONTACT_V1)
    if value not in CONTACT_MODES:
        raise ValueError(f"Unknown contact placement mode: {value}")
    return value


def passable_graph(m: dict[str, Any], group: list[int] | tuple[int, ...]) -> dict[int, tuple[int, ...]]:
    width = m["width"]
    allowed = set(group)
    neighbors: dict[int, tuple[int, ...]] = {}
    for tile in group:
        x, y = tile % width, tile // width
        adjacent_tiles = []
        for direction in range(6):
            xy = adjacent(m, x, y, direction)
            if xy is not None:
                other = xy[1] * width + xy[0]
                if other in allowed and passable(m["tiles"][other]):
                    adjacent_tiles.append(other)
        neighbors[tile] = tuple(adjacent_tiles)
    return neighbors


def graph_distances_from(sources: tuple[int, ...], neighbors: dict[int, tuple[int, ...]]) -> dict[int, int]:
    distances = {source: 0 for source in sources}
    queue = deque(sources)
    while queue:
        tile = queue.popleft()
        next_distance = distances[tile] + 1
        for neighbor in neighbors[tile]:
            if neighbor not in distances:
                distances[neighbor] = next_distance
                queue.append(neighbor)
    return distances


def directed_visibility_tables(
    m: dict[str, Any],
    group: list[int] | tuple[int, ...],
    battle_seed: int = 1,
) -> tuple[dict[int, set[int]], dict[int, set[int]]]:
    """Compute single-unit directed sight tables (`sees`, `seen_by`) across `group`."""
    first, *remaining = group
    unit = {"type": "warrior", "side": 0, "tile": first, "hp": 100}
    probe = {
        "schema": 1,
        "kind": "civ5sim.scenario",
        "map": m,
        "fog": True,
        "random_damage": True,
        "seed": battle_seed,
        "units": [unit],
    }
    sees: dict[int, set[int]] = {}
    seen_by: dict[int, set[int]] = {tile: set() for tile in group}
    with Battle(probe) as battle:
        vis = battle.observe(0)["visibility"]
        visible_first = {other for other in group if vis[other]}
        sees[first] = visible_first
        for other in visible_first:
            seen_by[other].add(first)
        for tile in remaining:
            unit["tile"] = tile
            battle.reset(probe)
            vis = battle.observe(0)["visibility"]
            visible = {other for other in group if vis[other]}
            sees[tile] = visible
            for other in visible:
                seen_by[other].add(tile)
    return sees, seen_by


def visible_enemy_counts(scenario: dict[str, Any]) -> list[int]:
    with Battle(scenario) as battle:
        return [sum(u["side"] != side for u in battle.observe(side)["units"]) for side in (0, 1)]


def _grow_hidden_cluster(
    vanguards: tuple[int, int],
    allowed_set: set[int],
    need_hidden: int,
    dist_a: dict[int, int],
    dist_b: dict[int, int],
    dist_enemy: dict[int, int],
    neighbors: dict[int, tuple[int, ...]],
    rng,
    *,
    accept_tile=None,
) -> list[int] | None:
    if need_hidden == 0:
        return []
    chosen: list[int] = []
    chosen_set = set(vanguards)
    frontier: set[int] = set()
    for vanguard in vanguards:
        for nb in neighbors[vanguard]:
            if nb in allowed_set and nb not in chosen_set:
                frontier.add(nb)

    while len(chosen) < need_hidden:
        pool = frontier if frontier else (allowed_set - chosen_set)
        if not pool:
            return None
        ranked = sorted(
            pool,
            key=lambda tile: (
                0 if min(dist_a.get(tile, 999), dist_b.get(tile, 999)) <= dist_enemy.get(tile, 999) else 1,
                min(dist_a.get(tile, 999), dist_b.get(tile, 999)),
                dist_a.get(tile, 999) + dist_b.get(tile, 999),
                -dist_enemy.get(tile, 0),
                rng.random(),
            ),
        )
        picked = None
        for candidate in ranked:
            if accept_tile is None or accept_tile(candidate):
                picked = candidate
                break
            frontier.discard(candidate)
            allowed_set.discard(candidate)
        if picked is None:
            if frontier:
                frontier.clear()
                continue
            return None
        chosen.append(picked)
        chosen_set.add(picked)
        frontier.discard(picked)
        for nb in neighbors[picked]:
            if nb in allowed_set and nb not in chosen_set:
                frontier.add(nb)
    return chosen


def sample_two_visible_compact_positions(
    group: list[int] | tuple[int, ...],
    n: int,
    rng,
    sees: dict[int, set[int]],
    seen_by: dict[int, set[int]],
    neighbors: dict[int, tuple[int, ...]],
    *,
    reasons=None,
    candidate_budget: int = 16,
) -> tuple[list[list[int]], dict[str, int]] | None:
    """Place two armies of size `n >= 2` such that:
    - Each side sees exactly 2 enemy units at turn 0 before any command.
    - Each side's remaining `n - 2` units are outside enemy sight and packed
      as closely as possible along passable tiles behind their 2 visible units.
    """
    if n < 2:
        raise ValueError("two_visible_compact_v1 requires at least 2 units per side")
    need_hidden = n - 2
    first_side = rng.randrange(2)
    anchors = list(group)
    rng.shuffle(anchors)

    best = None
    found = 0
    target_max_dist = 2 if n <= 8 else 3

    for a0 in anchors:
        b0_cands = list(neighbors[a0])
        if not b0_cands:
            b0_cands = [t for nb in neighbors[a0] for t in neighbors[nb] if t != a0]
        if not b0_cands:
            if reasons is not None:
                reasons["no_vanguard_pair"] += 1
            continue
        rng.shuffle(b0_cands)
        for b0 in b0_cands[:2]:
            v_first = (a0, b0)
            vf_set = set(v_first)
            vis_vf = sees[a0] | sees[b0]
            vs_pool = [t for t in vis_vf if t not in vf_set and (a0 in sees[t] or b0 in sees[t])]
            if len(vs_pool) < 2:
                if reasons is not None:
                    reasons["no_enemy_vanguard"] += 1
                continue
            rng.shuffle(vs_pool)
            vs_pool.sort(key=lambda t: 1 if (t in neighbors[a0] or t in neighbors[b0]) else 0)

            for a1 in vs_pool[:6]:
                b1_cands = [t for t in vs_pool if t != a1 and t in neighbors[a1] and vf_set <= (sees[a1] | sees[t])]
                if not b1_cands:
                    b1_cands = [
                        t
                        for t in vs_pool
                        if t != a1
                        and any(t in neighbors[mid] for mid in neighbors[a1])
                        and vf_set <= (sees[a1] | sees[t])
                    ]
                if not b1_cands:
                    continue
                b1_cands.sort(key=lambda t: (1 if (t in neighbors[a0] or t in neighbors[b0]) else 0, rng.random()))
                b1 = b1_cands[0]
                v_second = (a1, b1)
                vs_set = set(v_second)

                forbid_hf = sees[a1] | sees[b1] | vf_set | vs_set
                allowed_hf = {t for t in group if t not in forbid_hf}
                if len(allowed_hf) < need_hidden:
                    if reasons is not None:
                        reasons["hidden_capacity"] += 1
                    continue
                forbid_hs_base = vis_vf | vf_set | vs_set
                allowed_hs = {t for t in group if t not in forbid_hs_base}
                if len(allowed_hs) < need_hidden:
                    if reasons is not None:
                        reasons["hidden_capacity"] += 1
                    continue

                da0 = graph_distances_from((a0,), neighbors)
                db0 = graph_distances_from((b0,), neighbors)
                da1 = graph_distances_from((a1,), neighbors)
                db1 = graph_distances_from((b1,), neighbors)
                d_vf = {t: min(da0.get(t, 999), db0.get(t, 999)) for t in group}
                d_vs = {t: min(da1.get(t, 999), db1.get(t, 999)) for t in group}

                def accept_hf(tile: int, _hs=allowed_hs) -> bool:
                    remaining = _hs - sees[tile] - seen_by[tile] - {tile}
                    if len(remaining) < need_hidden:
                        return False
                    _hs.intersection_update(remaining)
                    return True

                hf = _grow_hidden_cluster(
                    v_first,
                    set(allowed_hf),
                    need_hidden,
                    da0,
                    db0,
                    d_vs,
                    neighbors,
                    rng,
                    accept_tile=accept_hf,
                )
                if hf is None:
                    if reasons is not None:
                        reasons["hidden_cluster"] += 1
                    continue
                hs = _grow_hidden_cluster(
                    v_second,
                    set(allowed_hs),
                    need_hidden,
                    da1,
                    db1,
                    d_vf,
                    neighbors,
                    rng,
                )
                if hs is None:
                    if reasons is not None:
                        reasons["hidden_cluster"] += 1
                    continue

                df_min = [d_vf[u] for u in hf] if hf else [0]
                ds_min = [d_vs[v] for v in hs] if hs else [0]
                max_h_dist = max(max(df_min), max(ds_min))
                sum_h_dist = sum(df_min) + sum(ds_min)
                adj_enemies = sum(v in neighbors[u] for u in (*v_first, *hf) for v in (*v_second, *hs))
                wrong_half = sum(d_vf[u] > d_vs[u] for u in hf) + sum(d_vs[v] > d_vf[v] for v in hs)
                score = (wrong_half, adj_enemies, max_h_dist, sum_h_dist)
                if best is None or score < best[0]:
                    best = (score, v_first, hf, v_second, hs, max_h_dist, adj_enemies)
                found += 1
                if found >= candidate_budget and best[0][:3] <= (0, 0, target_max_dist):
                    break
            if found >= candidate_budget and best is not None and best[0][:3] <= (0, 0, target_max_dist):
                break
        if found >= candidate_budget and best is not None and best[0][:3] <= (0, 0, target_max_dist):
            break

    if best is None:
        return None
    _, vf, hf, vs, hs, max_h_dist, adj_enemies = best
    pos_first = [*vf, *hf]
    pos_second = [*vs, *hs]
    rng.shuffle(pos_first)
    rng.shuffle(pos_second)
    positions: list[list[int]] = [[], []]
    positions[first_side] = pos_first
    positions[1 - first_side] = pos_second
    return positions, {
        "max_hidden_distance": max_h_dist,
        "adjacent_enemy_pairs": adj_enemies,
        "candidates_checked": found,
    }
