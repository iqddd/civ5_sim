"""Only public inputs enter this module. History is local to one army/battle."""

import warnings
from dataclasses import dataclass

import numpy as np

from civ5map.maps import adjacent, passable, validate
from civ5rl.core.schema import (
    ACTION,
    ACTION_DIM,
    GLOBAL,
    GLOBAL_DIM,
    HEX_DIRECTIONS,
    MAX_MAP_AXIS,
    PREV_ACTION_DIM,
    PREVIOUS,
    TILE,
    TILE_DIM,
)
from civ5sim.engine import ELEVATIONS, FEATURES, KINDS, TERRAINS, UNIT_TYPES


def ordered_actions(actions):
    result = tuple(
        sorted(set(actions), key=lambda a: (KINDS.index(a.kind), a.unit, a.target))
    )
    if len(result) != len(actions):
        warnings.warn("Duplicate public actions removed", RuntimeWarning, stacklevel=2)
    if not result:
        raise ValueError("Empty legal action list at decision")
    return result


def axial(m, tile):
    x, y = tile % m["width"], tile // m["width"]
    r = y + m["row_parity"]
    return x - (r - (r & 1)) // 2, r


def distance(m, a, b):
    aq, ar = axial(m, a)
    bq, br = axial(m, b)
    return (abs(aq - bq) + abs(ar - br) + abs(aq + ar - bq - br)) // 2


def public_map(m):
    keys = (
        "schema",
        "width",
        "height",
        "row_parity",
        "wrap_x",
        "tiles",
        "boundary_rivers",
    )
    from copy import deepcopy

    return {k: deepcopy(m[k]) for k in keys if k in m}


@dataclass
class Encoded:
    cells: np.ndarray
    globals: np.ndarray
    previous: np.ndarray
    neighbors: np.ndarray
    sources: np.ndarray
    targets: np.ndarray
    action_features: np.ndarray


class ObservationEncoder:
    def __init__(self, context, variant="C"):
        self.map = validate(public_map(context["map"]))
        m = self.map
        w, h = m["width"], m["height"]
        if max(w, h) > MAX_MAP_AXIS or min(w, h) < 2 or m["wrap_x"]:
            raise ValueError("RL-01 requires a non-wrapping map with axes 2..16")
        if variant not in ("A", "B", "C"):
            raise ValueError("Unknown RL variant")
        self.variant = variant
        self.side = context["side"]
        self.max_rounds = context["max_rounds"]
        self.random_damage = context["random_damage"]
        self.rosters = context["armies"]
        if any(len(a) > 8 for a in self.rosters):
            raise ValueError("RL-01 roster capacity exceeded")
        self.static = np.zeros((w * h, TILE_DIM), np.float32)
        self.neighbors = np.full((w * h, HEX_DIRECTIONS), -1, np.int64)
        for i, tile in enumerate(m["tiles"]):
            self.static[i, TILE["terrain"].start + TERRAINS.index(tile["terrain"])] = 1
            self.static[
                i, TILE["elevation"].start + ELEVATIONS.index(tile["elevation"])
            ] = 1
            self.static[i, TILE["feature"].start + FEATURES.index(tile["feature"])] = 1
            self.static[i, TILE["position_passable"]] = (*self.xy(i), passable(tile))
            for d in range(HEX_DIRECTIONS):
                xy = adjacent(m, i % w, i // w, d)
                if xy is not None:
                    n = xy[1] * w + xy[0]
                    self.neighbors[i, d] = n
                    self.static[i, TILE["neighbors"].start + d] = 1
                if d in (1, 2, 3):
                    river = tile["rivers"][d - 1] >= 0
                elif xy is not None:
                    river = m["tiles"][n]["rivers"][(d + 3) % 6 - 1] >= 0
                else:
                    river = any(
                        e["x"] == i % w and e["y"] == i // w and e["direction"] == d
                        for e in m.get("boundary_rivers", [])
                    )
                self.static[i, TILE["rivers"].start + d] = river
        self.contacts = {}
        self.last_visible = np.full(w * h, -1, np.int64)
        self.last_clock = 0
        self.last_turn = None
        self.previous = self.start_action()

    @staticmethod
    def start_action():
        result = np.zeros(PREV_ACTION_DIM, np.float32)
        result[PREVIOUS["kind"].stop - 1] = result[PREVIOUS["source_type"].stop - 1] = 1
        return result

    def xy(self, tile):
        w, h = self.map["width"], self.map["height"]
        return 2 * (tile % w) / (w - 1) - 1, 2 * (tile // w) / (h - 1) - 1

    def remember_action(self, action, observation):
        result = np.zeros(PREV_ACTION_DIM, np.float32)
        result[PREVIOUS["kind"].start + KINDS.index(action.kind)] = 1
        own = {u["id"]: u for u in observation["units"] if u["side"] == self.side}
        source = own.get(action.unit)
        result[
            PREVIOUS["source_type"].start
            + (UNIT_TYPES.index(source["type"]) if source else 5)
        ] = 1
        if source:
            result[PREVIOUS["source_xy"]] = self.xy(source["tile"])
            result[PREVIOUS["has_source"]] = 1
        if action.target >= 0:
            result[PREVIOUS["target_xy"]] = self.xy(action.target)
            result[PREVIOUS["has_target"]] = 1
        self.previous = result

    def encode(self, observation, actions):
        if observation["side"] != self.side or observation["active_side"] != self.side:
            raise ValueError("Observation belongs to another army's decision")
        turn = observation["turn"]
        clock = (turn + self.side) // 2
        if clock < self.last_clock:
            raise ValueError("Observation history moved backwards")
        visibility = np.asarray(observation["visibility"], dtype=bool)
        cells = self.static.copy()
        cells[:, TILE["visible"]] = visibility
        self.last_visible[visibility] = clock
        seen = self.last_visible >= 0
        cells[:, TILE["visibility_age"]] = np.where(
            seen, np.minimum(clock - self.last_visible, 32) / 32, 1
        )
        cells[:, TILE["seen"]] = seen
        own, enemy = [], []
        for u in observation["units"]:
            if u["hp"] <= 0:
                raise ValueError("Dead unit exposed in public observation")
            i, kind = u["tile"], UNIT_TYPES.index(u["type"])
            if u["side"] == self.side:
                own.append(u)
                cells[i, TILE["own_type"].start + kind] = 1
                cells[i, TILE["own_status"]] = (
                    u["hp"] / 100,
                    u["moves"] / 240,
                    u["attacked"],
                    u["acted"],
                    u["fortified"],
                    u["waiting"],
                    np.clip(u["damage_taken_this_turn"] / 100, 0, 1),
                    np.clip(u["damage_taken_last_turn"] / 100, 0, 1),
                )
            else:
                if not visibility[i]:
                    raise ValueError("Hidden enemy in public observation")
                enemy.append(u)
                cells[i, TILE["enemy_type"].start + kind] = 1
                cells[i, TILE["enemy_status"]] = (u["hp"] / 100, u["fortified"])
                self.contacts[u["id"]] = (kind, i, u["hp"], clock)
        visible_ids = {u["id"] for u in enemy}
        historical = {}
        for uid, (kind, tile, hp, when) in self.contacts.items():
            if uid not in visible_ids:
                historical.setdefault((tile, kind), []).append((hp, when))
        for (tile, kind), records in historical.items():
            cells[tile, TILE["contact_count"].start + kind] = len(records) / 8
            cells[tile, TILE["contact_age"].start + kind] = (
                min(clock - max(r[1] for r in records), 32) / 32
            )
            cells[tile, TILE["contact_hp"].start + kind] = (
                np.mean([r[0] for r in records]) / 100
            )
        globals_ = np.zeros(GLOBAL_DIM, np.float32)
        for offset, types in (
            (GLOBAL["own_roster"].start, self.rosters[self.side]),
            (GLOBAL["enemy_roster"].start, self.rosters[1 - self.side]),
            (GLOBAL["own_alive"].start, [u["type"] for u in own]),
            (GLOBAL["enemy_visible"].start, [u["type"] for u in enemy]),
        ):
            for kind in types:
                globals_[offset + UNIT_TYPES.index(kind)] += 1 / 8
        for kind, *_ in self.contacts.values():
            globals_[GLOBAL["contacts"].start + kind] += 1 / 8
        globals_[GLOBAL["turn"] :] = (
            turn / (2 * self.max_rounds),
            min(clock - self.last_clock, 32) / 32,
            self.last_turn != turn,
            self.map["width"] / 16,
            self.map["height"] / 16,
            self.side == 0,
            self.random_damage,
            self.max_rounds / 60,
        )
        self.last_clock, self.last_turn = clock, turn
        previous = self.previous.copy()
        if self.variant == "A":
            cells[:, TILE["visibility_age"] : TILE["seen"] + 1] = 0
            cells[:, TILE["contact_count"].start : TILE["contact_hp"].stop] = 0
            globals_[GLOBAL["contacts"]] = globals_[
                GLOBAL["elapsed"] : GLOBAL["new_turn"] + 1
            ] = 0
            previous = self.start_action()
        own_by_id = {u["id"]: u for u in own}
        sources, targets = (
            np.full(len(actions), -1, np.int64),
            np.full(len(actions), -1, np.int64),
        )
        features = np.zeros((len(actions), ACTION_DIM), np.float32)
        for j, a in enumerate(actions):
            features[j, ACTION["kind"].start + KINDS.index(a.kind)] = 1
            source = own_by_id.get(a.unit)
            if source:
                sources[j] = source["tile"]
                features[
                    j, ACTION["source_type"].start + UNIT_TYPES.index(source["type"])
                ] = 1
                features[j, ACTION["has_source"]] = 1
            if a.target >= 0:
                targets[j] = a.target
                features[j, ACTION["has_target"]] = 1
            if sources[j] >= 0 and targets[j] >= 0:
                w = self.map["width"]
                features[j, ACTION["displacement"]] = (
                    (targets[j] % w - sources[j] % w) / 15,
                    (targets[j] // w - sources[j] // w) / 15,
                    distance(self.map, int(sources[j]), int(targets[j])) / 32,
                )
        return Encoded(
            cells, globals_, previous, self.neighbors, sources, targets, features
        )
