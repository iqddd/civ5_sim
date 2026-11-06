"""Only public inputs enter this module. History is local to one army/battle."""

import warnings
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from civ5map.maps import adjacent, passable, validate
from civ5rl.core.schema import (
    ACTION,
    ACTION_DIM,
    AGE_CAP,
    GLOBAL,
    GLOBAL_DIM,
    HEX_DIRECTIONS,
    MAX_MAP_AXIS,
    MAX_ROSTER_UNITS,
    PREV_ACTION_DIM,
    PREVIOUS,
    PREVIOUS_START_KIND,
    PREVIOUS_START_SOURCE_TYPE,
    ROSTER_SCALE,
    TILE,
    TILE_DIM,
    opposite_direction,
)
from civ5rl.core.contracts import AgentContext
from civ5rl.core.variants import PolicyVariant
from civ5sim.engine import ELEVATIONS, FEATURES, KINDS, MAX_MOVES, TERRAINS, UNIT_TYPES

HP_SCALE = 100
MOVEMENT_SCALE = 240
if max(MAX_MOVES) != MOVEMENT_SCALE:
    raise ValueError("Engine movement scale does not match RL-01 preprocessing")
OWNED_RIVER_DIRECTIONS = (1, 2, 3)
MAX_ROUNDS_SCALE = 60
DISTANCE_SCALE = 32


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


class ContactRecord(NamedTuple):
    type_index: int
    tile: int
    last_hp: int
    last_clock: int


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
    def __init__(
        self,
        context: AgentContext,
        variant: PolicyVariant | str = PolicyVariant.CONTACTS_GRU,
    ):
        self.map = validate(public_map(context["map"]))
        m = self.map
        w, h = m["width"], m["height"]
        if max(w, h) > MAX_MAP_AXIS or min(w, h) < 2 or m["wrap_x"]:
            raise ValueError("RL-01 requires a non-wrapping map with axes 2..16")
        self.variant = PolicyVariant(variant)
        self.side = context["side"]
        self.max_rounds = context["max_rounds"]
        self.random_damage = context["random_damage"]
        self.rosters = [list(army) for army in context["armies"]]
        if any(len(a) > MAX_ROSTER_UNITS for a in self.rosters):
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
                river = self._has_river(i, d, xy)
                self.static[i, TILE["rivers"].start + d] = river
        for tile_index, neighbors in enumerate(self.neighbors):
            for direction, neighbor in enumerate(neighbors):
                if (
                    neighbor >= 0
                    and self.neighbors[neighbor, opposite_direction(direction)]
                    != tile_index
                ):
                    raise ValueError("Map neighbors are not reciprocal")
        self.contacts: dict[int, ContactRecord] = {}
        self.last_visible = np.full(w * h, -1, np.int64)
        self.last_clock = 0
        self.last_turn = None
        self.previous = self.start_action()

    def _has_river(self, tile_index, direction, adjacent_xy):
        m = self.map
        w = m["width"]
        if direction in OWNED_RIVER_DIRECTIONS:
            return m["tiles"][tile_index]["rivers"][direction - 1] >= 0
        if adjacent_xy is not None:
            neighbor = adjacent_xy[1] * w + adjacent_xy[0]
            return (
                m["tiles"][neighbor]["rivers"][opposite_direction(direction) - 1] >= 0
            )
        return any(
            edge["x"] == tile_index % w
            and edge["y"] == tile_index // w
            and edge["direction"] == direction
            for edge in m.get("boundary_rivers", [])
        )

    @staticmethod
    def start_action():
        result = np.zeros(PREV_ACTION_DIM, np.float32)
        result[PREVIOUS_START_KIND] = 1
        result[PREVIOUS_START_SOURCE_TYPE] = 1
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
            + (UNIT_TYPES.index(source["type"]) if source else len(UNIT_TYPES))
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
        cells, own, enemy = self._encode_cells(observation, clock)
        globals_ = self._encode_globals(own, enemy, turn, clock)
        previous = self.previous.copy()
        if self.variant == PolicyVariant.CURRENT_ONLY_MLP:
            for name in (
                "visibility_age",
                "seen",
                "contact_count",
                "contact_age",
                "contact_hp",
            ):
                cells[:, TILE[name]] = 0
            for name in ("contacts", "elapsed", "new_turn"):
                globals_[GLOBAL[name]] = 0
            previous = self.start_action()
        sources, targets, features = self._encode_actions(actions, own)
        self.last_clock = clock
        self.last_turn = turn
        return Encoded(
            cells, globals_, previous, self.neighbors, sources, targets, features
        )

    def _encode_cells(self, observation, clock):
        visibility = np.asarray(observation["visibility"], dtype=bool)
        cells = self.static.copy()
        cells[:, TILE["visible"]] = visibility
        self.last_visible[visibility] = clock
        seen = self.last_visible >= 0
        cells[:, TILE["visibility_age"]] = np.where(
            seen, np.minimum(clock - self.last_visible, AGE_CAP) / AGE_CAP, 1
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
                    u["hp"] / HP_SCALE,
                    u["moves"] / MOVEMENT_SCALE,
                    u["attacked"],
                    u["acted"],
                    u["fortified"],
                    u["waiting"],
                    np.clip(u["damage_taken_this_turn"] / HP_SCALE, 0, 1),
                    np.clip(u["damage_taken_last_turn"] / HP_SCALE, 0, 1),
                )
            else:
                if not visibility[i]:
                    raise ValueError("Hidden enemy in public observation")
                enemy.append(u)
                cells[i, TILE["enemy_type"].start + kind] = 1
                cells[i, TILE["enemy_status"]] = (u["hp"] / HP_SCALE, u["fortified"])
                self.contacts[u["id"]] = ContactRecord(kind, i, u["hp"], clock)
        visible_ids = {u["id"] for u in enemy}
        historical = {}
        for uid, contact in self.contacts.items():
            if uid not in visible_ids:
                historical.setdefault((contact.tile, contact.type_index), []).append(
                    contact
                )
        for (tile, kind), records in historical.items():
            cells[tile, TILE["contact_count"].start + kind] = (
                len(records) / ROSTER_SCALE
            )
            cells[tile, TILE["contact_age"].start + kind] = (
                min(clock - max(record.last_clock for record in records), AGE_CAP)
                / AGE_CAP
            )
            cells[tile, TILE["contact_hp"].start + kind] = (
                np.mean([record.last_hp for record in records]) / HP_SCALE
            )
        return cells, own, enemy

    def _encode_globals(self, own, enemy, turn, clock):
        globals_ = np.zeros(GLOBAL_DIM, np.float32)
        for offset, types in (
            (GLOBAL["own_roster"].start, self.rosters[self.side]),
            (GLOBAL["enemy_roster"].start, self.rosters[1 - self.side]),
            (GLOBAL["own_alive"].start, [u["type"] for u in own]),
            (GLOBAL["enemy_visible"].start, [u["type"] for u in enemy]),
        ):
            for kind in types:
                globals_[offset + UNIT_TYPES.index(kind)] += 1 / ROSTER_SCALE
        for contact in self.contacts.values():
            globals_[GLOBAL["contacts"].start + contact.type_index] += 1 / ROSTER_SCALE
        globals_[GLOBAL["turn"] :] = (
            turn / (2 * self.max_rounds),
            min(clock - self.last_clock, AGE_CAP) / AGE_CAP,
            self.last_turn != turn,
            self.map["width"] / MAX_MAP_AXIS,
            self.map["height"] / MAX_MAP_AXIS,
            self.side == 0,
            self.random_damage,
            self.max_rounds / MAX_ROUNDS_SCALE,
        )
        return globals_

    def _encode_actions(self, actions, own):
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
                    (targets[j] % w - sources[j] % w) / (MAX_MAP_AXIS - 1),
                    (targets[j] // w - sources[j] // w) / (MAX_MAP_AXIS - 1),
                    distance(self.map, int(sources[j]), int(targets[j]))
                    / DISTANCE_SCALE,
                )
        return sources, targets, features
