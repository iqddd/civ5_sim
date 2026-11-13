"""Scenario, observation, snapshot and replay boundary around the native kernel."""

import ctypes as C
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

from civ5map.maps import adjacent
from civ5map.maps import validate as validate_map

from . import _native as N

UNIT_TYPES = ("warrior", "archer", "chariot_archer", "spearman", "horseman")
KINDS = ("move", "melee", "ranged", "fortify", "skip", "end_turn", "swap")
ERRORS = (
    None,
    "bad_action",
    "bad_unit",
    "not_active",
    "exhausted",
    "bad_target",
    "blocked",
    "no_attack",
    "no_fortify",
)
FEATURES = (
    None,
    "forest",
    "jungle",
    "marsh",
    "oasis",
    "ice",
    "flood_plains",
    "fallout",
    "atoll",
)
TERRAINS = ("grass", "plains", "desert", "tundra", "snow", "coast", "ocean")
ELEVATIONS = ("flat", "hills", "mountain", "water")
MAX_MOVES = (120, 120, 240, 120, 240)


def integer(value, name, low=0, high=2**31 - 1):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")
    return value


def boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Action:
    kind: str
    unit: int = -1
    target: int = -1

    def native(self):
        if self.kind not in KINDS:
            raise ValueError(f"Unknown action: {self.kind}")
        integer(self.unit, "unit", -1)
        integer(self.target, "target", -1)
        return N.Action(KINDS.index(self.kind), self.unit, self.target)


@dataclass(frozen=True)
class StepResult:
    error: str | None
    moved: int = 0
    damage: int = 0
    retaliation: int = 0
    attacker_strength: int = 0
    defender_strength: int = 0

    @property
    def ok(self):
        return self.error is None

    @classmethod
    def from_native(cls, value):
        return cls(
            ERRORS[value.code],
            value.moved,
            value.damage,
            value.retaliation,
            value.attacker_strength,
            value.defender_strength,
        )


def load_scenario(path):
    """Resolve a scenario's relative map reference; returned data is self-contained."""
    path = Path(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value.get("map"), str):
        value["map"] = json.loads(
            (path.parent / value["map"]).read_text(encoding="utf-8")
        )
    return value


def action_value(value):
    if isinstance(value, Action):
        return value
    if isinstance(value, dict):
        return Action(**value)
    raise ValueError("Action must be an Action or dict")


class Battle:
    """Owned native state. Use as a context manager; independent instances may run concurrently.

    Unit IDs are stable zero-based positions in the scenario's units list. Moves
    use integer sixtieths of a movement point. The same instance is not thread safe.
    """

    def __init__(self, scenario, *, log=False):
        self._handle = None
        self._action_buffer = None
        self._observation_buffers = {}
        if isinstance(scenario, (str, Path)):
            scenario = load_scenario(scenario)
        self._check_header(scenario)
        self._map = deepcopy(validate_map(scenario["map"]))
        w, h = self._map["width"], self._map["height"]
        if w > 128 or h > 128 or w * h > 4096:
            raise ValueError(
                "Combat maps support at most 4096 cells and 128 cells per axis"
            )
        self._lib = N.library()
        self._rules = json.loads(
            (N.ROOT / "build/combat_rules.json").read_text(encoding="utf-8")
        )
        if self._rules["fingerprint"] != self._lib.sim_rules().decode():
            raise RuntimeError(
                "Rules passport does not match the loaded kernel; rebuild"
            )
        if (
            scenario.get("kind") == "civ5sim.snapshot"
            and scenario.get("rules") != self._rules
        ):
            raise ValueError("Snapshot uses a different rules passport")
        tiles = self._encode_tiles()
        units = self._encode_units(scenario["units"], require_history=scenario["kind"] == "civ5sim.snapshot")
        self._handle = self._lib.sim_new(
            w,
            h,
            self._map["row_parity"],
            self._map["wrap_x"],
            tiles,
            units,
            len(units),
            0,
            0,
            0,
        )
        if not self._handle:
            raise ValueError(
                "Invalid battle state: check unit positions and state fields"
            )
        try:
            self._restore_state(scenario, units)
            self._initial = self._snapshot(copy_static=False)
            self._logging = boolean(log, "log")
            self._journal = []
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check_header(scenario):
        if (
            not isinstance(scenario, dict)
            or type(scenario.get("schema")) is not int
        ):
            raise ValueError("Expected versioned battle data")
        if scenario.get("kind") not in ("civ5sim.scenario", "civ5sim.snapshot"):
            raise ValueError("Expected civ5sim.scenario or civ5sim.snapshot")
        expected = 2 if scenario["kind"] == "civ5sim.snapshot" else 1
        if scenario["schema"] != expected:
            raise ValueError(f"Expected {scenario['kind']} schema {expected}")

    def _encode_tiles(self):
        tiles = self._map["tiles"]
        result = (N.Tile * len(tiles))()
        for i, tile in enumerate(tiles):
            if tile["feature"] not in FEATURES:
                raise ValueError(f"Unsupported combat feature: {tile['feature']}")
            mask = 0
            for d in range(6):
                if d in (1, 2, 3):
                    exists = tile["rivers"][d - 1] >= 0
                else:
                    neighbor = adjacent(
                        self._map, i % self._map["width"], i // self._map["width"], d
                    )
                    if neighbor:
                        other = tiles[neighbor[1] * self._map["width"] + neighbor[0]]
                        exists = other["rivers"][(d + 3) % 6 - 1] >= 0
                    else:
                        exists = any(
                            e["x"] == i % self._map["width"]
                            and e["y"] == i // self._map["width"]
                            and e["direction"] == d
                            for e in self._map.get("boundary_rivers", [])
                        )
                mask |= int(exists) << d
            result[i] = N.Tile(
                TERRAINS.index(tile["terrain"]),
                ELEVATIONS.index(tile["elevation"]),
                FEATURES.index(tile["feature"]),
                mask,
            )
        return result

    def _encode_units(self, values, *, require_history=False):
        if not isinstance(values, list) or len(values) > 512:
            raise ValueError("units must be a list with at most 512 units")
        result = (N.Unit * len(values))()
        w, h = self._map["width"], self._map["height"]
        for i, value in enumerate(values):
            if (
                not isinstance(value, dict)
                or not {"type", "side", "tile"} <= value.keys()
            ):
                raise ValueError("Each unit must have type, side and tile")
            if "id" in value and integer(value["id"], "id") != i:
                raise ValueError("Unit ID must match its position in the units list")
            if value["type"] not in UNIT_TYPES:
                raise ValueError(f"Unknown unit type: {value['type']}")
            kind = UNIT_TYPES.index(value["type"])
            tile = value["tile"]
            if isinstance(tile, (list, tuple)) and len(tile) == 2:
                tile = integer(tile[1], "y", 0, h - 1) * w + integer(
                    tile[0], "x", 0, w - 1
                )
            tile = integer(tile, "tile", 0, w * h - 1)
            history_fields = ("damage_taken_this_turn", "damage_taken_last_turn")
            if require_history and not set(history_fields) <= value.keys():
                raise ValueError("Snapshot units must include complete damage history")
            history = [integer(value.get(name, 0), name) for name in history_fields]
            if not require_history and any(history):
                raise ValueError("New scenarios start with zero damage history; use a snapshot to resume")
            result[i] = N.Unit(
                kind,
                integer(value["side"], "side", 0, 1),
                tile,
                integer(value.get("hp", 100), "hp", 0, 100),
                integer(
                    value.get("moves", MAX_MOVES[kind]), "moves", 0, MAX_MOVES[kind]
                ),
                *[
                    int(boolean(value.get(n, False), n))
                    for n in ("attacked", "acted", "fortified", "waiting")
                ],
                *history,
            )
        return result

    def _restore_state(self, scenario, units):
        masks = None
        if scenario["kind"] == "civ5sim.snapshot":
            state = scenario["state"]
            rows = state.get("visibility")
            count = len(self._map["tiles"])
            if not isinstance(rows, list) or len(rows) != 2 or any(
                not isinstance(row, list) or len(row) != count for row in rows
            ):
                raise ValueError("Snapshot must include two complete visibility masks")
            values = [int(boolean(value, "visibility")) for row in rows for value in row]
            masks = (C.c_int * len(values))(*values)
            info = N.Info(
                integer(state["side"], "side", 0, 1),
                integer(state["turn"], "turn"),
                boolean(state["fog"], "fog"),
                boolean(state["random_damage"], "random_damage"),
                0,
                0,
                integer(state["rng"], "rng", 0, 2**64 - 1),
            )
        else:
            info = N.Info(
                integer(scenario.get("active_side", 0), "active_side", 0, 1),
                0,
                boolean(scenario.get("fog", True), "fog"),
                boolean(scenario.get("random_damage", True), "random_damage"),
                0,
                0,
                integer(scenario.get("seed", 0), "seed", 0, 2**64 - 1),
            )
        temporary = self._lib.sim_clone(self._handle)
        if not temporary:
            raise MemoryError("Cannot restore battle")
        try:
            if self._lib.sim_reset(temporary, units, len(units), C.byref(info)):
                raise ValueError("Invalid snapshot or scenario state")
            if masks is not None and self._lib.sim_restore_visibility(temporary, masks, len(masks)):
                raise ValueError("Snapshot visibility excludes a current sight source")
        except BaseException:
            self._lib.sim_free(temporary)
            raise
        self._lib.sim_free(self._handle)
        self._handle = temporary

    def _require_open(self):
        if not self._handle:
            raise RuntimeError("Battle is closed")

    def _info(self):
        self._require_open()
        value = N.Info()
        self._lib.sim_info(self._handle, C.byref(value))
        return value

    def _units(self, observer=-1):
        self._require_open()
        count = self._lib.sim_units(self._handle, observer, None)
        buffer = (N.Unit * count)()
        self._lib.sim_units(self._handle, observer, buffer)
        return buffer

    @staticmethod
    def _unit_dict(u, i):
        return {
            "id": i,
            "type": UNIT_TYPES[u.type],
            "side": u.side,
            "tile": u.tile,
            "hp": u.hp,
            "moves": u.moves,
            "damage_taken_this_turn": u.damage_taken_this_turn,
            "damage_taken_last_turn": u.damage_taken_last_turn,
            **{
                n: bool(getattr(u, n))
                for n in ("attacked", "acted", "fortified", "waiting")
            },
        }

    def observe(self, side=None):
        """Player-safe observation; no hidden health, movement, deaths or RNG."""
        packed = self.observation_buffer(side)
        side = packed["side"]
        buffer = packed["visibility"]
        units = []
        for i, u in enumerate(packed["units"]):
            if u.type < 0:
                continue
            value = self._unit_dict(u, i)
            if u.side != side:
                value = {
                    k: value[k]
                    for k in ("id", "type", "side", "tile", "hp", "fortified")
                }
            units.append(value)
        return {
            "side": side,
            "active_side": packed["active_side"],
            "turn": packed["turn"],
            "visibility": [bool(v) for v in buffer],
            "units": units,
        }

    def observation_buffer(self, side=None):
        """Player-safe reusable native arrays; overwritten on the next call for this side.

        Hidden/dead slots have all fields -1. Visible enemies have -1 for private
        moves/attacked/acted/waiting fields. Initial roster size is public.
        """
        info = self._info()
        side = info.side if side is None else integer(side, "side", 0, 1)
        if side not in self._observation_buffers:
            count = self._lib.sim_units(self._handle, side, None)
            self._observation_buffers[side] = (
                (N.Unit * count)(),
                (C.c_int * len(self._map["tiles"]))(),
            )
        units, visibility = self._observation_buffers[side]
        self._lib.sim_units(self._handle, side, units)
        self._lib.sim_visibility(self._handle, side, visibility)
        return {
            "side": side,
            "active_side": info.side,
            "turn": info.turn,
            "units": units,
            "visibility": visibility,
        }

    def available_actions(self):
        buffer, count = self.action_buffer()
        return [Action(KINDS[a.kind], a.unit, a.target) for a in buffer[:count]]

    def action_buffer(self):
        """Reusable packed action array and count; overwritten by the next call.

        Suitable for collectors that select actions without constructing Python
        objects for the complete action list. kind is the integer KINDS index.
        """
        self._require_open()
        if self._action_buffer is None:
            capacity = 1 + len(self._units()) * (len(self._map["tiles"]) + 27)
            self._action_buffer = (N.Action * capacity)()
        buffer = self._action_buffer
        count = self._lib.sim_actions(self._handle, buffer, len(buffer))
        if not 0 <= count <= len(buffer):
            raise RuntimeError("Native action buffer overflow")
        return buffer, count

    def preview_attack(self, action):
        """Fixed-roll estimate from the current player's information; never consumes RNG.

        Under fog, undiscovered flank support can make actual damage differ.
        """
        self._require_open()
        return StepResult.from_native(
            self._lib.sim_preview(self._handle, action_value(action).native())
        )

    def actions_with_previews(self):
        """Sorted legal commands and owned public FP32 fixed-roll features.

        One native call; no state mutation, RNG use or hidden-information payload.
        Returned arrays remain valid after subsequent steps and buffer reuse.
        """
        import numpy as np

        self._require_open()
        if self._action_buffer is None:
            capacity = 1 + len(self._units()) * (len(self._map["tiles"]) + 27)
            self._action_buffer = (N.Action * capacity)()
        if not hasattr(self, "_preview_buffer"):
            self._preview_buffer = (C.c_float * (5 * len(self._action_buffer)))()
            self._preview_array = np.ctypeslib.as_array(self._preview_buffer).reshape(-1, 5)
        count = self._lib.sim_actions_with_previews(
            self._handle, self._action_buffer, self._preview_buffer, len(self._action_buffer)
        )
        if not 0 <= count <= len(self._action_buffer):
            raise RuntimeError("Native preview buffer overflow")
        actions = tuple(Action(KINDS[a.kind], a.unit, a.target) for a in self._action_buffer[:count])
        features = self._preview_array[:count].copy()
        return actions, features

    def step(self, action):
        self._require_open()
        action = action_value(action)
        result = StepResult.from_native(
            self._lib.sim_step(self._handle, action.native())
        )
        self._record(action, result)
        return result

    def _record(self, action, result):
        if self._logging:
            self._journal.append(
                {
                    "action": asdict(action),
                    "result": asdict(result),
                    "state_sha256": digest(self.snapshot()),
                }
            )

    def snapshot(self):
        """Privileged complete state, not an input for a fog-of-war policy."""
        return self._snapshot(copy_static=True)

    def _snapshot(self, *, copy_static):
        """Internal snapshots share owned immutable terrain; public snapshots detach it."""
        info = self._info()
        units = [self._unit_dict(u, i) for i, u in enumerate(self._units())]
        return {
            "schema": 2,
            "kind": "civ5sim.snapshot",
            "rules": deepcopy(self._rules) if copy_static else self._rules,
            "map": deepcopy(self._map) if copy_static else self._map,
            "units": units,
            "state": {
                "side": info.side,
                "turn": info.turn,
                "fog": bool(info.fog),
                "random_damage": bool(info.random),
                "rng": info.rng,
                "visibility": self.visibility_history(),
            },
        }

    def visibility_history(self):
        """Privileged raw history; unlike observation it is not forced on by fog=False."""
        self._require_open()
        buffer = (C.c_int * len(self._map['tiles']))()
        rows = []
        for side in (0, 1):
            self._lib.sim_visibility_history(self._handle, side, buffer)
            rows.append([bool(v) for v in buffer])
        return rows

    def army_status(self):
        """Privileged counts; caller decides whether and when the episode ends."""
        info = self._info()
        return {
            "alive": [info.alive0, info.alive1],
            "destroyed": [info.alive0 == 0, info.alive1 == 0],
        }

    def replay(self):
        if not self._logging:
            raise RuntimeError("Enable log=True to collect a replay")
        return {
            "schema": 2,
            "kind": "civ5sim.replay",
            "initial": deepcopy(self._initial),
            "steps": deepcopy(self._journal),
        }

    def clone(self, *, log=False):
        """Copy mutable state; share immutable geometry and visibility tables."""
        self._require_open()
        other = object.__new__(Battle)
        other._handle = None
        other._action_buffer = None
        other._observation_buffers = {}
        other._lib, other._map, other._rules = self._lib, self._map, self._rules
        other._handle = self._lib.sim_clone(self._handle)
        if not other._handle:
            raise MemoryError("Cannot clone battle")
        other._initial = other._snapshot(copy_static=False)
        other._logging = boolean(log, "log")
        other._journal = []
        return other

    def reset(self, scenario=None):
        """Reset on the same map, retaining precomputed geometry."""
        self._require_open()
        value = self._initial if scenario is None else scenario
        if isinstance(value, (str, Path)):
            value = load_scenario(value)
        self._check_header(value)
        if value["map"] != self._map:
            raise ValueError(
                "Reset requires the same map; create a new Battle for another map"
            )
        if value["kind"] == "civ5sim.snapshot" and value.get("rules") != self._rules:
            raise ValueError("Snapshot uses a different rules passport")
        self._restore_state(value, self._encode_units(value["units"], require_history=value["kind"] == "civ5sim.snapshot"))
        self._action_buffer = None
        self._observation_buffers = {}
        self._initial = self._snapshot(copy_static=False)
        self._journal.clear()

    def close(self):
        if self._handle:
            self._lib.sim_free(self._handle)
            self._handle = None

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        self.close()


def step_batch(battles, actions):
    """One command per independent state in one C call; returns ordered results."""
    battles, actions = list(battles), [action_value(a) for a in actions]
    if len(battles) != len(actions):
        raise ValueError("Expected one action per battle")
    if len({id(b) for b in battles}) != len(battles):
        raise ValueError("Batch requires distinct battle instances")
    for b in battles:
        b._require_open()
    n = len(battles)
    states = (C.c_void_p * n)(*[b._handle for b in battles])
    commands = (N.Action * n)(*[a.native() for a in actions])
    results = (N.Result * n)()
    N.library().sim_batch(states, commands, results, n)
    values = [StepResult.from_native(r) for r in results]
    for b, a, r in zip(battles, actions, values):
        b._record(a, r)
    return values
