"""CP-06 tactical and support planners, branch model and danger estimates.

This module is an internal privileged boundary for the CP-source v1 agent.
Normal Agent.act observations never acquire a PlanningState implicitly.
"""

import ctypes as C
import os
from dataclasses import dataclass, field, fields
from enum import IntEnum, StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from .engine import Action, Battle, boolean, integer

ROOT = Path(__file__).resolve().parents[1]
LIBRARY_PATH = (
    ROOT / "build" / ("combat_ai.dll" if os.name == "nt" else "libcombat_ai.so")
)


class Aggression(IntEnum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    BRAVEHEART = 4


class SearchReason(StrEnum):
    COMPLETED = "completed"
    NO_PLAN = "no_plan"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TURN_CHANGED = "turn_changed"
    INVALID_REQUEST = "invalid_request"
    NOT_IMPLEMENTED = "not_implemented"


class AssignmentKind(IntEnum):
    INITIAL = 0
    MOVE = 1
    MELEE = 2
    RANGED = 3
    FINISH = 4
    BLOCKED = 5
    RESTART = 6
    SWAP = 7
    SWAP_REVERSE = 8
    FINISH_TEMP = 9


class SupportMode(IntEnum):
    HEAL_BEFORE = 0
    HEAL_AFTER = 1
    APPROACH = 2


@dataclass(frozen=True)
class SearchConfig:
    position_capacity: int = 6000
    max_completed: int = 2560
    max_branches: int = 5
    choices_per_unit: int = 5
    aggression: Aggression = Aggression.MEDIUM
    privileged_cp_queries: bool = True
    risk_budget_base: int = 0
    risk_budget_per_kill: int = 1


@dataclass(frozen=True)
class PlanningTile:
    terrain: int
    elevation: int
    feature: int
    rivers: int


@dataclass(frozen=True)
class PlanningUnit:
    id: int
    type: int
    side: int
    tile: int
    hp: int = 100
    moves: int = 120
    attacked: int = 0
    acted: int = 0
    fortified: int = 0
    waiting: int = 0


@dataclass(frozen=True)
class PlanningState:
    """Owned, RNG-free root input; no handle to the real battle is retained.

    Native unit types, feature/terrain enums and six-bit river masks match the
    combat kernel. Visibility is for the side in the search request. Dead units
    keep their stable roster slots. Tuple fields are immutable by contract.
    """

    width: int
    height: int
    tiles: tuple[PlanningTile, ...]
    units: tuple[PlanningUnit, ...]
    visibility: tuple[int, ...]
    active_side: int = 0
    turn: int = 0
    row_parity: int = 0
    wrap_x: bool = False
    fog: bool = True

    @classmethod
    def from_battle(cls, battle: Battle, side: int):
        """Organizer-only copy for privileged CP v1; never stores RNG."""
        integer(side, "side", 0, 1)
        observed = battle.observe(side)
        return cls(
            width=battle._map["width"],
            height=battle._map["height"],
            tiles=tuple(
                PlanningTile(*(getattr(t, f.name) for f in fields(PlanningTile)))
                for t in battle._encode_tiles()
            ),
            units=tuple(
                PlanningUnit(i, *(getattr(u, f.name) for f in fields(PlanningUnit)[1:]))
                for i, u in enumerate(battle._units())
            ),
            visibility=tuple(int(v) for v in observed["visibility"]),
            active_side=observed["active_side"],
            turn=observed["turn"],
            row_parity=battle._map["row_parity"],
            wrap_x=battle._map["wrap_x"],
            fog=bool(battle._info().fog),
        )


@dataclass(frozen=True)
class Contact:
    id: int
    last_seen_own_turn: int


def own_turn(turn: int, active_side: int, observer: int) -> int:
    """Observer-turn ordinal; initial offset depends on the starting side."""
    integer(turn, "turn", 0, 2**31 - 1)
    integer(active_side, "active_side", 0, 1)
    integer(observer, "observer", 0, 1)
    return (turn + int(active_side == observer)) // 2


@dataclass(frozen=True)
class ContactMemory:
    """Active contacts; update only from real observations in the CP agent.

    A hidden contact expires two observer-turn starts after its last sighting.
    The default unbound empty memory can initialize either side of a new match.
    """

    turn: int = 0
    side: int = -1
    known_enemies: tuple[Contact, ...] = ()

    def updated(self, observation: dict) -> "ContactMemory":
        """Return detached real history; neither hidden queries nor search update it."""
        side = integer(observation["side"], "side", 0, 1)
        turn = integer(observation["turn"], "turn", 0, 2**31 - 1)
        active = integer(observation["active_side"], "active_side", 0, 1)
        integer(self.turn, "contact turn", 0, turn)
        integer(self.side, "contact side", -1, 1)
        if self.side != side and not (
            self.side == -1 and self.turn == 0 and not self.known_enemies
        ):
            raise ValueError("Contact memory belongs to another side")
        if len(self.known_enemies) > 512:
            raise ValueError("Too many contacts")
        now = own_turn(turn, active, side)
        previous = own_turn(self.turn, active ^ ((turn - self.turn) & 1), side)
        remembered = {}
        seen = set()
        for c in self.known_enemies:
            integer(c.id, "contact id", 0, 511)
            integer(c.last_seen_own_turn, "last seen turn", 0, previous)
            if c.id in seen:
                raise ValueError("Duplicate contact")
            seen.add(c.id)
            if now - c.last_seen_own_turn <= 1:
                remembered[c.id] = c
        for u in observation["units"]:
            unit_side = integer(u["side"], "unit side", 0, 1)
            if unit_side != side and u["hp"] > 0:
                uid = integer(u["id"], "contact id", 0, 511)
                remembered[uid] = Contact(uid, now)
        return ContactMemory(turn, side, tuple(remembered[k] for k in sorted(remembered)))


@dataclass(frozen=True)
class SearchRequest:
    state: PlanningState
    side: int
    expected_turn: int
    target: int
    selected_units: tuple[int, ...]
    contacts: ContactMemory = field(default_factory=ContactMemory)
    aggression: Aggression | None = None


@dataclass(frozen=True)
class Assignment:
    kind: AssignmentKind
    unit: int
    from_tile: int
    to_tile: int
    remaining_moves: int
    movement_role: int
    score: int
    damage: int
    self_damage: int
    killed_unit: int
    command_target: int = -1


@dataclass(frozen=True)
class SearchStats:
    checked: int = 0
    generated: int = 0
    completed: int = 0
    pruned: int = 0
    peak_positions: int = 0


@dataclass(frozen=True)
class SearchResult:
    assignments: tuple[Assignment, ...]
    reason: SearchReason
    stats: SearchStats


class PlanningServices(Protocol):
    """Call-scoped capability supplied by the match organizer.

    Implementation owns the detached input, borrows the agent's planner, checks call lifetime,
    and expires when act_with_planning returns. No Battle/RNG property.
    """

    def search(
        self, target: int, selected_units: tuple[int, ...], *, contacts: ContactMemory,
        aggression: Aggression | None = None,
    ) -> SearchResult: ...

    def support(
        self, target: int, selected_units: tuple[int, ...], *, contacts: ContactMemory,
        mode: SupportMode,
    ) -> SearchResult: ...


class PlanningAgent(Protocol):
    """Explicit opt-in dispatch, alongside the ordinary Agent protocol.

    The organizer dispatches agents declaring this interface; ordinary policies
    continue receiving act(observation, actions). Exactly one Action is returned.
    """

    planner: "Planner"

    def reset(self, context: dict) -> None: ...

    def act_with_planning(
        self, observation: dict, actions: tuple[Action, ...], planning: PlanningServices
    ) -> Action: ...


class CallPlanningServices:
    """Organizer-owned, expiring search capability; contains no Battle or RNG."""

    def __init__(self, planner: "Planner", state: PlanningState, side: int, turn: int):
        self._planner = planner
        self._state = state
        self._side = side
        self._turn = turn

    def search(self, target, selected_units, *, contacts, aggression=None):
        if self._state is None:
            raise RuntimeError("Planning service has expired")
        return self._planner.search(SearchRequest(
            self._state, self._side, self._turn, target, tuple(selected_units), contacts, aggression
        ))

    def support(self, target, selected_units, *, contacts, mode):
        if self._state is None:
            raise RuntimeError("Planning service has expired")
        return self._planner.support(SearchRequest(
            self._state, self._side, self._turn, target, tuple(selected_units), contacts
        ), mode)

    def expire(self):
        self._state = None
        self._planner = None


def _structure(name, names):
    return type(name, (C.Structure,), {"_fields_": [(n, C.c_int32) for n in names]})


_Config = _structure("_Config", [f.name for f in fields(SearchConfig)])
_Tile = _structure("_Tile", [f.name for f in fields(PlanningTile)])
_Unit = _structure("_Unit", [f.name for f in fields(PlanningUnit)])
_Assignment = _structure("_Assignment", [f.name for f in fields(Assignment)])
_Stats = _structure("_Stats", [f.name for f in fields(SearchStats)])


class _State(C.Structure):
    _fields_ = [
        ("width", C.c_int32),
        ("height", C.c_int32),
        ("row_parity", C.c_int32),
        ("wrap_x", C.c_int32),
        ("tiles", C.POINTER(_Tile)),
        ("tile_count", C.c_int32),
        ("units", C.POINTER(_Unit)),
        ("unit_count", C.c_int32),
        ("visibility", C.POINTER(C.c_int32)),
        ("visibility_count", C.c_int32),
        ("active_side", C.c_int32),
        ("turn", C.c_int32),
        ("fog", C.c_int32),
    ]


_Contact = _structure("_Contact", [f.name for f in fields(Contact)])


class _Contacts(C.Structure):
    _fields_ = [
        ("turn", C.c_int32),
        ("side", C.c_int32),
        ("known_enemies", C.POINTER(_Contact)),
        ("known_enemy_count", C.c_int32),
    ]


class _Request(C.Structure):
    _fields_ = [
        ("state", C.POINTER(_State)),
        ("side", C.c_int32),
        ("expected_turn", C.c_int32),
        ("target", C.c_int32),
        ("selected_units", C.POINTER(C.c_int32)),
        ("selected_count", C.c_int32),
        ("contacts", _Contacts),
        ("aggression_override", C.c_int32),
    ]


class _Result(C.Structure):
    _fields_ = [
        ("reason", C.c_int32),
        ("assignment_count", C.c_int32),
        ("stats", _Stats),
    ]


@lru_cache(maxsize=1)
def _library():
    if not LIBRARY_PATH.exists():
        raise RuntimeError("Build the planner first: python tools/build_combat_ai.py")
    lib = C.CDLL(str(LIBRARY_PATH))
    for name, arguments, result in (
        ("cai_abi", [], C.c_int32),
        ("cai_version", [], C.c_char_p),
        ("cai_default_config", [], _Config),
        ("cai_new", [C.POINTER(_Config)], C.c_void_p),
        ("cai_free", [C.c_void_p], None),
        (
            "cai_search",
            [
                C.c_void_p,
                C.POINTER(_Request),
                C.POINTER(_Assignment),
                C.c_int32,
                C.POINTER(_Result),
            ],
            C.c_int32,
        ),
    ):
        fn = getattr(lib, name)
        fn.argtypes, fn.restype = arguments, result
    if lib.cai_abi() != 5:
        raise RuntimeError("Unsupported combat AI ABI; rebuild the planner")
    lib.cai_support.argtypes = [C.c_void_p, C.POINTER(_Request), C.c_int32,
                               C.POINTER(_Assignment), C.c_int32, C.POINTER(_Result)]
    lib.cai_support.restype = C.c_int32
    return lib


def _i32(value):
    return integer(value, "ABI value", -(2**31), 2**31 - 1)


def _pack(structure, value):
    return structure(*(_i32(getattr(value, f.name)) for f in fields(value)))


class Planner:
    """Owned search context. Independent instances may run concurrently.

    Use as a context manager. Calls/close on the same instance must be serialized.
    Returns a local plan; CPAgent executes it through simulator Actions.
    """

    def __init__(self, config: SearchConfig | None = None):
        self._handle = None
        config = SearchConfig() if config is None else config
        self._lib = _library()
        boolean(config.privileged_cp_queries, "privileged_cp_queries")
        if not isinstance(config.aggression, Aggression):
            raise TypeError("aggression must be an Aggression value")
        packed = _Config(
            *(_i32(getattr(config, f.name)) for f in fields(config)[:4]),
            int(config.aggression),
            int(config.privileged_cp_queries),
            _i32(config.risk_budget_base),
            _i32(config.risk_budget_per_kill),
        )
        self._handle = self._lib.cai_new(C.byref(packed))
        if not self._handle:
            raise ValueError("Invalid search configuration or native allocation failed")
        self.config = config

    @property
    def version(self):
        return self._lib.cai_version().decode("utf-8")

    def close(self):
        if self._handle:
            self._lib.cai_free(self._handle)
            self._handle = None

    def __enter__(self):
        if not self._handle:
            raise RuntimeError("Planner is closed")
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    def search(self, request: SearchRequest) -> SearchResult:
        return self._plan(request)

    def support(self, request: SearchRequest, mode: SupportMode) -> SearchResult:
        if not isinstance(mode, SupportMode):
            raise TypeError("mode must be a SupportMode value")
        return self._plan(request, mode)

    def _plan(self, request: SearchRequest, mode=None) -> SearchResult:
        if not self._handle:
            raise RuntimeError("Planner is closed")
        s = request.state
        if request.aggression is not None and not isinstance(request.aggression, Aggression):
            raise TypeError("aggression must be an Aggression value or None")
        boolean(s.wrap_x, "wrap_x")
        boolean(s.fog, "fog")
        # Bound Python-side allocations before constructing any ctypes array.
        if len(s.tiles) > 4096 or len(s.units) > 512 or len(s.visibility) > 4096:
            raise ValueError("Planning state exceeds kernel limits")
        if (
            len(request.selected_units) > 13
            or len(request.contacts.known_enemies) > 512
        ):
            return SearchResult((), SearchReason.INVALID_REQUEST, SearchStats())
        tiles = (_Tile * len(s.tiles))(*(_pack(_Tile, t) for t in s.tiles))
        units = (_Unit * len(s.units))(*(_pack(_Unit, u) for u in s.units))
        visibility = (C.c_int32 * len(s.visibility))(*map(_i32, s.visibility))
        selected = (C.c_int32 * len(request.selected_units))(
            *map(_i32, request.selected_units)
        )
        known = (_Contact * len(request.contacts.known_enemies))(
            *(_pack(_Contact, c) for c in request.contacts.known_enemies)
        )
        state = _State(
            _i32(s.width),
            _i32(s.height),
            _i32(s.row_parity),
            int(s.wrap_x),
            tiles,
            len(tiles),
            units,
            len(units),
            visibility,
            len(visibility),
            _i32(s.active_side),
            _i32(s.turn),
            int(s.fog),
        )
        packed = _Request(
            C.pointer(state),
            _i32(request.side),
            _i32(request.expected_turn),
            _i32(request.target),
            selected,
            len(selected),
            _Contacts(_i32(request.contacts.turn), _i32(request.contacts.side), known, len(known)),
            0 if request.aggression is None else int(request.aggression) + 1,
        )
        def invoke(buffer, capacity, result):
            args = (self._handle, C.byref(packed))
            if mode is None:
                return self._lib.cai_search(*args, buffer, capacity, C.byref(result))
            return self._lib.cai_support(*args, int(mode), buffer, capacity, C.byref(result))
        result = _Result()
        status = invoke(None, 0, result)
        assignments = ()
        if (
            status == 3
        ):  # Repeat the deterministic search with caller-owned output storage.
            capacity = result.assignment_count
            if not 0 < capacity <= 1_000_000:
                raise RuntimeError("Invalid native plan size")
            buffer = (_Assignment * capacity)()
            status = invoke(buffer, capacity, result)
            if not 0 <= result.assignment_count <= capacity:
                raise RuntimeError("Native plan buffer overflow")
            assignments = tuple(
                Assignment(
                    AssignmentKind(a.kind),
                    *(getattr(a, f.name) for f in fields(Assignment)[1:]),
                )
                for a in buffer[: result.assignment_count]
            )
        if status != 0:
            raise RuntimeError(f"Native planner failed with status {status}")
        return SearchResult(
            assignments,
            list(SearchReason)[result.reason],
            SearchStats(*(getattr(result.stats, f.name) for f in fields(SearchStats))),
        )
