"""CP-07 tactical and support planners, branch model and danger estimates.

This module is an internal privileged boundary for the CP-source v1 agent.
Normal Agent.act observations never acquire a PlanningState implicitly.
"""

import ctypes as C
import json
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
    RECOVERY = 3


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
    damage_taken_last_turn: int = 0


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
    visibility_history: tuple[tuple[int, ...], ...] = ()
    turn_reset_mode: str = "legacy"

    @classmethod
    def from_battle(cls, battle: Battle, side: int, *, observation: dict | None = None):
        """Organizer-only copy; observation must be from this state and side."""
        integer(side, "side", 0, 1)
        observed = battle.observe(side) if observation is None else observation
        if observed["side"] != side:
            raise ValueError("Planning observation belongs to another side")
        # Battle owns immutable terrain; unit/visibility state is always fresh.
        tiles = getattr(battle, "_planning_tiles", None)
        if tiles is None:
            tiles = tuple(
                PlanningTile(*(getattr(t, name) for name in _field_names(PlanningTile)))
                for t in battle._encode_tiles()
            )
            battle._planning_tiles = tiles
        return cls(
            width=battle._map["width"],
            height=battle._map["height"],
            tiles=tiles,
            units=tuple(
                PlanningUnit(i, *(0 if name == "damage_taken_last_turn" and u.side != side else getattr(u, name)
                                  for name in _field_names(PlanningUnit)[1:]))
                for i, u in enumerate(battle._units())
            ),
            visibility=tuple(int(v) for v in observed["visibility"]),
            active_side=observed["active_side"],
            turn=observed["turn"],
            row_parity=battle._map["row_parity"],
            wrap_x=battle._map["wrap_x"],
            fog=bool(battle._info().fog),
            turn_reset_mode=battle.turn_reset_mode,
            visibility_history=tuple(tuple(map(int, row)) for row in battle.visibility_history()),
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
class SearchDiagnostics:
    budget_exhausted: bool = False
    risk_rejections: int = 0
    improvement_rejections: int = 0
    bad_units: tuple[int, ...] = ()
    recovery_kind: str = "none"


@dataclass(frozen=True)
class SearchResult:
    assignments: tuple[Assignment, ...]
    reason: SearchReason
    stats: SearchStats
    diagnostics: SearchDiagnostics = field(default_factory=SearchDiagnostics)


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

    def attack_candidates(self, target: int, selected_units: tuple[int, ...], *,
                          contacts: ContactMemory) -> tuple[int, ...]: ...


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

    def __init__(self, planner: "Planner", state: PlanningState, side: int, turn: int,
                 *, metrics=None):
        self._planner = planner
        self._state = state
        self._side = side
        self._turn = turn
        self._metrics = metrics
        self._packed_state = None

    def _packed(self):
        # This capability owns the buffers, including when services sharing a
        # planner are interleaved. Nothing dynamic is cached on the planner.
        if self._packed_state is None:
            self._packed_state = self._planner._pack_state(self._state)
        return self._packed_state

    def search(self, target, selected_units, *, contacts, aggression=None):
        if self._state is None:
            raise RuntimeError("Planning service has expired")
        from .telemetry import measure

        with measure(self._metrics, "search", side=self._side, turn=self._turn):
            return self._planner.search(SearchRequest(
                self._state, self._side, self._turn, target, tuple(selected_units), contacts, aggression
            ), _packed_state=self._packed())

    def support(self, target, selected_units, *, contacts, mode):
        if self._state is None:
            raise RuntimeError("Planning service has expired")
        from .telemetry import measure

        with measure(self._metrics, "support", side=self._side, turn=self._turn):
            return self._planner.support(SearchRequest(
                self._state, self._side, self._turn, target, tuple(selected_units), contacts
            ), mode, _packed_state=self._packed())

    def attack_candidates(self, target, selected_units, *, contacts):
        if self._state is None:
            raise RuntimeError("Planning service has expired")
        from .telemetry import measure

        with measure(self._metrics, "candidates", side=self._side, turn=self._turn):
            return self._planner.attack_candidates(SearchRequest(
                self._state, self._side, self._turn, target, tuple(selected_units), contacts
            ), _packed_state=self._packed())

    def expire(self):
        self._state = None
        self._planner = None
        self._metrics = None
        self._packed_state = None


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
        ("visibility_history", C.POINTER(C.c_int32)),
        ("visibility_history_count", C.c_int32),
        ("turn_reset_mode", C.c_int32),
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


class _Diagnostics(C.Structure):
    _fields_ = [("budget_exhausted", C.c_int32), ("risk_rejections", C.c_int32),
                ("improvement_rejections", C.c_int32), ("bad_unit_count", C.c_int32),
                ("bad_units", C.c_int32 * 13), ("recovery_kind", C.c_int32)]


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
    if lib.cai_abi() != 8:
        raise RuntimeError("Unsupported combat AI ABI; rebuild the planner")
    lib.cai_support.argtypes = [C.c_void_p, C.POINTER(_Request), C.c_int32,
                               C.POINTER(_Assignment), C.c_int32, C.POINTER(_Result)]
    lib.cai_support.restype = C.c_int32
    # ABI6 additions are optional for read-only comparisons to older planners.
    if hasattr(lib, "cai_search_diagnostics"):
        lib.cai_search_diagnostics.argtypes = [C.c_void_p, C.POINTER(_Diagnostics)]
        lib.cai_search_diagnostics.restype = C.c_int32
    if hasattr(lib, "cai_attack_candidates"):
        lib.cai_attack_candidates.argtypes = [C.c_void_p, C.POINTER(_Request), C.POINTER(C.c_int32),
                                            C.c_int32, C.POINTER(C.c_int32)]
        lib.cai_attack_candidates.restype = C.c_int32
    return lib


def _i32(value):
    return integer(value, "ABI value", -(2**31), 2**31 - 1)


@lru_cache(maxsize=None)
def _field_names(value_type):
    return tuple(field.name for field in fields(value_type))


def _pack(structure, value):
    return structure(*(_i32(getattr(value, name)) for name in _field_names(type(value))))


@dataclass
class _PackedState:
    native: _State
    # Explicit ownership keeps every array alive through all native calls.
    buffers: tuple


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
        self._assignment_buffer = (_Assignment * 256)()
        self._tile_cache = None

    @property
    def version(self):
        return self._lib.cai_version().decode("utf-8")

    def close(self):
        if self._handle:
            self._lib.cai_free(self._handle)
            self._handle = None
        self._tile_cache = None

    def __enter__(self):
        if not self._handle:
            raise RuntimeError("Planner is closed")
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    def search(self, request: SearchRequest, *, _packed_state=None) -> SearchResult:
        return self._plan(request, packed_state=_packed_state)

    def support(self, request: SearchRequest, mode: SupportMode, *, _packed_state=None) -> SearchResult:
        if not isinstance(mode, SupportMode):
            raise TypeError("mode must be a SupportMode value")
        return self._plan(request, mode, packed_state=_packed_state)

    def attack_candidates(self, request: SearchRequest, *, _packed_state=None) -> tuple[int, ...]:
        return self._plan(request, candidates_only=True, packed_state=_packed_state)

    def set_trace(self, enabled, max_bytes=64 * 1024 * 1024):
        if not hasattr(self._lib, "cai_trace_enable"):
            if enabled:
                raise RuntimeError("Rebuild combat AI for decision tracing")
            return
        self._lib.cai_trace_enable.argtypes = [C.c_void_p, C.c_int32, C.c_int32]
        self._lib.cai_trace_enable.restype = C.c_int32
        self._lib.cai_trace_data.argtypes = [C.c_void_p]
        self._lib.cai_trace_data.restype = C.c_char_p
        self._lib.cai_trace_status.argtypes = [C.c_void_p]
        self._lib.cai_trace_status.restype = C.c_int32
        if self._lib.cai_trace_enable(self._handle, int(enabled), max_bytes):
            raise ValueError("Invalid diagnostic trace settings")

    def read_trace(self):
        if not hasattr(self._lib, "cai_trace_data"):
            return {"available": False, "events": []}
        self._lib.cai_trace_data.argtypes = [C.c_void_p]
        self._lib.cai_trace_data.restype = C.c_char_p
        self._lib.cai_trace_status.argtypes = [C.c_void_p]
        self._lib.cai_trace_status.restype = C.c_int32
        data = self._lib.cai_trace_data(self._handle) or b""
        return {"available": True, "truncated": bool(self._lib.cai_trace_status(self._handle)),
                "events": [json.loads(row) for row in data.splitlines()]}

    def _plan(self, request: SearchRequest, mode=None, *, candidates_only=False, packed_state=None):
        recorder = getattr(self, "_decision_recorder", None)
        if recorder is not None:
            try:
                self.set_trace(recorder.mode == "full")
            except Exception as exc:  # noqa: BLE001 -- diagnostics must not change decisions
                recorder.errors.append(f"trace setup: {exc}")
        result = self._execute_plan(request, mode, candidates_only=candidates_only,
                                    packed_state=packed_state)
        if recorder is not None:
            trace = None
            if recorder.mode == "full":
                try:
                    trace = self.read_trace()
                except Exception as exc:  # noqa: BLE001 -- preserve planner result
                    recorder.errors.append(f"trace read: {exc}")
            recorder.result(request, self.config, "candidates" if candidates_only else
                            "search" if mode is None else int(mode), result, trace, self.version)
        return result

    def _pack_state(self, s: PlanningState) -> _PackedState:
        if not self._handle:
            raise RuntimeError("Planner is closed")
        boolean(s.wrap_x, "wrap_x")
        boolean(s.fog, "fog")
        if s.turn_reset_mode not in ("legacy", "end_turn"):
            raise ValueError("turn_reset_mode must be legacy or end_turn")
        # Bound Python-side allocations before constructing any ctypes array.
        if len(s.tiles) > 4096 or len(s.units) > 512 or len(s.visibility) > 4096:
            raise ValueError("Planning state exceeds kernel limits")
        cached = self._tile_cache
        if cached is not None and cached[0] is s.tiles:
            tiles = cached[1]
        else:
            tiles = (_Tile * len(s.tiles))(*(_pack(_Tile, t) for t in s.tiles))
            # Only immutable terrain survives between planning capabilities.
            self._tile_cache = (
                (s.tiles, tiles)
                if isinstance(s.tiles, tuple)
                and all(type(tile) is PlanningTile for tile in s.tiles)
                else None
            )
        units = (_Unit * len(s.units))(*(_pack(_Unit, u) for u in s.units))
        visibility = (C.c_int32 * len(s.visibility))(*map(_i32, s.visibility))
        if s.visibility_history and (len(s.visibility_history) != 2 or any(
                len(row) != len(s.tiles) for row in s.visibility_history)):
            raise ValueError("Invalid planning visibility history")
        history_values = [v for row in s.visibility_history for v in row]
        history = (C.c_int32 * len(history_values))(*map(_i32, history_values))
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
            history if history_values else None,
            len(history_values),
            int(s.turn_reset_mode == "end_turn"),
        )
        return _PackedState(state, (tiles, units, visibility, history))

    def _execute_plan(self, request: SearchRequest, mode=None, *, candidates_only=False,
                      packed_state=None):
        if not self._handle:
            raise RuntimeError("Planner is closed")
        if request.aggression is not None and not isinstance(request.aggression, Aggression):
            raise TypeError("aggression must be an Aggression value or None")
        if (
            len(request.selected_units) > (512 if candidates_only else 13)
            or len(request.contacts.known_enemies) > 512
        ):
            if candidates_only:
                raise ValueError("Invalid attack candidate query")
            return SearchResult((), SearchReason.INVALID_REQUEST, SearchStats())
        if packed_state is None:
            packed_state = self._pack_state(request.state)
        selected = (C.c_int32 * len(request.selected_units))(
            *map(_i32, request.selected_units)
        )
        known = (_Contact * len(request.contacts.known_enemies))(
            *(_pack(_Contact, c) for c in request.contacts.known_enemies)
        )
        packed = _Request(
            C.pointer(packed_state.native),
            _i32(request.side),
            _i32(request.expected_turn),
            _i32(request.target),
            selected,
            len(selected),
            _Contacts(_i32(request.contacts.turn), _i32(request.contacts.side), known, len(known)),
            0 if request.aggression is None else int(request.aggression) + 1,
        )
        if candidates_only:
            if not hasattr(self._lib, "cai_attack_candidates"):
                raise RuntimeError("Rebuild combat AI for attack candidate queries")
            output = (C.c_int32 * 512)()
            count = C.c_int32()
            status = self._lib.cai_attack_candidates(self._handle, C.byref(packed), output, 512,
                                                     C.byref(count))
            if status or not 0 <= count.value <= len(request.selected_units):
                raise ValueError(f"Invalid attack candidate query (native status {status})")
            return tuple(output[:count.value])

        def invoke(buffer, capacity, result):
            args = (self._handle, C.byref(packed))
            if mode is None:
                return self._lib.cai_search(*args, buffer, capacity, C.byref(result))
            return self._lib.cai_support(*args, int(mode), buffer, capacity, C.byref(result))
        result = _Result()
        buffer = self._assignment_buffer
        capacity = len(buffer)
        status = invoke(buffer, capacity, result)
        assignments = ()
        if status == 3:  # Unusually long plans grow the retained output buffer.
            capacity = result.assignment_count
            if not 0 < capacity <= 1_000_000:
                raise RuntimeError("Invalid native plan size")
            buffer = (_Assignment * capacity)()
            self._assignment_buffer = buffer
            status = invoke(buffer, capacity, result)
        if status == 0:
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
        diagnostics = SearchDiagnostics()
        if hasattr(self._lib, "cai_search_diagnostics"):
            packed_diagnostics = _Diagnostics()
            if self._lib.cai_search_diagnostics(self._handle, C.byref(packed_diagnostics)) != 0:
                raise RuntimeError("Native diagnostics failed")
            if not 0 <= packed_diagnostics.bad_unit_count <= 13 or not 0 <= packed_diagnostics.recovery_kind <= 3:
                raise RuntimeError("Invalid native diagnostics")
            diagnostics = SearchDiagnostics(bool(packed_diagnostics.budget_exhausted),
                packed_diagnostics.risk_rejections, packed_diagnostics.improvement_rejections,
                tuple(packed_diagnostics.bad_units[:packed_diagnostics.bad_unit_count]),
                ("none", "attack", "retreat", "wait")[packed_diagnostics.recovery_kind])
        return SearchResult(
            assignments,
            list(SearchReason)[result.reason],
            SearchStats(*(getattr(result.stats, f.name) for f in fields(SearchStats))),
            diagnostics,
        )
