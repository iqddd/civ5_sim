"""Planner ABI/ownership tests, independent of installed Civ V sources."""

import ctypes as C
import unittest
from dataclasses import replace

from civ5sim import Battle
from civ5sim import _native as N
from civ5sim.combat_ai import (
    LIBRARY_PATH,
    Aggression,
    Contact,
    ContactMemory,
    Planner,
    PlanningState,
    PlanningTile,
    PlanningUnit,
    SearchConfig,
    SearchReason,
    SearchRequest,
    _Assignment,
    _Config,
    _Contacts,
    _Request,
    _Result,
    _State,
    _Tile,
    _Unit,
)


def state():
    return PlanningState(
        16,
        16,
        tiles=(PlanningTile(0, 0, 0, 0),) * 256,
        units=(PlanningUnit(0, 0, 0, 255), PlanningUnit(1, 1, 1, 0)),
        visibility=(1,) * 256,
        fog=False,
    )


def request(**changes):
    return replace(SearchRequest(state(), 0, 0, 255, (0,)), **changes)


@unittest.skipUnless(LIBRARY_PATH.exists(), "Build tools/build_combat_ai.py first")
class PlannerTests(unittest.TestCase):
    def planner(self, **changes):
        p = Planner(replace(SearchConfig(), **changes))
        self.addCleanup(p.close)
        return p

    def test_unit_zero_tile_255_and_complete_board(self):
        p = self.planner()
        self.assertIn("CP-07", p.version)
        result = p.search(request())
        self.assertEqual(result.reason, SearchReason.NO_PLAN)
        self.assertEqual(result.assignments, ())
        self.assertGreater(result.stats.checked, 0)

    def test_invalid_requests_do_not_poison_later_calls(self):
        p = self.planner()
        for changes in (
            {"selected_units": (-1,)},
            {"selected_units": (2,)},
            {"selected_units": (1,)},
            {"selected_units": (0, 0)},
            {"selected_units": (0,) * 14},
            {"target": -1},
            {"target": 256},
            {"side": 2},
            {"expected_turn": -1},
            {"state": replace(state(), visibility=(1,) * 255)},
            {"state": replace(state(), visibility=(2,) * 256)},
            {"state": replace(state(), width=15)},
            {"state": replace(state(), units=(replace(state().units[0], id=1),))},
            {"state": replace(state(), units=(replace(state().units[0], hp=0),))},
            {"state": replace(state(), units=(replace(state().units[0], hp=101),))},
            {
                "state": replace(
                    state(),
                    units=(state().units[0], replace(state().units[1], tile=255)),
                )
            },
        ):
            with self.subTest(changes=changes):
                self.assertEqual(
                    p.search(request(**changes)).reason, SearchReason.INVALID_REQUEST
                )
        self.assertEqual(p.search(request()).reason, SearchReason.NO_PLAN)

    def test_empty_group_and_turn_change(self):
        p = self.planner()
        self.assertEqual(
            p.search(request(selected_units=())).reason, SearchReason.NO_PLAN
        )
        for r in (request(expected_turn=1), request(side=1, selected_units=(1,))):
            self.assertEqual(p.search(r).reason, SearchReason.TURN_CHANGED)

    def test_contact_memory_crosses_abi_and_rejects_invalid_history(self):
        p = self.planner()
        for memory in (ContactMemory(0, 0, (Contact(1, 0),)), ContactMemory()):
            self.assertEqual(
                p.search(request(contacts=memory)).reason, SearchReason.NO_PLAN
            )
        for memory in (
            ContactMemory(-1),
            ContactMemory(1),
            ContactMemory(0, 1),
            ContactMemory(0, -1, (Contact(1, 0),)),
            ContactMemory(0, 0, (Contact(0, 0),)),
            ContactMemory(0, 0, (Contact(-1, 0),)),
            ContactMemory(0, 0, (Contact(2, 0),)),
            ContactMemory(0, 0, (Contact(1, -1),)),
            ContactMemory(0, 0, (Contact(1, 1),)),
            ContactMemory(0, 0, (Contact(1, 0), Contact(1, 0))),
            ContactMemory(0, 0, (Contact(1, 0),) * 513),
        ):
            with self.subTest(memory=memory):
                self.assertEqual(
                    p.search(request(contacts=memory)).reason,
                    SearchReason.INVALID_REQUEST,
                )
        dead_enemy = replace(state().units[1], hp=0)
        self.assertEqual(
            p.search(
                request(
                    state=replace(state(), units=(state().units[0], dead_enemy)),
                    contacts=ContactMemory(0, 0, (Contact(1, 0),)),
                )
            ).reason,
            SearchReason.NO_PLAN,
        )

    def test_thirteen_selected_does_not_limit_army_or_enemy_count(self):
        units = tuple(PlanningUnit(i, 0, 0 if i < 20 else 1, i) for i in range(256))
        p = self.planner(position_capacity=1)
        self.assertEqual(
            p.search(
                request(
                    state=replace(state(), units=units), selected_units=tuple(range(13))
                )
            ).reason,
            SearchReason.BUDGET_EXHAUSTED,
        )

    def test_config_defaults_match_native_and_invalid_budgets_fail(self):
        p = self.planner()
        native = p._lib.cai_default_config()
        self.assertEqual(
            [getattr(native, name) for name, _ in _Config._fields_],
            [6000, 2560, 5, 5, 2, 1, 0, 1],
        )
        for changes in (
            {"position_capacity": 0},
            {"position_capacity": 1_000_001},
            {"max_completed": 0},
            {"max_completed": 4001},
            {"max_branches": 1},
            {"max_branches": 10},
            {"choices_per_unit": 1},
            {"choices_per_unit": 10},
            {"privileged_cp_queries": False},
            {"aggression": 99},
            {"risk_budget_base": -1},
            {"risk_budget_base": 14},
            {"risk_budget_per_kill": -1},
            {"risk_budget_per_kill": 14},
        ):
            with (
                self.subTest(changes=changes),
                self.assertRaises((ValueError, TypeError)),
            ):
                Planner(replace(SearchConfig(), **changes))

    def test_i32_overflow_and_bool_cannot_silently_become_an_id(self):
        p = self.planner()
        for ids in ((2**32,), (True,), (-(2**31) - 1,)):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                p.search(request(selected_units=ids))

    def test_independent_contexts_close_and_input_ownership(self):
        first, second = self.planner(aggression=Aggression.HIGH), self.planner()
        r = request()
        expected = second.search(r)
        first.search(request(target=-1))
        first.close()
        first.close()
        with self.assertRaises(RuntimeError):
            first.search(r)
        self.assertEqual(second.search(r), expected)
        self.assertEqual(r, request())
        with Planner() as temporary:
            self.assertEqual(temporary.search(r), expected)
        with self.assertRaises(RuntimeError):
            temporary.__enter__()

    def test_c_api_bad_pointer_resets_result(self):
        p = self.planner()
        result = _Result(0, 37)
        status = p._lib.cai_search(p._handle, None, None, 0, C.byref(result))
        self.assertEqual(status, 1)
        self.assertEqual(result.assignment_count, 0)
        self.assertEqual(result.reason, 4)
        p._lib.cai_free(None)

    def test_c_api_buffer_retry_is_deterministic_and_never_partial(self):
        p = self.planner(position_capacity=100, max_completed=10)
        tiles = (_Tile * 35)(*(_Tile(0, 0, 0, 0) for _ in range(35)))
        units = (_Unit * 2)(
            _Unit(0, 0, 0, 16, 100, 120, 0, 0, 0, 0),
            _Unit(1, 0, 1, 17, 1, 120, 0, 0, 0, 0),
        )
        visibility = (C.c_int32 * 35)(*([1] * 35))
        selected = (C.c_int32 * 1)(0)
        state = _State(7, 5, 0, 0, tiles, 35, units, 2, visibility, 35, 0, 0, 0)
        request = _Request(C.pointer(state), 0, 0, 17, selected, 1, _Contacts())
        request.contacts.side = -1
        result = _Result()
        call = p._lib.cai_search
        self.assertEqual(call(p._handle, C.byref(request), None, 0, C.byref(result)), 3)
        required = result.assignment_count
        self.assertGreater(required, 1)
        expected_stats = bytes(result.stats)
        buffer = (_Assignment * required)()
        buffer[0].unit = 123456
        self.assertEqual(call(p._handle, C.byref(request), buffer, 1, C.byref(result)), 3)
        self.assertEqual(result.assignment_count, required)
        self.assertEqual(buffer[0].unit, 123456)
        self.assertEqual(call(p._handle, C.byref(request), buffer, required, C.byref(result)), 0)
        self.assertEqual(bytes(result.stats), expected_stats)
        self.assertTrue(any(a.killed_unit == 1 for a in buffer))

    def test_accumulated_visibility_valid_but_missing_current_source_invalid(self):
        p = self.planner()
        historical = replace(state(), fog=True)
        self.assertNotEqual(p.search(request(state=historical)).reason, SearchReason.INVALID_REQUEST)
        missing = replace(historical, visibility=(0,) * len(historical.visibility))
        self.assertEqual(p.search(request(state=missing)).reason, SearchReason.INVALID_REQUEST)

    def test_later_turn_future_attack_keeps_hypothetical_contact_memory(self):
        from civ5sim.combat_ai import AssignmentKind

        s = PlanningState(
            7, 5, (PlanningTile(0, 0, 0, 0),) * 35,
            (PlanningUnit(0, 4, 0, 16, moves=240),
             PlanningUnit(1, 0, 1, 17, hp=1), PlanningUnit(2, 3, 1, 18)),
            (1,) * 35, turn=2, fog=False,
        )
        p = self.planner(position_capacity=1000, max_completed=80)
        for memory in (ContactMemory(), ContactMemory(0, 0, (Contact(1, 0),))):
            r = p.search(SearchRequest(s, 0, 2, 17, (0,), memory))
            self.assertIn(r.reason, (SearchReason.COMPLETED, SearchReason.BUDGET_EXHAUSTED))
            self.assertTrue(any(a.kind == AssignmentKind.MELEE for a in r.assignments))

    def test_unavailable_selected_units_do_not_veto_active_units(self):
        from civ5sim.combat_ai import AssignmentKind

        s = PlanningState(
            7, 5, (PlanningTile(0, 0, 0, 0),) * 35,
            (PlanningUnit(0, 1, 0, 16, hp=1, moves=0, attacked=1, acted=1),
             PlanningUnit(1, 1, 0, 9), PlanningUnit(2, 0, 1, 17)),
            (1,) * 35, fog=False,
        )
        p = self.planner(position_capacity=1000, max_completed=80, risk_budget_per_kill=0)
        expected = p.search(SearchRequest(s, 0, 0, 17, (1,)))
        self.assertTrue(expected.assignments)
        for passive in (s.units[0], replace(s.units[0], moves=120, waiting=1)):
            current = replace(s, units=(passive, *s.units[1:]))
            plan = p.search(SearchRequest(current, 0, 0, 17, (0, 1)))
            self.assertTrue(any(a.unit == 1 and a.kind in (
                AssignmentKind.MOVE, AssignmentKind.MELEE, AssignmentKind.RANGED,
            ) for a in plan.assignments))
            self.assertFalse(any(a.unit == 0 and a.kind not in (
                AssignmentKind.INITIAL, AssignmentKind.BLOCKED,
            ) for a in plan.assignments))

    @unittest.skipUnless(
        (N.ROOT / "build/combat.dll").exists()
        or (N.ROOT / "build/libcombat.so").exists(),
        "Build tools/build_combat.py first",
    )
    def test_detached_privileged_input_does_not_change_battle_or_rng(self):
        from test_combat import scenario, unit
        from test_maps import field

        with Battle(
            scenario(
                unit(x=0, y=0),
                unit(side=1, x=15, y=15),
                fog=True,
                random_damage=True,
                terrain=field(16, 16),
            )
        ) as battle:
            before = battle.snapshot()
            copied = PlanningState.from_battle(battle, 0)
            self.assertEqual(len(battle.observe(0)["units"]), 1)
            self.assertEqual(len(copied.units), 2)
            self.assertFalse(hasattr(copied, "rng"))
            p = self.planner()
            result = p.search(SearchRequest(copied, 0, copied.turn, 255, (0,)))
            # No destination in this turn's reach belongs to the distant CP
            # tactical area. This is an approach task, not a tactical plan.
            self.assertEqual(result.reason, SearchReason.NO_PLAN)
            self.assertEqual(battle.snapshot(), before)
        self.assertEqual(
            p.search(SearchRequest(copied, 0, 0, 255, (0,))).reason,
            result.reason,
        )


if __name__ == "__main__":
    unittest.main()
