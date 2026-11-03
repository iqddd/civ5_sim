"""Checked execution, capability lifetime and real-match regressions."""

import unittest
from dataclasses import asdict, replace
from pathlib import Path

from test_maps import field

from civ5sim import Action, Battle, MatchConfig, run_match
from civ5sim.agents import EndTurnAgent, load_agent
from civ5sim.combat_ai import (
    LIBRARY_PATH,
    Assignment,
    CallPlanningServices,
    ContactMemory,
    PlanningState,
    SearchConfig,
    SearchReason,
    SearchResult,
    SearchStats,
    SupportMode,
)
from civ5sim.combat_ai import (
    AssignmentKind as K,
)
from civ5sim.cp_agent import CPAgent, hex_distance
from civ5sim.engine import digest
from civ5sim.match import load_match_config

ROOT = Path(__file__).resolve().parents[1]
SMALL = SearchConfig(position_capacity=180, max_completed=12)


def scenario(units, *, fog=False, random_damage=False, width=8, height=5):
    return {"schema": 1, "kind": "civ5sim.scenario", "map": field(width, height),
            "units": units, "fog": fog, "random_damage": random_damage, "seed": 42}


def unit(kind, side, tile, **extra):
    return {"type": kind, "side": side, "tile": tile, **extra}


def assignment(kind, uid, start, end, moves=0, *, damage=0, retaliation=0,
               killed=-1, target=None):
    return Assignment(kind, uid, start, end, moves, 0, 0, damage, retaliation,
                      killed, end if target is None else target)


class ScriptedPlanning:
    def __init__(self, *plans):
        self.plans = iter(plans)
        self.calls = []

    def search(self, target, selected, *, contacts, aggression=None):
        self.calls.append((target, selected, contacts))
        return next(self.plans, plan())

    def support(self, target, selected, *, contacts, mode):
        return plan()

    def attack_candidates(self, target, selected, *, contacts):
        # Dispatch tests script recruitment separately from physical rules.
        return tuple(selected)


def plan(*assignments, reason=SearchReason.COMPLETED):
    return SearchResult(assignments, reason, SearchStats())


@unittest.skipUnless(LIBRARY_PATH.exists(), "Build combat and AI libraries first")
class ExecutorTests(unittest.TestCase):
    def agent(self, battle):
        a = CPAgent(SMALL)
        a.reset({"side": 0, "map": battle._map})
        self.addCleanup(a.close)
        return a

    def act(self, a, b, planning):
        return a.act_with_planning(b.observe(0), tuple(b.available_actions()), planning)

    def execute(self, a, b, command):
        result = b.step(command)
        self.assertTrue(result.ok, result)
        a.on_action_result(command, result, b.observe(0))
        return result

    def assert_turn_ends(self, a, b, planning):
        for _ in range(40):
            command = self.act(a, b, planning)
            if command.kind == "end_turn":
                return
            self.assertEqual(command.kind, "skip")
            self.execute(a, b, command)
        self.fail("Coordinator did not finish the turn")

    def test_queued_focus_fire_and_finish_one_command_at_a_time(self):
        with Battle(scenario([unit("archer", 0, 17), unit("archer", 0, 10),
                              unit("warrior", 1, 19)])) as b:
            a = self.agent(b)
            preview = b.preview_attack(Action("ranged", 0, 19))
            p = ScriptedPlanning(plan(
                assignment(K.INITIAL, 0, 17, 17, 120, target=-1),
                assignment(K.RANGED, 0, 17, 19, 0, damage=preview.damage),
                assignment(K.FINISH, 0, 17, 17, 0, target=-1),
                assignment(K.FINISH, 1, 10, 10, 120, target=-1),
            ))
            command = self.act(a, b, p)
            self.assertEqual(command, Action("ranged", 0, 19))
            self.execute(a, b, command)
            self.assertTrue(a._queue)
            self.assertEqual(self.act(a, b, p), Action("skip", 1))
            self.execute(a, b, Action("skip", 1))
            self.assert_turn_ends(a, b, p)
            self.assertEqual(len(p.calls), 1)

    def test_swap_is_atomic_and_reverse_is_only_checked(self):
        with Battle(scenario([unit("warrior", 0, 17), unit("archer", 0, 18),
                              unit("warrior", 1, 20)])) as b:
            a = self.agent(b)
            p = ScriptedPlanning(plan(
                assignment(K.SWAP, 0, 17, 18, 60),
                assignment(K.SWAP_REVERSE, 1, 18, 17, 60, target=-1),
                assignment(K.FINISH, 0, 18, 18, 60, target=-1),
                assignment(K.FINISH, 1, 17, 17, 60, target=-1),
            ))
            command = self.act(a, b, p)
            self.assertEqual(command, Action("swap", 0, 18))
            self.execute(a, b, command)
            self.assertEqual(self.act(a, b, p), Action("skip", 0))
            self.execute(a, b, Action("skip", 0))
            self.assertEqual(self.act(a, b, p), Action("skip", 1))
            self.execute(a, b, Action("skip", 1))
            self.assert_turn_ends(a, b, p)
            self.assertEqual(len(p.calls), 1)

    def test_damage_mismatch_and_unexpected_survival_replan(self):
        with Battle(scenario([unit("horseman", 0, 17), unit("warrior", 1, 18)])) as b:
            a = self.agent(b)
            p = ScriptedPlanning(plan(
                assignment(K.MELEE, 0, 17, 18, 180, damage=100, killed=1),
                assignment(K.MOVE, 0, 17, 18, 0),
            ), plan())
            command = self.act(a, b, p)
            self.execute(a, b, command)
            self.assertFalse(a._queue)
            self.assert_turn_ends(a, b, p)
            self.assertEqual(len(p.calls), 2)
            self.assertTrue(any(e.get("reason") == "damage_changed" for e in a.drain_diagnostics()))

    def test_new_contact_cancels_even_when_predicted(self):
        with Battle(scenario([unit("horseman", 0, 16), unit("warrior", 1, 24),
                              unit("warrior", 1, 21)], fog=True)) as b:
            a = self.agent(b)
            before = b.observe(0)
            self.assertNotIn(2, [u["id"] for u in before["units"]])
            # Long move stops when the second enemy enters sight.
            with b.clone() as clone:
                result = clone.step(Action("move", 0, 19))
                state = clone.observe(0)
                moved = next(u for u in state["units"] if u["id"] == 0)
            self.assertTrue(result.ok)
            p = ScriptedPlanning(plan(
                assignment(K.MOVE, 0, 16, moved["tile"], moved["moves"], target=19),
                assignment(K.RESTART, 0, -1, -1, target=-1),
                assignment(K.FINISH, 0, moved["tile"], moved["tile"], moved["moves"]),
            ), plan())
            self.execute(a, b, self.act(a, b, p))
            self.assertFalse(a._queue)
            self.assertTrue(any(e.get("reason") == "new_contact" for e in a.drain_diagnostics()))

    def test_invalid_precondition_no_progress_and_empty_budget(self):
        with Battle(scenario([unit("warrior", 0, 17), unit("warrior", 1, 19)])) as b:
            a = self.agent(b)
            p = ScriptedPlanning(plan(assignment(K.MOVE, 0, 99, 18, 60)))
            self.assert_turn_ends(a, b, p)
            self.assertEqual(len(p.calls), 1)
            self.assertTrue(any(e.get("reason") == "no_progress" for e in a.drain_diagnostics()))
            b.step(Action("end_turn"))
            b.step(Action("end_turn"))
            a.reset({"side": 0, "map": b._map})
            p = ScriptedPlanning(plan(reason=SearchReason.BUDGET_EXHAUSTED))
            self.assert_turn_ends(a, b, p)
            self.assertTrue(any(e.get("reason") == "budget_exhausted" for e in a.drain_diagnostics()))

    def test_nonempty_budget_plan_and_unit_zero_tile_255(self):
        with Battle(scenario([unit("warrior", 0, 254), unit("warrior", 1, 239)],
                             width=16, height=16)) as b:
            a = self.agent(b)
            with b.clone() as clone:
                clone.step(Action("move", 0, 255))
                moved = clone.observe(0)["units"][0]
            p = ScriptedPlanning(plan(assignment(K.MOVE, 0, 254, 255, moved["moves"]),
                                      reason=SearchReason.BUDGET_EXHAUSTED))
            self.assertEqual(self.act(a, b, p), Action("move", 0, 255))
            self.execute(a, b, Action("move", 0, 255))
            self.assert_turn_ends(a, b, p)

    def test_no_contact_and_thirteen_selected_from_larger_army(self):
        with Battle(scenario([unit("warrior", 0, 0), unit("warrior", 1, 39)], fog=True)) as b:
            a = self.agent(b)
            p = ScriptedPlanning()
            self.assert_turn_ends(a, b, p)
            self.assertEqual(p.calls, [])
        with Battle(scenario([unit("warrior", 0, i) for i in range(15)] +
                             [unit("warrior", 1, 20)])) as b:
            a = self.agent(b)
            p = ScriptedPlanning(plan())
            self.act(a, b, p)
            self.assertEqual(len(p.calls[0][1]), 13)

    def test_reset_clears_memory_plan_and_closes_old_planner(self):
        with Battle(scenario([unit("warrior", 0, 17), unit("warrior", 1, 19)])) as b:
            a = self.agent(b)
            self.act(a, b, ScriptedPlanning(plan()))
            old = a.planner
            self.assertTrue(a.memory.known_enemies)
            a.reset({"side": 0, "map": b._map})
            self.assertFalse(old._handle)
            self.assertEqual(a.memory, ContactMemory())
            self.assertFalse(a._queue)
            self.assertEqual(a.drain_diagnostics(), [])

    def test_service_expires_and_search_does_not_change_real_rng(self):
        with Battle(scenario([unit("warrior", 0, 17), unit("warrior", 1, 19)],
                             random_damage=True)) as b:
            a = self.agent(b)
            original = b.snapshot()
            service = CallPlanningServices(a.planner, PlanningState.from_battle(b, 0), 0, 0)
            result = service.search(19, (0,), contacts=ContactMemory())
            self.assertIsInstance(result, SearchResult)
            self.assertEqual(b.snapshot(), original)
            service.expire()
            service.expire()
            with self.assertRaisesRegex(RuntimeError, "expired"):
                service.search(19, (0,), contacts=ContactMemory())

    def test_search_limit_bounds_replanning_despite_progress(self):
        with Battle(scenario([unit("horseman", 0, 0), unit("warrior", 1, 19)])) as b:
            a = self.agent(b)
            # Wrong predicted movement forces cancellation after each real move.
            p = ScriptedPlanning(*(plan(assignment(K.MOVE, 0, i, i + 1, 0))
                                   for i in range(3)))
            for target in (1, 2, 3):
                command = self.act(a, b, p)
                self.assertEqual(command, Action("move", 0, target))
                self.execute(a, b, command)
            self.assert_turn_ends(a, b, p)
            self.assertEqual(len(p.calls), 3)
            self.assertTrue(any(e.get("reason") == "search_limit" for e in a.drain_diagnostics()))

    def test_turn_change_discards_old_queue_and_integration_error_is_not_hidden(self):
        with Battle(scenario([unit("warrior", 0, 17), unit("warrior", 1, 19)])) as b:
            a = self.agent(b)
            p = ScriptedPlanning(plan(assignment(K.MOVE, 0, 17, 16, 60)))
            self.act(a, b, p)
            b.step(Action("end_turn"))
            b.step(Action("end_turn"))
            p = ScriptedPlanning(plan(reason=SearchReason.INVALID_REQUEST))
            with self.assertRaisesRegex(RuntimeError, "invalid_request"):
                self.act(a, b, p)
            self.assertFalse(a._queue)
            self.assertEqual(a._turn, 2)
            self.assertTrue(any(e.get("reason") == "turn_changed" for e in a.drain_diagnostics()))


@unittest.skipUnless(LIBRARY_PATH.exists(), "Build combat and AI libraries first")
class MatchIntegrationTests(unittest.TestCase):
    def config(self, **changes):
        m = field(8, 5)
        m["spawn_candidates"] = [[[2, 2], [2, 3]], [[4, 2], [4, 3]]]
        return MatchConfig(m, [["warrior", "archer"]] * 2, max_rounds=3, **changes)

    def verify_replay(self, match):
        with Battle(match.replay["initial"]) as b:
            for entry in match.replay["steps"]:
                self.assertEqual(asdict(b.step(entry["action"])), entry["result"])
                self.assertEqual(digest(b.snapshot()), entry["state_sha256"])
            self.assertEqual(b.snapshot(), match.final)

    def test_32_seeded_matches_repeatable_and_replayable(self):
        for seed in range(32):
            with self.subTest(seed=seed):
                c = self.config(seed=seed, fog=bool(seed & 1), random_damage=bool(seed & 2))
                c.map["row_parity"] = (seed >> 2) & 1
                c.map["wrap_x"] = bool(seed & 8)
                outcomes = []
                for _ in range(2):
                    agents = [CPAgent(SMALL), CPAgent(SMALL)]
                    try:
                        outcomes.append(run_match(c, agents))
                    finally:
                        for a in agents:
                            a.close()
                self.assertEqual(outcomes[0].result["status"], "completed", outcomes[0].result)
                self.assertEqual(outcomes[0], outcomes[1])
                self.assertTrue(outcomes[0].replay["ai_events"])
                self.verify_replay(outcomes[0])

    def test_service_expires_on_return_and_exception(self):
        for fail in (False, True):
            class Recorder(CPAgent):
                def act_with_planning(self, observation, actions, planning, fail=fail):
                    self.saved = planning
                    if fail:
                        raise RuntimeError("test failure")
                    return Action("end_turn")

            a = Recorder(SMALL)
            self.addCleanup(a.close)
            match = run_match(self.config(), [a, EndTurnAgent()])
            self.assertEqual(match.result["status"], "error" if fail else "completed")
            with self.assertRaisesRegex(RuntimeError, "expired"):
                a.saved.search(0, (), contacts=ContactMemory())
            with self.assertRaisesRegex(RuntimeError, "expired"):
                a.saved.support(0, (), contacts=ContactMemory(), mode=SupportMode.APPROACH)
            self.verify_replay(match)

    def test_final_action_checked_and_callback_error_preserves_prefix(self):
        class Fighter(CPAgent):
            def act_with_planning(self, observation, actions, planning):
                return next((a for a in actions if a.kind == "melee"), Action("end_turn"))

            def on_action_result(self, action, result, observation):
                self.checked = True
                raise RuntimeError("callback failure")

        a = Fighter(SMALL)
        self.addCleanup(a.close)
        c = self.config(fog=False, random_damage=False)
        c.map["spawn_candidates"] = [[[2, 2]], [[3, 2]]]
        c = replace(c, armies=[["horseman"], ["warrior"]])
        match = run_match(c, [a, EndTurnAgent()])
        self.assertTrue(a.checked)
        self.assertEqual(match.result["error"]["phase"], "on_action_result")
        self.assertEqual(match.result["commands"], 1)
        self.verify_replay(match)

    def test_builtin_and_missing_or_closed_planner(self):
        a = load_agent("cp")
        self.addCleanup(a.close)
        self.assertIsInstance(a, CPAgent)
        for closed in (False, True):
            class Broken(CPAgent):
                def reset(self, context, closed=closed):
                    if closed:
                        super().reset(context)
                        self.close()

            broken = Broken(SMALL)
            self.addCleanup(broken.close)
            match = run_match(self.config(), [broken, EndTurnAgent()])
            self.assertEqual(match.result["error"]["phase"], "reset")

    def test_builtin_contact_example_checks_last_command(self):
        config, agents = load_match_config(ROOT / "scenarios/cp-contact-match.json")
        match = run_match(config, agents)
        self.assertEqual(match.result["status"], "completed")
        self.assertIn(match.result["reason"], ("army_destroyed", "turn_limit"))
        # The callback may also cancel the remaining plan after the terminal
        # command; its action_result must still describe that exact command.
        last = [e for e in match.replay["ai_events"] if e['event'] == 'action_result'][-1]
        self.assertEqual(last["command"], match.result["commands"])
        self.assertEqual(last["action"], match.replay["steps"][-1]["action"])
        self.verify_replay(match)


class GeometryTests(unittest.TestCase):
    def test_parity_and_wrap_distance(self):
        m = field(8, 5)
        self.assertEqual(hex_distance(m, 0, 7), 7)
        m["wrap_x"] = True
        self.assertEqual(hex_distance(m, 0, 7), 1)
        from civ5map.maps import adjacent

        for parity in (0, 1):
            m["row_parity"] = parity
            for a in range(40):
                for direction in range(6):
                    xy = adjacent(m, a % 8, a // 8, direction)
                    if xy is not None:
                        self.assertEqual(hex_distance(m, a, xy[1] * 8 + xy[0]), 1)


if __name__ == "__main__":
    unittest.main()
