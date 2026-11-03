"""Failed combat rosters recover together without consuming unrelated armies."""

import unittest
from dataclasses import replace

from test_cp_agent import SMALL, ScriptedPlanning, assignment, plan, scenario, unit

from civ5sim import Action, Battle
from civ5sim.combat_ai import LIBRARY_PATH, SearchDiagnostics, SearchReason, SupportMode
from civ5sim.combat_ai import AssignmentKind as K
from civ5sim.cp_agent import CPAgent


@unittest.skipUnless(LIBRARY_PATH.exists(), "Build combat and AI libraries first")
class GroupRecoveryTests(unittest.TestCase):
    def setup_battle(self, units, **kwargs):
        self.battle = self.enterContext(Battle(scenario(units, **kwargs)))
        self.agent = CPAgent(SMALL)
        self.agent.reset({"side": 0, "map": self.battle._map})
        self.addCleanup(self.agent.close)

    def act(self, planning):
        before = self.battle.snapshot()
        action = self.agent.act_with_planning(self.battle.observe(0),
                                             tuple(self.battle.available_actions()), planning)
        self.assertEqual(self.battle.snapshot(), before)
        self.assertIn(action, self.battle.available_actions())
        return action

    def execute(self, action):
        result = self.battle.step(action)
        self.assertTrue(result.ok)
        self.agent.on_action_result(action, result, self.battle.observe(0))

    def finish_support(self, target, selected, *, contacts, mode):
        if mode != SupportMode.RECOVERY:
            return plan()
        uid = selected[0]
        u = next(u for u in self.battle.observe(0)["units"] if u["id"] == uid)
        return replace(plan(assignment(K.FINISH, uid, u["tile"], u["tile"], u["moves"])),
                       diagnostics=SearchDiagnostics(recovery_kind="wait"))

    def test_failure_only_recovers_its_roster_then_other_group_gets_search(self):
        self.setup_battle([unit("warrior", 0, t) for t in (1, 2, 13, 14)]
                          + [unit("warrior", 1, t) for t in (0, 15)], width=16, height=4)
        service = ScriptedPlanning(plan(), plan(assignment(K.FINISH, 2, 13, 13, 120)))
        service.attack_candidates = lambda target, *args, **kwargs: (0, 1) if target == 0 else (2, 3)
        service.support = self.finish_support
        for uid in (0, 1):
            action = self.act(service)
            self.assertEqual(action, Action("skip", uid))
            self.assertEqual(self.agent._recovery_units, {0, 1})
            self.execute(action)
        self.assertEqual(self.act(service), Action("skip", 2))
        self.assertEqual([set(call[1]) for call in service.calls], [{0, 1}, {2, 3}])

    def test_whole_failed_roster_is_locked_before_first_recovery_action(self):
        self.setup_battle([unit("warrior", 0, 17), unit("warrior", 0, 25),
                           unit("warrior", 1, 18)])
        failure = replace(plan(), diagnostics=SearchDiagnostics(bad_units=(0,)))
        service = ScriptedPlanning(failure)
        service.support = self.finish_support
        self.act(service)
        self.assertEqual(self.agent.diagnostic_state()["recovery_units"], [0, 1])
        own = [u for u in self.battle.observe(0)["units"] if u["side"] == 0]
        enemy = [u for u in self.battle.observe(0)["units"] if u["side"] != 0]
        self.agent._phase = 1
        self.assertIsNone(self.agent._next_task(own, enemy, self.battle.observe(0)))
        self.assertEqual(len(service.calls), 1)
        event = next(e for e in self.agent.drain_diagnostics() if e["event"] == "group_recovery")
        self.assertEqual(event["selected_units"], [0, 1])
        self.assertEqual(event["attempts"], 1)

    def test_retreat_keeps_opportunity_shot_but_never_approach(self):
        self.setup_battle([unit("archer", 0, 17), unit("warrior", 1, 18)])
        service = ScriptedPlanning(plan())
        calls = []
        def support(target, selected, *, contacts, mode):
            calls.append(mode)
            if mode == SupportMode.RECOVERY:
                return replace(plan(assignment(K.MOVE, 0, 17, 16, 60)),
                               diagnostics=SearchDiagnostics(recovery_kind="retreat"))
            return plan()
        service.support = support
        action = self.act(service)
        self.assertEqual(action, Action("move", 0, 16))
        self.execute(action)
        shot = Action("ranged", 0, 18)
        prediction = self.battle.preview_attack(shot)
        service.plans = iter([plan(assignment(K.RANGED, 0, 16, 18, 0, damage=prediction.damage))])
        self.assertEqual(self.act(service), shot)
        self.assertNotIn(SupportMode.APPROACH, calls)
        self.assertEqual(self.agent._recovery_units, {0})

    def test_new_contact_preserves_recovery_membership_until_next_turn(self):
        self.setup_battle([unit("horseman", 0, 16), unit("warrior", 0, 17),
                           unit("warrior", 1, 21)], fog=True)
        self.agent._turn = 0
        own = [u for u in self.battle.observe(0)["units"] if u["side"] == 0]
        self.agent._enter_recovery({"target": 16, "selected": (0, 1)}, own, "no_plan")
        with self.battle.clone() as witness:
            witness.step(Action("move", 0, 19))
            moved = witness.snapshot()["units"][0]
        service = ScriptedPlanning()
        def support(target, selected, *, contacts, mode):
            if mode == SupportMode.RECOVERY:
                return plan(assignment(K.MOVE, 0, 16, moved["tile"], moved["moves"], target=19))
            return plan()
        service.support = support
        self.execute(self.act(service))
        self.assertEqual(self.agent._recovery_units, {0, 1})
        self.assertIsNone(self.agent._task)
        self.assertTrue(any(e.get("reason") == "new_contact" for e in self.agent.drain_diagnostics()))
        self.agent._new_turn()
        self.assertFalse(self.agent._recovery_units)

    def test_damage_cancellation_does_not_restore_ordinary_tasks(self):
        self.setup_battle([unit("archer", 0, 17), unit("warrior", 1, 18)])
        self.agent._turn = 0
        self.agent._recovery_units.add(0)
        self.agent._task = {"kind": "recovery", "selected": (0,), "target": 17}
        predicted = self.battle.preview_attack(Action("ranged", 0, 18))
        self.agent._queue.append(assignment(K.RANGED, 0, 17, 18, 0, damage=predicted.damage + 1))
        self.execute(self.act(ScriptedPlanning()))
        self.assertEqual(self.agent._recovery_units, {0})
        self.assertIsNone(self.agent._task)

    def test_global_budget_is_recorded_separately_from_tactical_failure(self):
        self.setup_battle([unit("warrior", 0, 17), unit("warrior", 1, 18)])
        self.agent._turn = 0
        self.agent._phase = 1
        self.agent._search_limit = self.agent._searches = 1
        self.agent._recovery_limit = 10
        service = ScriptedPlanning()
        service.support = self.finish_support
        self.assertEqual(self.act(service), Action("skip", 0))
        self.assertFalse(service.calls)
        events = self.agent.drain_diagnostics()
        self.assertTrue(any(e["event"] == "group_recovery" and
                            e["reason"] == "search_budget_exhausted" for e in events))

    def test_native_empty_budget_result_is_not_global_budget_exhaustion(self):
        self.setup_battle([unit("warrior", 0, 17), unit("warrior", 1, 18)])
        service = ScriptedPlanning(plan(reason=SearchReason.BUDGET_EXHAUSTED))
        service.support = self.finish_support
        self.act(service)
        self.assertLess(self.agent._searches, self.agent._search_limit)
        event = next(e for e in self.agent.drain_diagnostics() if e["event"] == "group_recovery")
        self.assertEqual(event["reason"], "budget_exhausted")

    def test_metadata_only_combat_plan_transfers_the_roster(self):
        self.setup_battle([unit("warrior", 0, 17), unit("warrior", 1, 18)])
        service = ScriptedPlanning(plan(assignment(K.INITIAL, 0, 17, 17, 120),
                                        assignment(K.BLOCKED, 0, 17, 17, 120)))
        service.support = self.finish_support
        self.assertEqual(self.act(service), Action("skip", 0))
        self.assertEqual(self.agent._recovery_units, {0})
        event = next(e for e in self.agent.drain_diagnostics() if e["event"] == "group_recovery")
        self.assertEqual(event["reason"], "no_executable_plan")

    def test_recovery_opportunity_quota_never_consumes_an_untried_group(self):
        self.setup_battle([unit("chariot_archer", 0, 1), unit("chariot_archer", 0, 2),
                           unit("warrior", 0, 13), unit("warrior", 1, 15)], width=16, height=4)
        self.agent._turn = 0
        self.agent._phase = 1
        self.agent._recovery_units.update((0, 1))
        self.agent._search_limit = self.agent._recovery_limit = 20
        service = ScriptedPlanning()
        service.attack_candidates = lambda *args, **kwargs: (2,)
        def support(target, selected, *, contacts, mode):
            if mode == SupportMode.RECOVERY:
                uid = selected[0]
                u = next(u for u in self.battle.observe(0)["units"] if u["id"] == uid)
                # Metadata must not loop or cause an unrelated unit to wait.
                return plan(assignment(K.INITIAL, uid, u["tile"], u["tile"], u["moves"]))
            return plan()
        service.support = support
        first = self.act(service)
        self.assertEqual(first, Action("skip", 0))
        self.assertEqual(self.agent._call_searches, 2)
        self.assertEqual(self.agent._recovery_units, {0, 1})
        self.assertEqual([call[1] for call in service.calls], [(0,), (1,)])
        self.execute(first)
        self.execute(self.act(service))
        events = self.agent.drain_diagnostics()
        self.assertTrue(all(e["unit"] in (0, 1) for e in events if e["event"] == "waiting"))
        self.act(service)
        events = self.agent.drain_diagnostics()
        self.assertTrue(any(e["event"] == "search" and e.get("task_kind") == "combat"
                            and e["selected_units"] == [2] for e in events))
