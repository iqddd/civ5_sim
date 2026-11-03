"""Coordinator targets maximize the joint roster, with one empty-plan fallback."""

import unittest
from dataclasses import replace

from test_cp_agent import SMALL, ScriptedPlanning, assignment, plan, scenario, unit

from civ5sim import Action, Battle
from civ5sim.combat_ai import LIBRARY_PATH, Aggression, SearchReason
from civ5sim.combat_ai import AssignmentKind as K
from civ5sim.cp_agent import CPAgent


@unittest.skipUnless(LIBRARY_PATH.exists(), "Build combat and AI libraries first")
class TargetSelectionTests(unittest.TestCase):
    def setUp(self):
        self.battle = self.enterContext(Battle(scenario([
            unit("archer", 0, 17), unit("warrior", 0, 10), unit("warrior", 0, 11),
            unit("warrior", 1, 18), unit("warrior", 1, 13), unit("warrior", 1, 29),
        ])))
        self.agent = CPAgent(SMALL)
        self.agent.reset({"side": 0, "map": self.battle._map})
        self.addCleanup(self.agent.close)
        self.agent._turn = 0
        self.agent._phase = 1
        self.agent._search_limit = 20
        self.obs = self.battle.observe(0)
        self.own = {u["id"]: u for u in self.obs["units"] if u["side"] == 0}
        self.enemies = [u for u in self.obs["units"] if u["side"] == 1]

    def shortlist(self, eligibility):
        return self.agent._next_task(list(self.own.values()), self.enemies, self.obs,
                                     eligibility, tuple(self.battle.available_actions()))

    def choose(self, task, service):
        before = self.battle.snapshot()
        result = self.agent._choose_combat_plan(task, self.obs, self.own, self.enemies, service)
        self.assertEqual(self.battle.snapshot(), before)
        self.assertLessEqual(self.agent._probe_searches, self.agent._searches)
        return result

    def task(self, target=18, selected=(0,)):
        return {"kind": "combat", "target": target, "selected": selected,
                "enemy_id": next(e["id"] for e in self.enemies if e["tile"] == target)}

    def test_shortlist_prefers_larger_group_even_without_immediate_attackers(self):
        task = self.shortlist({3: {0}, 4: {1, 2}, 5: {0}})
        self.assertEqual(task["target"], 13)
        self.assertNotIn("alternative", task)
        self.assertEqual(set(task["selected"]), {1, 2})
        self.assertEqual(task["ranking"]["immediate_attackers"], 0)
        self.assertEqual(len(task["ranked_candidates"]), 3)
        self.assertFalse(self.agent._combat_attempts)

    def test_rank_uses_readiness_hp_exact_target_mean_then_enemy_id(self):
        group = self.task(selected=(0, 1))
        rank = self.agent._group_rank(group, self.own, (Action("ranged", 0, 18),))
        other = self.agent._group_rank(self.task(selected=(0, 1)), self.own, ())
        self.assertGreater(rank, other)
        wounded = {k: {**v, "hp": 50} for k, v in self.own.items()}
        self.assertGreater(other, self.agent._group_rank(self.task(selected=(0, 1)), wounded, ()))
        # Exact means: 4/3 must beat 5/3, without integer rounding.
        original_distance = self.agent._distance
        distances = {(17, 13): 1, (10, 13): 1, (11, 13): 2,
                     (17, 18): 1, (10, 18): 2, (11, 18): 2}
        self.agent._distance = lambda a, b: distances[a, b]
        near = self.agent._group_rank(self.task(13, (0, 1, 2)), self.own, ())
        far = self.agent._group_rank(self.task(18, (0, 1, 2)), self.own, ())
        self.assertGreater(near, far)
        self.agent._distance = lambda a, b: 1
        self.assertGreater(self.agent._group_rank(self.task(18), self.own, ()),
                           self.agent._group_rank(self.task(13), self.own, ()))
        self.agent._distance = original_distance

    def test_native_admission_and_thirteen_unit_limit_precede_ranking(self):
        available = [{**self.own[0], "id": i} for i in range(16)]
        task = self.agent._next_task(available, self.enemies, self.obs,
                                    {3: set(range(15)), 4: {15}, 5: set()}, ())
        self.assertEqual(task["target"], 18)
        self.assertEqual(task["selected"], tuple(range(13)))
        self.assertEqual(task["ranking"]["participants"], 13)
        self.assertNotIn("alternative", task)

    def test_excluded_and_attempted_pairs_cannot_enter_alternative(self):
        self.agent._combat_attempts.update({(1, 4), (2, 4)})
        self.agent._target_blocked[0, 5] = 1
        task = self.shortlist({3: {0}, 4: {1, 2}, 5: {0}})
        self.assertNotIn("alternative", task)

    def test_single_target_calls_search_once(self):
        task = self.shortlist({3: {0}, 4: set(), 5: set()})
        service = ScriptedPlanning(plan())
        self.choose(task, service)
        self.assertEqual(len(service.calls), 1)
        self.assertEqual(self.agent._probe_searches, 0)

    def test_nonempty_larger_group_does_not_search_smaller_group(self):
        task = self.shortlist({3: {0}, 4: {1, 2}, 5: {0}})
        first = plan(replace(assignment(K.MOVE, 1, 10, 9), score=-99999))
        second = plan(replace(assignment(K.RANGED, 0, 17, 18, damage=100), score=99999))
        service = ScriptedPlanning(first, second)
        self.assertIs(self.choose(task, service), first)
        self.assertEqual(self.agent._task["target"], 13)
        self.assertEqual(len(service.calls), 1)
        self.assertEqual(self.agent._probe_searches, 0)
        # A reduced-group retry cannot spawn a third candidate.
        self.choose(self.agent._task, service)
        self.assertEqual(len(service.calls), 1)

    def test_empty_primary_tries_next_ranked_target_once(self):
        task = self.shortlist({3: {0}, 4: {1, 2}, 5: {1, 2}})
        second = plan(assignment(K.MELEE, 2, 11, 29, damage=20))
        service = ScriptedPlanning(plan(reason=SearchReason.NO_PLAN), second)
        self.assertIs(self.choose(task, service), second)
        self.assertEqual(self.agent._task["target"], 29)
        self.assertEqual(set(service.calls[0][1]), set(service.calls[1][1]))
        self.assertEqual(len(service.calls), 2)
        self.choose(self.agent._task, service)
        self.assertEqual(len(service.calls), 2)

    def test_empty_result_and_budget_exhaustion_never_beat_available_plan(self):
        for first_empty in (True, False):
            with self.subTest(first_empty=first_empty):
                task = self.task()
                task["alternative"] = self.task(13)
                empty = plan(reason=SearchReason.BUDGET_EXHAUSTED)
                usable = plan(assignment(K.RANGED, 0, 17, 18, damage=1, retaliation=2),
                              reason=SearchReason.BUDGET_EXHAUSTED)
                service = ScriptedPlanning(*( (empty, usable) if first_empty else (usable, empty)))
                self.agent._search_cache.clear()
                self.assertIs(self.choose(task, service), usable)

    def test_two_empty_plans_preserve_primary_bad_unit_diagnostics(self):
        from civ5sim.combat_ai import SearchDiagnostics

        task = self.task()
        task["alternative"] = self.task(13)
        first = replace(plan(reason=SearchReason.NO_PLAN), diagnostics=SearchDiagnostics(bad_units=(0,)))
        self.assertIs(self.choose(task, ScriptedPlanning(first, plan())), first)
        self.assertEqual(self.agent._task["target"], 18)

    def test_prefix_stops_at_first_restart_and_caps_hp_changes(self):
        result = plan(assignment(K.RANGED, 0, 17, 18, damage=120, killed=3),
                      assignment(K.RESTART, 0, -1, -1),
                      assignment(K.MELEE, 1, 10, 13, damage=100, retaliation=100, killed=4))
        value = self.agent._prefix_value(result, self.obs)
        self.assertEqual((value["net_hp"], value["net_kills"], value["commands"]), (100, 1, 1))

    def test_found_plan_before_restart_is_used_without_probe(self):
        task = self.task()
        task["alternative"] = self.task(13, (1, 2))
        first = plan(assignment(K.MOVE, 0, 17, 16), assignment(K.RESTART, 0, -1, -1))
        service = ScriptedPlanning(first)
        self.assertIs(self.choose(task, service), first)
        self.assertEqual(len(service.calls), 1)
        event = next(e for e in self.agent.drain_diagnostics() if e["event"] == "target_comparison")
        self.assertEqual(event["reason"], "max_participants")
        self.assertEqual(event["candidates"][0]["prefix"]["commands"], 1)

    def test_cache_identity_includes_group_target_config_contacts_and_observation(self):
        from civ5sim.combat_ai import ContactMemory

        service = ScriptedPlanning(*(plan() for _ in range(8)))
        task = self.task()
        self.choose(task, service)
        self.choose(dict(task), service)
        self.assertEqual(len(service.calls), 1)
        self.choose(self.task(selected=(0, 1)), service)
        self.choose(self.task(13), service)
        self.agent.config = replace(self.agent.config, choices_per_unit=2)
        self.choose(dict(task), service)
        self.agent.memory = ContactMemory(turn=1, side=0)
        self.choose(dict(task), service)
        self.obs = {**self.obs, "turn": self.obs["turn"] + 1}
        self.choose(dict(task), service)
        self.assertEqual(len(service.calls), 6)

    def test_probe_budget_never_exceeds_ordinary_queries(self):
        service = ScriptedPlanning(*(plan() for _ in range(8)))
        for _ in range(4):
            task = self.task()
            task["alternative"] = self.task(13)
            self.choose(task, service)
        self.assertEqual(len(service.calls), 2)
        self.assertEqual((self.agent._searches, self.agent._probe_searches), (1, 1))

    def test_invalid_probe_is_integration_error(self):
        task = self.task()
        task["alternative"] = self.task(13)
        with self.assertRaisesRegex(RuntimeError, "invalid_request"):
            self.choose(task, ScriptedPlanning(plan(), plan(reason=SearchReason.INVALID_REQUEST)))

    def test_different_roster_cannot_be_used_as_fallback(self):
        class RecordingPlanning(ScriptedPlanning):
            def search(self, *args, aggression=None, **kwargs):
                self.aggressions.append(aggression)
                return super().search(*args, aggression=aggression, **kwargs)

        task = self.task()
        task["alternative"] = self.task(13, (0, 1, 2))
        service = RecordingPlanning(plan(), plan())
        service.aggressions = []
        self.choose(task, service)
        self.assertEqual(service.aggressions, [Aggression.LOW])

    def test_equal_size_different_members_cannot_be_fallback(self):
        task = self.shortlist({3: {0, 1}, 4: {1, 2}, 5: {0, 1}})
        self.assertEqual(set(task["alternative"]["selected"]), set(task["selected"]))
        self.assertNotEqual(task["alternative"]["target"], 13)


if __name__ == "__main__":
    unittest.main()
