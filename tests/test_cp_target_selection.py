"""Coordinator target experiment: compare physical prefixes, not CP scores."""

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

    def test_shortlist_keeps_nearest_and_adds_larger_group(self):
        task = self.shortlist({3: {0}, 4: {1, 2}, 5: {0}})
        self.assertEqual(task["target"], 18)
        self.assertEqual(task["alternative"]["target"], 13)
        self.assertEqual(set(task["alternative"]["selected"]), {1, 2})
        self.assertEqual(task["alternative"]["ranking"]["immediate_attackers"], 0)
        self.assertFalse(self.agent._combat_attempts)

    def test_rank_uses_readiness_hp_and_exact_pair_and_target_means(self):
        group = self.task(selected=(0, 1))
        rank = self.agent._group_rank(group, self.own, (Action("ranged", 0, 18),))
        other = self.agent._group_rank(self.task(selected=(0, 1)), self.own, ())
        self.assertGreater(rank, other)
        wounded = {k: {**v, "hp": 50} for k, v in self.own.items()}
        self.assertGreater(other, self.agent._group_rank(self.task(selected=(0, 1)), wounded, ()))
        # Equal head count/HP: the closer pair wins independently of ID order.
        close = self.agent._group_rank(self.task(13, (1, 2)), self.own, ())
        wide = self.agent._group_rank(self.task(13, (0, 2)), self.own, ())
        self.assertGreater(close, wide)

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

    def test_larger_group_can_win_without_comparing_tactical_scores(self):
        task = self.shortlist({3: {0}, 4: {1, 2}, 5: {0}})
        first = plan(replace(assignment(K.RANGED, 0, 17, 18, damage=20), score=99999))
        second = plan(replace(assignment(K.MELEE, 1, 10, 13, damage=40, retaliation=5), score=-99999))
        service = ScriptedPlanning(first, second)
        self.assertIs(self.choose(task, service), second)
        self.assertEqual(self.agent._task["target"], 13)
        self.assertEqual(len(service.calls), 2)
        # A reduced-group retry cannot spawn a third candidate.
        self.choose(self.agent._task, service)
        self.assertEqual(len(service.calls), 2)

    def test_tie_keeps_nearest_even_for_different_size_groups(self):
        task = self.shortlist({3: {0}, 4: {1, 2}, 5: {0}})
        first = plan(assignment(K.RANGED, 0, 17, 18, damage=20))
        second = plan(assignment(K.MELEE, 1, 10, 13, damage=20))
        self.assertIs(self.choose(task, ScriptedPlanning(first, second)), first)
        self.assertEqual(self.agent._task["target"], 18)

    def test_empty_result_and_budget_exhaustion_never_beat_available_plan(self):
        for first_empty in (True, False):
            with self.subTest(first_empty=first_empty):
                task = self.task()
                task["alternative"] = self.task(13, (1, 2))
                empty = plan(reason=SearchReason.BUDGET_EXHAUSTED)
                usable = plan(assignment(K.RANGED, 0, 17, 18, damage=1, retaliation=2),
                              reason=SearchReason.BUDGET_EXHAUSTED)
                service = ScriptedPlanning(*( (empty, usable) if first_empty else (usable, empty)))
                self.agent._search_cache.clear()
                self.assertIs(self.choose(task, service), usable)

    def test_two_empty_plans_preserve_primary_bad_unit_diagnostics(self):
        from civ5sim.combat_ai import SearchDiagnostics

        task = self.task()
        task["alternative"] = self.task(13, (1, 2))
        first = replace(plan(reason=SearchReason.NO_PLAN), diagnostics=SearchDiagnostics(bad_units=(0,)))
        self.assertIs(self.choose(task, ScriptedPlanning(first, plan())), first)
        self.assertEqual(self.agent._task["target"], 18)

    def test_prefix_stops_at_first_restart_and_caps_hp_changes(self):
        result = plan(assignment(K.RANGED, 0, 17, 18, damage=120, killed=3),
                      assignment(K.RESTART, 0, -1, -1),
                      assignment(K.MELEE, 1, 10, 13, damage=100, retaliation=100, killed=4))
        value = self.agent._prefix_value(result, self.obs)
        self.assertEqual((value["net_hp"], value["net_kills"], value["commands"]), (100, 1, 1))

    def test_net_kills_then_own_damage_break_hp_ties(self):
        task = self.task()
        task["alternative"] = self.task(13, (1, 2))
        first = plan(assignment(K.RANGED, 0, 17, 18, damage=50))
        second = plan(assignment(K.MELEE, 1, 10, 13, damage=100, retaliation=50))
        self.assertIs(self.choose(task, ScriptedPlanning(first, second)), second)
        self.agent._search_cache.clear()
        task = self.task()
        task["alternative"] = self.task(13, (1, 2))
        second = plan(assignment(K.MELEE, 1, 10, 13, damage=60, retaliation=10))
        self.assertIs(self.choose(task, ScriptedPlanning(first, second)), first)

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
            task["alternative"] = self.task(13, (1, 2))
            self.choose(task, service)
        self.assertEqual(len(service.calls), 2)
        self.assertEqual((self.agent._searches, self.agent._probe_searches), (1, 1))

    def test_invalid_probe_is_integration_error(self):
        task = self.task()
        task["alternative"] = self.task(13, (1, 2))
        with self.assertRaisesRegex(RuntimeError, "invalid_request"):
            self.choose(task, ScriptedPlanning(plan(), plan(reason=SearchReason.INVALID_REQUEST)))

    def test_aggression_is_computed_separately_for_each_group(self):
        class RecordingPlanning(ScriptedPlanning):
            def search(self, *args, aggression=None, **kwargs):
                self.aggressions.append(aggression)
                return super().search(*args, aggression=aggression, **kwargs)

        task = self.task()
        task["alternative"] = self.task(13, (0, 1, 2))
        service = RecordingPlanning(plan(), plan())
        service.aggressions = []
        self.choose(task, service)
        self.assertEqual(service.aggressions, [Aggression.LOW, self.agent.config.aggression])


if __name__ == "__main__":
    unittest.main()
