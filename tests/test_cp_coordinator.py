"""CP-06 army coverage, exploration, support boundary and full matches."""
import unittest
from copy import deepcopy
from dataclasses import asdict, replace

from test_cp_agent import SMALL, ScriptedPlanning, assignment, plan, scenario, unit
from test_maps import field

from civ5sim import Action, Battle, MatchConfig, run_match
from civ5sim.combat_ai import (
    LIBRARY_PATH,
    Aggression,
    Assignment,
    CallPlanningServices,
    ContactMemory,
    Planner,
    PlanningState,
    SearchDiagnostics,
    SearchReason,
    SearchRequest,
    SearchResult,
    SearchStats,
    SupportMode,
)
from civ5sim.combat_ai import (
    AssignmentKind as K,
)
from civ5sim.cp_agent import CPAgent
from civ5sim.engine import digest


@unittest.skipUnless(LIBRARY_PATH.exists(), "Build combat and AI libraries first")
class CoordinatorTests(unittest.TestCase):
    def agent(self, battle, side=0):
        a = CPAgent(SMALL)
        a.reset({"map": battle._map, "side": side})
        self.addCleanup(a.close)
        return a

    def act(self, agent, battle, support_calls=None):
        before = battle.snapshot()
        obs = battle.observe(agent.side)
        service = CallPlanningServices(agent.planner, PlanningState.from_battle(battle, agent.side),
                                       agent.side, obs["turn"])
        if support_calls is not None:
            native_support = service.support
            def record(target, selected, **kwargs):
                support_calls.append((target, selected, kwargs["mode"]))
                return native_support(target, selected, **kwargs)
            service.support = record
        try:
            action = agent.act_with_planning(obs, tuple(battle.available_actions()), service)
        finally:
            service.expire()
        self.assertEqual(before, battle.snapshot(), "Planning changed real state or RNG")
        self.assertIn(action, battle.available_actions())
        return action

    def turn(self, agent, battle):
        actions = []
        for _ in range(500):
            action = self.act(agent, battle)
            result = battle.step(action)
            self.assertTrue(result.ok)
            agent.on_action_result(action, result, battle.observe(agent.side))
            actions.append(action)
            if action.kind == "end_turn" or any(battle.army_status()["destroyed"]):
                return actions
        self.fail("Unbounded coordinator turn")

    def test_no_contact_moves_and_wide_groups_get_distinct_targets(self):
        units = [unit("warrior", 0, y*16+x) for y in (1, 2) for x in range(4)]
        units.append(unit("warrior", 1, 255))
        with Battle(scenario(units, width=16, height=16, fog=True)) as b:
            a = self.agent(b)
            self.assertEqual(len(b.observe(0)["units"]), 8)
            commands = self.turn(a, b)
            self.assertTrue(any(c.kind == "move" for c in commands))
            tasks = [e for e in a.drain_diagnostics()
                     if e["event"] == "task_assigned" and e["task_kind"] == "approach"]
            self.assertGreaterEqual(len(tasks), 2)
            self.assertNotEqual(tasks[0]["target"], tasks[1]["target"])
            self.assertTrue(all(len(e["selected_units"]) <= 4 for e in tasks))

    def test_all_units_of_14_and_26_armies_receive_actions(self):
        for count in (14, 26):
            with self.subTest(count=count), Battle(scenario(
                [unit("warrior", 0, i) for i in range(count)] + [unit("warrior", 1, 255)],
                width=16, height=16,
            )) as b:
                a = self.agent(b)
                commands = self.turn(a, b)
                self.assertEqual({c.unit for c in commands if c.unit >= 0}, set(range(count)))
                events = a.drain_diagnostics()
                searches = [e for e in events if e["event"] == "search"]
                approaches = [e for e in events if e["event"] == "support" and e["task_kind"] == "approach"]
                # These distant armies cannot reach a fighting position yet.
                # Recruitment must defer them to APPROACH, not feed them into
                # a local attack that can hand out premature FINISH records.
                self.assertFalse(searches)
                recruitment = [e for e in events if e["event"] == "attack_candidates"]
                self.assertTrue(recruitment)
                self.assertTrue(all(not e["selected_units"] for e in recruitment))
                self.assertTrue(all(len(e["selected_units"]) <= 13 for e in searches))
                self.assertTrue(all(e["assignments"] == 0 for e in searches))
                self.assertGreaterEqual(len(approaches), 2)
                self.assertTrue(all(len(e["selected_units"]) <= 13 for e in approaches))
                self.assertLessEqual(a._searches, 2*count+(count+12)//13)

    def test_wounded_unit_waits_and_naturally_heals(self):
        with Battle(scenario([unit("warrior", 0, 0, hp=60), unit("warrior", 1, 127)],
                             width=16, height=8, fog=True)) as b:
            a = self.agent(b)
            commands = self.turn(a, b)
            self.assertEqual(commands, [Action("skip", 0), Action("end_turn")])
            self.assertTrue(b.step(Action("end_turn")).ok)
            self.assertEqual(b.observe(0)["units"][0]["hp"], 70)

    def test_empty_search_still_covers_army_and_bounds_calls(self):
        with Battle(scenario([unit("warrior", 0, i) for i in range(41) if i != 27] +
                             [unit("warrior", 1, 27)], width=8, height=8)) as b:
            a = self.agent(b)
            scripted = ScriptedPlanning()
            commanded = set()
            for _ in range(45):
                command = a.act_with_planning(b.observe(0), tuple(b.available_actions()), scripted)
                if command.kind == "end_turn":
                    break
                self.assertEqual(command.kind, "skip")
                commanded.add(command.unit)
                result = b.step(command)
                a.on_action_result(command, result, b.observe(0))
            else:
                self.fail("Empty searches did not end turn")
            self.assertEqual(commanded, set(range(40)))
            self.assertGreaterEqual(len(scripted.calls), 2)
            self.assertLessEqual(len(scripted.calls), 84)

    def test_lost_contact_uses_last_observed_tile_and_expires(self):
        with Battle(scenario([unit("warrior", 0, 17), unit("warrior", 1, 18)])) as b:
            a = self.agent(b)
            obs = b.observe(0)
            a._observe(obs)
            self.assertEqual(a._last_seen, {1: 18})
            hidden = deepcopy(obs)
            hidden["units"] = [hidden["units"][0]]
            hidden["visibility"][18] = False
            hidden["turn"] = 2
            a._observe(hidden)
            self.assertEqual(a._last_seen, {1: 18})
            hidden["turn"] = 4
            a._observe(hidden)
            self.assertFalse(a._last_seen)
            a.reset({"map": b._map, "side": 0})
            a._observe(obs)
            hidden["turn"] = 0
            hidden["visibility"][18] = True
            a._observe(hidden)
            self.assertFalse(a._last_seen)

    def test_stalled_exploration_changes_target_and_reset_clears_history(self):
        with Battle(scenario([unit("warrior", 0, 0), unit("warrior", 1, 127)],
                             width=16, height=8, fog=True)) as b:
            a = self.agent(b)
            obs = b.observe(0)
            a._observe(obs)
            group = [u for u in obs["units"] if u["side"] == 0]
            target = a._explore_target(group, obs)
            obs["turn"] = 2
            self.assertEqual(a._explore_target(group, obs), target)
            obs["turn"] = 4
            self.assertNotEqual(a._explore_target(group, obs), target)
            a.reset({"map": b._map, "side": 0})
            self.assertFalse(a._objectives)
            self.assertFalse(a._blocked)
            self.assertEqual(set(a._seen_at), {-1})

    def test_never_seen_enemy_position_does_not_choose_exploration(self):
        first = None
        for tile in (239, 255):
            units = [unit("warrior", 0, y*16+x) for y in (1, 2) for x in range(4)]
            with Battle(scenario(units+[unit("warrior", 1, tile)],
                                 width=16, height=16, fog=True)) as b:
                a = self.agent(b)
                commands = self.turn(a, b)
                choices = [(e["target"], e["selected_units"]) for e in a.drain_diagnostics()
                           if e["event"] == "task_assigned" and e["task_kind"] == "approach"]
                self.assertGreaterEqual(len(choices), 2)
                decisions = commands, choices
                if first is None:
                    first = decisions
                self.assertEqual(decisions, first)

    def test_lost_contact_approach_targets_last_seen_not_hidden_current_tile(self):
        with Battle(scenario([unit("warrior", 0, 0), unit("horseman", 1, 2)],
                             width=16, height=8, fog=True)) as b:
            a = self.agent(b)
            a._observe(b.observe(0))
            for command in (Action("move", 0, 32), Action("end_turn"),
                            Action("move", 1, 6), Action("end_turn")):
                self.assertTrue(b.step(command).ok)
            self.assertEqual(len(b.observe(0)["units"]), 1)
            calls = []
            self.act(a, b, calls)
            approaches = [target for target, _, mode in calls if mode == SupportMode.APPROACH]
            self.assertEqual(approaches, [2])
            self.assertEqual(a._last_seen, {1: 2})

    def test_approach_ignores_blocked_target_and_retries_empty_alternative(self):
        with Battle(scenario([unit("warrior", 0, 8), unit("warrior", 0, 9, waiting=True),
                              unit("warrior", 1, 10), unit("warrior", 1, 0)],
                             width=16, height=1)) as b:
            a = self.agent(b)
            a.close()
            a.config = replace(SMALL, position_capacity=1)
            a.reset({"map": b._map, "side": 0})
            a._target_blocked[0, 2] = 1
            calls = []
            command = self.act(a, b, calls)
            self.assertEqual(command, Action("move", 0, 6))
            self.assertEqual([t for t, _, mode in calls if mode == SupportMode.APPROACH], [0])
        # An allowed but locally blocked target must not prevent another approach.
        with Battle(scenario([unit("warrior", 0, 8), unit("warrior", 0, 9, waiting=True),
                              unit("warrior", 1, 10), unit("warrior", 1, 0)],
                             width=16, height=1)) as b:
            a = self.agent(b)
            a.close()
            a.config = replace(SMALL, position_capacity=1)
            a.reset({"map": b._map, "side": 0})
            calls = []
            command = self.act(a, b, calls)
            self.assertEqual(command, Action("move", 0, 6))
            self.assertEqual([t for t, _, mode in calls if mode == SupportMode.APPROACH], [10, 0])

    def test_partial_approach_reassigns_unused_group_member(self):
        with Battle(scenario([unit("warrior", 0, 8), unit("warrior", 0, 9, waiting=True),
                              unit("warrior", 0, 14), unit("warrior", 1, 10),
                              unit("warrior", 1, 0)], width=16, height=1)) as b:
            a = self.agent(b)
            a.close()
            a.config = replace(SMALL, position_capacity=1)
            a.reset({"map": b._map, "side": 0})
            calls = []
            first = self.act(a, b, calls)
            self.assertEqual(first, Action("move", 2, 12))
            result = b.step(first)
            self.assertTrue(result.ok)
            a.on_action_result(first, result, b.observe(0))
            self.assertEqual(self.act(a, b, calls), Action("move", 0, 6))
            approaches = [(t, selected) for t, selected, mode in calls if mode == SupportMode.APPROACH]
            self.assertEqual(approaches[0], (10, (0, 2)))
            self.assertEqual(approaches[-1], (0, (0,)))
            self.assertTrue(all(selected == (0,) for _, selected in approaches[1:]))

    def test_exploration_contact_discards_finish_and_schedules_combat(self):
        with Battle(scenario([unit("horseman", 0, 16), unit("warrior", 0, 17),
                              unit("warrior", 1, 21)], fog=True)) as b:
            a = self.agent(b)
            a._explore_target = lambda group, observation: 19
            obs = b.observe(0)
            self.assertEqual(len(obs["units"]), 2)
            service = CallPlanningServices(a.planner, PlanningState.from_battle(b, 0), 0, 0)
            native_support = service.support
            def support(target, selected, **kwargs):
                result = native_support(target, selected, **kwargs)
                if kwargs["mode"] == SupportMode.APPROACH:
                    # A queued finish must not survive a newly revealed enemy.
                    stale = Assignment(K.FINISH, 0, 16, 16, 240, 0, 0, 0, 0, -1)
                    return replace(result, assignments=result.assignments+(stale,))
                return result
            service.support = support
            try:
                command = a.act_with_planning(obs, tuple(b.available_actions()), service)
            finally:
                service.expire()
            self.assertEqual(command.kind, "move")
            self.assertTrue(a._queue)
            result = b.step(command)
            self.assertTrue(result.ok)
            a.on_action_result(command, result, b.observe(0))
            self.assertFalse(a._queue)
            self.act(a, b)
            events = a.drain_diagnostics()
            self.assertTrue(any(e["event"] == "plan_cancelled" and e["reason"] == "new_contact"
                                for e in events))
            self.assertTrue(any(e["event"] == "search" and e["target"] == 21 for e in events))

    def test_support_lifetime_errors_rng_and_per_request_aggression(self):
        with Battle(scenario([unit("warrior", 0, 17), unit("warrior", 1, 18, hp=1)],
                             random_damage=True)) as b:
            a = self.agent(b)
            initial = b.snapshot()
            state = PlanningState.from_battle(b, 0)
            request = SearchRequest(state, 0, 0, 18, (0,))
            baseline = a.planner.search(request)
            peaceful = a.planner.search(replace(request, aggression=Aggression.NONE))
            self.assertFalse(any(x.kind in (K.MELEE, K.RANGED) for x in peaceful.assignments))
            self.assertTrue(any(x.kind == K.MELEE for x in baseline.assignments))
            self.assertEqual(a.planner.search(request), baseline)
            service = CallPlanningServices(a.planner, state, 0, 0)
            service.support(18, (0,), contacts=ContactMemory(), mode=SupportMode.HEAL_BEFORE)
            self.assertEqual(b.snapshot(), initial)
            self.assertEqual(a.planner.support(replace(request, selected_units=(0,)*14),
                                              SupportMode.APPROACH).reason, SearchReason.INVALID_REQUEST)
            service.expire()
            with self.assertRaisesRegex(RuntimeError, "expired"):
                service.support(18, (0,), contacts=ContactMemory(), mode=SupportMode.APPROACH)


@unittest.skipUnless(LIBRARY_PATH.exists(), "Build combat and AI libraries first")
class ContactRequirementTests(unittest.TestCase):
    """Coordinator obligations, independent of native search quality or old replays.

    Each obligation also has a passing witness test running the same episode,
    checking setup, legality and termination independently. See combat-ai-port-plan.md, section
    "Восстановленные содержательные проверки".
    """

    def agent(self, battle):
        agent = CPAgent(SMALL)
        agent.reset({"map": battle._map, "side": 0})
        self.addCleanup(agent.close)
        return agent

    def act(self, agent, battle, planning):
        before = battle.snapshot()
        action = agent.act_with_planning(battle.observe(0), tuple(battle.available_actions()), planning)
        self.assertEqual(before, battle.snapshot(), "Planning changed real state or RNG")
        self.assertIn(action, battle.available_actions())
        return action

    def execute(self, agent, battle, action):
        result = battle.step(action)
        self.assertTrue(result.ok, result)
        agent.on_action_result(action, result, battle.observe(0))

    def finish_turn(self, agent, battle, planning):
        for _ in range(16):
            action = self.act(agent, battle, planning)
            if action.kind == "end_turn":
                return
            self.execute(agent, battle, action)
        self.fail("Coordinator did not finish the bounded scenario")

    def contact_episode(self):
        battle = self.enterContext(Battle(scenario([
            unit("horseman", 0, 16), unit("warrior", 0, 17), unit("warrior", 1, 21),
        ], fog=True)))
        agent = self.agent(battle)
        agent._explore_target = lambda group, observation: 19
        before = battle.observe(0)
        self.assertEqual({u["id"] for u in before["units"]}, {0, 1})
        with battle.clone() as witness:
            self.assertTrue(witness.step(Action("move", 0, 19)).ok)
            moved = next(u for u in witness.observe(0)["units"] if u["id"] == 0)

        planning = ScriptedPlanning(*(plan(reason=SearchReason.NO_PLAN) for _ in range(8)))
        approaches = []

        def support(target, selected, *, contacts, mode):
            if mode == SupportMode.APPROACH:
                approaches.append(selected)
                if len(approaches) == 1:
                    return plan(assignment(K.MOVE, 0, 16, moved["tile"], moved["moves"], target=19))
            return plan(reason=SearchReason.NO_PLAN)

        planning.support = support
        action = self.act(agent, battle, planning)
        self.assertEqual(action, Action("move", 0, 19))
        self.assertEqual(set(approaches[0]), {0, 1})
        self.execute(agent, battle, action)
        observation = battle.observe(0)
        self.assertIn(2, {u["id"] for u in observation["units"]})
        unused = next(u for u in observation["units"] if u["id"] == 1)
        self.assertEqual(unused, next(u for u in before["units"] if u["id"] == 1))
        self.assertTrue(any(a.kind == "move" and a.unit == 1 for a in battle.available_actions()))
        self.assertTrue(any(e["event"] == "plan_cancelled" and e["reason"] == "new_contact"
                            for e in agent.drain_diagnostics()))
        self.finish_turn(agent, battle, planning)
        self.assertLessEqual(len(planning.calls), agent._search_limit)
        return approaches[1:]

    def search_limit_episode(self):
        battle = self.enterContext(Battle(scenario([
            unit("archer", 0, 17), unit("warrior", 1, 18),
            unit("warrior", 1, 19), unit("warrior", 1, 20),
        ])))
        agent = self.agent(battle)
        attacks = [a for a in battle.available_actions() if a.kind == "ranged" and a.unit == 0]
        self.assertTrue(attacks, "The shooter must have a legal attack before dispatch")
        with battle.clone() as witness:
            self.assertTrue(witness.step(attacks[0]).ok)
        planning = ScriptedPlanning(*(plan(reason=SearchReason.NO_PLAN) for _ in range(8)))
        def support(target, selected, *, contacts, mode):
            if mode == SupportMode.RECOVERY:
                return agent.planner.support(SearchRequest(
                    PlanningState.from_battle(battle, 0), 0, battle.observe(0)["turn"],
                    target, selected, contacts), mode)
            return plan(reason=SearchReason.NO_PLAN)
        planning.support = support
        action = self.act(agent, battle, planning)
        events = agent.drain_diagnostics()
        searches = [e for e in events if e["event"] == "search"]
        self.assertEqual(sum(not e.get("probe", False) for e in searches), 2)
        self.assertLessEqual(sum(e.get("probe", False) for e in searches), 2)
        self.assertEqual(len(planning.calls), len(searches))
        self.execute(agent, battle, action)
        self.finish_turn(agent, battle, planning)
        self.assertLessEqual(agent._searches, agent._search_limit)
        self.assertLessEqual(agent._probe_searches, agent._searches)
        return action, events

    def test_contact_scenario_reveals_enemy_and_leaves_unused_unit_mobile(self):
        self.contact_episode()

    def test_new_contact_returns_unused_explorer_to_approach(self):
        approaches = self.contact_episode()
        self.assertTrue(any(1 in selected for selected in approaches),
                        f"Unused explorer never reconsidered for approach: {approaches}")

    def test_search_limit_scenario_has_legal_shot_and_two_empty_searches(self):
        self.search_limit_episode()

    def test_call_search_limit_does_not_order_available_shooter_to_wait(self):
        action, events = self.search_limit_episode()
        self.assertFalse(any(e["event"] == "waiting" and e.get("reason") == "call_search_limit"
                             for e in events), "Computational limit became a wait order")
        self.assertNotEqual(action, Action("skip", 0))
        # CP may prefer retreat when firing would spend the movement needed
        # to escape. The budget itself must not turn either option into wait.
        self.assertIn(action.kind, ("ranged", "move"))

    def test_problematic_unit_is_removed_before_retry(self):
        with Battle(scenario([unit("warrior", 0, 17), unit("archer", 0, 25),
                              unit("warrior", 1, 18)])) as battle:
            agent = self.agent(battle)
            planning = ScriptedPlanning(SearchResult((), SearchReason.NO_PLAN, SearchStats(),
                SearchDiagnostics(risk_rejections=1, bad_units=(1,))))
            original_search = planning.search
            def search(target, selected, **kwargs):
                result = original_search(target, selected, **kwargs)
                if len(planning.calls) == 2:
                    request = SearchRequest(PlanningState.from_battle(battle, 0), 0, 0,
                                            target, selected, kwargs["contacts"])
                    return agent.planner.search(request)
                return result
            planning.search = search
            action = self.act(agent, battle, planning)
            self.assertEqual(set(planning.calls[0][1]), {0, 1})
            self.assertEqual(planning.calls[1][1], (0,))
            self.assertEqual(action.kind, "melee")

    def test_old_target_does_not_outrank_closer_enemy(self):
        with Battle(scenario([unit("warrior", 0, 17), unit("warrior", 1, 18),
                              unit("warrior", 1, 21)])) as battle:
            agent = self.agent(battle)
            agent._turn = 0
            agent._phase = 1
            agent._search_limit = 3
            agent._combat_targets[0] = (2, 4, 0, 0)
            observation = battle.observe(0)
            available = [u for u in observation["units"] if u["side"] == 0]
            enemies = [u for u in observation["units"] if u["side"] == 1]
            task = agent._next_task(available, enemies, observation)
            self.assertEqual(task["target"], 18)
            self.assertFalse(agent._combat_attempts, "Assignment is not an executed request")

    def test_actual_attack_updates_attacker_target(self):
        with Battle(scenario([unit("archer", 0, 17), unit("warrior", 1, 18),
                              unit("warrior", 1, 20)])) as battle:
            agent = self.agent(battle)
            with battle.clone() as witness:
                result = witness.step(Action("ranged", 0, 18))
                self.assertTrue(result.ok)
            planning = ScriptedPlanning(plan(assignment(K.RANGED, 0, 17, 18, 0,
                                           damage=result.damage, target=18)))
            action = self.act(agent, battle, planning)
            agent._combat_targets[0] = (2, 3, 0, 0)
            self.execute(agent, battle, action)
            self.assertEqual(agent._combat_targets[0][0], 1)

    def test_recovery_cancellation_returns_to_normal_dispatch(self):
        with Battle(scenario([unit("archer", 0, 17), unit("warrior", 1, 18)])) as battle:
            agent = self.agent(battle)
            agent._turn = 0
            agent._task = {"kind": "recovery", "target": 18, "selected": (0,)}
            with battle.clone() as witness:
                prediction = witness.step(Action("ranged", 0, 18))
            agent._queue.append(assignment(K.RANGED, 0, 17, 18, 0,
                                          damage=prediction.damage + 1, target=18))
            planning = ScriptedPlanning()
            action = self.act(agent, battle, planning)
            self.execute(agent, battle, action)
            self.assertIsNone(agent._task)
            self.assertTrue(any(e["event"] == "plan_cancelled" and e["reason"] == "damage_changed"
                                for e in agent.drain_diagnostics()))
            self.act(agent, battle, planning)

    def test_returned_units_are_reconsidered_after_group_action(self):
        for blocked in (False, True):
            with self.subTest(blocked=blocked), Battle(scenario([
                    unit("horseman", 0, 16), unit("archer", 0, 25), unit("warrior", 1, 20)])) as battle:
                agent = self.agent(battle)
                initial = battle.snapshot()
                with battle.clone() as witness:
                    self.assertTrue(witness.step(Action("move", 0, 17)).ok)
                    moved = witness.snapshot()["units"][0]
                records = [assignment(K.INITIAL, 0, 16, 16, 240),
                           assignment(K.BLOCKED, 1, 25, 25, 120),
                           assignment(K.MOVE, 0, 16, 17, moved["moves"])]
                if blocked:
                    records.append(assignment(K.BLOCKED, 0, 17, 17, moved["moves"]))
                planning = ScriptedPlanning(plan(*records))
                first = self.act(agent, battle, planning)
                self.execute(agent, battle, first)
                self.assertEqual(initial["units"][1], battle.snapshot()["units"][1])
                self.act(agent, battle, planning)
                self.assertGreaterEqual(len(planning.calls), 2)
                self.assertIn(0, planning.calls[1][1])
                self.assertIn(1, planning.calls[1][1])

    def test_new_combat_contact_restarts_same_target_roster_and_aggression(self):
        with Battle(scenario([unit("horseman", 0, 16), unit("warrior", 0, 17),
                              unit("warrior", 1, 21), unit("warrior", 1, 0)], fog=True)) as battle:
            agent = self.agent(battle)
            agent._turn = 0
            agent._search_limit = agent._support_limit = agent._recovery_limit = 30
            agent._phase = 1
            agent._task = {"kind": "combat", "target": 0, "selected": (0,),
                           "aggression": Aggression.HIGH}
            agent._group = (0,)
            agent._replan = True
            planning = ScriptedPlanning(
                plan(assignment(K.MOVE, 0, 16, 19, 60), assignment(K.FINISH, 0, 19, 19, 60)),
                plan(assignment(K.FINISH, 0, 19, 19, 60)))
            aggressions = []
            search = planning.search
            def record(target, selected, **kwargs):
                aggressions.append(kwargs["aggression"])
                return search(target, selected, **kwargs)
            planning.search = record
            first = self.act(agent, battle, planning)
            self.execute(agent, battle, first)
            self.assertIn(2, {u["id"] for u in battle.observe(0)["units"]})
            self.assertTrue(any(a.unit == 1 and a.kind == "move" for a in battle.available_actions()))
            self.assertEqual(agent._task["target"], 0)
            self.assertEqual(agent._task["selected"], (0,))
            self.assertTrue(agent._replan)
            second = self.act(agent, battle, planning)
            self.assertEqual(second, Action("skip", 0))
            self.assertEqual([(t, s) for t, s, _ in planning.calls], [(0, (0,)), (0, (0,))])
            self.assertEqual(aggressions, [Aggression.HIGH, Aggression.HIGH])
            self.assertEqual(agent._task["attempts"], 2)

    def test_opportunity_search_uses_ordinary_budget_and_low_aggression(self):
        with Battle(scenario([unit("chariot_archer", 0, 17), unit("warrior", 1, 19)])) as battle:
            agent = self.agent(battle)
            agent._search_limit = agent._recovery_limit = 1
            planning = ScriptedPlanning(plan())
            values = []
            search = planning.search
            def record(target, selected, **kwargs):
                values.append(kwargs["aggression"])
                return search(target, selected, **kwargs)
            planning.search = record
            own = [u for u in battle.observe(0)["units"] if u["side"] == 0]
            before = battle.snapshot()
            agent._recover(own, battle.observe(0), planning)
            agent._recover(own, battle.observe(0), planning)
            self.assertEqual(agent._searches, 1)
            self.assertEqual(values, [Aggression.LOW])
            self.assertEqual(before, battle.snapshot())
            agent._search_limit = 3
            agent._call_searches = 2
            agent._recovery_search_attempts.clear()
            agent._recover(own, battle.observe(0), planning)
            self.assertEqual(values, [Aggression.LOW], "Recovery bypassed the per-call search ceiling")

    def test_contact_and_group_reduction_share_four_attempt_ceiling(self):
        with Battle(scenario([unit("horseman", 0, 16), unit("warrior", 0, 17),
                              unit("warrior", 1, 21), unit("warrior", 1, 0)], fog=True)) as battle:
            agent = self.agent(battle)
            agent._turn = 0
            agent._search_limit = agent._support_limit = agent._recovery_limit = 30
            agent._task = {"kind": "combat", "target": 0, "selected": (0,),
                           "aggression": Aggression.LOW, "attempts": 3}
            agent._replan = True
            planning = ScriptedPlanning(plan(assignment(K.MOVE, 0, 16, 19, 60)))
            self.execute(agent, battle, self.act(agent, battle, planning))
            self.assertEqual(agent._task["attempts"], 4)
            agent._next_task = lambda *args: None
            self.act(agent, battle, planning)
            self.assertEqual(len(planning.calls), 1)
            self.assertTrue(any(e["event"] == "task_finished" and e["reason"] == "task_attempt_limit"
                                for e in agent.drain_diagnostics()))

    def test_real_retreat_enables_low_search_for_slow_shooter(self):
        with Battle(scenario([unit("archer", 0, 17), unit("warrior", 1, 19)])) as battle:
            agent = self.agent(battle)
            agent._search_limit = agent._recovery_limit = 2
            planning = ScriptedPlanning(plan())
            own = [u for u in battle.observe(0)["units"] if u["side"] == 0]
            agent._recover(own, battle.observe(0), planning)
            self.assertFalse(planning.calls)
            agent._turn = 0
            agent._task = {"kind": "recovery", "target": 17, "selected": (0,)}
            agent._queue.append(assignment(K.MOVE, 0, 17, 16, 60))
            self.execute(agent, battle, self.act(agent, battle, planning))
            self.assertIn(0, agent._recovery_moved)
            own = [u for u in battle.observe(0)["units"] if u["side"] == 0]
            self.assertGreater(own[0]["moves"], 0)
            agent._recover(own, battle.observe(0), planning)
            self.assertEqual(len(planning.calls), 1)


@unittest.skipUnless(LIBRARY_PATH.exists(), "Build combat and AI libraries first")
class RecoveryTests(unittest.TestCase):
    def recover(self, battle, side, selected=(0,)):
        before = battle.snapshot()
        with Planner(SMALL) as planner:
            request = SearchRequest(PlanningState.from_battle(battle, side), side,
                battle.observe(side)["turn"], battle.snapshot()["units"][selected[0]]["tile"], selected,
                ContactMemory().updated(battle.observe(side)))
            result = planner.support(request, SupportMode.RECOVERY)
        self.assertEqual(before, battle.snapshot())
        return result

    def execute(self, battle, result):
        for record in result.assignments:
            action = Action({K.MOVE: "move", K.MELEE: "melee", K.RANGED: "ranged",
                             K.FINISH: "skip"}[record.kind], record.unit,
                            -1 if record.kind == K.FINISH else record.command_target)
            self.assertIn(action, battle.available_actions())
            step = battle.step(action)
            self.assertTrue(step.ok)
            self.assertEqual((step.damage, step.retaliation), (record.damage, record.self_damage))

    def test_fatal_melee_is_rejected_even_when_legal(self):
        for side in (0, 1):
            for fog in (False, True):
                with self.subTest(side=side, fog=fog), Battle(scenario([
                        unit("warrior", side, 17, hp=1), unit("warrior", 1-side, 18)], fog=fog)) as battle:
                    if side: self.assertTrue(battle.step(Action("end_turn")).ok)
                    result = self.recover(battle, side)
                    self.assertFalse(any(a.kind == K.MELEE for a in result.assignments))
                    self.execute(battle, result)
                    self.assertGreater(battle.snapshot()["units"][0]["hp"], 0)

    def test_approach_then_attack_and_quiet_wait(self):
        with Battle(scenario([unit("warrior", 0, 17), unit("warrior", 1, 19)])) as battle:
            result = self.recover(battle, 0)
            self.assertEqual([a.kind for a in result.assignments], [K.MOVE, K.MELEE])
            self.execute(battle, result)

    def test_stationary_shot_before_wait_and_half_hp_condition(self):
        for hp in (100, 50):
            with self.subTest(hp=hp), Battle(scenario([
                    unit("archer", 0, 0, hp=hp), unit("warrior", 0, 1, waiting=True),
                    unit("warrior", 1, 2)], width=3, height=1)) as battle:
                self.assertIn(Action("ranged", 0, 2), battle.available_actions())
                result = self.recover(battle, 0)
                self.assertEqual(result.assignments[0].kind, K.RANGED if hp > 50 else K.FINISH)
                self.execute(battle, result)

    def test_opportunity_kill_but_not_enemy_cluster_or_bad_exchange(self):
        for extra in ([], [unit("warrior", 1, 19)]):
            with self.subTest(cluster=bool(extra)), Battle(scenario([
                    unit("warrior", 0, 17), unit("warrior", 1, 18, hp=1), *extra])) as battle:
                result = self.recover(battle, 0)
                self.assertEqual(any(a.kind == K.MELEE for a in result.assignments), not extra)
                self.execute(battle, result)
        with Battle(scenario([unit("warrior", 0, 17, hp=30),
                              unit("warrior", 1, 18)])) as battle:
            result = self.recover(battle, 0)
            self.assertFalse(any(a.kind == K.MELEE for a in result.assignments))
            self.execute(battle, result)

    def test_firing_does_not_take_movement_needed_for_retreat(self):
        with Battle(scenario([unit("archer", 0, 17), unit("warrior", 1, 18),
                              unit("warrior", 1, 19), unit("warrior", 1, 20)])) as battle:
            self.assertIn(Action("ranged", 0, 18), battle.available_actions())
            result = self.recover(battle, 0)
            self.assertEqual(result.assignments[0].kind, K.MOVE)
            self.execute(battle, result)
        with Battle(scenario([unit("warrior", 0, 0), unit("warrior", 1, 39)], fog=True)) as battle:
            result = self.recover(battle, 0)
            self.assertEqual(result.diagnostics.recovery_kind, "wait")
            self.assertEqual(result.assignments[0].kind, K.FINISH)
            self.execute(battle, result)

    def test_new_contact_is_not_used_to_plan_a_hidden_attack(self):
        with Battle(scenario([unit("horseman", 0, 16), unit("warrior", 1, 21)], fog=True)) as battle:
            self.assertEqual(len(battle.observe(0)["units"]), 1)
            result = self.recover(battle, 0)
            self.assertFalse(any(a.kind in (K.MELEE, K.RANGED) for a in result.assignments))

    def test_already_attacked_unit_retreats_and_diagnostics_clear(self):
        with Battle(scenario([unit("warrior", 0, 17, hp=20, attacked=True),
                              unit("warrior", 1, 18)])) as battle:
            result = self.recover(battle, 0)
            self.assertEqual(result.diagnostics.recovery_kind, "retreat")
            self.assertEqual(len(result.assignments), 1)
            self.assertEqual(result.assignments[0].kind, K.MOVE)
            self.execute(battle, result)
        with (Battle(scenario([unit("warrior", 0, 17), unit("warrior", 1, 18)])) as battle,
              Planner(SMALL) as planner):
            request = SearchRequest(PlanningState.from_battle(battle, 0), 0, 0, 18, (0,),
                                     ContactMemory().updated(battle.observe(0)))
            self.assertEqual(planner.support(request, SupportMode.RECOVERY).diagnostics.recovery_kind,
                             "attack")
            self.assertEqual(planner.search(request).diagnostics.recovery_kind, "none")
            invalid = replace(request, target=999)
            self.assertEqual(planner.search(invalid).diagnostics, SearchDiagnostics())

    def test_recovery_requires_one_unit_and_diagnostics_reset(self):
        with Battle(scenario([unit("warrior", 0, 17), unit("warrior", 0, 16),
                              unit("warrior", 1, 18)])) as battle:
            result = self.recover(battle, 0, (0, 1))
            self.assertEqual(result.reason, SearchReason.INVALID_REQUEST)
            self.assertEqual(result.diagnostics, SearchDiagnostics())

    def test_native_rejection_diagnostics_are_selected_and_not_a_plan_change(self):
        with Battle(scenario([unit("archer", 0, 17, hp=50), unit("horseman", 0, 25),
                              unit("warrior", 1, 18), unit("horseman", 1, 19),
                              unit("spearman", 1, 26)])) as battle, Planner(SMALL) as planner:
            request = SearchRequest(PlanningState.from_battle(battle, 0), 0, 0, 18, (0, 1),
                                     ContactMemory().updated(battle.observe(0)), Aggression.LOW)
            before = battle.snapshot()
            result = planner.search(request)
            self.assertGreater(result.diagnostics.risk_rejections, 0)
            self.assertIn(0, result.diagnostics.bad_units)
            # The exact offender set changes with threat estimates; diagnostics
            # must report only selected IDs, once each, in selection order.
            self.assertEqual(result.diagnostics.bad_units, tuple(
                uid for uid in request.selected_units if uid in result.diagnostics.bad_units))
            self.assertTrue(result.diagnostics.budget_exhausted)
            self.assertEqual(result, planner.search(request))
            self.assertEqual(before, battle.snapshot())


@unittest.skipUnless(LIBRARY_PATH.exists(), "Build combat and AI libraries first")
class AttackRecruitmentTests(unittest.TestCase):
    def query(self, battle, target, selected, side=0):
        with Planner(SMALL) as planner:
            before = battle.snapshot()
            request = SearchRequest(PlanningState.from_battle(battle, side), side,
                battle.observe(side)["turn"], target, tuple(selected),
                ContactMemory().updated(battle.observe(side)))
            result = planner.attack_candidates(request)
            self.assertEqual(before, battle.snapshot())
            return result

    def test_melee_adjacency_and_ranged_next_turn_positions(self):
        for kind, target, expected in (("warrior", 3, (0,)), ("warrior", 4, ()),
                                        ("archer", 4, (0,)), ("archer", 5, ())):
            with self.subTest(kind=kind, target=target), Battle(scenario([
                    unit(kind, 0, 0), unit("warrior", 1, target)], width=10, height=1)) as battle:
                self.assertEqual(self.query(battle, target, (0,)), expected)

    def test_nearby_blockers_remain_in_group_even_without_path(self):
        value = scenario([unit("warrior", 0, 0), unit("warrior", 1, 2)], width=10, height=1)
        value["map"]["tiles"][1]["elevation"] = "mountain"
        with Battle(value) as battle:
            self.assertEqual(self.query(battle, 2, (0,)), (0,))

    def test_friendly_stacking_is_ignored_but_enemy_and_sight_are_not(self):
        with Battle(scenario([unit("warrior", 0, 0), unit("warrior", 0, 1),
                              unit("warrior", 0, 2), unit("warrior", 1, 3)], width=10, height=1)) as battle:
            self.assertEqual(self.query(battle, 3, (0,)), (0,))
        with Battle(scenario([unit("warrior", 0, 0), unit("warrior", 1, 1),
                              unit("warrior", 1, 3)], width=10, height=1)) as battle:
            self.assertEqual(self.query(battle, 3, (0,)), ())
        value = scenario([unit("archer", 0, 0), unit("warrior", 1, 3)], width=10, height=1)
        value["map"]["tiles"][2]["elevation"] = "mountain"
        with Battle(value) as battle:
            self.assertEqual(self.query(battle, 3, (0,)), ())

    def test_filter_is_applied_before_thirteen_unit_limit(self):
        with Battle(scenario([unit("warrior", 0, i) for i in range(25)] +
                             [unit("horseman", 0, 254), unit("warrior", 1, 255)],
                             width=16, height=16)) as battle:
            self.assertEqual(self.query(battle, 255, tuple(range(26))), (25,))

    def test_both_sides_parity_and_wrap_use_native_geometry(self):
        for parity in (0, 1):
            for side in (0, 1):
                value = scenario([unit("warrior", side, 7), unit("warrior", 1-side, 1)],
                                 width=8, height=5)
                value["map"].update(row_parity=parity, wrap_x=True)
                with self.subTest(parity=parity, side=side), Battle(value) as battle:
                    if side:
                        self.assertTrue(battle.step(Action("end_turn")).ok)
                    self.assertEqual(self.query(battle, 1, (0,), side), (0,))

    def test_far_infantry_is_deferred_and_moves_toward_the_battle(self):
        value = scenario([unit("warrior", 0, 2), unit("archer", 0, 162),
                          unit("chariot_archer", 0, 83), unit("spearman", 0, 18),
                          unit("horseman", 0, 6, hp=73), unit("chariot_archer", 1, 8)],
                         width=16, height=16, fog=True)
        with Battle(value) as battle:
            self.assertEqual(self.query(battle, 8, tuple(range(5))), (4,))
            agent = CPAgent(SMALL)
            self.addCleanup(agent.close)
            agent.reset({"map": battle._map, "side": 0})
            actions = []
            for _ in range(40):
                observation = battle.observe(0)
                service = CallPlanningServices(agent.planner, PlanningState.from_battle(battle, 0),
                                                0, observation["turn"])
                try:
                    action = agent.act_with_planning(observation, tuple(battle.available_actions()), service)
                finally:
                    service.expire()
                self.assertIn(action, battle.available_actions())
                actions.append(action)
                result = battle.step(action)
                self.assertTrue(result.ok)
                agent.on_action_result(action, result, battle.observe(0))
                if action.kind == "end_turn": break
            else:
                self.fail("Recruitment did not lead to a bounded turn")
            first = next(e for e in agent.drain_diagnostics() if e["event"] == "search")
            self.assertEqual(first["selected_units"], [4])
            for uid in (0, 3):
                self.assertTrue(any(a.kind == "move" and a.unit == uid for a in actions), actions)

    def test_query_expires_with_service_and_preserves_search_diagnostics(self):
        with Battle(scenario([unit("warrior", 0, 0), unit("warrior", 1, 3)],
                             width=10, height=1)) as battle, Planner(SMALL) as planner:
            state = PlanningState.from_battle(battle, 0)
            service = CallPlanningServices(planner, state, 0, 0)
            self.assertEqual(service.attack_candidates(3, (0,), contacts=ContactMemory()), (0,))
            service.expire()
            with self.assertRaisesRegex(RuntimeError, "expired"):
                service.attack_candidates(3, (0,), contacts=ContactMemory())
            with self.assertRaises(ValueError):
                planner.attack_candidates(SearchRequest(state, 0, 0, 3, (0, 0)))


@unittest.skipUnless(LIBRARY_PATH.exists(), "Build combat and AI libraries first")
class FullBattleTests(unittest.TestCase):
    def config(self, shape, fog, random_damage, reverse):
        m = field(12, 8)
        if shape != "open":
            for y in range(8):
                if y not in ((3,) if shape == "passage" else (1, 4, 6)):
                    m["tiles"][y*12+5]["elevation"] = "mountain"
        m["spawn_candidates"] = [[[0, 3], [0, 4]], [[11, 3], [11, 4]]]
        armies = [["warrior", "archer"], ["spearman", "chariot_archer"]]
        if reverse:
            armies.reverse()
        return MatchConfig(m, armies, max_rounds=30, seed=17+int(reverse), fog=fog,
                           random_damage=random_damage, max_calls_per_turn=200)

    def run_one(self, config):
        agents = [CPAgent(SMALL), CPAgent(SMALL)]
        try:
            outcome = run_match(config, agents)
        finally:
            for a in agents:
                a.close()
        self.assertEqual(outcome.result["status"], "completed", outcome.result.get("error"))
        self.assertTrue(any(e["action"]["kind"] in ("melee", "ranged")
                            for e in outcome.replay["steps"]), "Never reached combat")
        with Battle(outcome.replay["initial"]) as b:
            for entry in outcome.replay["steps"]:
                self.assertEqual(asdict(b.step(entry["action"])), entry["result"])
                self.assertEqual(digest(b.snapshot()), entry["state_sha256"])
            self.assertEqual(b.snapshot(), outcome.final)
        return outcome

    def test_full_matches_three_maps_modes_and_sides_repeat_and_replay(self):
        for shape in ("open", "obstacles", "passage"):
            for fog in (False, True):
                for random_damage in (False, True):
                    for reverse in (False, True):
                        with self.subTest(shape=shape, fog=fog, damage=random_damage, reverse=reverse):
                            config = self.config(shape, fog, random_damage, reverse)
                            first = self.run_one(config)
                            self.assertEqual(first, self.run_one(config))


if __name__ == "__main__":
    unittest.main()
