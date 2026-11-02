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
    PlanningState,
    SearchReason,
    SearchRequest,
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
                # The native CP destination filter now decides applicability;
                # the coordinator no longer excludes starts at distance > 6.
                self.assertTrue(searches)
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
            self.assertGreaterEqual(len(scripted.calls), 4)
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
            self.assertEqual([(t, selected) for t, selected, mode in calls
                              if mode == SupportMode.APPROACH], [(10, (0, 2)), (0, (0,))])

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

    Each expectedFailure has a passing witness test running the same episode,
    so setup, legality and termination errors cannot silently replace the
    intended requirement failure. See combat-ai-port-plan.md, section
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
        planning.support = lambda *args, **kwargs: plan(reason=SearchReason.NO_PLAN)
        action = self.act(agent, battle, planning)
        self.assertEqual(len(planning.calls), 2)
        events = agent.drain_diagnostics()
        self.execute(agent, battle, action)
        self.finish_turn(agent, battle, planning)
        self.assertLessEqual(len(planning.calls), agent._search_limit)
        return action, events

    def test_contact_scenario_reveals_enemy_and_leaves_unused_unit_mobile(self):
        self.contact_episode()

    @unittest.expectedFailure
    def test_new_contact_returns_unused_explorer_to_approach(self):
        # Known coordinator defect: _phase_seen[3] retains the interrupted group.
        approaches = self.contact_episode()
        self.assertTrue(any(1 in selected for selected in approaches),
                        f"Unused explorer never reconsidered for approach: {approaches}")

    def test_search_limit_scenario_has_legal_shot_and_two_empty_searches(self):
        self.search_limit_episode()

    @unittest.expectedFailure
    def test_call_search_limit_does_not_order_available_shooter_to_wait(self):
        # Known coordinator defect: the per-call search cap issues a skip.
        action, events = self.search_limit_episode()
        self.assertFalse(any(e["event"] == "waiting" and e.get("reason") == "call_search_limit"
                             for e in events), "Computational limit became a wait order")
        self.assertNotEqual(action, Action("skip", 0))


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
