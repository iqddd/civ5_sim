"""Frozen CP-07 contact episodes: desired behavior and known, unfixed defects.

The fixture is a verified prefix of the original replay, independent of build/.
expectedFailure tests specify requirements; they do not accept the old behavior.
Remove that marker when fixing the corresponding production defect.
"""

import json
import unittest
from dataclasses import asdict
from pathlib import Path

from test_cp_agent import ScriptedPlanning, assignment, plan

from civ5sim import Action, Battle
from civ5sim.combat_ai import (
    LIBRARY_PATH,
    Aggression,
    AssignmentKind,
    CallPlanningServices,
    ContactMemory,
    Planner,
    PlanningState,
    SearchConfig,
    SearchReason,
    SearchRequest,
    SupportMode,
)
from civ5sim.cp_agent import CPAgent, hex_distance
from civ5sim.engine import digest
from civ5sim.greedy_agent import GreedyAgent

FIXTURE = Path(__file__).parent / "fixtures/cp07_contact.json"


@unittest.skipUnless(LIBRARY_PATH.exists(), "Build combat and AI libraries first")
class ContactRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.snapshots, cls.contacts = {}, {}
        memory = ContactMemory()
        with Battle(cls.fixture["initial"]) as battle:
            for count, step in enumerate(cls.fixture["steps"], 1):
                result = battle.step(step["action"])
                if asdict(result) != step["result"] or digest(battle.snapshot()) != step["state_sha256"]:
                    raise AssertionError(f"Frozen replay diverged at command {count}")
                memory = memory.updated(battle.observe(1))
                if count in (34, 42, 50, 51, 64):
                    cls.snapshots[count] = battle.snapshot()
                    cls.contacts[count] = memory

    def battle(self, after):
        return self.enterContext(Battle(self.snapshots[after]))

    def agent(self, battle, after):
        agent = CPAgent()
        agent.reset({"map": battle._map, "side": 1})
        agent.memory = self.contacts[after]
        self.addCleanup(agent.close)
        return agent

    def act(self, agent, battle):
        before = battle.snapshot()
        obs = battle.observe(1)
        service = CallPlanningServices(agent.planner, PlanningState.from_battle(battle, 1),
                                       1, obs["turn"])
        try:
            action = agent.act_with_planning(obs, tuple(battle.available_actions()), service)
        finally:
            service.expire()
        self.assertEqual(before, battle.snapshot(), "Planning changed the real battle")
        self.assertIn(action, battle.available_actions())
        return action

    def execute(self, agent, battle, action):
        result = battle.step(action)
        self.assertTrue(result.ok)
        agent.on_action_result(action, result, battle.observe(1))
        return result

    def search(self, battle, after, selected, target):
        planner = self.enterContext(Planner())
        before = battle.snapshot()
        result = planner.search(SearchRequest(
            PlanningState.from_battle(battle, 1), 1, battle.observe(1)["turn"],
            target, selected, self.contacts[after], Aggression.LOW,
        ))
        self.assertEqual(before, battle.snapshot())
        self.assertLessEqual(result.stats.generated, SearchConfig().position_capacity)
        return result

    def test_fixture_preserves_contact_and_available_responses(self):
        before = self.battle(34)
        self.assertFalse([u for u in before.observe(1)["units"] if u["side"] == 0])
        self.assertTrue(before.step(Action(**self.fixture["steps"][34]["action"])).ok)
        self.assertTrue([u for u in before.observe(1)["units"] if u["side"] == 0])
        surrounded = self.battle(50)
        shots = [a for a in surrounded.available_actions() if a.unit == 7 and a.kind == "ranged"]
        self.assertTrue(shots, "The trapped unit still has an attack")
        self.assertEqual(next(u["hp"] for u in surrounded.observe(1)["units"] if u["id"] == 7), 36)

    @unittest.expectedFailure
    def test_new_contact_returns_unused_explorers_to_approach(self):
        battle = self.battle(34)
        agent = self.agent(battle, 34)
        obs = battle.observe(1)
        # The interrupted exploration task, as recorded before command 035.
        agent._turn = obs["turn"]
        agent._phase = 3
        agent._task = {"kind": "approach", "target": 6, "selected": (5, 8, 6, 7)}
        agent._group = agent._task["selected"]
        agent._phase_seen[3].update(agent._group)
        agent._queue.append(assignment(AssignmentKind.MOVE, 7, 9, 8, 180, target=6))
        service = ScriptedPlanning()
        approaches = []

        def support(target, selected, *, contacts, mode):
            if mode == SupportMode.APPROACH:
                approaches.append(set(selected))
            return plan(reason=SearchReason.NO_PLAN)

        service.support = support
        action = agent.act_with_planning(obs, tuple(battle.available_actions()), service)
        self.execute(agent, battle, action)
        self.assertTrue(any(e.get("reason") == "new_contact" for e in agent.drain_diagnostics()))
        # Empty searches isolate dispatch from tactical quality; every unit must
        # still get an opportunity to approach before the bounded turn ends.
        for _ in range(16):
            action = agent.act_with_planning(battle.observe(1), tuple(battle.available_actions()), service)
            self.assertIn(action, battle.available_actions())
            self.execute(agent, battle, action)
            if action.kind == "end_turn":
                break
        else:
            self.fail("Unbounded coordinator turn")
        self.assertTrue({5, 6, 8} <= set().union(*approaches),
                        f"Unused explorers never reconsidered for approach: {approaches}")

    def test_greedy_043_moves_to_enable_an_attack_this_turn(self):
        battle = self.battle(42)
        greedy = GreedyAgent()
        greedy.reset({"map": battle._map, "side": 0})
        # Establish a legal witness independently of the policy's choice.
        with battle.clone() as witness:
            self.assertTrue(witness.step(Action("move", 0, 38)).ok)
            self.assertIn(Action("melee", 0, 39), witness.available_actions())
        before = battle.snapshot()
        action = greedy.act(battle.observe(0), tuple(battle.available_actions()))
        self.assertEqual(before, battle.snapshot())
        self.assertIn(action, battle.available_actions())
        self.assertEqual(action.kind, "move")
        self.assertTrue(battle.step(action).ok)
        attacks = [a for a in battle.available_actions()
                   if a.unit == action.unit and a.kind in ("melee", "ranged")]
        self.assertTrue(attacks, "Greedy spent its movement while a move+attack was available")
        followup = greedy.act(battle.observe(0), tuple(battle.available_actions()))
        self.assertIn(followup, attacks)
        self.assertTrue(battle.step(followup).ok)

    def test_search_call_limit_does_not_order_surrounded_shooter_to_wait(self):
        battle = self.battle(50)
        agent = self.agent(battle, 50)
        action = self.act(agent, battle)
        events = agent.drain_diagnostics()
        self.assertLessEqual(agent._searches, agent._search_limit)
        self.assertFalse(any(e.get("reason") == "call_search_limit" and
                             e["event"] == "waiting" for e in events),
                         "Computational limit became a wait order")
        self.assertNotEqual(action, Action("skip", 7))

    @unittest.expectedFailure
    def test_unsafe_participant_does_not_block_useful_group_attacks(self):
        battle = self.battle(50)
        result = self.search(battle, 50, (7, 9), 23)
        attackers = {a.unit for a in result.assignments
                     if a.kind in (AssignmentKind.MELEE, AssignmentKind.RANGED)}
        self.assertIn(9, attackers, "One unsafe participant blocked the other fighter")
        self.assertIn(7, attackers, "The trapped shooter must get an attacking continuation")

    def test_target_area_separates_far_targets_from_horse_local_attack(self):
        battle = self.battle(50)
        for target in (23, 22):
            with self.subTest(target=target):
                result = self.search(battle, 50, (9,), target)
                # Enemy at 118 was admitted by the old "near any selected
                # unit" approximation. It is outside these CP target areas.
                self.assertFalse(any(a.command_target == 118 and a.kind == AssignmentKind.MELEE
                                     for a in result.assignments))
        local = self.search(battle, 50, (9,), 118)
        self.assertTrue(any(a.unit == 9 and a.kind == AssignmentKind.MELEE
                            for a in local.assignments))

    def test_052_fixture_distinguishes_group_proximity_from_target_radius(self):
        battle = self.battle(51)
        own = {u["id"]: u for u in battle.observe(1)["units"] if u["side"] == 1}
        participants = (5, 6, 8, 9)
        self.assertTrue(all(hex_distance(battle._map, own[a]["tile"], own[b]["tile"]) <= 6
                            for a in participants for b in participants))
        excluded = {uid for uid in participants
                    if hex_distance(battle._map, own[uid]["tile"], 54) > 6}
        self.assertEqual(excluded, {5, 6, 8})
        # Preserve evidence of the old decision, not a requirement that the
        # future coordinator must keep this group or always include all four.
        recorded = [e for e in self.fixture["decisions"]
                    if e["command"] == 51 and e["event"] == "search"]
        self.assertEqual(recorded[0]["selected_units"], [9])

    def test_065_joint_plan_attacks_repositions_and_brings_friendly_cover(self):
        battle = self.battle(64)
        agent = self.agent(battle, 64)
        agent._combat_targets = {9: (3, 4, 0, 4), 8: (2, 6, 0, 4)}
        original = {u["id"]: u for u in battle.observe(1)["units"]}
        commands = []
        for _ in range(12):
            action = self.act(agent, battle)
            if action.kind == "end_turn":
                break
            self.execute(agent, battle, action)
            commands.append(action)
            now = {u["id"]: u for u in battle.observe(1)["units"]}
            if all(now[uid]["tile"] != original[uid]["tile"] for uid in (5, 8, 9)):
                break
        else:
            self.fail("No bounded attack/withdrawal/cover continuation")
        events = agent.drain_diagnostics()
        first = next(e for e in events if e["event"] == "search")
        self.assertTrue({5, 8, 9} <= set(first["selected_units"]))
        self.assertLessEqual(len(first["selected_units"]), 13)
        self.assertLessEqual(first["stats"]["generated"], agent.config.position_capacity)
        horse_actions = [a.kind for a in commands if a.unit == 9]
        self.assertIn("melee", horse_actions)
        self.assertIn("move", horse_actions[horse_actions.index("melee") + 1:])
        now = {u["id"]: u for u in battle.observe(1)["units"]}
        self.assertGreater(now[9]["hp"], 0)
        # ABI5 has no previous-turn damage input; CP's historical THIRDLINE
        # predicate receives zero. A forced increase in enemy distance would
        # encode the removed danger>=HP surrogate rather than the source rule.
        self.assertNotEqual(now[9]["tile"], original[9]["tile"])
        for uid in (5, 8):
            self.assertNotEqual(now[uid]["tile"], original[uid]["tile"])
            self.assertLess(hex_distance(battle._map, now[uid]["tile"], now[9]["tile"]),
                            hex_distance(battle._map, original[uid]["tile"], original[9]["tile"]))


if __name__ == "__main__":
    unittest.main()
