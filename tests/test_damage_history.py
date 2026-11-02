"""CP damage history is real battle state, independent of coordinator policy."""

import ctypes as C
import json
import unittest
from copy import deepcopy
from dataclasses import asdict, replace

from test_combat import scenario, unit

from civ5sim import Action, Battle, step_batch
from civ5sim import _native as N
from civ5sim.combat_ai import (
    Aggression,
    AssignmentKind,
    ContactMemory,
    Planner,
    PlanningState,
    SearchConfig,
    SearchReason,
    SearchRequest,
)
from civ5sim.engine import digest


class DamageHistoryTests(unittest.TestCase):
    def battle(self, *units, **settings):
        battle = Battle(scenario(*units, **settings), log=True)
        self.addCleanup(battle.close)
        return battle

    def focused(self):
        battle = self.battle(unit('archer', x=2), unit('archer', x=3),
                             unit('archer', x=4), unit(side=1, x=3, y=3))
        target = 3 + 3 * 12
        total = 0
        for attacker in range(3):
            result = battle.step(Action('ranged', attacker, target))
            self.assertTrue(result.ok)
            total += result.damage
        self.assertGreater(battle.snapshot()['units'][3]['hp'], 0)
        return battle, total

    def test_defender_damage_excludes_attackers_retaliation_and_rotates(self):
        battle = self.battle(unit(), unit(side=1, x=3))
        result = battle.step(Action('melee', 0, 3 + 2 * 12))
        self.assertTrue(result.ok)
        self.assertGreater(result.retaliation, 0)
        before = battle.snapshot()['units']
        self.assertEqual(before[0]['damage_taken_this_turn'], 0)
        self.assertEqual(before[1]['damage_taken_this_turn'], result.damage)
        self.assertTrue(battle.step(Action('end_turn')).ok)
        after = battle.snapshot()['units']
        self.assertEqual(after[1]['damage_taken_last_turn'], result.damage)
        self.assertEqual(after[1]['damage_taken_this_turn'], 0)
        self.assertEqual(after[1]['hp'], min(100, before[1]['hp'] + 10))
        self.assertEqual(after[0]['damage_taken_last_turn'], 0)
        battle.step(Action('end_turn'))
        self.assertEqual(battle.snapshot()['units'][0]['damage_taken_last_turn'], 0)
        battle.step(Action('end_turn'))
        self.assertEqual(battle.snapshot()['units'][1]['damage_taken_last_turn'], 0)

    def test_multiple_attacks_snapshot_clone_batch_reset_and_replay(self):
        battle, total = self.focused()
        snapshot = battle.snapshot()
        self.assertEqual(snapshot['schema'], 2)
        self.assertEqual(snapshot['units'][3]['damage_taken_this_turn'], total)
        with battle.clone() as clone, Battle(json.loads(json.dumps(snapshot))) as restored:
            outcomes = step_batch([clone, restored], [Action('end_turn')] * 2)
            self.assertTrue(all(result.ok for result in outcomes))
            self.assertEqual(clone.snapshot(), restored.snapshot())
            self.assertEqual(clone.snapshot()['units'][3]['damage_taken_last_turn'], total)
            self.assertEqual(battle.snapshot(), snapshot)
            battle.step(Action('end_turn'))
            self.assertEqual(battle.snapshot(), clone.snapshot())
        battle.reset(snapshot)
        self.assertEqual(battle.snapshot(), snapshot)
        battle.reset(scenario(unit(), unit(side=1, x=3)))
        self.assertTrue(all(u['damage_taken_this_turn'] == u['damage_taken_last_turn'] == 0
                            for u in battle.snapshot()['units']))
        fresh, _ = self.focused()
        replay = fresh.replay()
        self.assertEqual(replay['schema'], 2)
        with Battle(replay['initial']) as restored:
            for row in replay['steps']:
                self.assertEqual(asdict(restored.step(row['action'])), row['result'])
                self.assertEqual(digest(restored.snapshot()), row['state_sha256'])

    def test_invalid_commands_previews_and_history_validation_are_atomic(self):
        battle, _ = self.focused()
        initial = battle.snapshot()
        self.assertFalse(battle.step(Action('melee', 0, 3 + 3 * 12)).ok)
        battle.preview_attack(Action('ranged', 0, 3 + 3 * 12))
        self.assertEqual(battle.snapshot(), initial)
        for field in ('damage_taken_this_turn', 'damage_taken_last_turn'):
            for value in (-1, True, 2**31, None):
                invalid = deepcopy(initial)
                invalid['units'][0][field] = value
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    battle.reset(invalid)
                self.assertEqual(battle.snapshot(), initial)
            invalid = deepcopy(initial)
            del invalid['units'][0][field]
            with self.assertRaisesRegex(ValueError, 'complete damage history'):
                battle.reset(invalid)
            self.assertEqual(battle.snapshot(), initial)
        old = deepcopy(initial)
        old['schema'] = 1
        with self.assertRaisesRegex(ValueError, 'schema 2'):
            Battle(old)
        self.assertEqual(battle.snapshot(), initial)

    def test_private_enemy_history_is_masked_in_observation_and_ai_input(self):
        battle, _ = self.focused()
        own = next(u for u in battle.observe(1)['units'] if u['side'] == 1)
        self.assertGreater(own['damage_taken_this_turn'], 0)
        enemy = next(u for u in battle.observe(0)['units'] if u['side'] == 1)
        self.assertNotIn('damage_taken_this_turn', enemy)
        self.assertNotIn('damage_taken_last_turn', enemy)
        native = battle.observation_buffer(0)['units'][3]
        self.assertEqual((native.damage_taken_this_turn, native.damage_taken_last_turn), (-1, -1))
        battle.step(Action('end_turn'))
        self.assertEqual(PlanningState.from_battle(battle, 0).units[3].damage_taken_last_turn, 0)
        self.assertGreater(PlanningState.from_battle(battle, 1).units[3].damage_taken_last_turn, 0)
        self.assertEqual(N.library().sim_abi(), 2)

    def test_real_history_reaches_public_tactical_search_without_coordinator(self):
        battle, total = self.focused()
        battle.step(Action('end_turn'))
        state = PlanningState.from_battle(battle, 1)
        self.assertEqual(state.units[3].damage_taken_last_turn, total)
        self.assertGreater(total, state.units[3].hp)
        request = SearchRequest(state, 1, state.turn, state.units[1].tile, (3,),
                                ContactMemory().updated(battle.observe(1)))
        before = battle.snapshot()
        # Keep the damaged participant available so the public result exposes
        # its role; otherwise root pruning may legitimately exclude it.
        with Planner(SearchConfig(aggression=Aggression.BRAVEHEART, risk_budget_base=13)) as planner:
            self.assertEqual(planner._lib.cai_abi(), 6)
            result = planner.search(request)
            initial = next(a for a in result.assignments if a.kind == AssignmentKind.INITIAL)
            self.assertEqual(initial.movement_role, 3)
            self.assertEqual(planner.search(request), result)
            self.assertEqual(battle.snapshot(), before)
            negative = replace(state.units[3], damage_taken_last_turn=-1)
            invalid = replace(request, state=replace(state, units=(*state.units[:3], negative)))
            self.assertEqual(planner.search(invalid).reason, SearchReason.INVALID_REQUEST)
            self.assertEqual(planner.search(request), result)

    def test_native_reset_rejects_negative_counter_without_mutation(self):
        battle, _ = self.focused()
        before = battle.snapshot()
        units = battle._units()
        units[0].damage_taken_last_turn = -1
        info = battle._info()
        self.assertNotEqual(N.library().sim_reset(battle._handle, units, len(units), C.byref(info)), 0)
        self.assertEqual(battle.snapshot(), before)

    def test_previous_interval_is_frozen_during_owners_attack(self):
        battle = self.battle(unit(), unit(side=1, x=3))
        first = battle.step(Action('melee', 0, 3 + 2 * 12))
        battle.step(Action('end_turn'))
        second = battle.step(Action('melee', 1, 2 + 2 * 12))
        self.assertTrue(second.ok)
        units = battle.snapshot()['units']
        self.assertEqual(units[1]['damage_taken_last_turn'], first.damage)
        self.assertEqual(units[1]['damage_taken_this_turn'], 0)
        self.assertEqual(units[0]['damage_taken_this_turn'], second.damage)

    def test_fresh_corpus_requires_explicit_history_in_serialized_requests(self):
        from tools.tactical_corpus import fresh_cases, load_request

        cases = fresh_cases()
        self.assertEqual(len(cases), 12)
        for case in cases.values():
            self.assertEqual(case['snapshot']['schema'], 2)
            self.assertTrue(load_request(case['request']).state.units)
        legacy = deepcopy(next(iter(cases.values()))['request'])
        del legacy['state']['units'][0]['damage_taken_last_turn']
        with self.assertRaisesRegex(ValueError, 'Legacy tactical request'):
            load_request(legacy)
