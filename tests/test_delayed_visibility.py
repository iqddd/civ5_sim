"""CP CORE_DELAYED_VISIBILITY is state, including mid-turn search roots."""

import json
import unittest
from dataclasses import asdict

from test_combat import scenario, unit

from civ5sim import Action, Battle, step_batch
from civ5sim.combat_ai import ContactMemory, Planner, PlanningState, SearchRequest
from civ5sim.engine import digest
from tools.view_combat import prepare_replay


class DelayedVisibilityTests(unittest.TestCase):
    def battle(self, side=0, fog=True):
        value = scenario(unit('horseman', side=side), unit('archer', side=1-side, x=4), fog=fog)
        value['active_side'] = side
        battle = Battle(value, log=True)
        self.addCleanup(battle.close)
        return battle

    def test_retreat_keeps_enemy_visible_and_return_does_not_rediscover(self):
        for side in (0, 1):
            b = self.battle(side)
            self.assertIn(1, [u['id'] for u in b.observe(side)['units']])
            self.assertTrue(b.step(Action('move', 0, 24)).ok)
            self.assertTrue(b.observe(side)['visibility'][28])
            self.assertIn(1, [u['id'] for u in b.observe(side)['units']])
            result = b.step(Action('move', 0, 26))
            self.assertEqual(result.moved, 2)
            self.assertEqual(b.snapshot()['units'][0]['tile'], 26)

    def test_turn_boundaries_expire_sight_for_both_sides(self):
        for side in (0, 1):
            b = self.battle(side)
            b.step(Action('move', 0, 24))
            b.step(Action('end_turn'))
            self.assertFalse(b.observe(side)['visibility'][28])
            self.assertNotIn(1, [u['id'] for u in b.observe(side)['units']])
            b.step(Action('move', 1, 29))
            b.step(Action('end_turn'))
            self.assertFalse(b.observe(side)['visibility'][28])

    def test_snapshot_clone_batch_reset_and_replay_preserve_history(self):
        b = self.battle()
        b.step(Action('move', 0, 24))
        snapshot = b.snapshot()
        self.assertTrue(snapshot['state']['visibility'][0][28])
        with b.clone() as clone, Battle(json.loads(json.dumps(snapshot))) as restored:
            self.assertEqual(clone.snapshot(), snapshot)
            self.assertEqual(restored.snapshot(), snapshot)
            results = step_batch([clone, restored], [Action('end_turn')]*2)
            self.assertTrue(all(r.ok for r in results))
            self.assertEqual(clone.snapshot(), restored.snapshot())
            self.assertFalse(restored.observe(0)['visibility'][28])
        b.step(Action('end_turn'))
        b.reset(snapshot)
        self.assertEqual(b.snapshot(), snapshot)
        self.assertFalse(b.step(Action('move', 0, -1)).ok)
        self.assertEqual(b.snapshot(), snapshot)
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'replay.json'
            path.write_text(json.dumps(b.replay()), encoding='utf-8')
            # reset() begins a fresh replay at the resumed mid-turn snapshot.
            prepared = prepare_replay(path)
            self.assertEqual(prepared['visibility_mode'], 'delayed')
            self.assertEqual(prepared['frames'][0]['visibility'][0][28], '1')
        replay = b.replay()
        with Battle(replay['initial']) as restored:
            for step in replay['steps']:
                self.assertEqual(asdict(restored.step(step['action'])), step['result'])
                self.assertEqual(digest(restored.snapshot()), step['state_sha256'])

    def test_invalid_snapshot_mask_does_not_mutate_real_battle(self):
        b = self.battle()
        before = b.snapshot()
        for value in (None, [], [[False]*96]*2, [[True]*96, [True]*95], [[1]*96]*2):
            bad = json.loads(json.dumps(before))
            bad['state']['visibility'] = value
            with self.assertRaises(ValueError):
                b.reset(bad)
            self.assertEqual(b.snapshot(), before)

    def test_search_accepts_history_and_leaves_real_state_unchanged(self):
        b = self.battle()
        b.step(Action('move', 0, 24))
        before = b.snapshot()
        state = PlanningState.from_battle(b, 0)
        self.assertTrue(state.visibility[28])
        with Planner() as planner:
            planner.search(SearchRequest(state, 0, 0, 28, (0,), ContactMemory().updated(b.observe(0))))
        self.assertEqual(b.snapshot(), before)

    def test_no_fog_stays_fully_visible(self):
        b = self.battle(fog=False)
        b.step(Action('move', 0, 24))
        mid_turn = b.snapshot()
        self.assertFalse(all(mid_turn['state']['visibility'][0]))
        with Battle(mid_turn) as restored:
            self.assertEqual(restored.snapshot(), mid_turn)
            self.assertTrue(all(restored.observe(0)['visibility']))
        b.step(Action('end_turn'))
        self.assertTrue(all(b.observe(0)['visibility']))
        with Battle(b.snapshot()) as restored:
            self.assertEqual(restored.snapshot(), b.snapshot())
