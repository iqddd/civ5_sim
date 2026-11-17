"""Scientific boundaries: turn openings, world-balanced selection and paired RNG."""
import unittest
from civ5rl.experiments.turn_comparison import position_candidates,select_positions,branch_spec


class TurnComparisonTests(unittest.TestCase):
    def test_never_skip_forced_opening_or_include_unseen_contact(self):
        case=dict(learner_side=0,parent_world=1)
        def row(turn,command,n=2,contact=True):
            return dict(command=command,observation=dict(turn=turn,units=[dict(side=1)] if contact else []),actions=[{}]*n)
        history=[row(0,0),row(2,2,1),row(2,3),row(4,5,contact=False),row(6,8)]
        replay=dict(steps=[dict(action=dict(kind='end_turn')) for _ in range(9)])
        self.assertEqual([p['key'] for p in position_candidates(12,case,history,replay)],[[12,6,0]])

    def test_one_per_battle_precedes_second_and_order_is_stable(self):
        groups={w:{e:[dict(key=[e,t,0]) for t in (2,4,6)] for e in range(w*10,w*10+3)} for w in (1,2)}
        positions=select_positions(groups,10)
        self.assertEqual(len({p['key'][0] for p in positions[:6]}),6)
        self.assertEqual(positions,select_positions(dict(reversed(list(groups.items()))),10))
        self.assertEqual(len({tuple(p['key']) for p in positions}),10)
        self.assertEqual(len(select_positions(groups,100)),12)

    def test_sampling_and_damage_streams_have_distinct_dependencies(self):
        p=dict(key=[111068,2,0],command=19)
        branches=[[branch_spec(p,v,r,'source') for r in range(4)] for v in range(4)]
        self.assertEqual(len({b['sampling_seed'] for b in branches[1]}),1)
        self.assertEqual(len({row[0]['sampling_seed'] for row in branches}),4)
        self.assertEqual(len({row[0]['seed'] for row in branches}),1)
        self.assertEqual(len({b['seed'] for b in branches[0]}),4)


if __name__=='__main__':unittest.main()
