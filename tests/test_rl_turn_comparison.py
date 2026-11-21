"""Scientific boundaries: turn openings, world-balanced selection and paired RNG."""
import unittest
from civ5rl.experiments.comparison.turn_collection import position_candidates,select_positions,branch_spec


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

    def test_confirmation_keeps_sampling_but_uses_unseen_paired_damage(self):
        # A changed namespace would silently test a different selected policy.
        p=dict(key=[111068,2,0],command=19)
        original=[branch_spec(p,2,r,'source') for r in range(4)]
        fresh=[branch_spec(p,2,r,'source') for r in range(4,12)]
        self.assertEqual({b['sampling_seed'] for b in fresh},{original[0]['sampling_seed']})
        self.assertFalse({b['seed'] for b in original}&{b['seed'] for b in fresh})
        self.assertEqual(len({b['seed'] for b in fresh}),8)
        self.assertEqual([b['seed'] for b in fresh],
                         [branch_spec(p,0,r,'source')['seed'] for r in range(4,12)])

    def test_cluster_bootstrap_retains_position_weighting(self):
        # Equal-world averaging would silently turn this mean from .5 into 0.
        from civ5rl.experiments.comparison.turn_confirmation import paired_statistics
        from itertools import product
        import numpy as np
        result=paired_statistics([1,1,1,2],[1,1,1,-1],samples=10000,seed=20261103)
        draws=np.random.default_rng(20261103).integers(2,size=(10000,2))
        possible={d:(sum(3 if i==0 else -1 for i in d)/sum(3 if i==0 else 1 for i in d))
                  for d in product(range(2),repeat=2)}
        expected=np.quantile([possible[tuple(d)] for d in draws],[.025,.975]).tolist()
        self.assertEqual(result['mean_win_delta'],.5)
        self.assertEqual(result['ci95'],expected)


if __name__=='__main__':unittest.main()
