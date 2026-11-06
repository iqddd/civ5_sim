"""New risks only: two CP histories and bounded file input preserving BC math."""
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import torch

from civ5rl.core.model import Policy
from civ5rl.engine.imitation import DemonstrationCache, backward_minibatch, label_minibatches
from civ5rl.engine.imitation_stream import PreparedDemonstrations, prepare_episode
from civ5rl.engine.ppo_fast import prefixes
from civ5rl.experiments.imitation_target_scale import load_config
from civ5rl.protocols.imitation import RecordingCP, dump_episode, run_imitation_match, search_config
from civ5rl.protocols.imitation_target_scale import KIND, match_task
from civ5rl.protocols.ppo_target_scale import TargetFactory
from civ5rl.provenance.common import ROOT, configure, read_json
from civ5sim import Action, Battle
from civ5sim.cp_agent import CPAgent
from civ5sim.engine import digest


class TargetImitationChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure(17,'cpu')
        cls.config=load_config(ROOT/'configs/rl-cp-imitation-target-scale.json')
        cls.plan=read_json(ROOT/cls.config['target_manifest'])

    def test_both_CP_sides_record_executed_public_history(self):
        # Two representative full matches; no corpus-wide replay audit.
        factory=TargetFactory(self.config,self.plan['base'])
        for contact in (False,True):
            case=factory.case(900+contact,self.plan['base']['worlds']['training'][0],10,0,contact,0,
                              'target_BC_recording_check_v1',n=5)
            case.update(source=KIND,case_id=case['episode_id']);case['case_hash']=digest(case)
            with tempfile.TemporaryDirectory() as temporary:
                directory=Path(temporary)
                row=match_task((self.config,case,str(directory/'recorded')))
                from civ5rl.protocols.imitation import load_episode
                episodes=[load_episode(Path(e['path'])) for e in row['entries']]
                replay=read_json(directory/'recorded/replay.json');indices=[0,0]
                with Battle(replay['initial']) as battle:
                    for step in replay['steps']:
                        side=battle.snapshot()['state']['side'];decision=episodes[side]['decisions'][indices[side]]
                        self.assertEqual(decision['observation'],battle.observe(side))
                        self.assertEqual(decision['actions'][decision['chosen']],step['action'])
                        self.assertEqual(asdict(battle.step(step['action'])),step['result'])
                        self.assertEqual(digest(battle.snapshot()),step['state_sha256'])
                        indices[side]+=1
                self.assertEqual(indices,[len(e['decisions']) for e in episodes])
                ordinary=[CPAgent(search_config(self.config)) for _ in (0,1)]
                outcome,_,_=run_imitation_match(self.config,case,ordinary,directory/'ordinary')
                self.assertEqual(digest(outcome.replay['initial']),digest(replay['initial']))
                self.assertEqual(digest(outcome.replay['steps']),digest(replay['steps']))
                # Reset must clear recording and CP memory for a different side.
                recorder=RecordingCP(search_config(self.config))
                try:
                    recorder.reset(episodes[0]['context'])
                    recorder.decisions.append({'sentinel':True})
                    recorder.reset(episodes[1]['context'])
                    self.assertEqual(recorder.decisions,[])
                    self.assertEqual(recorder.public_context['side'],1)
                finally:recorder.close()

    def test_file_eviction_preserves_loss_gradients_Adam_and_prefix(self):
        # A new file cache could silently alter previous-action history/label order.
        factory=TargetFactory(self.config,self.plan['base'])
        case=factory.case(901,self.plan['base']['worlds']['training'][0],10,0,True,0,'file_BC_math',n=5)
        with Battle(case['scenario']) as battle:
            observation=battle.observe(0);legal=tuple(battle.available_actions())
        context={'map':case['scenario']['map'],'side':0,'armies':case['armies'],
                 'max_rounds':30,'fog':True,'random_damage':True}
        free={'observation':observation,'actions':[asdict(a) for a in legal],
              'chosen':legal.index(Action('end_turn')),'supervised':True}
        forced={**free,'actions':[asdict(Action('end_turn'))],'chosen':0,'supervised':False}
        episodes=[{'context':context,'decisions':[free,forced,free,forced,free,free]},
                  {'context':{**context,'side':0},'decisions':[forced,free,forced,free,free]}]
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);entries=[]
            for i,ep in enumerate(episodes):
                path=f'{i}.json.gz';dump_episode(root/path,ep)
                entries.append({'path':path,'supervised':[r['supervised'] for r in ep['decisions']]})
            prepared=[prepare_episode((str(root),e,str(root/f'{i}.npz'))) for i,e in enumerate(entries)]
            file_cache=PreparedDemonstrations(root,prepared,'cpu',limit_bytes=1)
            reference=DemonstrationCache(episodes,'cpu')
            spans=[(0,2,6),(1,1,5)]
            count=sum(sum(reference.masks[e][a:b]) for e,a,b in spans)
            model=Policy('C');clone=deepcopy(model)
            for p in model.value.parameters():p.requires_grad_(False)
            for p in clone.value.parameters():p.requires_grad_(False)
            a=backward_minibatch(model,reference,spans,count,micro_size=2)
            b=backward_minibatch(clone,file_cache,spans,count,micro_size=1)
            self.assertAlmostEqual(a,b,places=6)
            for p,q in zip(model.parameters(),clone.parameters()):
                if p.requires_grad:torch.testing.assert_close(p.grad,q.grad,rtol=3e-5,atol=2e-7)
            before={k:v.clone() for k,v in clone.value.state_dict().items()}
            for policy in (model,clone):
                torch.nn.utils.clip_grad_norm_(policy.parameters(),.5)
                torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=1e-4,eps=1e-5).step()
            for p,q in zip(model.parameters(),clone.parameters()):torch.testing.assert_close(p,q,rtol=3e-5,atol=2e-7)
            self.assertTrue(all(torch.equal(v,clone.value.state_dict()[k]) for k,v in before.items()))
            self.assertGreater(file_cache.counters['prefix_positions'],0)
            packed=list(label_minibatches(file_cache.chunks(),file_cache.masks,size=4))
            self.assertEqual([n for _,n in packed],[4,3])


if __name__=='__main__':unittest.main()
