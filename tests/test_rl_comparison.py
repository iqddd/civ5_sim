"""Silent math/selection failures would invalidate counterfactual labels."""
import unittest
import torch
from civ5rl.experiments.comparison_collection import preference,select_forks
from civ5rl.engine.comparison_training import pair_loss,categorical_kl

class ComparisonTests(unittest.TestCase):
    def test_finalized_prefix_collection_imports_and_restores_training_histories(self):
        # Prefix captures are useful during collection, but retaining them in
        # the source packet makes its two-file training import partial.
        import tempfile
        from pathlib import Path
        from dataclasses import asdict
        from types import SimpleNamespace
        from civ5rl.provenance.common import ROOT,read_json,write_json,file_sha256
        from civ5rl.provenance.checkpoint import passport
        from civ5rl.experiments.rollout_comparison_collection import finalize_collection
        from civ5rl.experiments.rollout_comparison import prepare_collection_input
        from civ5rl.reporting.comparison_storage import restore_tree
        from civ5rl.engine.comparison_inputs import Histories
        from civ5rl.experiments.comparison_prefix import SourcePrefixes
        from civ5rl.engine.evaluation import EvaluationRuntime
        from civ5rl.core.model import Policy
        from tests.test_combat import scenario,unit
        from civ5sim import Battle
        with tempfile.TemporaryDirectory(dir=ROOT/'.tmp',prefix='prefix-import-') as tmp:
            source=Path(tmp)/'collection';destination=Path(tmp)/'training';episode=1
            folder=source/'sources'/str(episode)
            spec=scenario(unit(x=0,y=0),unit(side=1,x=1,y=1))
            case=dict(scenario=spec,armies=[['warrior'],['warrior']],learner_side=0,
                      max_rounds=60,agent_seeds=[17,18])
            with Battle(spec) as battle:
                actions,previews=battle.actions_with_previews();observation=battle.observe(0)
                observation['attack_previews']=previews.tolist()
            row=dict(command=1,chosen=0,alternative=1,actions=[asdict(a) for a in actions],observation=observation)
            cfg=dict(seed=17,namespace='prefix-import-smoke',collection_seconds=30,parent_sha256=[])
            write_json(folder/'case.json',case);write_json(folder/'history.json',[row])
            write_json(source/'manifest.json',{})
            write_json(source/'match-config.json',{})
            passport(source,cfg,{})
            policy=Policy('C2P').eval();weight_file=Path(tmp)/'parent.pt'
            weight_file.write_bytes(b'smoke checkpoint identity')
            prefixes=SourcePrefixes(policy,weight_file,EvaluationRuntime('cpu',1,1,.001))
            prefixes.capture(folder,[torch.zeros(1,1,policy.hidden_size)])
            (folder/'prefix-states.tmp').write_bytes(b'interrupted cache write')
            self.assertTrue((folder/'prefix-states.npz').exists())
            state=dict(sources=[episode],forks=[dict(id=0,episode=episode,index=0,status='completed',
                       label='accepted',reason=None,accepted=True,better=0,weight=.5,executed_repetitions=8)])
            finalize_collection(cfg,source,state,SimpleNamespace(elapsed=lambda:1.))
            names=('dataset.json','anchors.json','collection.json','match-config.json','manifest.json',
                   'passport.json','resolved-config.json','corpus-manifest.json','sources.zip',
                   'sources.storage/manifest.json','sources.storage/json.7z')
            fit_cfg=dict(parents=[],parent_sha256=[],collection_input=dict(
                run=source.relative_to(ROOT).as_posix(),sha256={n:file_sha256(source/n) for n in names}))
            prepare_collection_input(fit_cfg,destination)
            restore_tree(destination,'sources')
            self.assertEqual({p.name for p in (destination/'sources'/'1').iterdir()}, {'case.json','history.json'})
            data=read_json(destination/'dataset.json')
            self.assertEqual(len(data['pairs']),1)
            histories=Histories(destination,source_hashes=data['source_hashes'])
            encoded,_=histories.get(episode)
            self.assertEqual(len(encoded),1)
            self.assertEqual(len(encoded[0].sources),len(actions))

    def test_branch_prefix_restore_preserves_hidden_and_local_encoder(self):
        # A wrong boundary/forced-action history can silently change every
        # continuation label. Compare with stepwise inference and reject stale data.
        import tempfile
        from pathlib import Path
        from dataclasses import asdict
        import numpy as np
        from civ5rl.core.model import Policy,batch_inputs
        from civ5rl.core.context import context_from_spec
        from civ5rl.core.observation import ObservationEncoder,ordered_actions
        from civ5rl.engine.evaluation import EvaluationRuntime
        from civ5rl.engine.evaluation_proxy import ProxyAgent
        from civ5rl.experiments.comparison_prefix import SourcePrefixes
        from civ5rl.provenance.common import configure,write_json
        from civ5rl.protocols.memory import flat_map
        from civ5sim import Battle,Action
        from tests.test_combat import scenario,unit
        configure(17,'cpu')
        s=scenario(unit(x=0,y=0),unit(side=1,x=1,y=1),terrain=flat_map(2))
        case=dict(scenario=s,armies=[['warrior'],['warrior']],learner_side=0,
                  max_rounds=60,agent_seeds=[17,18])
        context=context_from_spec(case,seed=17)
        encoder=ObservationEncoder(context,'C2P');policy=Policy('C2P').eval()
        rows=[];frames=[];hidden=None
        with Battle(s) as battle,torch.inference_mode():
            for command in (0,2,4):
                legal,preview=battle.actions_with_previews();legal=ordered_actions(legal)
                obs=battle.observe(0);obs['attack_previews']=preview.tolist()
                encoded=encoder.encode(obs,legal)
                _,_,hidden=policy(batch_inputs([encoded],'cpu'),hidden)
                frames.append(hidden.clone())
                selected=legal.index(Action('end_turn'))
                rows.append(dict(command=command,observation=obs,actions=[asdict(a) for a in legal],chosen=selected))
                encoder.remember_action(legal[selected],obs)
                battle.step(Action('end_turn'));battle.step(Action('end_turn'))
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp);write_json(source/'case.json',case);write_json(source/'history.json',rows)
            checkpoint=source/'weights';checkpoint.write_bytes(b'fixed-parent')
            runtime=EvaluationRuntime('cpu').resolve()
            prefixes=SourcePrefixes(policy,checkpoint,runtime)
            restored,count=prefixes.restore(source,4)
            self.assertEqual(count,3)
            torch.testing.assert_close(restored,frames[-1],atol=2e-6,rtol=1e-5)
            self.assertEqual(prefixes.prefills,1)
            prefixes.capture(source,frames)
            fresh=SourcePrefixes(policy,checkpoint,runtime)
            captured,_=fresh.restore(source,4)
            self.assertTrue(torch.equal(captured,frames[-1]));self.assertEqual(fresh.prefills,0)
            with torch.inference_mode():captured.add_(1)
            again,_=fresh.restore(source,4)
            self.assertTrue(torch.equal(again,frames[-1]))
            # No connection exists: processing the prefix must stay local.
            proxy=ProxyAgent(None,'argmax');proxy.reset_public_state(context,'C2P');proxy.prefix_skip=count
            reference=ObservationEncoder(context,'C2P')
            for i,row in enumerate(rows):
                legal=[Action(**a) for a in row['actions']]
                action=legal[row['chosen']] if i<2 else next(a for a in legal if a.kind!='end_turn')
                reference.encode(row['observation'],legal);reference.remember_action(action,row['observation'])
                proxy.observe_step(row['observation'],legal,action)
            self.assertEqual(proxy.prefix_skip,0)
            np.testing.assert_array_equal(proxy.encoder.previous,reference.previous)
            np.testing.assert_array_equal(proxy.encoder.last_visible,reference.last_visible)
            self.assertEqual(proxy.encoder.contacts,reference.contacts)
            write_json(source/'history.json',rows[:-1])
            with self.assertRaisesRegex(ValueError,'history changed'):fresh.restore(source,4)

    def test_vector_losses_preserve_padding_multiplicity_and_fp64_control(self):
        # Zero padding probabilities must never multiply an infinite log-delta;
        # repeats and independent pair/anchor means must retain their weights.
        from civ5rl.engine.comparison_losses import preference_and_kl,full_kl_rows
        from civ5rl.engine.comparison_training import DeviceReferences
        lengths=[1,3,5]
        teacher={i:(torch.linspace(-.7,.6,n)/.5).log_softmax(-1) for i,n in enumerate(lengths)}
        raw=torch.tensor([[.2,0,0,0,0],[.3,-.6,.2,0,0],[-.5,.1,.7,-.4,.3]])
        mask=torch.arange(5)[None]<torch.tensor(lengths)[:,None]
        pairs=[(1,2,0),(1,1,2),(2,3,4)]
        weights=[.5/3,.25/3,1./3];counts=[1/4,1/4,2/4]
        reference=raw.clone().requires_grad_()
        a=sum(w*torch.nn.functional.softplus((reference[i,worse]-reference[i,better])/.5)
              for (i,better,worse),w in zip(pairs,weights))
        b=sum(c*categorical_kl(teacher[i],reference[i,:n]) for i,(n,c) in enumerate(zip(lengths,counts)))
        (a+b).backward()
        control=sum(c*categorical_kl(teacher[i],raw[i,:n],double=True) for i,(n,c) in enumerate(zip(lengths,counts)))
        for device in ('cpu',*(['cuda'] if torch.cuda.is_available() else [])):
            value=raw.detach().to(device).clone().requires_grad_();logits=value.masked_fill(~mask.to(device),-torch.inf)
            bank=DeviceReferences(teacher,device);parent,_=bank.gather([0,1,2],5)
            actual=preference_and_kl(logits,torch.tensor(pairs,device=device),torch.tensor(weights,device=device),parent,torch.tensor(counts,device=device))
            torch.testing.assert_close(actual[0].cpu(),a.detach());torch.testing.assert_close(actual[1].cpu(),b.detach(),atol=2e-7,rtol=1e-5)
            sum(actual).backward()
            torch.testing.assert_close(value.grad.cpu(),reference.grad,atol=2e-7,rtol=1e-5)
            self.assertTrue(torch.isfinite(value.grad).all());self.assertTrue(torch.equal(value.grad.cpu()[~mask],torch.zeros_like(value.grad.cpu()[~mask])))
            bank=DeviceReferences(teacher,device,control=True);parent,probability=bank.gather([0,1,2],5)
            rows=full_kl_rows(logits.detach(),parent,probability)
            current=(logits.detach()/.5).log_softmax(-1).cpu().double()
            cpu_control=sum(c*(teacher[i].double().exp()*(teacher[i].double()-current[i,:n])).sum()
                            for i,(n,c) in enumerate(zip(lengths,counts)))
            measured=(rows*torch.tensor(counts,device=device,dtype=torch.float64)).sum().cpu()
            torch.testing.assert_close(measured,cpu_control,rtol=1e-12,atol=1e-12)
            torch.testing.assert_close(measured,control,rtol=1e-6,atol=1e-7)
            zero=preference_and_kl(logits,torch.empty((0,3),device=device,dtype=torch.long),torch.empty(0,device=device),parent.float(),torch.zeros(3,device=device))
            self.assertEqual(float(sum(zero).detach()),0.)

    def test_checkpoint_cpu_image_is_independent_and_restores_adam_rng(self):
        # Checkpointing a live CUDA tree can silently retain aliases/device tags.
        # Validate the saved values and resumed optimizer, not serialization calls.
        import tempfile
        from pathlib import Path
        from copy import deepcopy
        from civ5rl.engine.comparison_training import atomic_torch_save
        from civ5rl.provenance.rng import capture_rng,restore_rng
        from civ5rl.experiments.imitation_helpers import assert_equal
        for device in ('cpu',*(['cuda'] if torch.cuda.is_available() else [])):
            model=torch.nn.Linear(4,3).to(device)
            optimizer=torch.optim.Adam(model.parameters(),lr=1e-5)
            model(torch.ones(2,4,device=device)).square().mean().backward();optimizer.step()
            state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng(),
                       transposed=torch.arange(12,device=device,dtype=torch.float64).view(3,4).t())
            with tempfile.TemporaryDirectory() as temporary:
                a,b=(Path(temporary)/name for name in ('verified.pt','training.pt'))
                image=atomic_torch_save(state,a,b)
                self.assertEqual(a.read_bytes(),b.read_bytes())
                saved=torch.load(a,weights_only=False)
                def check_cpu(value):
                    if torch.is_tensor(value):self.assertEqual(value.device.type,'cpu')
                    elif isinstance(value,dict):
                        for child in value.values():check_cpu(child)
                    elif isinstance(value,(list,tuple)):
                        for child in value:check_cpu(child)
                check_cpu(saved);assert_equal(saved,image)
                with torch.no_grad():
                    for p in model.parameters():p.add_(3)
                self.assertFalse(torch.equal(model.weight.cpu(),saved['model']['weight']))
                restored=deepcopy(model);other=torch.optim.Adam(restored.parameters(),lr=1e-5)
                restored.load_state_dict(saved['model']);other.load_state_dict(saved['optimizer']);restore_rng(saved['rng'])
                assert_equal(restored.state_dict(),image['model']);assert_equal(other.state_dict(),image['optimizer'])
                assert_equal(capture_rng(),image['rng'])

    def test_prepared_histories_preserve_public_prefixes_beyond_eight_episodes(self):
        # A small LRU silently repeats costly JSON/encoding work. Prepared data
        # must retain previous-action/forced history and survive a fresh process.
        import tempfile
        from pathlib import Path
        from dataclasses import asdict
        from copy import deepcopy
        from civ5rl.core.observation import ObservationEncoder,ordered_actions
        from civ5rl.core.context import context_from_spec
        from civ5rl.core.model import batch_inputs
        from civ5rl.engine.comparison_inputs import Histories
        from civ5rl.protocols.memory import flat_map
        from civ5rl.provenance.common import write_json,read_json
        from civ5sim import Battle,Action
        from tests.test_combat import scenario,unit
        s=scenario(unit(x=0,y=0),unit(side=1,x=1,y=1),terrain=flat_map(2))
        case=dict(scenario=s,armies=[['warrior'],['warrior']],learner_side=0,max_rounds=60,agent_seeds=[17,18])
        with Battle(s) as battle:
            actions,previews=battle.actions_with_previews();actions=ordered_actions(actions)
            obs=battle.observe(0);obs['attack_previews']=previews.tolist()
        rows=[dict(observation=obs,actions=[asdict(a) for a in actions],chosen=0),
              dict(observation={**obs,'attack_previews':[[0.]*5]},actions=[asdict(Action('end_turn'))],chosen=0),
              dict(observation=obs,actions=[],chosen=999)]
        encoder=ObservationEncoder(context_from_spec(case,seed=17),'C2P');expected=[]
        for row in rows[:2]:
            legal=[Action(**a) for a in row['actions']]
            expected.append(encoder.encode(row['observation'],legal));encoder.remember_action(legal[row['chosen']],row['observation'])
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)
            for episode in range(10):
                write_json(out/'sources'/str(episode)/'case.json',case)
                write_json(out/'sources'/str(episode)/'history.json',rows)
            store=Histories(out,limits={e:2 for e in range(10)})
            store.prepare([(e,1) for e in range(10)])
            for _ in range(2):
                for episode in range(10):
                    history,raw=store.get(episode)
                    self.assertIsNone(raw);self.assertEqual(len(history),2)
                    for a,b in zip(history,expected):
                        for name in vars(b):
                            import numpy as np
                            np.testing.assert_array_equal(getattr(a,name),getattr(b,name))
            history,_=store.get(0)
            for device in ('cpu',*(['cuda'] if torch.cuda.is_available() else [])):
                for gpu,pinned in ((0,0),(1024**2,0),(0,1024**2)):
                    packed=Histories(out,limits={0:2},device=device,gpu_bytes=gpu,pinned_bytes=pinned)
                    packed.prepare([(0,1)])
                    view,_=packed.get(0)
                    actual=packed.spatial_inputs([view[:1],view[1:]],device)
                    reference=batch_inputs(expected,device,spatial_only=True,compact_units=True)
                    for key in reference:torch.testing.assert_close(actual[key],reference[key],rtol=0,atol=0)
                    self.assertEqual(packed.statistics().get('encoded_episodes',0),0)
            self.assertEqual(store.statistics()['encoded_decisions'],20)
            self.assertEqual(store.statistics()['raw_rows_retained'],0)
            # A changed public source must select a new cache, never stale inputs.
            changed=deepcopy(rows)
            for row in changed[:2]:row['observation']['turn']=42
            write_json(out/'sources/0/history.json',changed)
            fresh=Histories(out,limits={0:2});altered,_=fresh.get(0)
            self.assertFalse(torch.equal(torch.from_numpy(altered[0].globals),torch.from_numpy(history[0].globals)))
    def test_batched_histories_keep_prefix_boundaries_and_preference_weights(self):
        # Wrong GRU grouping, padding, or loss multiplicity can silently train on
        # another episode's memory. Compare the actual accumulated gradients.
        from copy import deepcopy
        from dataclasses import replace
        from civ5rl.core.model import Policy
        from civ5rl.core.observation import ObservationEncoder,ordered_actions
        from civ5rl.engine.comparison_training import target_logits,logical_step,reference_distributions
        from civ5rl.provenance.common import configure
        from civ5rl.protocols.memory import flat_map
        from civ5sim import Battle
        from tests.test_combat import scenario,unit
        from tests.test_rl import context
        configure(17,'cpu')
        s=scenario(unit(x=0,y=0),unit(side=1,x=1,y=1),terrain=flat_map(2))
        with Battle(s) as battle:
            legal,previews=battle.actions_with_previews()
            obs=battle.observe(0);obs['attack_previews']=previews
            item=ObservationEncoder(context(s),'C2P').encode(obs,ordered_actions(legal))
        first=[replace(item,previous=item.previous+i*.001) for i in range(130)]
        second=[replace(item,previous=item.previous-i*.003) for i in range(3)]
        class Histories:
            def get(self,e):return ({1:first,2:second}[e],None)
        histories=Histories()
        a=Policy('C2P');a.configure_execution(inverse_routing=True,action_routing=True,maximum_routing=True,attention_backend='compact_fp32')
        b=deepcopy(a)
        pairs=[dict(episode=1,index=129,better=1,worse=0,weight=.5),dict(episode=2,index=2,better=0,worse=1,weight=.25),
               dict(episode=1,index=80,better=0,worse=1,weight=.25)]
        anchors=[(1,129),(1,128),(1,80),(2,0),(2,2),(2,2)]
        refs=reference_distributions(a,histories,anchors)
        with torch.no_grad():
            for key in set(anchors):
                h,_=histories.get(key[0]);z=target_logits(a,h,key[1])
                torch.testing.assert_close(refs[key],(z/.5).log_softmax(-1),atol=2e-6,rtol=2e-5)
        # Nonzero KL as well as preference gradients; one position appears three times.
        with torch.no_grad():a.action[-1].weight[0,0].add_(.01);b.load_state_dict(a.state_dict())
        for row in pairs:
            h,_=histories.get(row['episode'])
            z=target_logits(a,h,row['index'],gradient=True)
            (pair_loss(z,row['better'],row['worse'],row['weight'])/len(pairs)).backward()
        for key in anchors:
            h,_=histories.get(key[0]);z=target_logits(a,h,key[1],gradient=True)
            (categorical_kl(refs[key],z)/len(anchors)).backward()
        torch.nn.utils.clip_grad_norm_([p for p in a.parameters() if p.requires_grad],.5)
        optimizer=torch.optim.Adam(b.parameters(),lr=0.)
        logical_step(b,optimizer,histories,pairs,anchors,refs)
        for (name,p),(_,q) in zip(a.named_parameters(),b.named_parameters()):
            if p.grad is not None:torch.testing.assert_close(q.grad,p.grad,atol=2e-6,rtol=3e-4,msg=name)
        if torch.cuda.is_available():
            # Calling CoalescedSpatial directly must retain the executor's BF16
            # autocast boundary. A missing boundary silently changes precision.
            from civ5rl.engine.comparison_training import _encode_histories
            from civ5rl.core.model import batch_inputs
            p=deepcopy(b).cuda();q=deepcopy(b).cuda()
            for model in (p,q):
                model.zero_grad(set_to_none=True);model.enable_amp(True,spatial_batch_size=32)
                model.spatial_cell_padding=16
            sequences=[first[:16],second]
            inputs=batch_inputs([item for seq in sequences for item in seq],'cuda',spatial_only=True,compact_units=True)
            expected=p.encode_spatial(inputs);actual=_encode_histories(q,sequences)
            for x,y in zip(expected,actual):torch.testing.assert_close(y,x,atol=2e-6,rtol=2e-5)
            sum(x.square().mean() for x in expected).backward()
            sum(x.square().mean() for x in actual).backward()
            for (name,x),(_,y) in zip(p.named_parameters(),q.named_parameters()):
                if x.grad is not None:torch.testing.assert_close(y.grad,x.grad,atol=2e-6,rtol=3e-4,msg=name)

    def test_collection_recovers_dataset_publication(self):
        import tempfile
        from pathlib import Path
        from civ5rl.provenance.common import write_json,read_json
        from civ5rl.experiments.rollout_comparison import _collect
        class Budget:
            def elapsed(self):return 7205.
        with tempfile.TemporaryDirectory() as temporary:
            out=Path(temporary)
            folder=out/'sources/1'
            write_json(folder/'case.json',{'learner_side':0})
            write_json(folder/'history.json',[dict(actions=[{},{}])])
            # Legacy crash boundary: complete committed but dataset absent.
            write_json(out/'collection.json',dict(complete=True,sources=[1],forks=[],seconds=7200))
            cfg=dict(namespace='regression',collection_seconds=7200)
            _collect(cfg,out,budget=Budget())
            self.assertEqual(read_json(out/'dataset.json')['accepted_pairs'],0)
            self.assertTrue(read_json(out/'collection.json')['finalized'])
            self.assertEqual(read_json(out/'collection.json')['overshoot_seconds'],5.)
            before=(out/'dataset.json').read_bytes()
            _collect(cfg,out,budget=Budget())
            self.assertEqual((out/'dataset.json').read_bytes(),before)

    def test_strict_two_halves_and_reward(self):
        # First pair +1/+2; changing draws to losses cannot affect label.
        a=[0,None,1,1,0,None,1,1]
        b=[0,0,1,1,0,0,0,1]
        p=preference([a,b],0)
        self.assertTrue(p['accepted']);self.assertEqual(p['weight'],3/8)
        self.assertEqual(p['deltas'],[1,2])
        self.assertEqual(preference([[1 if x is None else x for x in a],b],0)['weight'],3/8)
        self.assertEqual(preference([a,a],0)['reason'],'both_zero')
        self.assertEqual(preference([a,b[:4]+a[4:]],0)['reason'],'one_zero')
        self.assertEqual(preference([a,b[:4]+[1]*4],0)['reason'],'opposite_signs')
    def test_pair_gradient_and_kl_direction(self):
        z=torch.tensor([1.,-.5,0.],requires_grad=True)
        loss=pair_loss(z,1,0,.5);loss.backward()
        self.assertLess(float(z.grad[1]),0);self.assertGreater(float(z.grad[0]),0);self.assertEqual(float(z.grad[2]),0)
        z2=z.detach().requires_grad_();half=pair_loss(z2,1,0,.25);half.backward()
        torch.testing.assert_close(z2.grad,z.grad/2)
        ref=(z.detach()/.5).log_softmax(-1)
        self.assertAlmostEqual(float(categorical_kl(ref,z.detach(),double=True)),0.,places=12)
        changed=torch.tensor([0.,1.,2.]);expected=(ref.double().exp()*(ref.double()-(changed/.5).log_softmax(-1).double())).sum()
        torch.testing.assert_close(categorical_kl(ref,changed,double=True),expected)
    def test_random_selection_balances_rounds_not_commands(self):
        rows=[dict(chosen=0,alternative=1,observation=dict(turn=t)) for t in range(6) for _ in range(t+1)]
        indices=select_forks(rows,123,'test')
        self.assertEqual(len(indices),4);self.assertEqual(len({rows[i]['observation']['turn'] for i in indices}),4)
        self.assertEqual(indices,select_forks(rows,123,'test'))
        rows[0]['chosen']=rows[0]['alternative']
        self.assertNotIn(0,select_forks(rows,123,'test',maximum=8))
        # Passing quota before sampling must leave late rounds reachable.
        rounds={rows[i]['observation']['turn'] for episode in range(128) for i in select_forks(rows,episode,'quota',maximum=2)}
        self.assertEqual(rounds,set(range(1,6)))

    def test_two_stage_label_matches_one_stage_preference(self):
        import random
        from civ5rl.experiments.comparison_queue import two_stage_label
        rng = random.Random(17)
        for _ in range(200):
            winners = {(r, a): rng.choice([0, 1, None]) for r in range(8) for a in (0, 1)}
            rows = [[winners[(r, a)] for r in range(8)] for a in (0, 1)]
            one = preference(rows, 0)
            two = two_stage_label(winners, 0)
            self.assertEqual(two['accepted'], one['accepted'])
            self.assertEqual(two['weight'], one['weight'])
            if two['accepted']:
                self.assertEqual(two['better'], one['better'])
            if one['deltas'][0] == 0:
                self.assertEqual(two['label'], 'first_group_zero')
                self.assertEqual(two['executed_repetitions'], 4)
                self.assertIsNone(two['deltas'][1])
            else:
                self.assertEqual(two['executed_repetitions'], 8)
                self.assertEqual(two['deltas'], one['deltas'])

    def test_selective_proposal_inference_isolates_second_policy(self):
        from types import SimpleNamespace
        from dataclasses import replace
        from civ5rl.core.model import Policy
        from civ5rl.core.observation import ObservationEncoder, ordered_actions
        from civ5rl.experiments.comparison_queue import SelectiveProposalInference
        from civ5rl.protocols.memory import flat_map
        from civ5sim import Battle
        from tests.test_combat import scenario, unit
        from tests.test_rl import context

        configure = __import__('civ5rl.provenance.common', fromlist=['configure']).configure
        configure(17, 'cpu')
        s = scenario(unit(x=0, y=0), unit(side=1, x=1, y=1), terrain=flat_map(2))
        with Battle(s) as battle:
            legal, previews = battle.actions_with_previews()
            obs = battle.observe(0)
            obs['attack_previews'] = previews
            encoded = ObservationEncoder(context(s), 'C2P').encode(obs, ordered_actions(legal))
        encoded_b = replace(encoded, previous=encoded.previous + 0.05)

        p1 = Policy('C2P').eval()
        p2 = Policy('C2P').eval()
        with torch.no_grad():
            p2.action[-1].weight.mul_(-1)
        infer = SelectiveProposalInference([p1, p2], 'cpu')

        def make_agent(proposal):
            return SimpleNamespace(
                mode='argmax', temperature=0.5, collect_selection_metrics=False, proposal=proposal,
                members=[SimpleNamespace(hidden=None), SimpleNamespace(hidden=None)],
            )

        src = make_agent(True)
        cont = make_agent(False)
        out = infer.infer_encoded([(src, 'act', encoded), (cont, 'act', encoded_b)])
        self.assertIn('events', out[0])
        self.assertNotIn('events', out[1])
        self.assertIsNotNone(src.members[0].hidden)
        self.assertIsNotNone(src.members[1].hidden)
        self.assertIsNotNone(cont.members[0].hidden)
        self.assertIsNone(cont.members[1].hidden)

        # Compare u171/u107 hidden states against isolated single-request runs.
        src_ref = make_agent(True)
        cont_ref = make_agent(False)
        out_src = infer.infer_encoded([(src_ref, 'act', encoded)])
        out_cont = infer.infer_encoded([(cont_ref, 'act', encoded_b)])
        self.assertEqual(out[0], out_src[0])
        self.assertEqual(out[1], out_cont[0])
        torch.testing.assert_close(src.members[0].hidden, src_ref.members[0].hidden)
        torch.testing.assert_close(src.members[1].hidden, src_ref.members[1].hidden)
        torch.testing.assert_close(cont.members[0].hidden, cont_ref.members[0].hidden)

if __name__=='__main__':unittest.main()
