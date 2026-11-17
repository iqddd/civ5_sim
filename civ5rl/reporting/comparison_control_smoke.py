"""Discarded real-history smoke for the full-corpus control path.

Only two public files from one source battle are extracted into disposable
storage, with their scientific SHA-256 checked before use. This is a small
correctness check, not the 330+660 timing measurement or an experiment.
"""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import tempfile

import torch

from civ5rl.deploy.archive import seven_zip
from civ5rl.engine.comparison_control import fit_full_corpus
from civ5rl.engine.phase_budget import PhaseBudget
from civ5rl.provenance.common import ROOT,configure,read_json,write_json,file_sha256
from civ5rl.provenance.checkpoint import verify_frozen_passport


def continuation_smoke(cfg,out):
    """Verify real inherited state and evaluator compatibility without fitting."""
    from civ5rl.engine.comparison_continuation import validate_continuation,continuation_seconds
    from civ5rl.engine.evaluation import EvaluationRuntime,evaluation_precision
    from civ5rl.experiments.ppo_target_scale import evaluation_key
    out=Path(out)
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty precheck output')
    state=validate_continuation(cfg);source=ROOT/cfg['fit_continuation']['run']
    configure(cfg['seed'],'cuda')
    with evaluation_precision():
        _,key=evaluation_key(read_json(source/'match-config.json'),read_json(source/'manifest.json'),
                             source,'checkpoints/parent.pt','development_512',
                             EvaluationRuntime('cuda',8,32,.001).resolve().identity())
    if key not in read_json(source/'evaluations.json'):
        raise ValueError('Parent Development evaluator/runtime/GPU is incompatible')
    result=dict(status='verified',inherited_steps=state['steps'],next_adam_step=state['steps']+1,
                inherited_seconds=continuation_seconds(cfg),remaining_seconds=cfg['training_seconds']-continuation_seconds(cfg),
                anchor_cursor=state['anchor_cursor'],head_adam_exact=True,parent_evaluation_compatible=True)
    write_json(out/'precheck.json',result);print(json.dumps(result,indent=2),flush=True)
    return result


def smoke(cfg,out):
    out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Use an empty smoke output')
    out.mkdir(parents=True,exist_ok=True)
    source=ROOT/cfg['collection_input']['run']
    for name,sha in cfg['collection_input']['sha256'].items():
        if file_sha256(source/name)!=sha:raise ValueError(f'Collection input changed: {name}')
    verify_frozen_passport(source)
    dataset=read_json(source/'dataset.json')
    example=dataset['pairs'][0];episode=example['episode']
    anchors=[key for key in dataset['anchors'] if key[0]==episode][:4]
    if len(anchors)<4:raise ValueError('Smoke requires four public target positions')
    configure(cfg['seed'],'cuda')
    with tempfile.TemporaryDirectory(prefix='comparison-control-',dir=ROOT/'.tmp') as tmp:
        scratch=Path(tmp)
        files=[f'{episode}/{name}' for name in ('case.json','history.json')]
        subprocess.run([seven_zip(),'x',str(source/'sources.storage/json.7z'),
                        '-o'+str(scratch/'sources'),'-y',*files],check=True,
                       stdout=subprocess.DEVNULL)
        for name,sha in dataset['source_hashes'][str(episode)].items():
            if file_sha256(scratch/'sources'/str(episode)/name)!=sha:
                raise ValueError('Smoke public source hash differs')
        tiny=dict(pairs=[example],anchors=anchors,
                  source_hashes={str(episode):dataset['source_hashes'][str(episode)]})
        tiny_cfg=deepcopy(cfg)
        tiny_cfg['training'].update(max_accepted_steps=2,anchors_per_step=2)
        with PhaseBudget(scratch,'training',1800) as budget:
            fit_full_corpus(tiny_cfg,scratch,tiny,{'discarded_smoke':True},budget)
        result=read_json(scratch/'training.json')
        before=file_sha256(scratch/'candidate.pt')
        with PhaseBudget(scratch,'training',1800) as budget:
            fit_full_corpus(tiny_cfg,scratch,tiny,{'discarded_smoke':True},budget)
        if before!=file_sha256(scratch/'candidate.pt'):
            raise AssertionError('Terminal smoke resume changed the candidate')
        candidate=torch.load(scratch/'candidate.pt',map_location='cpu',weights_only=False)
        parent=torch.load(ROOT/cfg['parents'][0],map_location='cpu',weights_only=False)
        if any(not torch.equal(v,parent['model'][k]) for k,v in candidate['model'].items() if k.startswith('value.')):
            raise AssertionError('Critic changed during comparison smoke')
        if cfg['training']['mode']=='frozen_heads_step_kl':
            if any(not torch.equal(v,parent['model'][k]) for k,v in candidate['model'].items()
                   if not k.startswith(('action.','attack_preview.'))):
                raise AssertionError('Frozen representation changed')
            from civ5rl.engine.comparison_heads import configure_heads
            model=configure_heads(cfg['parents'][0],device='cpu')
            parameters=[p for p in model.parameters() if p.requires_grad]
            if sum(p.numel() for p in parameters)!=68609:raise AssertionError('Wrong trainable head set')
            if len(candidate['optimizer']['param_groups'][0]['params'])!=len(parameters):
                raise AssertionError('Frozen parameters entered head Adam')
            result.update(frozen_weights_exact=True,trainable_parameters=68609)
        accepted=result['training_steps']
        if any(int(v['step'])!=accepted for v in candidate['optimizer']['state'].values()):
            raise AssertionError('Smoke Adam counters differ from accepted steps')
        if result['last_full_kl']>cfg['training']['kl_limit']:
            raise AssertionError('Smoke published an unverified model')
        result.update(discarded_smoke=True,source_pairs=1,training_anchors_per_step=2,
                      control_anchors=4,terminal_resume_exact=True,critic_unchanged=True,
                      parents_unchanged=all(file_sha256(ROOT/p)==sha for p,sha in
                                            zip(cfg['parents'],cfg['parent_sha256'])))
        import gc
        gc.collect()
    write_json(out/'precheck.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def benchmark(args,parser):
    """Measure disposable full work separately from the small correctness smoke."""
    import shutil
    import statistics
    from time import perf_counter
    from civ5rl.experiments.rollout_comparison import load_config
    from civ5rl.engine.comparison_training import (
        Histories,configure_policy,prepare_references,logical_step,full_kl,atomic_torch_save,
    )
    from civ5rl.engine.comparison_control import initial_state
    from civ5rl.engine.optimizer_transaction import AdamSnapshot
    from civ5rl.reporting.comparison_storage import restore_tree
    from civ5rl.provenance.common import seed_for
    from civ5rl.provenance.rng import capture_rng
    from civ5rl.engine.resources import ResourceMonitor
    if args.config is None:parser.error('Full-corpus comparison requires --config')
    cfg=load_config(args.config);out=args.out.resolve()
    if out.exists() and any(out.iterdir()):parser.error('Use an empty benchmark output')
    if args.steps<1 or args.warmup<1:parser.error('Require warmup and measured steps')
    out.mkdir(parents=True,exist_ok=True)
    source=ROOT/cfg['collection_input']['run']
    for name,sha in cfg['collection_input']['sha256'].items():
        if file_sha256(source/name)!=sha:raise ValueError(f'Collection input changed: {name}')
    verify_frozen_passport(source)
    data=read_json(source/'dataset.json');options=cfg['training']
    started=perf_counter();configure(cfg['seed'],'cuda')
    with tempfile.TemporaryDirectory(prefix='comparison-full-bench-',dir=ROOT/'.tmp') as tmp:
        scratch=Path(tmp)
        shutil.copytree(source/'sources.storage',scratch/'sources.storage')
        restore_tree(scratch,'sources')
        head_only=options['mode']=='frozen_heads_step_kl'
        if head_only:
            from civ5rl.engine.comparison_heads import configure_heads,HeadFeatures
            policy=configure_heads(cfg['parents'][0])
        else:policy=configure_policy(cfg['parents'][0])
        optimizer=torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=options['initial_lr'],
                                   betas=(.9,.999),eps=1e-5,weight_decay=0)
        keys=[(p['episode'],p['index']) for p in data['pairs']]+[tuple(k) for k in data['anchors']]
        limits={}
        for episode,index in keys:limits[episode]=max(limits.get(episode,0),index+1)
        histories=Histories(scratch,source_hashes=data['source_hashes'],limits=limits)
        histories.prepare(keys,device='cuda')
        if head_only:
            features=HeadFeatures(policy,histories,data)
            features.prepare(cfg['parents'][0])
        else:refs=prepare_references(policy,histories,data['anchors'],scratch/'references.pt')
        state=initial_state(data['anchors'],options,seed_for(cfg['seed'],0,'anchors',namespace=cfg['namespace']))
        anchors=[data['anchors'][i] for i in state['anchor_order'][:options['anchors_per_step']]]
        cold=perf_counter()-started;snapshot=AdamSnapshot(policy,optimizer);rows=[]
        with ResourceMonitor(out/'telemetry') as monitor:
            for index in range(args.warmup+args.steps):
                snapshot.restore();monitor.mark('warmup' if index<args.warmup else 'full_corpus_attempt')
                begin=perf_counter()
                if head_only:features.update(optimizer,data['pairs'],anchors)
                else:logical_step(policy,optimizer,histories,data['pairs'],anchors,refs['mixed'])
                step=perf_counter()-begin;begin=perf_counter()
                if head_only:kl,learning=features.measure()
                else:kl=full_kl(policy,histories,data['anchors'],refs['fp32'])
                control=perf_counter()-begin
                begin=perf_counter()
                atomic_torch_save(dict(model=policy.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng()),scratch/'discarded.pt')
                save=perf_counter()-begin;begin=perf_counter();snapshot.restore();rollback=perf_counter()-begin
                rows.append(dict(warmup=index<args.warmup,step_seconds=step,kl_seconds=control,
                                 save_seconds=save,rollback_seconds=rollback,full_kl=kl))
        if head_only:
            snapshot.restore()
            from torch.profiler import profile,ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as trace:
                features.update(optimizer,data['pairs'],anchors)
            trace.export_chrome_trace(str(out/'heads-trace.json'))
            kernels=sorted((dict(name=e.key,cuda_us=e.device_time_total,count=e.count)
                            for e in trace.key_averages() if e.device_time_total),
                           key=lambda e:e['cuda_us'],reverse=True)[:20]
            write_json(out/'top-kernels.json',kernels)
            snapshot.restore()
        measured=[r for r in rows if not r['warmup']]
        cycle=statistics.mean(r['step_seconds']+r['kl_seconds']+2*r['save_seconds'] for r in measured)
        warmup=sum(r['step_seconds']+r['kl_seconds']+r['save_seconds']+r['rollback_seconds']
                   for r in rows if r['warmup'])
        estimate=cold+warmup+options['max_accepted_steps']*cycle
        result=dict(discarded=True,pairs=len(data['pairs']),training_anchors=len(anchors),
                    control_anchors=len(data['anchors']),cold_preparation_seconds=cold,
                    warmup_seconds=warmup,measurements=rows,estimated_ten_steps_seconds=estimate,
                    estimated_thirty_attempts_seconds=cold+warmup+30*(cycle+statistics.mean(r['rollback_seconds'] for r in measured)),
                    training_seconds=cfg['training_seconds'],fits_nominal=estimate<=cfg['training_seconds'],
                    limitation='Measured batch and hardware only; retries and final packing can add time.')
        if head_only:
            result.update(feature_cache=features.statistics,feature_parity=features.parity,
                learning_after_discarded_step=learning,trainable_parameters=sum(p.numel() for p in policy.parameters() if p.requires_grad),
                speedup_vs_full_training_step=18.2342698/statistics.mean(r['step_seconds'] for r in measured),
                speedup_vs_full_kl=22.5921917/statistics.mean(r['kl_seconds'] for r in measured))
        del histories
        import gc
        gc.collect()
    write_json(out/'summary.json',result);print(json.dumps(result,indent=2),flush=True)
    return result
