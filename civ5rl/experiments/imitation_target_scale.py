"""One bounded local CP--CP behavioral-cloning experiment on target armies."""
import gc
import io
import json
import platform
import random
import shutil
import traceback
import zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import psutil
import torch
from torch.nn import functional as F

from civ5rl.config import load_config as resolve_config
from civ5rl.engine.control import RunLock
from civ5rl.engine.imitation import decision_indices, label_minibatches
from civ5rl.engine.imitation_performance import backward_minibatch_async
from civ5rl.engine.imitation_stream import PreparedDemonstrations, prepare_inputs
from civ5rl.engine.ppo_fast import prefixes
from civ5rl.engine.resources import ResourceMonitor, analyze_telemetry
from civ5rl.engine.training import FrozenParamsCheck, bc_optimizer_step
from civ5rl.experiments.evaluation import paired_world_bootstrap, verify_replay
from civ5rl.experiments.imitation_helpers import (CompilationClock, adam, assert_equal,
                                                isolated_rng, load_policy, pool, restore_rng)
from civ5rl.experiments.ppo_after_imitation import policy_hash
from civ5rl.experiments.ppo_target_scale import evaluate as target_evaluate
from civ5rl.protocols.imitation_target_scale import KIND, match_task
from civ5rl.protocols.ppo_target_scale import TargetFactory
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import ROOT, configure, file_sha256, read_json, seed_for, write_json
from civ5sim.engine import digest


def load_config(path):
    overlay=read_json(path); base=resolve_config(ROOT/overlay['base_config'])
    config={k:deepcopy(base[k]) for k in ('model','observation','numerics','optimizer','teacher_search','performance')}
    config.update({k:v for k,v in overlay.items() if k!='performance'})
    config['performance'].update(overlay['performance'])
    config['environment']={'max_calls_per_turn':2000}
    config['max_calls_per_turn']=2000
    if (config['kind']!=KIND or config['seed']!=17 or config['workers']!=4
            or config['labels_per_minibatch']!=1024 or config['sequence_length']!=128):
        raise ValueError('Unsupported target BC profile')
    if config['optimizer']!={'lr':1e-4,'betas':[.9,.999],'eps':1e-5,'weight_decay':0}:
        raise ValueError('Unsupported target BC optimizer')
    return config


def prepare(config,out):
    if (out/'run.json').exists():
        if read_json(out/'resolved-config.json')!=config:
            raise ValueError('Configuration changed')
        return read_json(out/'manifest.json')
    if file_sha256(ROOT/config['parent'])!=config['parent_sha256']:
        raise ValueError('Wrong continuation update 759 checkpoint')
    original=read_json(ROOT/config['target_manifest'])
    base=original['base']; worlds=base['worlds']
    if len(worlds['training'])!=57 or set(worlds['training']) & set(worlds['validation']):
        raise ValueError('Training/validation world leakage')
    plan={'schema':1,'kind':KIND,'base':base,
          'development':original['development'],'confirmation':original['confirmation']}
    for purpose,count in (('development',128),('confirmation',256)):
        if len(plan[purpose])!=count or any(c['split']!='validation' or c['parent_world'] not in worlds['validation']
                                         for c in plan[purpose]):
            raise ValueError('Closed or incomplete evaluation cases')
    plan['hash']=digest(plan)
    write_json(out/'manifest.json',plan);write_json(out/'resolved-config.json',config)
    # One source snapshot, not an environment certification or ancestor audit.
    paths=set()
    for pattern in ('civ5rl/**/*.py','civ5sim/**/*.py','native/**/*.cpp','native/**/*.h','native/**/*.inc'):
        paths.update(ROOT.glob(pattern))
    paths.update([ROOT/'tools/run_cp_imitation.py',ROOT/'configs/rl-cp-imitation-target-scale.json',
                  ROOT/config['base_config']])
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            if path.is_file():archive.write(path,path.relative_to(ROOT).as_posix())
    configure(17,'cpu')
    policy=load_policy(config,ROOT/config['parent'],'cpu')
    parent=torch.load(ROOT/config['parent'],map_location='cpu',weights_only=False)
    assert_equal(policy.state_dict(),parent['model'])
    optimizer=adam(policy,config,'cpu')
    assert not optimizer.state
    run_record={'kind':KIND,'parent':config['parent'],'parent_sha256':config['parent_sha256'],
                'parent_model_equal':True,'new_Adam_empty':True,'seed':17,
                'source_snapshot':file_sha256(out/'source.zip'),'manifest_hash':plan['hash'],
                'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
                'GPU':torch.cuda.get_device_name(0),'teacher_search':config['teacher_search'],
                'native':{str(p.relative_to(ROOT)):file_sha256(p) for p in
                          (ROOT/'build/combat.dll',ROOT/'build/combat_ai.dll',ROOT/'build/combat_rules.h')},
                'run_id':digest({'config':config,'manifest':plan['hash']})[:16]}
    write_json(out/'run.json',run_record)
    checkpoint.save(out/'checkpoints/initial.pt',policy,optimizer,
                    {'epoch':0,'Adam_steps':0,'active_seconds':0},random.Random(17),config,
                    {'run_id':run_record['run_id']})
    write_json(out/'status.json',{'status':'prepared'})
    return plan


def collect(config,plan,out):
    if (out/'collection.json').exists():
        result=read_json(out/'collection.json')
        if result['status']=='complete':return result
        raise ValueError('Incomplete collection exists; do not silently repeat it')
    factory=TargetFactory(config,plan['base'])
    started=perf_counter(); rows=[]; labels=0; index=0
    with ResourceMonitor(out/'collection-telemetry') as monitor, pool(config['workers']) as executor:
        monitor.mark('CP_generation')
        while labels<config['training_labels'] and perf_counter()-started<config['collection_seconds']:
            jobs=[]
            for _ in range(config['workers']):
                case=factory.make(index);case['source']=KIND;case['case_id']=index
                case['case_hash']=digest(case)
                jobs.append((config,case,str(out/f'data/match-{index:06}')));index+=1
            completed=list(executor.map(match_task,jobs))
            for row in completed:
                for entry in row['entries']:
                    entry['path']=Path(entry['path']).relative_to(out).as_posix()
            rows.extend(completed);labels+=sum(r['labels'] for r in completed)
            elapsed=perf_counter()-started
            write_json(out/'collection.json',{'status':'collecting','labels':labels,'seconds':elapsed,'matches':rows})
            print(f'CP--CP collection: {len(rows)} matches, {labels} free labels, {elapsed:.1f}s; '
                  f'{labels/elapsed:.1f} labels/s',flush=True)
    elapsed=perf_counter()-started
    entries=[e for r in rows for e in r['entries']]
    if not labels:raise RuntimeError('No free CP labels were collected')
    # A fixed training diagnostic subset; never used for candidate selection.
    ids=list(range(len(rows)));random.Random(17).shuffle(ids);selected=[];count=0
    for i in ids:
        selected.extend(rows[i]['entries']);count+=rows[i]['labels']
        if count>=config['fixed_training_labels']:break
    result={'status':'complete','stop_reason':'label_target' if labels>=config['training_labels'] else 'collection_time_limit',
            'labels':labels,'label_overshoot':max(0,labels-config['training_labels']),
            'seconds':elapsed,'time_overshoot':max(0,elapsed-config['collection_seconds']),
            'positions':sum(e['positions'] for e in entries),'entries':entries,'matches':rows,
            'fixed_training':selected,'fixed_labels':count,'physical_repeats':factory.duplicates,
            'coverage':{'conditions':dict(factory.conditions),'worlds':dict(factory.worlds),'crops':dict(factory.crops)}}
    write_json(out/'collection.json',result)
    return result


def fixed_metrics(policy,cache,micro):
    before=cache.mode;cache.mode='diagnostic'
    counts=Counter();correct=Counter();losses=Counter();unit_counts=Counter();unit_correct=Counter()
    try:
        with torch.no_grad():
            for spans,_ in label_minibatches(cache.chunks(),cache.masks):
                if policy.spatial_cuda_graphs:torch.compiler.cudagraph_mark_step_begin()
                hidden=prefixes(policy,cache,spans)
                for start in range(0,len(spans),micro):
                    part=spans[start:start+micro];batch,labels,lengths=cache.batch(part)
                    logits,_,_=policy.forward_sequences(batch,lengths,hidden[:,start:start+len(part)].contiguous())
                    ce=F.cross_entropy(logits,labels['indices'],reduction='none').cpu().tolist()
                    predicted=logits.argmax(-1).cpu().tolist()
                    rows=[r for e,a,b in part for r in cache.episodes[e]['decisions'][a:b]]
                    for i,row in enumerate(rows):
                        if not row['supervised']:continue
                        target,_=decision_indices(row);action=row['actions'][target];kind=action['kind']
                        counts[kind]+=1;correct[kind]+=predicted[i]==target;losses[kind]+=ce[i]
                        units={u['id']:u['type'] for u in row['observation']['units']}
                        unit=units.get(action.get('unit'),'no_unit')
                        unit_counts[unit]+=1;unit_correct[unit]+=predicted[i]==target
        n=sum(counts.values())
        return {'labels':n,'ce':sum(losses.values())/n,'accuracy':sum(correct.values())/n,
                'by_kind':{k:{'labels':counts[k],'ce':losses[k]/counts[k],
                              'accuracy':correct[k]/counts[k]} for k in counts},
                'by_unit_type':{k:{'labels':unit_counts[k],'accuracy':unit_correct[k]/unit_counts[k]} for k in unit_counts}}
    finally:cache.mode=before


def baseline(config,plan,out):
    if (out/'selection.json').exists():return
    # A cheap check of saved data and exact evaluator sources. No reconstruction.
    reused=False
    archive=ROOT/config['baseline_archive']
    if archive.exists():
        with zipfile.ZipFile(archive) as previous:
            prefix='artifacts/rl/ppo-target-scale/'
            old_plan=json.loads(previous.read(prefix+'manifest.json'))
            old_evals=json.loads(previous.read(prefix+'evaluations.json'))
            sources=zipfile.ZipFile(io.BytesIO(previous.read(prefix+'source.zip')))
            relevant=('civ5rl/core/agent.py','civ5rl/core/model.py','civ5rl/core/observation.py',
                      'civ5rl/experiments/ppo_after_imitation.py','civ5sim/match.py','civ5sim/cp_agent.py')
            same_sources=all(sources.read(name)==(ROOT/name).read_bytes() for name in relevant)
            saved=torch.load(out/'checkpoints/initial.pt',map_location='cpu',weights_only=False)
            fingerprint=policy_hash(saved['model'])
            key='development:'+fingerprint
            if old_plan['development']==plan['development'] and same_sources and key in old_evals:
                point=old_evals[key]; directory=out/point['runs']['argmax']['cp']['directory']
                old_directory=prefix+point['runs']['argmax']['cp']['directory']+'/'
                for name in previous.namelist():
                    if name.startswith(old_directory) and not name.endswith('/'):
                        target=directory/name[len(old_directory):]
                        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(previous.read(name))
                point={**point,'checkpoint':'checkpoints/initial.pt','epoch':0,'Adam_steps':0,'reused_Windows_baseline':True}
                write_json(out/'evaluations.json',{key:point});reused=True
    if not reused:
        point=target_evaluate(config,plan,out,'checkpoints/initial.pt','development')
        point.update(epoch=0,Adam_steps=0,reused_Windows_baseline=False)
        evaluations=read_json(out/'evaluations.json');evaluations['development:'+point['policy_hash']]=point
        write_json(out/'evaluations.json',evaluations)
    wins=point['runs']['argmax']['cp']['summary']['W']
    write_json(out/'selection.json',{'baseline':point,'best':point,'best_wins':wins})
    print('Development baseline: '+str(point['runs']['argmax']['cp']['summary']),flush=True)


def measure(config,plan,out,path,epoch,steps,policy,fixed,micro,shuffle):
    fingerprint=policy_hash(policy.state_dict())
    cache=read_json(out/'evaluations.json') if (out/'evaluations.json').exists() else {}
    previous=cache.get('development:'+fingerprint)
    if previous and 'fixed_training' in previous:return previous
    with isolated_rng(shuffle):
        diagnostic=fixed_metrics(policy,fixed,micro)
        point=target_evaluate(config,plan,out,path,'development')
    point={**point,'epoch':epoch,'Adam_steps':steps,'fixed_training':diagnostic}
    evaluations=read_json(out/'evaluations.json');evaluations['development:'+point['policy_hash']]=point
    write_json(out/'evaluations.json',evaluations)
    selection=read_json(out/'selection.json');wins=point['runs']['argmax']['cp']['summary']['W']
    if wins>selection['best_wins']:selection.update(best=point,best_wins=wins)
    write_json(out/'selection.json',selection)
    return point


class ActiveClock:
    def __init__(self,offset=0.):self.started=perf_counter();self.excluded=0.;self.offset=offset
    def seconds(self):return self.offset+perf_counter()-self.started-self.excluded
    @contextmanager
    def pause(self):
        start=perf_counter()
        try:yield
        finally:self.excluded+=perf_counter()-start


def train(config,plan,out):
    restart_path=out/'restart-input-v2.json'
    restart=read_json(restart_path) if restart_path.exists() else None
    if (out/'training.json').exists():
        if read_json(out/'training.json')['status']=='complete':return
        if not restart or restart.get('consumed'):
            raise ValueError('Partial training exists; optimization must not repeat automatically')
    baseline(config,plan,out)
    data=read_json(out/'collection.json')
    if data['status']!='complete':raise ValueError('Collection is incomplete')
    input_preparation=prepare_inputs(out,data['entries'],workers=config['workers'])
    prepared_entries=input_preparation['entries']
    lookup={e['path']:e for e in prepared_entries}
    fixed_entries=[lookup[e['path']] for e in data['fixed_training']]
    configure(17,'cuda');shuffle=random.Random(seed_for(17,0,'shuffle',namespace=config['experiment']['shuffle_namespace']))
    source=restart['checkpoint_boundary']['latest'] if restart else 'checkpoints/initial.pt'
    prepared=perf_counter();policy=load_policy(config,out/source,'cuda')
    saved=torch.load(out/source,map_location='cpu',weights_only=False)
    fixed_limit=min(config['cache_bytes']//16,256*1024**2)
    cache=PreparedDemonstrations(out,prepared_entries,'cuda',config['cache_bytes']-fixed_limit)
    fixed=PreparedDemonstrations(out,fixed_entries,'cuda',fixed_limit)
    warm=[]
    # The largest map/army first; use real fragments, including nonzero prefixes.
    largest=sorted(range(len(data['entries'])),key=lambda i:(data['matches'][i//2]['map_size'],
                                                           data['matches'][i//2]['army_size']),reverse=True)
    chunks=cache.chunks(); order=sorted(chunks,key=lambda x:largest.index(x[0]))
    probes=list(label_minibatches(order,cache.masks))[:2]
    with CompilationClock() as compilation, ResourceMonitor(out/'training-telemetry') as monitor:
        compilation.monitor=monitor
        selected=None
        for micro in (8,4,2,1):
            monitor.mark('initial_warmup',microbatch=micro)
            policy.load_state_dict(saved['model']);optimizer=adam(policy,config,'cuda')
            optimizer.load_state_dict(saved['optimizer'])
            def backward(p,c,s,n):return backward_minibatch_async(p,c,s,n,micro)
            try:
                start=perf_counter();swap=psutil.swap_memory()
                for spans,n in probes:
                    bc_optimizer_step(policy,optimizer,backward,cache,spans,n,
                                      max_gradient_norm=.5,asynchronous=True)
                    torch.cuda.synchronize()
                end_swap=psutil.swap_memory()
                paging=max(0,end_swap.sin-swap.sin)+max(0,end_swap.sout-swap.sout)
                if paging>512*1024**2 and psutil.virtual_memory().available<1024**3:
                    raise MemoryError('Pronounced paging during full BC steps')
                selected=micro
                warm.append({'microbatch':micro,'passed':True,'seconds':perf_counter()-start,
                             'free_VRAM':torch.cuda.mem_get_info()[0],'available_RAM':psutil.virtual_memory().available,
                             'paging_bytes':paging,'cache':cache.statistics()})
                break
            except (torch.cuda.OutOfMemoryError,MemoryError) as error:
                warm.append({'microbatch':micro,'passed':False,'error':str(error)})
                optimizer.zero_grad(set_to_none=True);del optimizer;gc.collect();torch.cuda.empty_cache()
        if selected is None:raise MemoryError('Target BC does not fit at microbatch 1')
        # Probe updates never enter the experiment.
        policy.load_state_dict(saved['model']);optimizer=adam(policy,config,'cuda')
        optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved,shuffle)
        optimizer.zero_grad(set_to_none=True)
        assert_equal(policy.state_dict(),saved['model']);assert_equal(optimizer.state_dict(),saved['optimizer'])
        frozen=FrozenParamsCheck(policy.value)
        if restart and (out/'memory-preparation.json').exists():
            shutil.copy2(out/'memory-preparation.json',out/'memory-preparation-before-input-fix.json')
        write_json(out/'memory-preparation.json',{'attempts':warm,'selected_microbatch':selected,
            'backend':'prepared_numpy_CPU','source_checkpoint':source,
            'probe_updates_discarded':True,'preparation_seconds':perf_counter()-prepared,
            'one_time_input_preparation_seconds':input_preparation['seconds']})
        monitor.mark('offline_metrics')
        if not restart:
            with isolated_rng(shuffle):initial_metrics=fixed_metrics(policy,fixed,selected)
            write_json(out/'fixed-training-initial.json',initial_metrics)
            evaluations=read_json(out/'evaluations.json');selection=read_json(out/'selection.json')
            initial_key='development:'+selection['baseline']['policy_hash']
            evaluations[initial_key]['fixed_training']=initial_metrics
            selection['baseline']=evaluations[initial_key]
            if selection['best']['policy_hash']==selection['baseline']['policy_hash']:
                selection['best']=selection['baseline']
            write_json(out/'evaluations.json',evaluations);write_json(out/'selection.json',selection)
        cache.counters.clear();fixed.clear()
        origin_steps=saved['state']['Adam_steps']
        history=[];start_epoch=1
        if restart:
            boundary=restart['checkpoint_boundary']
            if not saved['state']['epoch_completed'] or saved['state']['Adam_steps']!=boundary['Adam_steps']:
                raise ValueError('Input-fix restart requires its complete saved epoch boundary')
            previous=[json.loads(line) for line in (out/'steps.jsonl').read_text().splitlines() if line.strip()]
            if len(previous)!=origin_steps:raise ValueError('Saved step log differs from checkpoint boundary')
            steps_per_epoch=(data['labels']+1023)//1024
            for e in range(1,boundary['epoch']+1):
                records=previous[(e-1)*steps_per_epoch:e*steps_per_epoch]
                if sum(r['labels'] for r in records)!=data['labels']:raise ValueError('Incomplete saved epoch')
                history.append({'epoch':e,'completed':True,'steps':len(records),'labels':data['labels'],
                    'online_CE':sum(r['ce']*r['labels'] for r in records)/data['labels']})
            start_epoch=boundary['epoch']+1
            steps=boundary['Adam_steps'];labels=boundary['free_labels_processed']
            for field in ('history_positions','prefix_positions'):cache.counters[field]=boundary[field]
            # Finish the planned evaluation interrupted by the requested stop.
            monitor.mark('evaluation')
            measure(config,plan,out,source,boundary['epoch'],steps,policy,fixed,selected,shuffle)
            fixed.clear()
            write_json(out/'restart-state-verification.json',{'model_equal':True,'Adam_equal':True,
                'RNG_restored':True,'source':source,'Adam_counter':steps,'next_Adam_counter':steps+1,
                'active_seconds_preserved':boundary['active_seconds'],'history_and_order_preserved':True})
            restart['consumed']=True;write_json(restart_path,restart)
        else:steps=labels=0
        clock=ActiveClock(restart['checkpoint_boundary']['active_seconds'] if restart else 0.)
        reason='epoch_limit';last_path=source
        utilization=None

        def save(path,epoch,cursor,order,complete):
            monitor.mark('checkpoint',epoch=epoch,Adam_steps=steps)
            state={'epoch':epoch,'epoch_completed':complete,'batch_cursor':cursor,'epoch_order':order,
                   'Adam_steps':steps,'free_labels_processed':labels,'active_seconds_at_save_start':clock.seconds(),
                   'history_positions':cache.counters['history_positions'],'prefix_positions':cache.counters['prefix_positions']}
            frozen.verify()
            checkpoint.save(out/path,policy,optimizer,state,shuffle,config,{'run_id':read_json(out/'run.json')['run_id']})
            write_json(out/'state.json',{**state,'active_seconds':clock.seconds(),'latest':path})

        for epoch in range(start_epoch,config['epochs']+1):
            if clock.seconds()>=config['training_seconds']:reason='active_time_limit';break
            order=cache.chunks();shuffle.shuffle(order);records=[];completed=True;cursor=0;epoch_labels=0
            monitor.mark('iterative_training',epoch=epoch,Adam_steps=steps)
            batches=iter(label_minibatches(order,cache.masks))
            while True:
                if clock.seconds()>=config['training_seconds']:
                    completed=False;reason='active_time_limit';break
                try:spans,n=next(batches)
                except StopIteration:break
                start=perf_counter();before=dict(cache.counters)
                ce,norm=bc_optimizer_step(policy,optimizer,backward,cache,spans,n,max_gradient_norm=.5,asynchronous=True)
                torch.cuda.synchronize();ce=float(ce);norm=float(norm)
                steps+=1;labels+=n;cursor+=1;epoch_labels+=n
                counters={int(s['step']) for s in optimizer.state.values()}
                if counters!={steps}:raise ValueError('BC Adam counters differ from completed steps')
                record={'step':steps,'labels':n,'ce':ce,'gradient_norm':norm,'seconds':perf_counter()-start,
                        'prefix_positions':cache.counters['prefix_positions']-before.get('prefix_positions',0),
                        'history_positions':cache.counters['history_positions']-before.get('history_positions',0),
                        'compile_seconds':compilation.seconds(start,perf_counter())}
                records.append(record)
                with (out/'steps.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
                if steps==1:write_json(out/'first-step-check.json',{'before':0,'after':1,'all_counters_equal':True})
                if restart and steps==origin_steps+1:
                    write_json(out/'first-restarted-step-check.json',{'before':origin_steps,'after':steps,'all_counters_equal':True})
                if steps%10==0:
                    print(f'BC epoch={epoch}, Adam={steps}, labels={labels}, CE={ce:.4f}, active={clock.seconds():.1f}s',flush=True)
                if clock.seconds()>=600 and utilization is None:
                    analysis=analyze_telemetry(out/'training-telemetry/resources.jsonl')
                    utilization={'metrics':analysis['metrics'],'last_60s':analysis['windows'][-1] if analysis['windows'] else None,
                        'measured_cache':cache.statistics(),'mean_step_seconds':sum(r['seconds'] for r in records)/len(records),
                        'prefix_seconds_not_separately_profiled':True,
                        'continuation_reason':'Bounded CPU episode input, full current-weight GRU-prefix and small fixed spatial blocks constrain dispatch. Preserve the validated 1024-label objective and precision; no extra optimization campaign or artificial GPU load.'}
                    write_json(out/'GPU-utilization-review.json',utilization)
                if epoch_labels==data['labels']:break
            history.append({'epoch':epoch,'completed':completed,'steps':len(records),
                'labels':sum(r['labels'] for r in records),
                'online_CE':sum(r['ce']*r['labels'] for r in records)/sum(r['labels'] for r in records) if records else None})
            if records:
                last_path=f'checkpoints/epoch-{epoch:02}.pt' if completed else f'checkpoints/partial-epoch-{epoch:02}-step-{steps:05}.pt'
                save(last_path,epoch,cursor,order,completed)
            terminal=not completed or clock.seconds()>=config['training_seconds']
            if terminal:reason='active_time_limit'
            if records and (epoch in config['evaluate_epochs'] or terminal):
                monitor.mark('evaluation')
                with clock.pause():measure(config,plan,out,last_path,epoch,steps,policy,fixed,selected,shuffle)
                fixed.clear();monitor.mark('iterative_training',epoch=epoch)
            write_json(out/'training.json',{'status':'running','epochs':history,'Adam_steps':steps,
                'free_labels_processed':labels,'active_seconds':clock.seconds(),'latest':last_path,
                'cache':cache.statistics(),'microbatch':selected})
            if terminal:break
        # A budget boundary between epochs must also evaluate the last changed policy.
        if steps:
            monitor.mark('evaluation')
            with clock.pause():measure(config,plan,out,last_path,history[-1]['epoch'],steps,policy,fixed,selected,shuffle)
        frozen.verify()
        elapsed=clock.seconds()
        write_json(out/'training.json',{'status':'complete','stop_reason':reason,'epochs':history,
            'Adam_steps':steps,'free_labels_processed':labels,'active_seconds':elapsed,
            'time_overshoot':max(0,elapsed-config['training_seconds']),'latest':last_path,
            'microbatch':selected,'cache':cache.statistics(),'value_head_unchanged':True,
            'GPU_utilization_review':utilization})
        cache.clear();fixed.clear()
    del policy,optimizer;gc.collect();torch.cuda.empty_cache()


def evaluate(config,plan,out):
    if (out/'confirmation.json').exists():return
    training=read_json(out/'training.json')
    if training['status']!='complete':raise ValueError('Training is incomplete')
    selection=read_json(out/'selection.json');best=selection['best'];base=selection['baseline']
    if selection['best_wins']<=base['runs']['argmax']['cp']['summary']['W']:
        write_json(out/'confirmation.json',{'status':'not_run_no_development_improvement','selected':best})
        return
    before=target_evaluate(config,plan,out,'checkpoints/initial.pt','confirmation')
    after=target_evaluate(config,plan,out,best['checkpoint'],'confirmation')
    def outcomes(point):
        directory=out/point['runs']['argmax']['cp']['directory']; result={};cases={}
        for path in sorted(directory.glob('*/metrics.json')):
            row=read_json(path);case=read_json(path.parent/'case.json')
            key=(case['parent_world'],case['case_id'],case['learner_side'])
            result[key]=int(row['winner']==row['side']);cases[key]=case
        return result,cases
    first,cases=outcomes(before);second,paired=outcomes(after)
    if cases!=paired:raise ValueError('Unpaired target BC confirmation')
    interval=paired_world_bootstrap(first,second,samples=10000,seed=20261103,cases=cases,cases_per_world=16)
    write_json(out/'confirmation.json',{'status':'complete','baseline':before,'candidate':after,'comparison':interval})


def report(config,out):
    collection=read_json(out/'collection.json') if (out/'collection.json').exists() else {}
    training=read_json(out/'training.json') if (out/'training.json').exists() else {}
    selection=read_json(out/'selection.json') if (out/'selection.json').exists() else {}
    lines=['# Имитация штатного CP на целевых армиях','',
           f"Статус: {read_json(out/'status.json')}; локальная RTX 4060 Ti; seed 17.",
           '',f"Сбор: {collection.get('labels')} свободных меток, {collection.get('positions')} позиций, "
           f"{len(collection.get('matches',[]))} полных CP–CP боёв, {collection.get('seconds')} с; "
           f"причина {collection.get('stop_reason')}.",
           '', 'Обе стороны обычного stateful CP сохранены отдельно; все исходы включены. '
           'Training-worlds: 57. Forced states и prefix не входят в знаменатель CE.',
           '',f"Обучение: {training.get('stop_reason')}; Adam steps {training.get('Adam_steps')}; "
           f"обработано свободных меток {training.get('free_labels_processed')}; active {training.get('active_seconds')} с; "
           f"превышение {training.get('time_overshoot')} с; microbatch {training.get('microbatch')}.",
           '', '| Эпоха | Checkpoint | W/D/L | Win rate | Fixed training CE / accuracy |',
           '|---:|---|---|---:|---|']
    if (out/'evaluations.json').exists():
        for point in read_json(out/'evaluations.json').values():
            if point['purpose']!='development':continue
            summary=point['runs']['argmax']['cp']['summary'];metric=point.get('fixed_training',{})
            lines.append(f"| {point.get('epoch',0)} | {point['checkpoint']} | {summary['W']}/{summary['D']}/{summary['L']} | "
                         f"{summary['win_rate']:.2%} | {metric.get('ce')} / {metric.get('accuracy')} |")
    lines+=['', 'Выбор только по development-победам, tie — более ранний, включая родителя.',
            '', 'Selection: '+json.dumps(selection,ensure_ascii=False),
            '', 'Value head неизменна: '+str(training.get('value_head_unchanged'))+'. Общий encoder менялся; калибровка critic не сохраняется.',
            '', 'Диагностическая CE/accuracy измеряется на фиксированном training-поднаборе, не на holdout.',
            '', 'PPO, другой seed, закрытый test и автоматическое продление не запускались. Vast не использовался.']
    if (out/'confirmation.json').exists():lines+=['','Additional: '+json.dumps(read_json(out/'confirmation.json'),ensure_ascii=False)]
    if collection.get('matches'):
        commands=Counter();units=Counter();outcomes=Counter();strata=Counter()
        for row in collection['matches']:
            commands.update(row['commands']);units.update(row['unit_types'])
            outcomes['draw' if row['winner'] is None else f"side_{row['winner']}_win"]+=1
            strata[f"{row['map_size']}/{row['army_size']}/{row['placement']}"]+=1
        lines+=['','## Корпус','',
            f"Forced positions: {collection.get('positions',0)-collection.get('labels',0)}; "
            f"превышение меток {collection.get('label_overshoot')}, времени {collection.get('time_overshoot')} с; "
            f"физические повторы {collection.get('physical_repeats')}; "
            f"миров фактически {len(collection.get('coverage',{}).get('worlds',{}))}; "
            f"crops фактически {len(collection.get('coverage',{}).get('crops',{}))}.",
            '', 'Исходы CP–CP: '+json.dumps(outcomes),
            '', 'Свободные команды: '+json.dumps(commands),
            '', 'Свободные метки по типу действующего юнита: '+json.dumps(units),
            '', 'Расстановки map/army/contact: '+json.dumps(strata)]
    if training.get('epochs'):
        lines+=['','## Обучение','', '| Эпоха | Полная | Свободные метки | Adam steps | Online CE |',
                '|---:|---|---:|---:|---:|']
        for epoch in training['epochs']:
            lines.append(f"| {epoch['epoch']} | {epoch['completed']} | {epoch['labels']} | {epoch['steps']} | {epoch['online_CE']} |")
        lines+=['', 'История/prefix и кеш: '+json.dumps(training.get('cache',{})),
                '', 'Online CE усредняется по меткам при изменявшихся весах. Fixed training CE в таблице оценок измеряется при фиксированных весах.']
    if (out/'training-telemetry/resources.jsonl').exists():
        analysis=analyze_telemetry(out/'training-telemetry/resources.jsonl')
        write_json(out/'gpu-analysis.json',{'metrics':analysis['metrics'],'windows':analysis['windows']})
        lines+=['','## GPU NVML','',json.dumps(analysis['metrics']),
                '',f"Непрерывных окон 60 s: {len(analysis['windows'])}. Подготовка, compilation, offline CE и боевая оценка исключены.",
                '', 'Разбор недогруза: '+json.dumps(training.get('GPU_utilization_review'),ensure_ascii=False)]
    if selection:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        points=[p for p in read_json(out/'evaluations.json').values() if p['purpose']=='development']
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        axes[0].plot([p.get('Adam_steps',0) for p in points],
                     [100*p['runs']['argmax']['cp']['summary']['win_rate'] for p in points],marker='o')
        axes[0].set(xlabel='BC Adam steps',ylabel='Development argmax wins vs CP, %')
        metrics=[p for p in points if p.get('fixed_training')]
        axes[1].plot([p.get('Adam_steps',0) for p in metrics],[p['fixed_training']['ce'] for p in metrics],marker='o')
        axes[1].set(xlabel='BC Adam steps',ylabel='Fixed training CE (not holdout)')
        for ax in axes:ax.grid(alpha=.25)
        fig.savefig(out/'learning-curves.png',dpi=150);plt.close(fig)
        lines+=['','![Кривые](learning-curves.png)']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(phase,config,out,workers,device):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if workers!=4 or device!='cuda':raise ValueError('This experiment uses four CPU workers and local CUDA')
    if phase not in ('prepare','collect','train','evaluate','report','run'):raise ValueError('Unsupported target BC phase')
    with RunLock(out):
        try:
            plan=prepare(config,out)
            if (out/'status.json').exists() and read_json(out/'status.json')['status']=='complete':
                report(config,out)
                return
            if phase in ('collect','run'):
                write_json(out/'status.json',{'status':'collecting'});collect(config,plan,out)
                write_json(out/'status.json',{'status':'collection_complete'})
            if phase in ('train','run'):
                write_json(out/'status.json',{'status':'training'});train(config,plan,out)
                write_json(out/'status.json',{'status':'training_complete'})
            if phase in ('evaluate','run'):
                write_json(out/'status.json',{'status':'evaluating'});evaluate(config,plan,out)
                selected=read_json(out/'selection.json')['best'];directory=out/selected['runs']['argmax']['cp']['directory']
                for point in read_json(out/'evaluations.json').values():
                    d=out/point['runs']['argmax']['cp']['directory'];paths=list(d.glob('*/metrics.json'))
                    if len(paths)!=point['runs']['argmax']['cp']['summary']['matches'] or any(
                        not (p.parent/name).is_file() for p in paths for name in ('case.json','result.json','final.json','replay.json')):
                        raise ValueError('Incomplete saved evaluation artifacts')
                chosen={}
                for path in sorted(directory.glob('*/metrics.json')):
                    r=read_json(path);kind='D' if r['winner'] is None else 'W' if r['winner']==r['side'] else 'L'
                    chosen.setdefault(kind,path.parent)
                write_json(out/'replay-verification.json',{kind:verify_replay(path) for kind,path in chosen.items()})
                write_json(out/'status.json',{'status':'complete','confirmation':read_json(out/'confirmation.json')['status']})
            report(config,out)
        except BaseException:
            (out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
            write_json(out/'status.json',{'status':'technical_error','phase':phase})
            report(config,out)
            raise
