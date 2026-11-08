"""One local target-scale PPO run, with bounded CPU-backed rollout storage."""

import gc
import json
import platform
import random
import subprocess
import sys
import traceback
import zipfile
from collections import Counter
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import torch

from civ5rl.core.context import context_from_spec
from civ5rl.core.model import batch_inputs
from civ5rl.core.observation import ObservationEncoder
from civ5rl.engine.collector import Collector
from civ5rl.engine.control import RunLock, assert_not_closed
from civ5rl.engine.ppo import update
from civ5rl.engine.ppo import encode_episode
from civ5rl.engine.ppo_fast import BatchedBackend
from civ5rl.experiments import ppo_scale_up as shared
from civ5rl.experiments.evaluation import paired_world_bootstrap, verify_replay
from civ5rl.experiments.imitation_helpers import assert_equal, isolated_rng, pool
from civ5rl.experiments.ppo_artifacts import load_rollout, save_rollout
from civ5rl.experiments.ppo_after_imitation import optimizer_for, play_case, policy_hash, match_summary
from civ5rl.protocols.corpus import validate_corpus
from civ5rl.protocols.ppo_target_scale import KIND, TargetFactory, InvalidDataset, evaluation_cases, validate_profile
from civ5rl.provenance import checkpoint
from civ5rl.provenance import case_manifest, bound_cases
from civ5rl.provenance.common import ROOT, configure, file_sha256, read_json, seed_for, write_json
from civ5sim.engine import digest, UNIT_TYPES


def math_check(config, out):
    episodes = load_rollout(ROOT/'artifacts/rl/ppo-controlled-updates/rollouts/first.pt')[:2]
    for ep, length in zip(episodes,(7,11)):
        for key in ('observations','actions','chosen','old_logp','old_log_probs','old_value','old_hidden','decision_metrics'):
            if key in ep:
                ep[key] = ep[key][:length]
    # Two independent encoders must leave the saved small-task representation intact.
    ep = episodes[0]
    encoder = ObservationEncoder(context_from_spec(ep['spec']))
    encoded = encoder.encode(ep['observations'][0],ep['actions'][0])
    # The only old/new encoder difference is the capacity guard. Emulate its
    # original value while encoding the same small input, then compare all fields.
    import civ5rl.core.observation as observation_module
    capacity = observation_module.MAX_ROSTER_UNITS
    try:
        observation_module.MAX_ROSTER_UNITS = 8
        old = ObservationEncoder(context_from_spec(ep['spec'])).encode(ep['observations'][0],ep['actions'][0])
    finally:
        observation_module.MAX_ROSTER_UNITS = capacity
    for field in encoded.__dataclass_fields__:
        np.testing.assert_array_equal(getattr(old,field),getattr(encoded,field))
    config = deepcopy(config)
    config['performance']['compile'] = False
    # Re-label the public fixture with this parent's own FP32 distributions;
    # the historical rollout was collected by a different policy.
    configure(17,'cpu')
    reference,_ = shared.load_parent(config)
    with torch.no_grad():
        for episode in episodes:
            inputs=encode_episode(episode,'C')
            logits,values,_=reference.forward_sequences(batch_inputs(inputs,'cpu'),[len(inputs)])
            logp=logits.log_softmax(-1).numpy()
            episode['old_log_probs']=[row[:len(actions)].copy() for row,actions in zip(logp,episode['actions'])]
            episode['old_logp']=[float(row[chosen]) for row,chosen in zip(episode['old_log_probs'],episode['chosen'])]
            episode['old_value']=values.numpy().tolist()
    del reference
    outputs = []
    for streamed in (False,True):
        configure(17,'cpu')
        policy, optimizer = shared.load_parent(config)
        options = deepcopy(config)
        options['performance'].update(cpu_rollout_labels=streamed,resident_rollout=not streamed)
        backend = BatchedBackend(policy,episodes,options,'cpu')
        replay = backend.precheck()
        spans = [(0,2,7),(1,1,11)]
        count = sum(b-a for _,a,b in spans)
        optimizer.zero_grad(set_to_none=True)
        backend.backward(spans,count,0)
        grads = {n:p.grad.clone() for n,p in policy.named_parameters()}
        backend.record_gradient(torch.nn.utils.clip_grad_norm_(policy.parameters(),.5))
        optimizer.step()
        stats = backend.statistics(count)[0]
        outputs.append((policy.state_dict(), optimizer.state_dict(), grads, stats, replay))
    for before,after in zip(outputs[0],outputs[1]):
        assert_equal(before,after)
    write_json(out/'math-check.json',{'loss_gradients_Adam_equal':True,'valid_tokens':count,
                                     'nonzero_GRU_prefixes':[2,1],'small_inputs_unchanged':True,
                                     'cpu_ragged_labels':True,'stats':outputs[1][3]})


def prepare(config, out):
    started = perf_counter()
    e = config['experiment']
    if file_sha256(ROOT/e['parent']) != e['parent_sha256']:
        raise ValueError('Parent checkpoint SHA256 differs')
    previous = read_json(ROOT/e['parent_manifest'])
    corpus = read_json(ROOT/'artifacts/rl/corpus/manifest.json')
    worlds = previous['base']['worlds']
    if len(worlds['training'])!=57 or set(worlds['training']) & set(worlds['validation']):
        raise ValueError('Invalid parent-world split')
    maps = [m for m in corpus['maps'] if (m['split']=='train' and m['parent_seed'] in worlds['training'])
            or (m['split']=='validation' and m['parent_seed'] in worlds['validation'])]
    if sum(m['split']=='train' for m in maps)!=684:
        raise ValueError('Expected 684 target training crops')
    validate_corpus({'maps':maps})
    base = {'worlds':worlds,'maps':maps,'hash':digest(maps)}
    factory = TargetFactory(config,base)
    capacity = []
    for index,entry in enumerate(maps):
        options = sorted((m for m in maps if m['split']==entry['split'] and m['parent_seed']==entry['parent_seed']
                          and m['size']==entry['size']),key=lambda m:m['path'])
        for contact in (False,True):
            try:
                case = factory.case(index,entry['parent_seed'],entry['size'],2,contact,0,
                    'ppo_target_scale_capacity_v1',split=entry['split'],crop=options.index(entry),n=13)
            except InvalidDataset:
                write_json(out/'capacity-checks.json',capacity)
                raise
            capacity.append({'map':entry['path'],'contact':contact,'attempts':case['placement_attempts'],
                             'algorithm':case['placement_algorithm'],'method':case['placement_method'],
                             'spawn_zones_revised':case['spawn_zones_revised']})
        if (index+1)%50==0:
            print(f'Placement capacity: {index+1}/{len(maps)} crops',flush=True)
    write_json(out/'capacity-checks.json',capacity)
    plan = {'schema':1,'kind':KIND,'base':base, **evaluation_cases(config,base),
            'small_control':previous['development']}
    for purpose in ('development','confirmation'):
        cases = plan[purpose]
        assert len(cases)==(128 if purpose=='development' else 256)
        counts = Counter((c['map_size'],c['army_bin']) for c in cases if c['learner_side']==0)
        assert max(counts.values())-min(counts.values())<=1
        assert {k for c in cases for k in c['armies'][0]}==set(UNIT_TYPES)
        for world in worlds['validation']:
            assert len([c for c in cases if c['parent_world']==world])==(8 if purpose=='development' else 16)
    plan['hash'] = digest(plan)
    write_json(out/'manifest.json',plan)
    math_check(config,out)
    write_json(out/'preparation.json',{'seconds':perf_counter()-started,'capacity_cases':len(capacity),
               'development':128,'confirmation':256,'training_crops':684,'test_used':False})
    print('Cases and CPU-backed PPO math verified; starting bounded memory calibration',flush=True)
    return plan


def calibrate(config, plan, out, device):
    attempts = []
    for envs in (16,8,4):
        config = deepcopy(config)
        config['collection'].update(num_envs=envs,envs_per_worker=envs//4)
        config['performance']['worker_reply_size']=envs//4
        factory = TargetFactory(config,plan['base'])
        specs = [factory.case(900000+i,plan['base']['worlds']['training'][i%57],16,2,i%2,i%2,
                             'ppo_target_scale_memory_v1',n=13) for i in range(envs)]
        episodes = None
        try:
            configure(17,device)
            policy,optimizer = shared.load_parent(config,device)
            torch.cuda.reset_peak_memory_stats()
            trusted=case_manifest(specs,source=KIND,split='train',namespace='ppo_target_scale_memory_v1')
            with Collector(config,factory,device,out/'preparation-errors') as collector:
                episodes,_,timings = collector.collect(policy,None,0,fixed_specs=bound_cases(trusted),trusted_manifest=trusted)
            free,total=torch.cuda.mem_get_info()
            collection_peak=torch.cuda.max_memory_reserved()+max(0,total-free-torch.cuda.memory_reserved())
            if total-collection_peak<2*1024**3:
                raise MemoryError('Less than 2 GiB VRAM reserve during collection')
            if psutil.virtual_memory().available < 4*1024**3:
                raise MemoryError('Less than 4 GiB available RAM after collection')
            save_rollout(episodes,out/'preparation-rollout',1)
            for micro in (8,4,2,1):
                config['ppo']['sequences_per_microbatch']=micro
                config['performance']['kl_sequences_per_batch']=micro
                del policy,optimizer
                gc.collect(); torch.cuda.empty_cache()
                configure(17,device)
                policy,optimizer = shared.load_parent(config,device)
                torch.cuda.reset_peak_memory_stats()
                try:
                    # One real Adam attempt on a disposable copy, including full KL.
                    backend = BatchedBackend(policy,episodes,config,device)
                    backend.precheck()
                    from civ5rl.engine.ppo import token_minibatches
                    spans = [(i,a,min(a+128,len(ep['chosen']))) for i,ep in enumerate(episodes)
                             for a in range(0,len(ep['chosen']),128)]
                    batch = next(iter(token_minibatches(spans,1024)))
                    optimizer.zero_grad(set_to_none=True)
                    backend.backward(batch,sum(b-a for _,a,b in batch),0)
                    torch.nn.utils.clip_grad_norm_(policy.parameters(),.5,error_if_nonfinite=True)
                    optimizer.step()
                    kl = backend.kl()
                    del backend
                    torch.cuda.synchronize()
                    free,total = torch.cuda.mem_get_info()
                    peak = max(collection_peak,torch.cuda.max_memory_reserved()+max(0,total-free-torch.cuda.memory_reserved()))
                    available_ram = psutil.virtual_memory().available
                    result={'envs':envs,'microbatch':micro,'peak_reserved_bytes':peak,'total_vram':total,
                            'available_ram':available_ram,'decisions':sum(len(ep['chosen']) for ep in episodes),
                            'collection':timings,'temporary_poststep_KL':kl}
                    attempts.append(result)
                    write_json(out/'memory-calibration.json',attempts)
                    if total-peak<2*1024**3:
                        raise torch.OutOfMemoryError('Less than 2 GiB VRAM reserve')
                    if available_ram<4*1024**3:
                        raise MemoryError('Less than 4 GiB RAM reserve')
                    # Mixed-size public history collected with unmodified parent weights.
                    del policy,optimizer,episodes
                    gc.collect(); torch.cuda.empty_cache()
                    configure(17,device)
                    policy,optimizer=shared.load_parent(config,device)
                    mixed=[factory.case(910000+i,plan['base']['worlds']['training'][i%57],(10,12,16)[i%3],i%3,
                                        i%2,i%2,'ppo_target_scale_mixed_v1') for i in range(envs)]
                    trusted=case_manifest(mixed,source=KIND,split='train',namespace='ppo_target_scale_mixed_v1')
                    with Collector(config,factory,device,out/'preparation-errors') as collector:
                        episodes,_,_=collector.collect(policy,None,0,fixed_specs=bound_cases(trusted),trusted_manifest=trusted)
                    save_rollout(episodes,out/'mixed-preparation-rollout',1)
                    backend=BatchedBackend(policy,episodes,config,device)
                    backend.precheck()
                    write_json(out/'mixed-behavior-check.json',backend.replay)
                    del backend,policy,optimizer,episodes
                    gc.collect(); torch.cuda.empty_cache()
                    write_json(out/'resolved-config.json',config)
                    print(f'Memory configuration fixed: {envs} envs, microbatch {micro}',flush=True)
                    return config
                except torch.OutOfMemoryError as error:
                    if 'backend' in locals():
                        del backend
                    attempts.append({'envs':envs,'microbatch':micro,'OOM':str(error)})
                    write_json(out/'memory-calibration.json',attempts)
                    gc.collect(); torch.cuda.empty_cache()
            raise MemoryError('Backward does not fit with microbatch 1')
        except (torch.OutOfMemoryError,MemoryError) as error:
            attempts.append({'envs':envs,'collection_or_RAM_error':str(error)})
            write_json(out/'memory-calibration.json',attempts)
            episodes=None
            gc.collect(); torch.cuda.empty_cache()
    raise MemoryError('Target-scale workload does not fit 4 envs / microbatch 1')


def initialize(config, plan, out):
    configure(17,'cpu')
    policy,optimizer=shared.load_parent(config)
    record={'kind':KIND,'config':config,'parent_sha256':config['experiment']['parent_sha256'],
            'initial_model_Adam_bitwise_equal':True,'parent_Adam_counter':45915,'critic_preserved':True,
            'python':platform.python_version(),'torch':torch.__version__,'cuda':torch.version.cuda,
            'GPU':torch.cuda.get_device_name(0),'seed':17,
            'shuffle_seed':seed_for(17,0,'shuffle',namespace=config['experiment']['shuffle_namespace'])}
    # One bounded source snapshot. Markdown is informational, never a phase barrier.
    names=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=ROOT).decode().split('\0')
    with zipfile.ZipFile(out/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(names)):
            if name and (ROOT/name).is_file(): archive.write(ROOT/name,name)
    record['source_snapshot']=file_sha256(out/'source.zip')
    record['run_id']=digest(record)
    state={'updates':0,'accepted_updates':0,'attempted_steps':0,'rejected_steps':0,'optimizer_steps':0,
           'origin_Adam_counter':45915,'decisions':0,'next_episode':350592,'zero_accepted_streak':0,
           'active_seconds':0.,'evaluation_seconds':0.,'preparation_seconds':0.,'evaluated':{},
           'latest':'checkpoints/initial.pt','pending_evaluation':True,
           'best':{'path':'checkpoints/initial.pt','wins':-1,'update':0},'stop_reason':None}
    write_json(out/'run.json',record)
    checkpoint.save(out/state['latest'],policy,optimizer,state,random.Random(record['shuffle_seed']),config,{'run_id':record['run_id']})
    write_json(out/'state.json',state)
    write_json(out/'status.json',{'status':'prepared'})


def evaluation_task(task):
    config,case,path,directory=task
    if case['split']!='validation' or digest({k:v for k,v in case.items() if k!='case_hash'})!=case['case_hash']:
        raise ValueError('Invalid validation case')
    result=play_case(config,case,path,'argmax','cp',directory)
    replay=read_json(Path(directory)/'replay.json')
    final=read_json(Path(directory)/'final.json')
    side=case['learner_side']
    damage=[0,0]
    owners={u['id']:u['side'] for u in replay['initial']['units']}
    for step in replay['steps']:
        detail=step['result']
        if step['action']['kind'] in ('ranged','melee'):
            attacker=owners[step['action']['unit']]
            damage[attacker]+=detail['damage']
            damage[1-attacker]+=detail['retaliation']
    result.update(map_size=case.get('map_size',10),army_size=len(case['armies'][side]),
                  army_bin=case.get('army_bin',-1),placement=case['placement'],
                  damage_dealt=damage,
                  units_lost=[sum(u['side']==s and u['hp']<=0 for u in final['units']) for s in (0,1)])
    write_json(Path(directory)/'metrics.json',result)
    return result


def evaluate(config,plan,out,path,purpose):
    fingerprint=policy_hash(torch.load(out/path,map_location='cpu',weights_only=False)['model'])
    cache=read_json(out/'evaluations.json') if (out/'evaluations.json').exists() else {}
    import platform
    import inspect
    import civ5rl.core.agent as agent_module
    from civ5sim.combat_ai import LIBRARY_PATH
    evaluator = digest({'platform':platform.platform(), 'torch':torch.__version__,
        'play_case':inspect.getsource(play_case), 'agent':file_sha256(Path(agent_module.__file__)),
        'match':file_sha256(ROOT/'civ5sim/match.py'), 'cp':file_sha256(ROOT/'civ5sim/cp_agent.py'),
        'model':file_sha256(ROOT/'civ5rl/core/model.py'), 'observation':file_sha256(ROOT/'civ5rl/core/observation.py'),
        'planner':file_sha256(LIBRARY_PATH),
        'kernel':file_sha256(ROOT/'build'/('combat.dll' if sys.platform=='win32' else 'libcombat.so'))})
    key=digest({'purpose':purpose,'policy':fingerprint,'mode':'argmax','temperature':config.get('policy_temperature',1.0),
        'cases':[c['case_hash'] for c in plan[purpose]], 'evaluator':evaluator, 'search':config['teacher_search']})
    if key in cache:return cache[key]
    cases=plan[purpose]
    started=perf_counter()
    relative=f'evaluation/{purpose}/{fingerprint[:16]}/{key[:16]}/argmax/cp'
    directory=out/relative
    with pool(4) as executor:
        tasks=[(config,case,str(out/path),str(directory/f"case-{case['case_id']:03}-side-{case['learner_side']}"))
               for case in cases if not (directory/f"case-{case['case_id']:03}-side-{case['learner_side']}"/'metrics.json').exists()]
        for first in range(0,len(tasks),4):
            list(executor.map(evaluation_task,tasks[first:first+4]))
            print(f'Evaluation {purpose}: {min(first+4,len(tasks))}/{len(tasks)} new matches',flush=True)
    summary=match_summary(directory,len(cases))
    rows=[read_json(p) for p in directory.glob('*/metrics.json')]
    slices={}
    for field in ('map_size','army_bin','army_size','placement','side'):
        slices[field]={}
        for value in sorted({r[field] for r in rows},key=str):
            selected=[r for r in rows if r[field]==value]
            wdl=Counter('D' if r['winner'] is None else 'W' if r['winner']==r['side'] else 'L' for r in selected)
            slices[field][str(value)]={'matches':len(selected),**{k:wdl[k] for k in ('W','D','L')},
                                      'seconds':sum(r['seconds'] for r in selected),
                                      'mean_rounds':float(np.mean([r['rounds'] for r in selected])),
                                      'draw_reasons':dict(Counter(r['reason'] for r in selected if r['winner'] is None))}
    result={'checkpoint':path,'policy_hash':fingerprint,'purpose':purpose,'seconds':perf_counter()-started,
            'runs':{'argmax':{'cp':{'directory':relative,'summary':summary}}},'slices':slices}
    cache[key]=result
    write_json(out/'evaluations.json',cache)
    print(f'Evaluation {purpose}: {summary}; wall={result["seconds"]:.1f}s',flush=True)
    return result


def development(config,plan,out,state,shuffle):
    started=perf_counter()
    with isolated_rng(shuffle): result=evaluate(config,plan,out,state['latest'],'development')
    state['evaluation_seconds']+=perf_counter()-started
    state['evaluated'][str(state['updates'])]={**result,'decisions':state['decisions']}
    wins=result['runs']['argmax']['cp']['summary']['W']
    if wins>state['best']['wins']:
        state['best']={'path':state['latest'],'wins':wins,'update':state['updates'],'policy_hash':result['policy_hash']}
    state['pending_evaluation']=False
    write_json(out/'state.json',state)
    if state['updates']==0:
        write_json(out/'evaluation-cost.json',{'baseline_wall_seconds':result['seconds'],
            'estimated_target_evaluation_seconds':result['seconds']*9,
            'slices':result['slices'],'uncertainty':'Trajectory length and army composition change CPU planning cost'})
    report(config,out)


def finish(config,plan,out):
    state=read_json(out/'state.json')
    if not state['stop_reason'] or state['pending_evaluation']:raise ValueError('Training is incomplete')
    write_json(out/'selected.json',state['best'])
    baseline=evaluate(config,plan,out,'checkpoints/initial.pt','confirmation')
    candidate=evaluate(config,plan,out,state['best']['path'],'confirmation')
    same=baseline['policy_hash']==candidate['policy_hash']
    def outcomes(evaluation):
        directory=out/evaluation['runs']['argmax']['cp']['directory']
        results,cases={},{}
        for p in directory.glob('*/metrics.json'):
            row,case=read_json(p),read_json(p.parent/'case.json')
            key=(case['parent_world'],case['case_id'],case['learner_side'])
            results[key]=int(row['winner']==row['side']);cases[key]=case
        return results,cases
    before,cases=outcomes(baseline);after,paired=outcomes(candidate)
    if cases!=paired:raise ValueError('Unpaired target validation')
    comparison=None if same else paired_world_bootstrap(before,after,samples=10000,seed=20261103,cases=cases,cases_per_world=16)
    write_json(out/'confirmation.json',{'baseline':baseline,'candidate':candidate,'comparison':comparison,'baseline_selected':same})
    small_before=evaluate(config,plan,out,'checkpoints/initial.pt','small_control')
    small_after=evaluate(config,plan,out,state['best']['path'],'small_control')
    write_json(out/'small-control.json',{'baseline':small_before,'candidate':small_after})
    directory=out/candidate['runs']['argmax']['cp']['directory']
    chosen={}
    for p in sorted(directory.glob('*/metrics.json')):
        r=read_json(p);outcome='D' if r['winner'] is None else 'W' if r['winner']==r['side'] else 'L'
        chosen.setdefault(outcome,p.parent)
    saved=torch.load(out/state['best']['path'],map_location='cpu',weights_only=False)
    counters={int(v['step']) for v in saved['optimizer']['state'].values()}
    assert counters=={45915+saved['state']['optimizer_steps']}
    write_json(out/'verification.json',{'selected_Adam_counter':next(iter(counters)),
        'replays':{kind:verify_replay(path) for kind,path in chosen.items()}})


def report(config,out):
    state=read_json(out/'state.json')
    lines=['# PPO target scale: локальный перенос на армии 5–13', '',
           f"Статус: {read_json(out/'status.json')['status']}; причина: {state['stop_reason']}.",
           f"Новые решения {state['decisions']}; rollouts {state['updates']}; active {state['active_seconds']:.2f} s; "
           f"accepted/rejected {state['optimizer_steps']}/{state['rejected_steps']}; Adam {45915+state['optimizer_steps']}.",
           '',f"Параллелизм: {config['collection']['num_envs']} сред / {config['collection']['cpu_workers']} workers; microbatch {config['ppo']['sequences_per_microbatch']}; CPU rollout.",
           '', '| Новые решения | Checkpoint | W/D/L CP argmax | Win rate |','|---:|---|---|---:|']
    for point in state['evaluated'].values():
        r=point['runs']['argmax']['cp']['summary']
        lines.append(f"| {point['decisions']} | {point['checkpoint']} | {r['W']}/{r['D']}/{r['L']} | {r['win_rate']:.2%} |")
    lines+=['',f"Выбран: {state['best']}."]
    for name in ('confirmation.json','small-control.json','evaluation-cost.json','memory-calibration.json','preparation.json'):
        if (out/name).exists():lines+=['',f'## {name}','',json.dumps(read_json(out/name),ensure_ascii=False)]
    if (out/'updates.jsonl').exists():
        rows=[json.loads(line) for line in (out/'updates.jsonl').read_text().splitlines()]
        lines+=['','## Оптимизация','', '| Rollout | W/D/L | KL final | Critic MSE/EV | Gradient | Epochs | Adam accept/reject | Collection/PPO s |',
                '|---:|---|---:|---|---:|---:|---|---|']
        for r in rows:
            p=r['ppo']
            lines.append(f"| {r['update']} | {r['W']}/{r['D']}/{r['L']} | {p.get('final_accepted_kl')} | "
                         f"{p.get('critic_mse')}/{p.get('critic_explained_variance')} | {p.get('gradient_norm_mean')} | "
                         f"{p.get('epochs_completed')} | {p['optimizer_steps']}/{p['rejected_optimizer_steps']} | "
                         f"{r['collection']['collection_seconds']:.2f}/{p.get('ppo_seconds',0):.2f} |")
    lines+=['','Оба validation-набора разработческие. Закрытый test, аренда, другие seeds и изменение условий не использовались.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(phase,config,out,device):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if phase=='pause':
        (out/'pause.request').write_text('user requested pause\n',encoding='utf-8')
        print('Pause requested; completes current update and preserves pending evaluation',flush=True)
        return
    if (out/'resolved-config.json').exists():config=read_json(out/'resolved-config.json')
    validate_profile(config)
    with RunLock(out):
        if phase in ('train','run','resume'):assert_not_closed(out)
        try:
            if phase=='resume':
                (out/'pause.request').unlink(missing_ok=True)
            if not (out/'run.json').exists():
                if phase not in ('prepare','run'):raise ValueError('Prepare first')
                write_json(out/'status.json',{'status':'preparing'})
                if (out/'manifest.json').exists() and (out/'preparation.json').exists():
                    plan=read_json(out/'manifest.json')
                    if digest({k:v for k,v in plan.items() if k!='hash'})!=plan['hash']:
                        raise ValueError('Prepared manifest differs')
                    if any(c.get('placement_algorithm')!='native_visibility_regions_v2'
                           for purpose in ('development','confirmation') for c in plan[purpose]
                           if c['placement']=='separated'):
                        raise ValueError('Prepared cases use the previous placement algorithm')
                    print('Reusing completed crop preparation; continuing memory calibration',flush=True)
                else:
                    plan=prepare(config,out)
                config=calibrate(config,plan,out,device)
                initialize(config,plan,out)
            else:
                record=read_json(out/'run.json')
                if record.get('active_config',record['config'])!=config:
                    raise ValueError('Configuration differs')
            plan=read_json(out/'manifest.json')
            if phase in ('train','run','resume'):
                write_json(out/'status.json',{'status':'training'})
                shared.train(config,plan,out,device,factory_type=TargetFactory,development_fn=development)
                if read_json(out/'status.json')['status']=='paused':
                    report(config,out)
                    return
            if phase in ('evaluate','run','resume'):
                finish(config,plan,out)
                write_json(out/'status.json',{'status':'complete','stop_reason':read_json(out/'state.json')['stop_reason']})
            if (out/'state.json').exists():report(config,out)
        except InvalidDataset as error:
            write_json(out/'invalid-dataset.json',error.details)
            write_json(out/'status.json',{'status':'invalid_dataset','phase':phase,'reason':str(error)})
            (out/'report.md').write_text('# PPO target scale: подготовка остановлена\n\n'
                +str(error)+'\n\nОсновное обучение и оценочные бои не запускались. '
                'Карта, состав и категория не заменены. Исчерпание попыток генератора не доказывает '
                'невозможность расстановки другим алгоритмом. Диагностика: invalid-dataset.json.\n',encoding='utf-8')
            raise
        except BaseException:
            (out/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
            write_json(out/'status.json',{'status':'technical_error','phase':phase})
            if (out/'state.json').exists():report(config,out)
            raise
