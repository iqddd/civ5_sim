"""One bounded rollout-comparison cycle, with separate collection/train budgets."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from collections import Counter
from copy import deepcopy
from pathlib import Path
from time import perf_counter
import json
import random
import hashlib
import torch
from civ5rl.engine.control import RunLock
from civ5rl.engine.evaluation import EvaluationRuntime, EvaluationExecutor, evaluation_precision
from civ5rl.engine.comparison_training import atomic_torch_save, fit, training_runtime
from civ5rl.experiments.comparison_collection import ProposalExecutor, match_task, select_forks, preference
from civ5rl.provenance.common import ROOT,read_json,write_json,file_sha256,configure,seed_for
from civ5rl.provenance import checkpoint
from civ5sim.engine import digest

KIND='rl01.rollout_comparison'


def load_config(path):
    cfg=read_json(path)
    if cfg['kind']!=KIND or cfg['seed']!=17:raise ValueError('Unknown comparison profile')
    training_limits=(1800,2400) if cfg.get('training',{}).get('mode')=='full_corpus_step_kl' else (1800,)
    if (not cfg.get('collection_input') and cfg['max_forks']!=1024) or cfg['collection_seconds']!=7200 or cfg['training_seconds'] not in training_limits:
        raise ValueError('This first-cycle protocol fixes its budgets')
    for p,sha in zip(cfg['parents'],cfg['parent_sha256']):
        if file_sha256(ROOT/p)!=sha:raise ValueError('Parent SHA mismatch')
    if cfg.get('training'):
        training=cfg['training']
        mode=training.get('mode')
        if mode not in ('full_corpus_step_kl','frozen_heads_step_kl'):raise ValueError('Unknown fitting mode')
        expected=dict(mode=mode,max_accepted_steps=10,anchors_per_step=660,
                      initial_lr=1e-5,kl_limit=.015,max_retries=2)
        if training!=expected or not cfg.get('collection_input'):
            raise ValueError('Unknown full-corpus comparison protocol')
        if cfg.get('evaluation_protocol')!='paired_development_only':
            raise ValueError('Full-corpus fitting requires paired Development only')
    return cfg


def prepare(cfg,out):
    if (out/'run.json').exists():return
    if cfg.get('collection_input'):
        return prepare_collection_input(cfg,out)
    out.mkdir(parents=True,exist_ok=True)
    source=ROOT/cfg['source_run'];saved=read_json(source/'run.json')
    source_state=read_json(source/'state.json')
    if cfg['start_episode']<source_state['next_episode']:raise ValueError('Episode IDs overlap')
    plan=read_json(source/'manifest.json')
    identity=checkpoint.passport(out,cfg,plan)
    write_json(out/'run.json',dict(kind=KIND,config=cfg,identity=identity,training_runtime=training_runtime(cfg),source_manifest_sha256=file_sha256(source/'manifest.json'),parent_sha256=cfg['parent_sha256']))
    write_json(out/'match-config.json',saved['config'])
    write_json(out/'manifest.json',plan)
    write_json(out/'collection.json',dict(seconds=0.,next_episode=cfg['start_episode'],forks=[],sources=[],complete=False,samples=[]))
    write_json(out/'status.json',dict(status='prepared',phase='collect'))


def prepare_collection_input(cfg,out):
    """Import immutable collection evidence into a separately passported fit.

    The collector's source snapshot is verified as frozen evidence, never against
    the current training executable. All consumed files are pinned by the fit
    config. Only the fit's copy of public histories may be unpacked or cached.
    Publishing run.json last makes interrupted preparation safely repeatable.
    """
    import shutil
    source=ROOT/cfg['collection_input']['run']
    expected=cfg['collection_input']['sha256']
    required={'dataset.json','anchors.json','collection.json','match-config.json',
              'manifest.json','passport.json','resolved-config.json','corpus-manifest.json',
              'sources.zip','sources.storage/manifest.json','sources.storage/json.7z'}
    if set(expected)!=required:raise ValueError('Incomplete frozen collection identity')
    with RunLock(source):
        for name,sha in expected.items():
            if file_sha256(source/name)!=sha:raise ValueError(f'Collection input changed: {name}')
        checkpoint.verify_frozen_passport(source)
        collection=read_json(source/'collection.json')
        original=read_json(source/'resolved-config.json')
        if not collection.get('finalized'):raise ValueError('Collection is not finalized')
        if original['parent_sha256']!=cfg['parent_sha256']:raise ValueError('Collection parents differ')
        dataset=read_json(source/'dataset.json')
        if dataset['completed_forks']!=sum(f['status']=='completed' for f in collection['forks']):
            raise ValueError('Collection dataset count differs')
        out.mkdir(parents=True,exist_ok=True)
        for name in ('dataset.json','anchors.json','collection.json','match-config.json',
                     'manifest.json','sources.storage/manifest.json','sources.storage/json.7z'):
            target=out/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source/name,target)
        write_json(out/'collection-lineage.json',cfg['collection_input'])
    plan=read_json(out/'manifest.json')
    identity=checkpoint.passport(out,cfg,plan)
    write_json(out/'status.json',dict(status='prepared',phase='train'))
    write_json(out/'run.json',dict(kind=KIND,config=cfg,identity=identity,
        training_runtime=training_runtime(cfg),collection_input=cfg['collection_input'],
        source_manifest_sha256=file_sha256(source/'manifest.json'),parent_sha256=cfg['parent_sha256']))


def collect(cfg,out,*,limit=None,budget=None):
    if budget is not None:return _collect(cfg,out,limit=limit,budget=budget)
    from civ5rl.engine.phase_budget import PhaseBudget
    with PhaseBudget(out,"collection",cfg["collection_seconds"]) as budget:
        return _collect(cfg,out,limit=limit,budget=budget)


def _collect(cfg,out,*,limit=None,budget):
    from civ5rl.protocols.ppo_target_scale import TargetFactory
    from civ5rl.experiments.comparison_collection import match_task
    state=read_json(out/'collection.json')
    if state.get('finalized') and (out/'dataset.json').exists():return
    if state['complete'] or state.get('stage')=='finalizing':
        return finalize_collection(cfg,out,state,budget,limit)
    started=perf_counter();base=state['seconds']
    config=read_json(out/'match-config.json');config=deepcopy(config)
    config['experiment'].update(start_episode=cfg['start_episode'],training_namespace=cfg['namespace']+'_training')
    factory=TargetFactory(config,read_json(out/'manifest.json')['base'])
    runtime=EvaluationRuntime('cuda',8,32,.001)
    maximum=limit or cfg['max_forks']
    def save():
        state['seconds']=budget.elapsed()
        write_json(out/'collection.json',state)
    def expired():return budget.expired()
    from contextlib import ExitStack
    with evaluation_precision(),ExitStack() as sessions:
        proposal=None
        continuation=None
        def get_proposal():
            nonlocal proposal
            if proposal is None:proposal=sessions.enter_context(ProposalExecutor([ROOT/p for p in cfg['parents']],runtime))
            return proposal
        def get_continuation():
            nonlocal continuation
            if continuation is None:continuation=sessions.enter_context(EvaluationExecutor(str(ROOT/cfg['parents'][0]),runtime))
            return continuation
        if any(f['status']=='running' for f in state['forks']):
            process_forks(cfg,out,state,config,get_continuation,expired,save)
        while (len(state['forks'])<maximum and not expired()) or state.get('pending_wave'):
            # Persist the wave before launch. A failed/resumed wave keeps its exact cases.
            if not state.get('pending_wave'):
                state['pending_wave']=[]
                for _ in range(8):
                    episode=state['next_episode'];state['next_episode']+=1
                    case=factory.make(episode)
                    directory=out/'sources'/str(episode)
                    write_json(directory/'case.json',case)
                    state['pending_wave'].append(episode)
                save()
            wave=state['pending_wave']
            tasks=[]
            for episode in wave:
                directory=out/'sources'/str(episode)
                if not (directory/'completed.json').exists():
                    tasks.append((config,read_json(directory/'case.json'),str(directory),str(ROOT/cfg['parents'][0]),None))
            if tasks:
                executor=get_proposal()
                list(executor.run(match_task,tasks))
            for episode in wave:
                if episode in state['sources']:continue
                state['sources'].append(episode)
                history=read_json(out/'sources'/str(episode)/'history.json')
                for index in select_forks(history,episode,cfg['namespace'],maximum=min(4,maximum-len(state['forks']))):
                    if len(state['forks'])>=maximum:break
                    row=history[index]
                    state['forks'].append(dict(id=len(state['forks']),episode=episode,index=index,command=row['command'],actions=[row['actions'][row['chosen']],row['actions'][row['alternative']]],status='not_started'))
            state['pending_wave']=[];save()
            if expired():break
            process_forks(cfg,out,state,config,get_continuation,expired,save)
        # Selected forks are never replaced when a label is ambiguous.
        if not expired():process_forks(cfg,out,state,config,get_continuation,expired,save)
    for fork in state['forks']:
        if fork['status']=='not_started':fork['status']='not_started_budget'
    state['stage']='finalizing';save()
    finalize_collection(cfg,out,state,budget,limit)


def finalize_collection(cfg,out,state,budget,limit=None):
    from civ5rl.reporting.comparison_storage import restore_tree,archive_tree
    if not (out/'dataset.json').exists():
        restore_tree(out,'sources');restore_tree(out,'continuations')
        write_json(out/'dataset.json',build_dataset(cfg,out,state))
    if limit is None:archive_tree(out,'continuations')
    state.update(complete=True,finalized=True,stage='complete',seconds=budget.elapsed(),overshoot_seconds=max(0,budget.elapsed()-cfg['collection_seconds']))
    write_json(out/'collection.json',state)


def process_forks(cfg,out,state,config,get_executor,expired,save):
    for fork in state['forks']:
        if fork['status']=='completed':continue
        if fork['status'] not in ('running',) and expired():break
        start=perf_counter();fork['status']='running';save()
        case=read_json(out/'sources'/str(fork['episode'])/'case.json')
        tasks=[]
        for j in range(8):
            seed=seed_for(cfg['seed'],fork['episode'],fork['id'],j,namespace=cfg['namespace']+'_futures') or 1
            for alternative in range(2):
                folder=out/'continuations'/str(fork['id'])/f'{j}-{alternative}'
                if (folder/'completed.json').exists():continue
                branch=dict(source=str(out/'sources'/str(fork['episode'])),command=fork['command'],seed=seed,action=fork['actions'][alternative])
                write_json(folder/'branch.json',branch)
                tasks.append((config,case,str(folder),str(ROOT/cfg['parents'][0]),branch))
        if tasks:
            executor=get_executor()
            list(executor.run(match_task,tasks))
        fork['status']='completed'
        state['samples'].append(dict(work=1,seconds=perf_counter()-start))
        save()
        print(f"Completed fork {fork['id']+1}/{len(state['forks'])}; collection={state['seconds']:.1f}s",flush=True)


def build_dataset(cfg,out,state):
    """Read each case/history once; hash and extract targets from the same bytes.

    Retain only free-position indices and chosen/alternative indices for selected
    forks. Anchor sampling remains blind to continuation outcomes and keeps its
    original RNG order; labels are read only after the anchor set is fixed.
    """
    # Fix A without reading a single continuation outcome; labels are read only below.
    random_stream=random.Random(seed_for(17,0,'anchors',namespace=cfg['namespace']))
    free=[];fork_rows={};sides={};hashes={}
    wanted={(f['episode'],f['index']) for f in state['forks']}
    def read_hashed(path):
        data=path.read_bytes()
        return json.loads(data),hashlib.sha256(data).hexdigest()
    for episode in state['sources']:
        folder=out/'sources'/str(episode)
        case,case_sha=read_hashed(folder/'case.json')
        rows,history_sha=read_hashed(folder/'history.json')
        sides[episode]=case['learner_side']
        hashes[str(episode)]={'case.json':case_sha,'history.json':history_sha}
        free.extend((episode,i) for i,r in enumerate(rows) if len(r['actions'])>1)
        for i,row in enumerate(rows):
            if (episode,i) in wanted:fork_rows[(episode,i)]=(row['chosen'],row['alternative'])
    selected=random_stream.sample(free,min(8192,len(free)))
    anchors=sorted(set(selected)|{(f['episode'],f['index']) for f in state['forks']})
    write_json(out/'anchors.json',dict(positions=anchors,selection_blind_to_continuation_outcomes=True))
    pairs=[];reasons=Counter()
    for fork in state['forks']:
        if fork['status']!='completed':continue
        winners=[[read_json(out/'continuations'/str(fork['id'])/f'{j}-{a}'/'completed.json')['winner'] for j in range(8)] for a in range(2)]
        label=preference(winners,sides[fork['episode']]);reasons[label['reason']]+=1
        write_json(out/'labels'/f"{fork['id']}.json",dict(fork=fork,**label))
        if label['accepted']:
            actions=fork_rows[(fork['episode'],fork['index'])]
            pairs.append(dict(episode=fork['episode'],index=fork['index'],better=actions[label['better']],worse=actions[1-label['better']],weight=label['weight']))
    return dict(pairs=pairs,anchors=anchors,source_hashes=hashes,selected_forks=len(state['forks']),completed_forks=sum(f['status']=='completed' for f in state['forks']),accepted_pairs=len(pairs),rejections=dict(reasons))


def evaluate(cfg,out):
    from civ5rl.experiments.ppo_target_scale import evaluate as ordinary_evaluate
    from civ5rl.engine.evaluation import evaluation_options
    from civ5rl.reporting.evaluation_audit import load_eval_rows,clustered_paired_bootstrap
    from civ5rl.reporting.evaluation_storage import readable_evaluations,archive_evaluations
    training=read_json(out/'training.json')
    if training.get('candidate_sha256') and file_sha256(out/'candidate.pt')!=training['candidate_sha256']:
        raise ValueError('Candidate checkpoint changed')
    if (out/'result.json').exists() and read_json(out/'result.json').get('status')=='complete':return
    if not training.get('changed'):
        write_json(out/'result.json',dict(status=training.get('status'),training_steps=training.get('training_steps',0),selected='parent',evaluations_skipped=True,**dataset_counts(out)));return
    config=read_json(out/'match-config.json');plan=read_json(out/'manifest.json')
    if cfg.get('evaluation_protocol')=='paired_development_only':
        return evaluate_paired_development(cfg,out,config,plan,ordinary_evaluate)
    # Evaluator understands existing checkpoint format; no experiment ledger is forged.
    with readable_evaluations(out),evaluation_options(device='cuda',workers=8,max_batch=32,wait_seconds=.001):
        candidate=ordinary_evaluate(config,plan,out,'candidate.pt','development_512')
        wins=candidate['runs']['argmax']['cp']['summary']['W']
        source=ROOT/cfg['source_run']
        baseline=next(e for e in read_json(source/'evaluations.json').values() if e['checkpoint']=='checkpoints/initial.pt' and e['purpose']=='development_512')
        with RunLock(source),readable_evaluations(source):
            parent_rows=load_eval_rows(source/baseline['runs']['argmax']['cp']['directory'],512,plan=plan['development_512'])
            rows=load_eval_rows(out/candidate['runs']['argmax']['cp']['directory'],512,plan=plan['development_512'])
            comparison=clustered_paired_bootstrap(parent_rows,rows)
        result=dict(status='complete',development=candidate,comparison=comparison,selected='candidate' if wins>361 else 'parent',historical_gpu_comparison=True,**dataset_counts(out))
        if wins>361:
            import shutil
            parent_path=out/'checkpoints/parent.pt';parent_path.parent.mkdir(exist_ok=True)
            if not parent_path.exists():shutil.copy2(ROOT/cfg['parents'][0],parent_path)
            parent=ordinary_evaluate(config,plan,out,'checkpoints/parent.pt','confirmation')
            additional=ordinary_evaluate(config,plan,out,'candidate.pt','confirmation')
            before=load_eval_rows(out/parent['runs']['argmax']['cp']['directory'],256,plan=plan['confirmation'])
            after=load_eval_rows(out/additional['runs']['argmax']['cp']['directory'],256,plan=plan['confirmation'])
            result['additional']=clustered_paired_bootstrap(before,after)
        else:result['additional']={'status':'not_run_no_development_gain'}
        write_json(out/'result.json',result)


def evaluate_paired_development(cfg,out,config,plan,evaluator):
    """Compare the last verified candidate and parent through one local runtime."""
    import shutil
    from civ5rl.engine.evaluation import evaluation_options
    from civ5rl.reporting.evaluation_storage import readable_evaluations
    from civ5rl.reporting.evaluation_audit import load_eval_rows,clustered_paired_bootstrap
    parent_path=out/'checkpoints/parent.pt';parent_path.parent.mkdir(exist_ok=True)
    if not parent_path.exists():shutil.copy2(ROOT/cfg['parents'][0],parent_path)
    if file_sha256(parent_path)!=cfg['parent_sha256'][0]:raise ValueError('Evaluation parent changed')
    with readable_evaluations(out),evaluation_options(device='cuda',workers=8,max_batch=32,wait_seconds=.001):
        parent=evaluator(config,plan,out,'checkpoints/parent.pt','development_512')
        candidate=evaluator(config,plan,out,'candidate.pt','development_512')
        def rows(record):
            return load_eval_rows(out/record['runs']['argmax']['cp']['directory'],512,plan=plan['development_512'])
        comparison=clustered_paired_bootstrap(rows(parent),rows(candidate))
    write_json(out/'result.json',dict(status='complete',development=candidate,parent_development=parent,
        comparison=comparison,candidate_selection='last_accepted',
        candidate_checkpoint='candidate.pt',parent_checkpoint='checkpoints/parent.pt',
        additional={'status':'disabled_by_protocol'},**dataset_counts(out)))


def dataset_counts(out):
    data=read_json(out/'dataset.json')
    return {**{k:data[k] for k in ('selected_forks','completed_forks','accepted_pairs')},
            'rejections':data.get('rejections',data.get('rejection_reasons',{}))}


def run(phase,cfg,out,*,preparation_seconds=0.):
    from contextlib import ExitStack
    from civ5rl.engine.phase_budget import PhaseBudget
    out=Path(out).resolve()
    with RunLock(out),ExitStack() as timing:
        collection=read_json(out/'collection.json') if (out/'collection.json').exists() else {}
        training=read_json(out/'training.json') if (out/'training.json').exists() else {}
        collect_needed=not cfg.get('collection_input') and (not collection.get('finalized') or not (out/'dataset.json').exists())
        if cfg.get('collection_input') and phase=='collect':
            raise ValueError('Frozen collection inputs cannot collect again')
        train_needed=training.get('status') not in ('complete','no_accepted_pairs')
        budget=None
        if phase in ('prepare','collect','resume') and collect_needed:
            budget=timing.enter_context(PhaseBudget(out,'collection',cfg['collection_seconds']))
        elif (phase in ('train','resume') or
              phase=='prepare' and cfg.get('training',{}).get('mode') in ('full_corpus_step_kl','frozen_heads_step_kl')) and train_needed:
            budget=timing.enter_context(PhaseBudget(out,'training',cfg['training_seconds']))
        if budget is not None:budget.base+=preparation_seconds
        configure(cfg['seed'],'cuda')
        prepare(cfg,out)
        checkpoint.verify_execution_passport(out,cfg,read_json(out/'manifest.json'))
        write_json(out/'status.json',dict(status='running',phase=phase))
        try:
            if phase in ('collect','resume') and collect_needed:
                collect(cfg,out,budget=budget)
                timing.close();budget=None
            if phase in ('train','resume'):
                if not (out/'dataset.json').exists():raise ValueError('Collection incomplete')
                if budget is None and train_needed:
                    budget=timing.enter_context(PhaseBudget(out,'training',cfg['training_seconds']))
                fit(cfg,out,read_json(out/'dataset.json'),read_json(out/'passport.json'),budget=budget)
                timing.close()
            if phase in ('evaluate','resume'):evaluate(cfg,out)
            write_json(out/'status.json',dict(status='complete' if phase in ('evaluate','resume') else 'prepared',phase=phase))
        except BaseException as exc:
            write_json(out/'status.json',dict(status='technical_error',phase=phase,error=f'{type(exc).__name__}: {exc}'))
            raise
    from civ5rl.reporting.comparison_report import report
    report(out)
    if phase in ('evaluate','resume') and (out/'evaluation').exists():
        from civ5rl.reporting.evaluation_storage import archive_evaluations
        archive_evaluations(out)
