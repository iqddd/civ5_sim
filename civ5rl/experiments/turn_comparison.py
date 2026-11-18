"""Outcome-blind whole-turn collection from immutable recorded source battles."""
from collections import defaultdict,Counter
from pathlib import Path
import random
import shutil
from time import perf_counter

from civ5rl.provenance.common import ROOT,read_json,write_json,file_sha256,seed_for,configure
from civ5rl.provenance import checkpoint
from civ5rl.engine.runtime.control import RunLock
from civ5rl.reporting.comparison_storage import readable_tree,restore_tree,archive_tree
from civ5sim.engine import digest

KIND='rl01.turn_comparison_collection'
NAMESPACE='c2p_turn_comparison_v1'


def position_candidates(episode,case,history,replay):
    """Require the actual first decision, never skip a forced turn opening."""
    candidates=[];seen=set();side=case['learner_side']
    for history_index,row in enumerate(history):
        turn=row['observation']['turn']
        if turn in seen:continue
        seen.add(turn)
        command=row['command']
        if command==0 or replay['steps'][command-1]['action']['kind']!='end_turn':continue
        obs=row['observation']
        if len(row['actions'])<=1 or not any(u['side']!=side for u in obs['units']):continue
        candidates.append(dict(key=[episode,turn,side],episode=episode,command=command,
            world=case['parent_world'],history_index=history_index,observation_sha256=digest(obs)))
    return candidates


def select_positions(by_world,maximum=512):
    """Round-robin shuffled worlds/battles; exhaust first picks before seconds."""
    rng=random.Random(seed_for(17,0,'positions',namespace=NAMESPACE+'_selection'))
    worlds=sorted(by_world);rng.shuffle(worlds);ordered={}
    for world in worlds:
        episodes=sorted(by_world[world]);rng.shuffle(episodes);ordered[world]=episodes
    battles=[]
    for index in range(max((len(v) for v in ordered.values()),default=0)):
        for world in worlds:
            if index<len(ordered[world]):battles.append((world,ordered[world][index]))
    choices={}
    for world,episode in battles:
        rows=sorted(by_world[world][episode],key=lambda r:tuple(r['key']))
        choices[episode]=rng.sample(rows,min(2,len(rows)))
    selected=[]
    for pick in range(2):
        for _,episode in battles:
            if len(choices[episode])>pick:selected.append(choices[episode][pick])
            if len(selected)>=maximum:return selected
    return selected


def branch_spec(position,variant,repetition,source):
    key=position['key']
    return dict(source=str(source),command=position['command'],turn_variant=variant,repetition=repetition,
        position_key=key,seed=seed_for(17,key,repetition,namespace=NAMESPACE+'_damage') or 1,
        sampling_seed=seed_for(17,key,variant,namespace=NAMESPACE+'_sampling'))


def load_config(path):
    cfg=read_json(path)
    expected=71 if 'confirmation' in cfg else 512
    if cfg['kind']!=KIND or cfg['seed']!=17 or cfg['positions']!=expected:raise ValueError('Unknown whole-turn protocol')
    if 'confirmation' in cfg and cfg['confirmation']['repetitions']!=list(range(4,12)):
        raise ValueError('Confirmation requires eight new repetitions 4-11')
    return cfg


def prepare(cfg,out):
    if 'confirmation' in cfg:
        from civ5rl.experiments.turn_confirmation import prepare_confirmation
        return prepare_confirmation(cfg,out)
    source=ROOT/cfg['collection_input']['run']
    for name,sha in cfg['collection_input']['sha256'].items():
        if file_sha256(source/name)!=sha:raise ValueError(f'Source changed: {name}')
    checkpoint.verify_frozen_passport(source,sources=False)
    if file_sha256(ROOT/cfg['parent'])!=cfg['parent_sha256']:raise ValueError('Parent changed')
    if file_sha256(ROOT/cfg['split'])!=cfg['split_sha256']:raise ValueError('World split changed')
    training=read_json(ROOT/cfg['split'])['training']
    lineage=dict(source=cfg['collection_input'],training_worlds=training,split_sha256=cfg['split_sha256'])
    if (out/'passport.json').exists():checkpoint.verify_execution_passport(out,cfg,lineage)
    else:
        identity=checkpoint.passport(out,cfg,lineage)
        write_json(out/'run.json',dict(kind=KIND,config=cfg,identity=identity))
    if (out/'positions.json').exists():
        if file_sha256(out/'positions.json')!=read_json(out/'selection.json')['sha256']:
            raise ValueError('Position manifest changed')
        return
    write_json(out/'status.json',dict(status='preparing',phase='selection'))
    target=out/'sources.storage'
    if not target.exists():shutil.copytree(source/'sources.storage',target)
    collection=read_json(source/'collection.json');by_world=defaultdict(dict);battles=0
    hashes=read_json(source/'dataset.json')['source_hashes'];started=perf_counter()
    with readable_tree(out,'sources') as sources:
        for episode in sorted(collection['sources']):
            folder=sources/str(episode);case=read_json(folder/'case.json')
            if not 111068<=episode<=112095:raise ValueError('Unexpected source episode')
            world=case['parent_world']
            if world!=case['map_entry']['parent_seed']:raise ValueError('World identity differs')
            if world not in training:continue
            battles+=1
            if any(file_sha256(folder/name)!=sha for name,sha in hashes[str(episode)].items()):
                raise ValueError('Public source history changed')
            replay=read_json(folder/'replay.json');history=read_json(folder/'history.json')
            candidates=position_candidates(episode,case,history,replay)
            if candidates:by_world[world][episode]=candidates
        if battles!=595:raise ValueError('Expected 595 training source battles')
    positions=select_positions(by_world,cfg['positions'])
    manifest=dict(schema=1,namespace=NAMESPACE,positions=positions,lineage=lineage,
        parent_sha256=cfg['parent_sha256'],variants=4,repetitions=4)
    write_json(out/'positions.json',manifest)
    write_json(out/'selection.json',dict(sha256=file_sha256(out/'positions.json'),source_battles=battles,
        eligible_battles=sum(len(v) for v in by_world.values()),
        eligible_positions=sum(len(rows) for v in by_world.values() for rows in v.values()),
        selected=len(positions),target_continuations=len(positions)*16,seconds=perf_counter()-started,
        world_counts=dict(Counter(p['world'] for p in positions))))
    write_json(out/'status.json',dict(status='prepared',phase='collection'))


def collect(cfg,out,workers=8):
    from civ5rl.engine.evaluation.executor import EvaluationRuntime
    from civ5rl.experiments.comparison_turns import TurnExecutor
    from civ5rl.experiments.comparison_match import match_task
    if (out/'result.json').exists() and read_json(out/'collection.json').get('complete'):
        archive_tree(out,'continuations')
        write_json(out/'status.json',dict(status='complete',phase='complete'))
        return
    manifest=read_json(out/'positions.json')
    runtime=EvaluationRuntime('cuda',workers,32,.001).resolve()
    runtime_path=out/'collection-runtime.json'
    if runtime_path.exists() and read_json(runtime_path)!=runtime.identity():
        raise ValueError('Collection runtime changed')
    write_json(runtime_path,runtime.identity())
    restore_tree(out,'continuations')
    state=read_json(out/'collection.json') if (out/'collection.json').exists() else dict(completed=0,seconds=0.)
    confirmation='confirmation' in cfg
    repetitions=manifest['repetitions'] if confirmation else range(4)
    state['target']=len(manifest['positions'])*16
    rows=[];tasks=[];started=perf_counter();base=state['seconds']
    from civ5rl.engine.runtime.telemetry import ResourceMonitor
    with ResourceMonitor(out/'telemetry') as monitor, readable_tree(out,'sources') as sources:
        monitor.mark('collection')
        for position in manifest['positions']:
            episode=position['episode'];case=read_json(sources/str(episode)/'case.json')
            variants=(0,position['selected_variant']) if confirmation else range(4)
            for variant in variants:
                for repetition in repetitions:
                    folder=out/'continuations'/f'{episode}-{position["key"][1]}'/f'{variant}-{repetition}'
                    branch=branch_spec(position,variant,repetition,sources/str(episode))
                    if (folder/'completed.json').exists():
                        if read_json(folder/'branch.json')!=branch or not (folder/'checked.json').exists():
                            raise ValueError('Saved continuation identity/check differs')
                        rows.append(read_json(folder/'completed.json'));continue
                    tasks.append((read_json(ROOT/cfg['collection_input']['run']/'match-config.json'),case,
                                  str(folder),str(ROOT/cfg['parent']),branch))
        write_json(out/'status.json',dict(status='collecting',phase='collection'))
        with TurnExecutor(str(ROOT/cfg['parent']),runtime) as executor:
            for folder in executor.run(match_task,tasks):
                rows.append(read_json(Path(folder)/'completed.json'))
                state.update(completed=len(rows),seconds=base+perf_counter()-started)
                if len(rows)%16==0:write_json(out/'collection.json',state)
    state.update(completed=len(rows),seconds=base+perf_counter()-started,complete=False)
    write_json(out/'collection.json',state)
    if len(rows)!=state['target']:raise ValueError('Collection incomplete')
    if confirmation:
        from civ5rl.experiments.turn_confirmation import summarize_confirmation
        summarize_confirmation(cfg,out,manifest,rows,state['seconds'])
        archive_tree(out,'continuations')
        state['complete']=True
        write_json(out/'collection.json',state)
        write_json(out/'status.json',dict(status='complete',phase='complete'))
        return
    summary={}
    for variant in range(4):
        items=[r for r in rows if r['turn_variant']==variant]
        summary[str(variant)]={k:sum(('D' if r['winner'] is None else 'W' if r['winner']==r['side'] else 'L')==k for r in items) for k in 'WDL'}
    indexed={(tuple(r['position_key']),r['turn_variant'],r['repetition']):r for r in rows}
    paired={}
    for variant in range(1,4):
        differences=[int(r['winner']==r['side'])-int(indexed[(key,0,repetition)]['winner']==r['side'])
            for (key,v,repetition),r in indexed.items() if v==variant]
        paired[str(variant)]=dict(mean_win_delta=sum(differences)/len(differences) if differences else None,
                                comparisons=len(differences))
    diversity=[]
    for position in manifest['positions']:
        episode,turn,_=position['key']
        for repetition in range(4):
            sequences=[]
            for variant in range(4):
                history=read_json(out/'continuations'/f'{episode}-{turn}'/f'{variant}-{repetition}'/'turn-history.json')
                sequences.append(digest([row['actions'][row['chosen']] for row in history]))
            diversity.append(len(set(sequences)))
    write_json(out/'result.json',dict(status='complete',variants=summary,positions=len(manifest['positions']),
        continuations=len(rows),seconds=state['seconds'],paired_vs_argmax=paired,
        unique_turn_sequences=dict(Counter(diversity)),note='Descriptive paired futures; no best-sample policy claim'))
    archive_tree(out,'continuations')
    state['complete']=True
    write_json(out/'collection.json',state)
    write_json(out/'status.json',dict(status='complete',phase='complete'))


def run(phase,cfg,out,workers=8):
    out=Path(out).resolve();configure(17,'cuda')
    with RunLock(out):
        if phase=='precheck':
            from civ5rl.reporting.turn_comparison import smoke
            return smoke(cfg,out)
        prepare(cfg,out)
        if phase in ('collect','resume'):collect(cfg,out,workers)
