"""Play an existing public showcase manifest with a new frozen checkpoint."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import time
from pathlib import Path
from collections import Counter
from civ5rl.protocols.imitation import search_config
from civ5rl.engine.evaluation import EvaluationRuntime, EvaluationExecutor, evaluation_agent, evaluation_precision
from civ5rl.provenance.common import configure, read_json, write_json, file_sha256
from civ5sim import MatchConfig, run_match
from civ5sim.cp_agent import CPAgent
from civ5sim.engine import digest
from tools.view_combat import prepare_replay

def write_viewer_passport(out, checkpoint, sha):
    from civ5sim.combat_ai import Planner, LIBRARY_PATH
    planner = Planner()
    try:
        version = planner.version
    finally:
        planner.close()
    write_json(Path(out)/'passport.json', dict(schema=1, kind='civ5rl.showcase_viewer',
               port=version, checkpoint=str(checkpoint), checkpoint_sha256=sha,
               cp_library_sha256=file_sha256(LIBRARY_PATH),
               purpose='viewer provenance; not a training execution passport'))

def play(task):
    case,teacher,folder,checkpoint=task
    folder=Path(folder)
    learner=evaluation_agent(checkpoint,'argmax')
    cp=CPAgent(search_config(dict(teacher_search=teacher)))
    agents=[None,None]
    side=case['learner_side']
    agents[side],agents[1-side]=learner,cp
    started=time.perf_counter()
    try:
        match=run_match(MatchConfig(case['scenario']['map'],case['armies'],case['max_rounds'],
                                    seed=case['streams']['battle'],max_calls_per_turn=case['max_calls_per_turn']),
                        agents,scenario=case['scenario'],agent_seeds=case['agent_seeds'])
        match.save(folder)
    finally:
        cp.close()
    if match.result['status']!='completed':
        raise RuntimeError(f'Match {folder.name} failed: {match.result.get("error")}')
    verified=prepare_replay(folder/'replay.json')
    assert verified['verified'] and len(verified['frames'])==len(match.replay['steps'])+1
    wdl='D' if match.result['winner'] is None else ('W' if match.result['winner']==side else 'L')
    row=dict(case_id=case['case_id'],directory=folder.name,side=side,army_size=len(case['armies'][0]),
             roster=case['armies'][0],parent_world=case['parent_world'],map_hash=digest(case['scenario']['map']),
             outcome=wdl,winner=match.result['winner'],reason=match.result['reason'],
             rounds=match.result['completed_rounds'],commands=match.result['commands'],
             seconds=time.perf_counter()-started,replay_verified=True)
    write_json(folder/'metrics.json',row)
    write_json(folder/'checked.json',dict(replay_verified=True,commands=match.result['commands'],
                                         replay_sha256=file_sha256(folder/'replay.json')))
    return row

def play_cases(manifest_path, checkpoint, out):
    manifest_path, checkpoint, out = Path(manifest_path).resolve(), Path(checkpoint).resolve(), Path(out).resolve()
    source = read_json(manifest_path)
    if source.get('kind') != 'civ5rl.showcase' or source.get('closed_test_used') is not False:
        raise ValueError('Expected an explicitly public showcase manifest')
    cases = source['cases']
    if any(c.get('split') not in ('train', 'development', 'dev') for c in cases):
        raise ValueError('Only open cases are allowed')
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty output directory')
    sha = file_sha256(checkpoint)
    configure(source['seed'], 'cuda')
    out.mkdir(parents=True, exist_ok=True)
    write_viewer_passport(out, checkpoint, sha)
    manifest = dict(source, checkpoint=str(checkpoint), checkpoint_sha256=sha,
                    source_manifest=str(manifest_path), source_manifest_sha256=file_sha256(manifest_path),
                    selected_update=None)
    write_json(out/'manifest.json', manifest)
    tasks=[]
    for c in cases:
        name=f"battle-{c['case_id']+1:02d}-{len(c['armies'][0])}v{len(c['armies'][0])}-RL-{'A' if c['learner_side']==0 else 'B'}-map{c['parent_world']}"
        folder=out/name
        write_json(folder/'case.json', c)
        tasks.append((c,source['teacher_search'],str(folder),str(checkpoint)))
    runtime=EvaluationRuntime('cuda',8,32,.001).resolve()
    rows=[]; started=time.perf_counter()
    with evaluation_precision():
        executor=EvaluationExecutor(str(checkpoint),runtime)
        write_json(out/'runtime.json',runtime.identity())
        for row in executor.run(play,tasks):
            rows.append(row)
            print(f"{len(rows)}/{len(cases)} {row['directory']}: {row['outcome']}; verified",flush=True)
        inference=executor.statistics()
    assert len(rows)==len(cases) and all(r['replay_verified'] for r in rows)
    assert file_sha256(checkpoint)==sha
    wdl=Counter(r['outcome'] for r in rows)
    result=dict(schema=1,checkpoint=str(checkpoint),checkpoint_sha256=sha,mode='argmax',
                seconds=time.perf_counter()-started,summary=dict(matches=len(rows),**{k:wdl[k] for k in ('W','D','L')}),
                matches=sorted(rows,key=lambda r:r['case_id']),runtime=runtime.identity(),inference=inference,
                all_replays_verified=True,closed_test_used=False,
                viewer_summary=f"{checkpoint.parent.parent.name} / {checkpoint.stem}: argmax vs CP, W/D/L {wdl['W']}/{wdl['D']}/{wdl['L']}; same saved cases")
    write_json(out/'summary.json',result)
    return result
