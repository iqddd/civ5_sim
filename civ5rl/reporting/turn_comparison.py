"""Discardable whole-turn smoke; short horizon is never a scientific collection."""
from pathlib import Path
from time import perf_counter
from civ5rl.provenance.common import ROOT,read_json,write_json


def smoke(cfg,out):
    from tests.test_combat import scenario,unit
    from civ5rl.engine.evaluation import EvaluationRuntime,EvaluationExecutor
    from civ5rl.experiments.comparison_collection import ProposalExecutor
    from civ5rl.experiments.comparison_match import match_task
    from civ5rl.experiments.turn_comparison import branch_spec,position_candidates
    from civ5rl.engine.resources import ResourceMonitor
    from civ5rl.engine.turn_collection import TurnExecutor
    from torch.profiler import profile,ProfilerActivity
    out=Path(out)
    if any(p.name not in ('run.lock','derived') for p in out.iterdir()):raise ValueError('Use empty smoke output')
    config=read_json(ROOT/cfg['collection_input']['run']/'match-config.json')
    config={**config,'_smoke_rounds':2}
    spec=scenario(unit(x=2,y=2),unit(side=1,x=5,y=2),random_damage=True)
    spec['turn_reset_mode']='end_turn'
    case=dict(scenario=spec,armies=[['warrior'],['warrior']],learner_side=0,max_rounds=2,
        max_calls_per_turn=200,agent_seeds=[17,18],parent_world=0)
    runtime=EvaluationRuntime('cuda',2,8,.001).resolve();parent=str(ROOT/cfg['parent'])
    source=out/'source';started=perf_counter()
    with ProposalExecutor([parent,parent],runtime) as executor:
        list(executor.run(match_task,[(config,case,str(source),parent,None)]))
    positions=position_candidates(0,case,read_json(source/'history.json'),read_json(source/'replay.json'))
    if not positions:raise AssertionError('Smoke needs a nonempty reconstructed prefix')
    position=positions[0]
    tasks=[]
    for variant in range(4):
        for repetition in range(4):
            branch=branch_spec(position,variant,repetition,source)
            tasks.append((config,case,str(out/'branches'/f'{variant}-{repetition}'),parent,branch))
    with ResourceMonitor(out/'telemetry') as monitor,EvaluationExecutor(parent,runtime) as executor:
        monitor.mark('turn_smoke')
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as trace:
            folders=list(executor.run(match_task,tasks))
        trace.export_chrome_trace(str(out/'trace.json'))
        repeat=[(a,b,str(out/'repeated'/Path(c).name),d,e) for a,b,c,d,e in reversed(tasks[::4])]
        list(executor.run(match_task,repeat))
    with TurnExecutor(parent,runtime) as executor:
        prefixed=[(a,b,str(out/'prefixed'/Path(c).name),d,e) for a,b,c,d,e in tasks[::4]]
        list(executor.run(match_task,prefixed))
        assert executor.prefixes.prefills==1 and executor.prefixes.hits>=3
    for task in tasks[::4]:
        a=read_json(Path(task[2])/'turn-history.json');b=read_json(out/'repeated'/Path(task[2]).name/'turn-history.json')
        assert a==b
        assert a==read_json(out/'prefixed'/Path(task[2]).name/'turn-history.json')
    rows=[read_json(Path(folder)/'completed.json') for folder in folders]
    assert len(rows)==16 and all(r['prefix_verified'] for r in rows)
    result=dict(status='passed',discarded=True,smoke_rounds=2,continuations=16,
        task_order_reproducible=True,replay_verified=True,seconds=perf_counter()-started,
        top_kernels=sorted([dict(name=e.key,cuda_us=e.device_time_total,count=e.count)
            for e in trace.key_averages() if e.device_time_total],key=lambda r:r['cuda_us'],reverse=True)[:12])
    write_json(out/'precheck.json',result);print(result,flush=True)
    return result
