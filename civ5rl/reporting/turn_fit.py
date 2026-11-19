"""Discardable FP32 correctness smoke on real turn histories, not a timing forecast."""

from pathlib import Path
from time import perf_counter

import torch

from civ5rl.provenance.common import ROOT,read_json,write_json
from civ5rl.experiments.turn_fit_data import build_dataset,history_inputs,read_json_lines
from civ5rl.engine.comparison.turn_inputs import TurnHistories
from civ5rl.engine.comparison.turn_fit import configure_policy,parent_references
from civ5rl.engine.comparison.turn_training import (
    control_kl, current_chosen, hidden_states_at, logical_step, pack_batch, preference_loss,
    prepare_anchor_subset, prepare_preferences,
)
from civ5rl.engine.runtime.transactions import AdamSnapshot,cpu_snapshot,atomic_torch_save


def top_cuda_kernels(trace,limit=12):
    """Aggregate device events, excluding inclusive CPU/ATen operator timings."""
    from collections import defaultdict
    grouped=defaultdict(lambda:dict(cuda_us=0.,count=0))
    for event in trace.events():
        if event.device_type==torch.autograd.DeviceType.CUDA:
            row=grouped[event.name];row['cuda_us']+=event.device_time_total;row['count']+=1
    return sorted([dict(name=name,**row) for name,row in grouped.items()],
                  key=lambda row:row['cuda_us'],reverse=True)[:limit]


def store_trace(out):
    """Keep large Kineto JSON in verified PPMd storage, outside measured timing."""
    import shutil
    from civ5rl.reporting.storage import pack_verified
    out=Path(out).resolve()
    if not (out/'trace.json').exists():return None
    raw=out/'trace'
    if raw.resolve().parent!=out or raw.exists():raise ValueError('Trace staging path is occupied')
    raw.mkdir();(out/'trace.json').rename(raw/'trace.json')
    pack_verified(raw,out/'trace.storage',token='turn-fit-trace',temporary_prefix='verify-turn-trace-',
                  changed_message='Trace changed during verification')
    if raw.resolve().parent!=out or raw.is_symlink():raise ValueError('Unsafe trace cleanup')
    shutil.rmtree(raw)
    return 'trace.storage'


def smoke(cfg,out,*,microbatch=2,compile_model=True):
    out=Path(out)
    if any(p.name not in ('run.lock','derived') for p in out.iterdir()):raise ValueError('Use a fresh smoke directory')
    started=perf_counter()
    manifest=read_json(ROOT/cfg['screening_run']/'positions.json')
    review={tuple(p['pos']):p for p in read_json_lines(ROOT/cfg['review'])}
    selected={tuple(p['position_key']) for p in read_json(ROOT/cfg['screening_run']/'derived/turn-review/candidates.json')['rows']}
    eligible=[]
    for i,p in enumerate(manifest['positions']):
        row=review[tuple(p['key'])]
        if len({r['wdl']=='W' for r in row['rows']})<2:continue
        cost=(p['history_index']+8)*sum(r['free'] for r in row['rows'])
        eligible.append((cost,i,tuple(p['key']) in selected))
    ids=[min(x for x in eligible if x[2])[1],
         min(x for x in eligible if not x[2] and manifest['positions'][x[1]]['history_index']>0)[1]]
    dataset=build_dataset(cfg,position_indices=ids,anchor_limit=32)
    write_json(out/'dataset.json',dataset)
    with history_inputs(cfg,dataset) as (sources,roots,branches):
        histories=TurnHistories(out,sources,roots,branches,dataset['anchors'])
        preferences = prepare_preferences(histories, dataset['positions'])
        chosen_keys = sorted(histories.targets)
        anchor_subset = prepare_anchor_subset(histories, dataset['anchors'], preferences.training_observations)
        histories.prepare([*chosen_keys, *map(tuple, dataset['anchors'])])
        batch = pack_batch(preferences.rows, 'cuda', histories=histories, branch_steps=preferences.branch_steps)
        policy=configure_policy(str(ROOT/cfg['parents'][0]),compile_model=compile_model)
        optimizer=torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=1e-5,
                                   betas=(.9,.999),eps=1e-5,weight_decay=0)
        parent, training_refs, control_refs = parent_references(
            policy, histories, dataset, batch.keys, out, cfg, branch_steps=preferences.branch_steps
        )
        reference=torch.stack([parent[k] for k in batch.keys])
        before = current_chosen(policy, histories, batch.keys, branch_steps=preferences.branch_steps)
        baseline=float(preference_loss(before,reference,batch.indices,batch.mask,batch.signs,batch.weights))
        initial_kl,_=control_kl(policy,histories,dataset['anchors'],control_refs,anchor_subset=anchor_subset)
        assert abs(initial_kl)<1e-6 and float((before-reference).abs().max())<1e-5
        value={k:v.clone() for k,v in policy.value.state_dict().items()}
        snapshot=AdamSnapshot(policy,optimizer)
        from civ5rl.engine.runtime.telemetry import ResourceMonitor
        from torch.profiler import profile,ProfilerActivity
        with ResourceMonitor(out/'telemetry') as monitor:
            monitor.mark('smoke_optimizer')
            with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as trace:
                metrics=logical_step(policy,optimizer,histories,batch,dataset['anchors'],reference,
                                     training_refs,microbatch=microbatch)
            trace.export_chrome_trace(str(out/'trace.json'))
        after = current_chosen(policy, histories, batch.keys, branch_steps=preferences.branch_steps)
        final=float(preference_loss(after,reference,batch.indices,batch.mask,batch.signs,batch.weights))
        scores=((after-reference)[batch.indices]*batch.mask).sum(-1)/batch.mask.sum(-1).clamp_min(1)
        margins=batch.signs*(scores[:,0]-scores[:,1])
        final_kl,learning=control_kl(policy,histories,dataset['anchors'],control_refs,anchor_subset=anchor_subset)
        assert final<baseline and bool((margins>0).any()),'Real winning executions must gain relative preference'
        assert all(torch.equal(v,policy.value.state_dict()[k]) for k,v in value.items())
        from civ5rl.engine.comparison.turn_fit import TurnTrainingState
        from civ5rl.provenance.rng import capture_rng,restore_rng
        resumed=TurnTrainingState(steps=1,cursor=64,mode='turn_preferences_step_kl',lr=1e-5,
            metadata=dict(steps_per_epoch=8,positions_per_step=64))
        accepted=cpu_snapshot({**resumed.to_checkpoint_dict(),
            'model':policy.state_dict(),'optimizer':optimizer.state_dict(),'rng':capture_rng()})
        atomic_torch_save(accepted,out/'smoke-resume.pt')
        import random,numpy as np
        random.random();np.random.random();torch.rand(());torch.rand((),device='cuda')
        snapshot.restore()
        assert not optimizer.state
        assert all(torch.equal(current,saved) for current,saved in snapshot.model)
        restored=capture_rng()
        assert restored['python_rng']==snapshot.rng['python_rng']
        assert np.array_equal(restored['numpy_rng'][1],snapshot.rng['numpy_rng'][1])
        assert torch.equal(restored['torch_rng'],snapshot.rng['torch_rng'])
        assert all(torch.equal(a,b) for a,b in zip(restored['cuda_rng'],snapshot.rng['cuda_rng']))
        loaded=torch.load(out/'smoke-resume.pt',map_location='cpu',weights_only=False)
        policy.load_state_dict(loaded['model']);optimizer.load_state_dict(loaded['optimizer'])
        restore_rng(loaded['rng'])
        state=TurnTrainingState.from_checkpoint_dict(loaded)
        assert (state.steps,state.cursor)==(resumed.steps,resumed.cursor)
        assert all(int(s['step'])==1 for s in optimizer.state.values())
        assert all(torch.equal(v.cpu(),loaded['model'][k]) for k,v in policy.state_dict().items())
        result=dict(status='passed',discarded=True,positions=ids,dataset=dataset['counts'],
            precision='deterministic_cuda_fp32_no_amp_no_tf32',initial_full_kl=initial_kl,final_full_kl=final_kl,
            pair_loss_before=baseline,pair_loss_after=final,winning_margins_positive=int((margins>0).sum()),
            value_weights_unchanged=True,rollback_restored=True,rng_restored=True,
            checkpoint_kept_cursor=True,loaded_adam_steps=1,nonempty_prefix_tested=True,
            seconds=perf_counter()-started,metrics=metrics,learning=learning,
            top_kernels=top_cuda_kernels(trace),
            performance_note='Correctness smoke; shared-device NVML can include a concurrently running game. No full-fit ETA inferred.')
        result['trace_archive']=store_trace(out)
        write_json(out/'precheck.json',result)
        print(result,flush=True)
        return result


def benchmark(args,parser):
    """Discard target-host full-size steps; report preparation, warmup, update and KL separately."""
    import random
    from statistics import mean
    from civ5rl.experiments.turn_fit import load_config
    from civ5rl.provenance.common import configure,seed_for
    from civ5rl.engine.runtime.telemetry import ResourceMonitor
    if args.steps<1 or args.warmup<0 or args.trace_steps<0 or args.history_microbatch<1:
        parser.error('steps/microbatch must be positive, warmup/trace-steps nonnegative')
    cfg=load_config(args.config);out=Path(args.out).resolve()
    if out==(ROOT/cfg['output']).resolve() or out.exists() and any(out.iterdir()):
        parser.error('Use a separate empty benchmark output')
    out.mkdir(parents=True,exist_ok=True);configure(cfg['seed'],'cuda');started=perf_counter()
    dataset=build_dataset(cfg)
    order=list(range(512));random.Random(seed_for(cfg['seed'],0,'positions',namespace=cfg['namespace'])).shuffle(order)
    workload={**dataset,'positions':[dataset['positions'][i] for i in order[:64]]}
    anchor_order=list(range(7000));random.Random(seed_for(cfg['seed'],0,'anchors',namespace=cfg['namespace'])).shuffle(anchor_order)
    anchors=[dataset['anchors'][i] for i in anchor_order[:660]]
    with history_inputs(cfg,workload) as (sources,roots,branches):
        histories=TurnHistories(out,sources,roots,branches,dataset['anchors'])
        preferences = prepare_preferences(histories, workload['positions'])
        chosen_keys = sorted(histories.targets)
        anchor_subset = prepare_anchor_subset(histories, dataset['anchors'], preferences.training_observations)
        histories.prepare([*chosen_keys, *map(tuple, dataset['anchors'])])
        batch = pack_batch(preferences.rows, 'cuda', histories=histories, branch_steps=preferences.branch_steps)
        policy=configure_policy(str(ROOT/cfg['parents'][0]),compile_model=args.compile)
        optimizer=torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=1e-5,
                                   betas=(.9,.999),eps=1e-5,weight_decay=0)
        parent, train_refs, control_refs = parent_references(
            policy, histories, workload, batch.keys, out, cfg, branch_steps=preferences.branch_steps
        )
        reference=torch.stack([parent[k] for k in batch.keys])
        needed_anchor_bp = [(ep, idx - 128) for ep, idx in set(map(tuple, anchors)) if idx >= 128]
        anchor_bp_h = hidden_states_at(policy, histories, needed_anchor_bp)
        anchor_prefixes = {(ep, idx): anchor_bp_h[(ep, idx - 128)] for ep, idx in set(map(tuple, anchors)) if idx >= 128}
        needed_branch_bp = [bk for pos in batch.positions for bk in pos.boundary_keys if bk is not None]
        branch_prefixes = hidden_states_at(policy, histories, needed_branch_bp, branch_steps=preferences.branch_steps)
        torch.cuda.synchronize();preparation=perf_counter()-started
        snapshot=AdamSnapshot(policy,optimizer);rows=[];warmup=[]
        def step():
            torch.cuda.synchronize();t=perf_counter()
            logical_step(
                policy,
                optimizer,
                histories,
                batch,
                anchors,
                reference,
                train_refs,
                microbatch=args.history_microbatch,
                anchor_prefixes=anchor_prefixes,
                branch_prefixes=branch_prefixes,
            )
            torch.cuda.synchronize();update=perf_counter()-t;t=perf_counter()
            kl, _ = control_kl(
                policy,
                histories,
                dataset['anchors'],
                control_refs,
                anchor_subset=anchor_subset,
                collect_prefixes=anchors,
                collected_prefixes={},
            )
            torch.cuda.synchronize()
            control=perf_counter()-t;t=perf_counter()
            from civ5rl.provenance.rng import capture_rng
            for _ in range(2):
                atomic_torch_save(cpu_snapshot(dict(model=policy.state_dict(),optimizer=optimizer.state_dict(),
                    rng=capture_rng())),out/'discarded-working.pt')
            return dict(step_seconds=update,full_kl_seconds=control,working_saves_seconds=perf_counter()-t,full_kl=kl)
        with ResourceMonitor(out/'telemetry') as monitor:
            monitor.mark('warmup')
            for _ in range(args.warmup):warmup.append(step());snapshot.restore()
            monitor.mark('measured_steps')
            for _ in range(args.steps):rows.append(step());snapshot.restore()
            kernels=[]
            if args.trace_steps:
                from torch.profiler import profile,ProfilerActivity
                monitor.mark('trace')
                with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as trace:
                    for _ in range(args.trace_steps):step();snapshot.restore()
                trace.export_chrome_trace(str(out/'trace.json'))
                kernels=top_cuda_kernels(trace,20)
        cycle=mean(r['step_seconds']+r['full_kl_seconds']+r['working_saves_seconds'] for r in rows)
        cold=sum(r['step_seconds']+r['full_kl_seconds']+r['working_saves_seconds'] for r in warmup)
        result=dict(status='complete',discarded=True,positions_per_step=64,anchors_per_step=660,control_anchors=7000,
            preparation_seconds=preparation,warmup=warmup,measured=rows,mean_cycle_seconds=cycle,
            forecast_16_accepted_seconds=preparation+cold+16*cycle,
            forecast_with_two_retries_every_step_seconds=preparation+cold+48*cycle,
            budget_seconds=cfg['training_seconds'],top_kernels=kernels,
            note='Fixed-shape discarded steps, two working saves per attempt. Forecast adds warmup conservatively; cleanup and actual KL rejection frequency remain uncertain.')
        result['trace_archive']=store_trace(out)
        write_json(out/'benchmark.json',result);print(result,flush=True)
        return result
