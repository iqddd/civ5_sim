"""Prepare and fit a policy from complete, recorded own-turn comparisons."""

from pathlib import Path
import shutil

from civ5rl.provenance.common import ROOT, read_json, write_json, configure, file_sha256
from civ5rl.provenance import checkpoint
from civ5rl.engine.runtime.control import RunLock
from civ5rl.engine.runtime.budget import PhaseBudget
from civ5rl.experiments.turn_fit_data import build_dataset, verify_inputs, history_inputs

KIND='rl01.turn_comparison_fit'


def load_config(path):
    cfg=read_json(path)
    if cfg['kind']!=KIND:raise ValueError('Expected whole-turn fit config')
    t=cfg['training']
    if t['positions_per_step']!=64 or t['max_epochs']!=2 or t['max_accepted_steps']!=16:
        raise ValueError('Expected 64 positions per step and two epochs / 16 steps')
    if t['anchors_per_step']!=660 or t['kl_limit']!=.015 or t['max_retries']!=2:
        raise ValueError('Unexpected whole-turn KL control')
    if cfg['training_seconds']!=3600 or t['initial_lr']!=1e-5 or cfg['seed']!=17:
        raise ValueError('Unexpected whole-turn training limits')
    return cfg


def prepare(cfg,out):
    verify_inputs(cfg)
    if file_sha256(ROOT/cfg['parents'][0])!=cfg['parent_sha256'][0]:raise ValueError('u171 changed')
    lineage=dict(input_sha256=cfg['input_sha256'],parent_sha256=cfg['parent_sha256'])
    if (out/'passport.json').exists():checkpoint.verify_execution_passport(out,cfg,lineage)
    else:
        identity=checkpoint.passport(out,cfg,lineage)
        write_json(out/'run.json',dict(kind=KIND,config=cfg,identity=identity))
    if not (out/'dataset.json').exists():
        dataset=build_dataset(cfg)
        write_json(out/'dataset.json',dataset)
    else:dataset=read_json(out/'dataset.json')
    if dataset['input_sha256']!=cfg['input_sha256']:raise ValueError('Prepared input changed')
    if not (out/'training.json').exists():write_json(out/'status.json',dict(status='prepared',phase='training'))
    return dataset


def evaluate(cfg,out):
    """Development is an explicit command and never a side effect of resume."""
    import torch
    from civ5rl.experiments.ppo_after_imitation import policy_hash
    from civ5rl.experiments.ppo_target_scale import evaluate as evaluator
    from civ5rl.engine.evaluation.executor import evaluation_options
    from civ5rl.reporting.evaluation_storage import readable_evaluations, archive_evaluations
    from civ5rl.reporting.evaluation_audit import load_eval_rows, clustered_paired_bootstrap
    training = read_json(out/'training.json')
    if training['status']!='complete':raise ValueError('Finish fit before evaluation')
    candidate=out/'candidate.pt'
    if file_sha256(candidate)!=training['candidate_sha256']:raise ValueError('Candidate changed')
    original = ROOT / cfg['parents'][0]
    if file_sha256(original)!=cfg['parent_sha256'][0]:raise ValueError('Parent changed')
    candidate_hash = policy_hash(torch.load(candidate, map_location='cpu', weights_only=False)['model'])
    parent_hash = policy_hash(torch.load(original, map_location='cpu', weights_only=False)['model'])
    if candidate_hash == parent_hash:
        skipped = {'status': 'skipped_identical_policy'}
        write_json(out/'result.json', dict(status='complete', development=skipped,
            parent_development=skipped, comparison=None, candidate_selection='last_accepted',
            additional={'status': 'disabled_by_protocol'}))
        write_json(out/'status.json', dict(status='complete', phase='complete', development=skipped['status']))
        return
    parent=out/'checkpoints/parent.pt';parent.parent.mkdir(exist_ok=True)
    if not parent.exists():shutil.copy2(original,parent)
    if file_sha256(parent)!=cfg['parent_sha256'][0]:raise ValueError('Parent changed')
    source=ROOT/cfg['source_run'];config=read_json(source/'run.json')['config']
    plan={'development_512':read_json(source/'manifest.json')['development_512']}
    write_json(out/'status.json',dict(status='evaluating',phase='development_512'))
    with readable_evaluations(out),evaluation_options(device='cuda',workers=8,max_batch=32,wait_seconds=.001):
        before=evaluator(config,plan,out,'checkpoints/parent.pt','development_512')
        after=evaluator(config,plan,out,'candidate.pt','development_512')
        def rows(record):
            return load_eval_rows(out/record['runs']['argmax']['cp']['directory'],512,plan=plan['development_512'])
        comparison=clustered_paired_bootstrap(rows(before),rows(after))
    write_json(out/'result.json',dict(status='complete',development=after,parent_development=before,
        comparison=comparison,candidate_selection='last_accepted',additional={'status':'disabled_by_protocol'}))
    archive_evaluations(out)
    write_json(out/'status.json',dict(status='complete',phase='complete'))


def run(phase,cfg,out,*,microbatch=2,compile_model=True,preparation_seconds=0.):
    out=Path(out).resolve();configure(cfg['seed'],'cuda')
    with RunLock(out):
        if phase=='precheck':
            from civ5rl.reporting.turn_fit import smoke
            return smoke(cfg,out,microbatch=microbatch,compile_model=compile_model)
        if phase=='evaluate':return evaluate(cfg,out)
        # Preparation is charged even when explicitly requested before train.
        with PhaseBudget(out,'training',cfg['training_seconds'],initial_seconds=preparation_seconds) as budget:
            dataset=prepare(cfg,out)
            if phase=='prepare':return
            runtime=dict(history_microbatch=microbatch,compile=compile_model,
                         precision='deterministic_cuda_fp32_no_amp_no_tf32')
            rp=out/'training-runtime.json'
            if rp.exists() and read_json(rp)!=runtime:raise ValueError('Resume runtime changed')
            write_json(rp,runtime)
            from civ5rl.engine.comparison.turn_inputs import TurnHistories
            from civ5rl.engine.comparison.turn_fit import train, publish, TurnTrainingState
            from civ5rl.engine.runtime.telemetry import ResourceMonitor
            write_json(out/'status.json',dict(status='training',phase='preparation'))
            with ResourceMonitor(out/'telemetry') as monitor:
                monitor.mark('preparation')
                try:
                    with history_inputs(cfg,dataset) as (sources,roots,branches):
                        histories=TurnHistories(out,sources,roots,branches,dataset['anchors'])
                        train(cfg,out,dataset,histories,budget,microbatch=microbatch,compile_model=compile_model,monitor=monitor)
                except TimeoutError:
                    import torch
                    from civ5rl.engine.runtime.transactions import atomic_torch_save
                    from civ5sim.engine import digest
                    identity=dict(dataset=digest(dataset),config=digest(cfg))
                    p=out/'training.pt'
                    payload=torch.load(p,map_location='cpu',weights_only=False) if p.exists() else TurnTrainingState().to_checkpoint_dict()
                    payload.update(stop_reason='time_limit',seconds=budget.elapsed(),identity=identity)
                    publish(cfg,out,payload,identity)
                    atomic_torch_save(payload,p)
        # Refresh final accounting after extraction cleanup and working saves.
        report=read_json(out/'training.json');report['seconds']=budget.elapsed()
        report['overshoot_seconds']=max(0.,report['seconds']-cfg['training_seconds'])
        write_json(out/'training.json',report)
        write_json(out/'status.json',dict(status='complete',phase='fit_complete',development='not_requested'))
