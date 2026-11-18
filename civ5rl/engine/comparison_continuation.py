"""Explicit continuation of a completed head fit, preserving its optimizer clock.

The predecessor is frozen evidence, not an executable resume target. Its verified
state is imported once under the new passport; all cumulative counters, shuffle
order, diagnostics and RNG survive. Derived caches can be discarded and rebuilt.
"""
from pathlib import Path
import math
import shutil
import torch

from civ5rl.provenance.common import ROOT,read_json,write_json,file_sha256
from civ5rl.provenance.checkpoint import verify_frozen_passport


PINNED = ('training.pt','resolved-config.json','dataset.json','phase-budgets/training.json',
          'manifest.json','match-config.json')


def continuation_seconds(cfg):
    spec=cfg.get('fit_continuation')
    if not spec:return 0.
    source=ROOT/spec['run']
    return max(read_json(source/'phase-budgets/training.json')['seconds'],
               read_json(source/'training.json')['seconds'])


def validate_continuation(cfg):
    spec=cfg['fit_continuation'];source=ROOT/spec['run']
    if source.resolve()==(ROOT/cfg['output']).resolve():raise ValueError('Use a separate continuation directory')
    if not set(PINNED).issubset(spec['sha256']):raise ValueError('Incomplete fit continuation lineage')
    for name in PINNED:
        sha=spec['sha256'][name]
        if file_sha256(source/name)!=sha:raise ValueError(f'Fit continuation source changed: {name}')
    verify_frozen_passport(source,sources=False)
    old=read_json(source/'resolved-config.json')
    expected={**old,'output':cfg['output'],'fit_continuation':spec,
              'training':{**old['training'],'max_accepted_steps':100,'cycle_anchors':True}}
    if cfg!=expected or old['training']['mode']!='frozen_heads_step_kl' or old['training']['max_accepted_steps']!=10:
        raise ValueError('Continuation changes scientific conditions beyond its limits')
    state=torch.load(source/'training.pt',map_location='cpu',weights_only=False)
    report=read_json(source/'training.json')
    if (read_json(source/'status.json')['status']!='complete' or report['status']!='complete'
            or state['stop_reason']!='accepted_step_limit' or state['steps']!=10
            or state['attempts']!=10 or state['anchor_cursor']!=6600 or state['pending'] is not None
            or state['lr']!=1e-5 or len(state['history'])!=10):
        raise ValueError('Expected the verified ten-step head fit')
    if sorted(state['anchor_order'])!=list(range(12096)):
        raise ValueError('Invalid inherited anchor permutation')
    adam=state['optimizer']
    if (len(adam['param_groups'])!=1 or len(adam['param_groups'][0]['params'])!=5
            or adam['param_groups'][0]['lr']!=state['lr'] or len(adam['state'])!=5
            or sum(v['exp_avg'].numel() for v in adam['state'].values())!=68609
            or any(int(v['step'])!=10 for v in adam['state'].values())):
        raise ValueError('Invalid inherited head Adam')
    candidate=torch.load(source/'candidate.pt',map_location='cpu',weights_only=False)
    parent=torch.load(ROOT/cfg['parents'][0],map_location='cpu',weights_only=False)
    if any(not torch.equal(value,candidate['model'][name]) for name,value in state['model'].items()):
        raise ValueError('Source candidate differs from its accepted checkpoint')
    if any(not torch.equal(value,parent['model'][name]) for name,value in state['model'].items()
           if not name.startswith(('action.','attack_preview.'))):
        raise ValueError('Source fit changed frozen representations')
    spent=max(continuation_seconds(cfg),state['seconds'])
    if not math.isfinite(spent) or not 0<=spent<cfg['training_seconds']:
        raise ValueError('No active budget remains for continuation')
    if report['candidate_sha256']!=file_sha256(source/'candidate.pt'):
        raise ValueError('Source candidate report differs')
    return state


def inherited_state(cfg,identity):
    state=validate_continuation(cfg)
    if state['identity']['dataset_hash']!=identity['dataset_hash']:
        raise ValueError('Continuation dataset changed')
    state.update(identity=identity,stop_reason=None,seconds=continuation_seconds(cfg),
                 inherited_seconds=continuation_seconds(cfg),inherited_steps=state['steps'])
    return state


def reuse_parent_development(cfg,out,config,plan):
    """Import only a cache entry whose ordinary evaluator key still matches.

    The key includes policy, cases/seeds, runtime/GPU, selection and evaluator
    implementation/binaries. No compatible key means an explicit error, never
    a fallback GPU evaluation. The original archive is restored after reading.
    """
    from civ5rl.engine.control import RunLock
    from civ5rl.engine.evaluation import EvaluationRuntime,evaluation_precision
    from civ5rl.experiments.ppo_target_scale import evaluation_key
    from civ5rl.reporting.evaluation_storage import readable_evaluations
    from civ5rl.deploy.archive import _inventory
    source=ROOT/cfg['fit_continuation']['run'];out=Path(out)
    if read_json(source/'match-config.json')!=config or read_json(source/'manifest.json')!=plan:
        raise ValueError('Parent evaluation cases/rules changed')
    with evaluation_precision():
        runtime=EvaluationRuntime('cuda',8,32,.001).resolve().identity()
        _,key=evaluation_key(config,plan,out,'checkpoints/parent.pt','development_512',runtime)
    saved=read_json(source/'evaluations.json')
    if key not in saved:raise ValueError('Parent Development evaluator/runtime/GPU is incompatible')
    parent=saved[key]
    if parent['checkpoint']!='checkpoints/parent.pt' or parent['runtime']!=runtime:
        raise ValueError('Parent evaluation policy/runtime changed')
    directory=Path(parent['runs']['argmax']['cp']['directory'])
    if directory.is_absolute() or '..' in directory.parts or directory.parts[0]!='evaluation':
        raise ValueError('Unsafe parent evaluation directory')
    receipt=out/'reused-parent.json'
    if not receipt.exists():
        with RunLock(source),readable_evaluations(source):
            shutil.copytree(source/directory,out/directory,dirs_exist_ok=True)
            if _inventory(source/directory)!=_inventory(out/directory):
                raise ValueError('Reused parent evaluation inventory differs')
        write_json(receipt,dict(source=cfg['fit_continuation'],cache_key=key))
    elif read_json(receipt)!=dict(source=cfg['fit_continuation'],cache_key=key):
        raise ValueError('Parent evaluation reuse identity changed')
    cache=read_json(out/'evaluations.json') if (out/'evaluations.json').exists() else {}
    cache[key]=parent;write_json(out/'evaluations.json',cache)
    return parent
