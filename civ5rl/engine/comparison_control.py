"""Transactional full-corpus updates with a fixed parent-distribution bound.

The checkpoint contains only a verified model and Adam, including while an
attempt is pending. Pending work owns a stable ID, LR and anchor slice. Resume
replays that ID from its saved RNG boundary rather than applying tentative
weights. Attempt history in the checkpoint is authoritative; JSONL views can
be rebuilt after a publication failure without duplicating accepted steps.
"""

import math
import random
from time import perf_counter

from civ5rl.engine.optimizer_transaction import AdamSnapshot
from civ5rl.provenance.rng import capture_rng, restore_rng


MODE = 'full_corpus_step_kl'


def initial_state(anchors, options, seed):
    order = list(range(len(anchors)))
    random.Random(seed).shuffle(order)
    return dict(mode=MODE, steps=0, attempts=0, rejected_attempts=0,
                epoch=0, cursor=0, anchor_cursor=0, anchor_order=order,
                lr=options['initial_lr'], pending=None, history=[],
                last_full_kl=0., attempted_final_kl=None, stop_reason=None, seconds=0.)


def run_updates(policy, optimizer, state, anchors, options, budget, save,
                update, measure):
    """Apply bounded updates; callbacks execute the existing loss and full KL.

    save() snapshots the current verified weights, optimizer and RNG. It is
    called before touching Adam and after the accept/restore decision. A
    deadline never interrupts an entered update or its required KL check.
    """
    if state.get('stop_reason'):
        return
    while state['steps'] < options['max_accepted_steps']:
        if budget.expired():
            if state['pending'] is not None and state['pending']['entered']:
                state['history'].append(dict(**state['pending'], accepted=False,
                                             outcome='interrupted_time_limit', seconds=0.))
            state['pending'] = None
            state['stop_reason'] = 'time_limit'
            break
        pending = state['pending']
        if pending is None:
            pending = state['pending'] = dict(attempt=state['attempts']+1,
                step=state['steps'] + 1, retry=0, lr=state['lr'],
                anchor_cursor=state['anchor_cursor'], entered=False)
        if not pending['entered']:
            state['attempts'] += 1
            pending['entered'] = True
        count = options['anchors_per_step']
        start = pending['anchor_cursor']
        selected = [anchors[i] for i in state['anchor_order'][start:start+count]]
        if len(selected) != count:
            raise ValueError('Not enough distinct anchors for the accepted-step limit')
        # A restored pending attempt already has its ID and RNG in the checkpoint.
        snapshot = AdamSnapshot(policy, optimizer)
        started = perf_counter()
        save()
        for group in optimizer.param_groups:
            group['lr'] = pending['lr']
        try:
            step_started = perf_counter()
            metrics = update(selected)
            step_seconds = perf_counter() - step_started
            kl_started = perf_counter()
            measured = measure()
            kl, learning = measured if isinstance(measured,tuple) else (measured,None)
            kl_seconds = perf_counter() - kl_started
            if not math.isfinite(kl):
                raise FloatingPointError('Non-finite full KL after Adam attempt')
        except BaseException:
            snapshot.restore()
            save()
            raise
        accepted = kl <= options['kl_limit']
        state['attempted_final_kl'] = kl
        rollback_started = perf_counter()
        if accepted:
            state['steps'] += 1
            state['anchor_cursor'] += count
            state['lr'] = pending['lr']
            state['last_full_kl'] = kl
        else:
            snapshot.restore()
            state['rejected_attempts'] += 1
        record = dict(**pending, accepted=accepted, outcome='accepted' if accepted else 'kl_rejected',
                      full_kl=kl, step_seconds=step_seconds, kl_seconds=kl_seconds,
                      rollback_seconds=0. if accepted else perf_counter()-rollback_started,
                      seconds=perf_counter()-started, **metrics)
        if accepted and learning is not None:
            record['learning'] = learning
            state['learning'] = learning
        state['history'].append(record)
        state['pending'] = None
        if not accepted:
            if pending['retry'] >= options['max_retries']:
                state['stop_reason'] = 'retry_limit'
            elif budget.expired():
                state['stop_reason'] = 'time_limit'
            else:
                # Reserve the retry ID only when entering it, after checking time.
                state['pending'] = dict(pending, attempt=state['attempts']+1,
                                        retry=pending['retry']+1, lr=pending['lr']/2,
                                        entered=False)
        save()
        if state['stop_reason']:
            break
    if state['steps'] >= options['max_accepted_steps']:
        state['stop_reason'] = 'accepted_step_limit'
    save()


def fit_full_corpus(cfg, out, dataset, identity, budget):
    from pathlib import Path
    import torch
    from civ5sim.engine import digest
    from civ5rl.provenance.common import read_json, write_json, seed_for, file_sha256
    from civ5rl.engine.comparison_training import (
        Histories, configure_policy, prepare_references, logical_step, full_kl,
        atomic_torch_save, publish_candidate,
    )
    from civ5rl.reporting.comparison_storage import restore_tree

    out = Path(out)
    options = cfg['training']
    anchors, pairs = dataset['anchors'], dataset['pairs']
    if len({tuple(key) for key in anchors}) != len(anchors):
        raise ValueError('Full-corpus control requires distinct anchors')
    if len(anchors) < options['anchors_per_step'] * options['max_accepted_steps']:
        raise ValueError('Anchor pool does not cover the accepted-step limit')
    identity = dict(passport=identity, dataset_hash=digest(dataset))
    path = out/'training.pt'
    state = (torch.load(path, map_location='cpu', weights_only=False) if path.exists() else
             initial_state(anchors, options, seed_for(cfg['seed'], 0, 'anchors', namespace=cfg['namespace'])))
    if state.get('identity', identity) != identity:
        raise ValueError('Training dataset/identity changed')
    if state.get('stop_reason'):
        if ((out/'candidate.pt').exists() and (out/'training.json').exists()
                and read_json(out/'training.json').get('status')=='complete'):
            report = read_json(out/'training.json')
            if file_sha256(out/'candidate.pt') != report['candidate_sha256']:
                raise ValueError('Candidate checkpoint changed')
            return out/'candidate.pt' if report['changed'] else None
        return publish_candidate(cfg,out,state,identity)
    restore_tree(out,'sources')
    head_only = options['mode']=='frozen_heads_step_kl'
    if head_only:
        from civ5rl.engine.comparison_heads import configure_heads,HeadFeatures
        policy = configure_heads(cfg['parents'][0])
    else:
        policy = configure_policy(cfg['parents'][0])
    optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],
        lr=options['initial_lr'],betas=(.9,.999),eps=1e-5,weight_decay=0)
    keys = [(p['episode'],p['index']) for p in pairs]+[tuple(k) for k in anchors]
    limits = {}
    for episode,index in keys:
        limits[episode] = max(limits.get(episode,0), index+1)
    histories = Histories(out,source_hashes=dataset.get('source_hashes'),limits=limits)
    histories.prepare(keys,device=next(policy.parameters()).device,budget=budget)
    features = None
    if head_only:
        features = HeadFeatures(policy,histories,dataset)
        try:
            features.prepare(cfg['parents'][0],budget=budget)
            _,baseline = features.measure()
            state.setdefault('baseline_learning',baseline)
            state.setdefault('learning',baseline)
            state['feature_parity'] = features.parity
        except TimeoutError:
            if not budget.expired():raise
        state['mode'] = options['mode']
    else:
        references = prepare_references(policy,histories,anchors,out/'references.pt',budget=budget)
    if 'model' in state:
        policy.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'])

    def jsonl(name, rows):
        import json
        target=out/name; temporary=target.with_suffix('.jsonl.tmp')
        temporary.write_text(''.join(json.dumps(row)+'\n' for row in rows),encoding='utf-8')
        temporary.replace(target)

    def save():
        state.update(model=policy.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng(),
                     seconds=budget.elapsed(),identity=identity,input_cache=histories.statistics())
        if features is not None:state['input_cache']['head_features']=features.statistics
        atomic_torch_save(state,path)
        cycle=0.; accepted=[]
        for row in state['history']:
            cycle+=row['seconds']
            if row['accepted']:
                accepted.append(dict(row,cycle_seconds=cycle));cycle=0.
        jsonl('attempts.jsonl',state['history'])
        jsonl('steps.jsonl',accepted)
        if features is not None:
            learning_rows=([dict(step=0,learning=state['baseline_learning'])] if 'baseline_learning' in state else [])
            learning_rows += [dict(step=row['step'],learning=row['learning']) for row in state['history'] if row['accepted']]
            jsonl('learning.jsonl',learning_rows)
        write_json(out/'training.json',{k:state[k] for k in (
            'mode','steps','attempts','rejected_attempts','lr','anchor_cursor','pending',
            'last_full_kl','seconds','stop_reason','input_cache')} | {'learning':state.get('learning')})

    def update(selected):
        if features is not None:return features.update(optimizer,pairs,selected)
        return logical_step(policy,optimizer,histories,pairs,selected,references['mixed'])

    run_updates(policy,optimizer,state,anchors,options,budget,save,update,
                features.measure if features is not None else lambda: full_kl(policy,histories,anchors,references['fp32']))
    frozen=torch.load(path,map_location='cpu',weights_only=False)
    return publish_candidate(cfg,out,frozen,identity)
