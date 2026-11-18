"""Offline action-value experiment; no simulator matches or policy updates.

Preparation precedes the soft training timer. A completed epoch is the only
selection boundary. The immutable candidate receipt precedes confirmation;
resumption may finish that calculation but cannot select another checkpoint.
"""
import random
import shutil
from pathlib import Path
from time import perf_counter
import numpy as np
import torch

from civ5rl.engine.action_value import (split_worlds, make_head, make_optimizer,
    predict, summarize, bootstrap, world_scores, fork_targets, public_action_vector)
from civ5rl.engine.comparison_training import atomic_torch_save
from civ5rl.engine.control import RunLock
from civ5rl.provenance import checkpoint
from civ5rl.provenance.common import ROOT, configure, read_json, write_json, file_sha256
from civ5rl.provenance.rng import capture_rng, restore_rng

KIND = 'rl01.action_value'
PROTOCOL = dict(head=[534,128,1],activation='gelu',precision='cuda_deterministic_fp32_no_amp_no_tf32',
    optimizer=dict(name='AdamW',lr=3e-4,betas=[.9,.999],eps=1e-8,weight_decay_weights=1e-4,
                   weight_decay_bias=0.,clip_norm=1.),
    shuffle_namespace='c2p_action_value_shuffle_v1',split_namespace='c2p_action_value_split_v1',
    world_counts=[33,12,12],evaluation_repetitions=[0,1,2,3],
    bootstrap=dict(samples=10000,seed=20261103),minimum_gain=.01)


def load_config(path):
    cfg = read_json(path)
    if cfg['kind'] != KIND or cfg['seed'] != 17:
        raise ValueError('Unknown action-value protocol')
    if cfg['epochs'] != 100 or cfg['batch_forks'] != 256 or cfg['training_seconds'] != 1800:
        raise ValueError('Unexpected first action-value experiment limits')
    if cfg['protocol']!=PROTOCOL:
        raise ValueError('Unexpected action-value scientific conditions')
    return cfg


def prepare(cfg, out):
    """Copy and verify input evidence, publishing the feature packet last."""
    from civ5rl.engine.action_value_features import FrozenActionFeatures
    from civ5rl.engine.comparison_inputs import Histories
    from civ5rl.reporting.comparison_storage import readable_tree
    source = ROOT/cfg['collection_input']['run']
    if source.resolve() == out.resolve():
        raise ValueError('Use a separate action-value output')
    for name, sha in cfg['collection_input']['sha256'].items():
        if file_sha256(source/name) != sha:
            raise ValueError(f'Collection input changed: {name}')
    checkpoint.verify_frozen_passport(source,sources=False)
    if file_sha256(ROOT/cfg['parent']) != cfg['parent_sha256']:
        raise ValueError('Frozen parent changed')
    plan = read_json(source/'manifest.json')
    split = split_worlds(plan['base']['worlds']['training'])
    if (out/'passport.json').exists():
        identity = checkpoint.verify_execution_passport(out, cfg, split)
    else:
        identity = checkpoint.passport(out, cfg, split)
        write_json(out/'run.json', dict(kind=KIND, config=cfg, identity=identity))
    if (out/'features.pt').exists() and (out/'data.json').exists():
        if file_sha256(out/'features.pt') != read_json(out/'data.json')['sha256']:
            raise ValueError('Action-value features changed')
        return
    started = perf_counter()
    write_json(out/'split.json', split)
    write_json(out/'status.json', dict(status='preparing', phase='preparation'))
    collection = read_json(source/'collection.json')
    if not collection.get('finalized') or len(collection['forks']) != 4096:
        raise ValueError('Require the completed 4096-fork collection')
    dataset = read_json(source/'dataset.json')
    target = out/'sources.storage'
    if not target.exists():
        shutil.copytree(source/'sources.storage', target)
    previous = Histories(ROOT/cfg['feature_cache_run'])
    limits = {}
    for episode, index in dataset['anchors']:
        limits[episode] = max(limits.get(episode, 0), index+1)
    histories = Histories(out, source_hashes=dataset['source_hashes'], limits=limits,
        cache_dir=previous.directory if previous.directory.exists() else None)
    with readable_tree(out, 'sources'):
        features = FrozenActionFeatures(histories, dataset, ROOT/cfg['parent']).prepare()
        cases, rows = {}, {}
        wanted_by_episode = {}
        for fork in collection['forks']:
            wanted_by_episode.setdefault(fork['episode'],set()).add(fork['index'])
        inputs, wins, counts, first, worlds, keys = [], [], [], [], [], []
        for fork in collection['forks']:
            episode, index = fork['episode'], fork['index']
            if fork['status'] != 'completed':
                raise ValueError('Incomplete fork')
            if episode not in cases:
                cases[episode] = read_json(out/'sources'/str(episode)/'case.json')
                raw = histories.raw(episode)
                wanted = wanted_by_episode[episode]
                rows[episode] = {i: r for i, r in enumerate(raw) if i in wanted}
            case = cases[episode]
            world = case['map_entry']['parent_seed']
            if world != case['parent_world'] or world not in sum(split.values(), []):
                raise ValueError('Untrusted source world')
            row = rows[episode][index]
            actions = [row['chosen'], row['alternative']]
            if [row['actions'][a] for a in actions] != fork['actions'] or actions[0] == actions[1]:
                raise ValueError('Fork public action order changed')
            _, location = features.locations[(episode, index)]
            record = features.records[episode]
            vectors = [public_action_vector(record,location,action) for action in actions]
            total, repeats, initial = fork_targets(fork)
            if fork['side'] != case['learner_side'] or total != [v['W'] for v in fork['wdl']]:
                raise ValueError('Outcome summary differs')
            inputs.append(torch.stack(vectors)); wins.append(total)
            counts.append(repeats); first.append(initial); worlds.append(world); keys.append([episode, index])
        packet = dict(inputs=torch.stack(inputs), wins=torch.tensor(wins, dtype=torch.float32),
            counts=torch.tensor(counts, dtype=torch.float32), first_wins=np.asarray(first),
            worlds=np.asarray(worlds), keys=keys)
        if (packet['inputs'].shape != (4096, 2, 534) or int(packet['counts'].sum()) != 38368
                or set(worlds)!=set(sum(split.values(),[]))):
            raise ValueError('Unexpected action-value corpus dimensions')
    atomic_torch_save(packet, out/'features.pt')
    write_json(out/'data.json', dict(sha256=file_sha256(out/'features.pt'), forks=4096,
        continuations=38368, preparation_seconds=perf_counter()-started, cache=features.statistics,
        split_counts={name: int(np.isin(packet['worlds'], ws).sum()) for name, ws in split.items()},
        parent_sha256=cfg['parent_sha256']))
    write_json(out/'status.json', dict(status='prepared', phase='training'))


def train(cfg, out, data):
    """Checkpoint at completed epochs; replay an interrupted epoch from its boundary.

    One epoch permutation upload and one loss readback replace per-batch host
    traffic. Confirmation features remain on CPU until candidate confirmation.
    """
    head = make_head()
    optimizer = make_optimizer(head)
    shuffle = random.Random('c2p_action_value_shuffle_v1|17')
    path = out/'training.pt'
    if path.exists():
        state = torch.load(path, map_location='cpu', weights_only=False)
        head.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'], shuffle)
    else:
        state = dict(epoch=0, steps=0, cursor=0, order=None, seconds=0., best=None, history=[], complete=False,
                     epoch_started_seconds=None)
    if state['complete']:
        return
    split = read_json(out/'split.json')
    training = np.flatnonzero(np.isin(data['worlds'], split['training'])).tolist()
    selection = np.flatnonzero(np.isin(data['worlds'], split['selection']))
    x=data['inputs'][training].cuda()
    n=data['counts'][training].cuda()
    y=(data['wins'][training]/data['counts'][training]).cuda()
    x_sel=data['inputs'][selection].cuda()
    started, base = perf_counter(), state['seconds']
    def save():
        state.update(model=head.state_dict(), optimizer=optimizer.state_dict(), rng=capture_rng(shuffle),
                     seconds=base+perf_counter()-started)
        atomic_torch_save(state, path)
        write_json(out/'training.json', {k:state[k] for k in ('epoch','steps','seconds','best','history','complete')})
    if not path.exists():save()
    while state['epoch'] < cfg['epochs']:
        if state['cursor'] == 0 and base+perf_counter()-started >= cfg['training_seconds']:
            break
        if state['order'] is None:
            # Both timestamps use cumulative active training time, so an epoch
            # spans resumed processes without counting downtime/preparation.
            state['epoch_started_seconds']=base+perf_counter()-started
            state['order'] = list(range(len(training))); shuffle.shuffle(state['order'])
        order=torch.tensor(state['order'],device='cuda')
        epoch_loss=torch.zeros((),dtype=torch.float64,device='cuda')
        epoch_count=torch.zeros_like(epoch_loss)
        write_json(out/'status.json', dict(status='training', phase='training'))
        for start in range(state['cursor'], len(training), cfg['batch_forks']):
            ids = order[start:start+cfg['batch_forks']]
            optimizer.zero_grad(set_to_none=True)
            logits = head(x[ids].flatten(0, 1)).reshape(-1, 2)
            counts=n[ids]
            count=counts.sum()
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y[ids],reduction='none')*counts).sum()/count
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
            torch._assert_async(torch.isfinite(norm), 'Non-finite action-value gradient')
            optimizer.step()
            epoch_loss+=loss.detach().double()*count
            epoch_count+=count
            state['steps'] += 1; state['cursor'] = min(start+cfg['batch_forks'], len(training))
        logits = predict(head, x_sel)
        metrics = summarize(logits, data['first_wins'][selection], data['worlds'][selection])
        state['epoch'] += 1
        if state['best'] is None or metrics['G'] > state['best']['G']:
            atomic_torch_save(dict(model=head.state_dict(), epoch=state['epoch'], selection=metrics), out/'best.pt')
            state['best'] = dict(epoch=state['epoch'], G=metrics['G'])
        state['order'] = None; state['cursor'] = 0
        state['history'].append(dict(epoch=state['epoch'],
            seconds=base+perf_counter()-started-state['epoch_started_seconds'], selection=metrics,
            training_bce=float(epoch_loss/epoch_count)))
        state['epoch_started_seconds']=None
        state['complete']=(state['epoch']>=cfg['epochs'] or base+perf_counter()-started>=cfg['training_seconds'])
        save()
        print(f"Epoch {state['epoch']}/{cfg['epochs']} selection G={metrics['G']:+.4f}; elapsed={state['seconds']:.1f}s", flush=True)
        if state['epoch'] == 1:
            print(f"First-epoch forecast: {state['history'][0]['seconds']*cfg['epochs']:.1f}s for {cfg['epochs']} epochs; soft limit={cfg['training_seconds']}s", flush=True)
        if state['complete']:break
    if state['best'] is None:
        raise RuntimeError('No completed epoch to select')
    if not state['complete']:
        state['complete'] = True
        save()
    freeze_candidate(out, state['best'])


def freeze_candidate(out, best):
    """Publish the immutable selection receipt before any confirmation metric."""
    path = out/'candidate.pt'
    if (out/'selected.json').exists():
        receipt = read_json(out/'selected.json')
        if file_sha256(path) != receipt['sha256'] or receipt['epoch'] != best['epoch']:
            raise ValueError('Selected candidate changed')
        return receipt
    shutil.copy2(out/'best.pt', path)
    receipt = dict(**best, sha256=file_sha256(path))
    write_json(out/'selected.json', receipt)
    return receipt


def evaluate(cfg, out, data):
    state = torch.load(out/'training.pt', map_location='cpu', weights_only=False)
    if not state['complete']:
        raise ValueError('Training must finish before confirmation')
    receipt = freeze_candidate(out, state['best'])
    if (out/'result.json').exists():
        return read_json(out/'result.json')
    started = perf_counter()
    write_json(out/'status.json', dict(status='evaluating', phase='confirmation'))
    head = make_head()
    head.load_state_dict(torch.load(out/'candidate.pt', map_location='cpu', weights_only=False)['model'])
    ids = np.flatnonzero(np.isin(data['worlds'], read_json(out/'split.json')['confirmation']))
    logits = predict(head, data['inputs'][ids].cuda())
    wins, worlds = data['first_wins'][ids], data['worlds'][ids]
    scores=world_scores(logits,wins,worlds)
    metrics = summarize(logits, wins, worlds,scores=scores)
    result = dict(status='complete', selected=receipt, confirmation=metrics,
        bootstrap=bootstrap(scores),
        confirmation_seconds=perf_counter()-started, preparation_seconds=read_json(out/'data.json')['preparation_seconds'],
        training_seconds=state['seconds'])
    # FP64 export agrees with diagnostic probabilities; policy ranking continues
    # to use logits even when sigmoid saturates at the numerical endpoints.
    probs=torch.as_tensor(logits,dtype=torch.float64).sigmoid().numpy()
    atomic_torch_save(dict(logits=logits,probs=probs,delta_prob=probs[:,1]-probs[:,0],
        chosen_alternative=logits[:,1]>logits[:,0],
        keys=[data['keys'][i] for i in ids]), out/'confirmation-predictions.pt')
    write_json(out/'result.json', result)
    write_json(out/'status.json', dict(status='complete', phase='complete'))
    return result


def run(phase, cfg, out):
    out = Path(out).resolve()
    configure(cfg['seed'], 'cuda')
    if not torch.cuda.is_available():
        raise RuntimeError('Action-value fitting requires CUDA')
    with RunLock(out):
        if phase == 'precheck':
            from civ5rl.reporting.action_value import smoke
            return smoke(cfg, out)
        prepare(cfg, out)
        if phase == 'prepare':return
        data = torch.load(out/'features.pt', map_location='cpu', weights_only=False)
        # Cache preparation must not affect the new head's initialization.
        configure(cfg['seed'], 'cuda')
        if phase in ('train', 'resume'):train(cfg, out, data)
        if phase in ('evaluate', 'resume'):evaluate(cfg, out, data)
        from civ5rl.reporting.action_value import report
        report(out)
