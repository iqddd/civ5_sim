"""Bounded pairwise policy fitting on complete recorded public histories."""
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter
import random
import torch
from torch.nn import functional as F
from civ5rl.core.agent import RLAgent
from civ5rl.core.context import context_from_spec
from civ5rl.core.model import batch_inputs
from civ5rl.core.observation import ObservationEncoder
from civ5rl.engine.evaluation import evaluation_precision
from civ5rl.engine.optimizer_transaction import require_finite_state
from civ5rl.provenance.common import read_json, write_json
from civ5rl.provenance.rng import capture_rng, restore_rng
from civ5sim import Action

HISTORY_BATCH = 8
ATTENTION_BACKEND = 'compact_fp32'


def training_runtime():
    return dict(attention_backend=ATTENTION_BACKEND,history_batch=HISTORY_BATCH,
                spatial_batch=32,bptt=128,compile_mode='default',cuda_graphs=False,
                coalesced_spatial_gradients=True,
                precision='mixed_bf16_spatial_fp32_gru_heads_loss')


def pair_loss(logits, better, worse, weight):
    return weight*F.softplus((logits[worse]-logits[better])/.5)


def categorical_kl(reference, logits, *, double=False):
    # Both inputs contain only the target's real legal actions (no padding).
    current=(logits/.5).log_softmax(-1)
    if double:
        reference,current=reference.double(),current.double()
    return (reference.exp()*(reference-current)).sum()


def atomic_torch_save(value,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(value,temp);temp.replace(path)


class Histories:
    def __init__(self,out,capacity=8,source_hashes=None):
        self.out=Path(out);self.cache=OrderedDict();self.capacity=capacity;self.source_hashes=source_hashes
    def get(self,episode):
        if episode not in self.cache:
            folder=self.out/'sources'/str(episode)
            if self.source_hashes is not None:
                from civ5rl.provenance.common import file_sha256
                expected=self.source_hashes[str(episode)]
                if any(file_sha256(folder/name)!=sha for name,sha in expected.items()):
                    raise ValueError('Recorded public history changed')
            case=read_json(folder/'case.json');rows=read_json(folder/'history.json')
            encoder=ObservationEncoder(context_from_spec(case,seed=case['agent_seeds'][case['learner_side']]),'C2P')
            encoded=[]
            for row in rows:
                legal=[Action(**a) for a in row['actions']]
                encoded.append(encoder.encode(row['observation'],legal))
                encoder.remember_action(legal[row['chosen']],row['observation'])
            self.cache[episode]=(encoded,rows)
        self.cache.move_to_end(episode)
        while len(self.cache)>self.capacity:self.cache.popitem(last=False)
        return self.cache[episode]


def target_logits(policy,history,index,*,gradient=False,block=32):
    """Whole current-weight prefix; only final <=128 decisions retain a graph."""
    device=next(policy.parameters()).device
    boundary=max(0,index+1-128) if gradient else index+1
    hidden=None
    with torch.no_grad():
        for start in range(0,boundary,block):
            part=history[start:min(boundary,start+block)]
            batch=batch_inputs(part,device,compact_units=policy.attention_backend=='compact_fp32')
            x,context=policy.encode_spatial(batch)
            h,hidden=policy.sequence_memory(context,[len(part)],hidden)
        if not gradient:
            target={k:v[-1:] for k,v in batch.items()}
            logits,_=policy.score_heads(target,x[-1:],h[-1:])
            return logits[0,:len(history[index].sources)]
    hidden=None if hidden is None else hidden.detach()
    for start in range(boundary,index+1,block):
        part=history[start:min(index+1,start+block)]
        batch=batch_inputs(part,device,compact_units=policy.attention_backend=='compact_fp32')
        x,context=policy.encode_spatial(batch)
        h,hidden=policy.sequence_memory(context,[len(part)],hidden)
    target={k:v[-1:] for k,v in batch.items()}
    logits,_=policy.score_heads(target,x[-1:],h[-1:])
    return logits[0,:len(history[index].sources)]


def configure_policy(checkpoint,device='cuda',mixed=True):
    policy=RLAgent.from_checkpoint(checkpoint,'cpu','argmax').policy.to(device).train()
    policy.enable_amp(mixed,spatial_batch_size=32 if mixed else 0)
    policy.enable_inverse_routing()
    policy.enable_action_routing()
    policy.enable_maximum_routing()
    policy.configure_execution(activation_checkpointing=True,attention_backend=ATTENTION_BACKEND)
    if str(device).startswith('cuda'):
        # Fuse eager kernels without introducing private CUDA Graph pools.
        policy.enable_compile(dynamic=True,mode='default',spatial_mode='default')
    for parameter in policy.value.parameters():parameter.requires_grad_(False)
    return policy


def _history_groups(histories,keys,size=HISTORY_BATCH):
    # Maps with different cell counts need not pad one another to the largest map.
    grouped={}
    for key in dict.fromkeys(map(tuple,keys)):
        history,_=histories.get(key[0])
        cells=len(history[0].cells)
        entries=grouped.setdefault(cells,[])
        entries.append((key,history))
        if len(entries)==size:
            yield entries
            grouped[cells]=[]
    for entries in grouped.values():
        if entries:yield entries


def _encode_histories(policy,sequences):
    device=next(policy.parameters()).device
    items=[item for sequence in sequences for item in sequence]
    inputs=batch_inputs(items,device,spatial_only=True,compact_units=policy.attention_backend=='compact_fp32')
    if torch.is_grad_enabled() and policy.spatial_batch_size:
        # Use the executor so cell padding and precision match the ordinary path.
        return policy.encode_spatial(inputs,coalesced=True)
    # FP32 full-KL has no mixed-padding setting. Keep its spatial activations
    # bounded too: an entire group of histories is not one attention matrix.
    encoded=[policy.encode_spatial({k:v[start:start+32] for k,v in inputs.items()})
             for start in range(0,len(items),32)]
    return tuple(torch.cat([row[i] for row in encoded],0) for i in (0,1))


def batched_target_logits(policy,entries,*,gradient=True):
    """Independent current-weight histories; detach each one's own BPTT boundary."""
    keys,histories=zip(*entries)
    indices=[key[1] for key in keys]
    boundaries=[max(0,i+1-128) if gradient else 0 for i in indices]
    hidden=None
    present=[i for i,boundary in enumerate(boundaries) if boundary]
    if present:
        with torch.no_grad():
            prefixes=[histories[i][:boundaries[i]] for i in present]
            _,context=_encode_histories(policy,prefixes)
            _,last=policy.sequence_memory(context,[len(p) for p in prefixes])
            lookup={i:last[:,j:j+1] for j,i in enumerate(present)}
            zero=torch.zeros_like(last[:,:1])
            hidden=torch.cat([lookup[i] if i in lookup else zero for i in range(len(entries))],1).detach()
    windows=[h[b:i+1] for h,b,i in zip(histories,boundaries,indices)]
    with torch.set_grad_enabled(gradient):
        x,context=_encode_histories(policy,windows)
        lengths=[len(w) for w in windows]
        h,_=policy.sequence_memory(context,lengths,hidden)
        ends=torch.tensor([sum(lengths[:i+1])-1 for i in range(len(lengths))],device=x.device)
        targets=[history[index] for history,index in zip(histories,indices)]
        batch=batch_inputs(targets,x.device)
        logits,_=policy.score_heads(batch,x[ends],h[ends])
        return [row[:len(target.sources)] for row,target in zip(logits,targets)]


def reference_distributions(policy,histories,anchors):
    grouped={}
    for episode,index in anchors:grouped.setdefault(episode,set()).add(index)
    result={}
    with torch.no_grad():
        for entries in _history_groups(histories,[(e,max(indices)) for e,indices in grouped.items()]):
            sequences=[history[:key[1]+1] for key,history in entries]
            x,context=_encode_histories(policy,sequences)
            lengths=[len(s) for s in sequences]
            h,_=policy.sequence_memory(context,lengths)
            targets=[];offset=0
            for (key,history),length in zip(entries,lengths):
                targets.extend(((key[0],i),history[i],offset+i) for i in sorted(grouped[key[0]]))
                offset+=length
            for start in range(0,len(targets),32):
                part=targets[start:start+32]
                selected=torch.tensor([p[2] for p in part],device=x.device)
                logits,_=policy.score_heads(batch_inputs([p[1] for p in part],x.device),x[selected],h[selected])
                logp=(logits/.5).log_softmax(-1).cpu()
                for row,(key,item,_) in zip(logp,part):result[key]=row[:len(item.sources)].clone()
    return result


def full_kl(policy,histories,anchors,references):
    policy.enable_amp(False)
    try:
        with evaluation_precision():
            current=reference_distributions(policy,histories,anchors)
        # FP64 subtraction/reduction matches the established full-KL contract.
        return sum(float((references[tuple(k)].double().exp()*(references[tuple(k)].double()-current[tuple(k)].double())).sum()) for k in anchors)/len(anchors)
    finally:policy.enable_amp(True,spatial_batch_size=32)


def prepare_references(parent,histories,anchors,path):
    from civ5rl.provenance.common import file_sha256
    from civ5sim.engine import digest
    import hashlib
    started=perf_counter()
    model_hash=hashlib.sha256()
    for name,value in parent.state_dict().items():
        model_hash.update(name.encode());model_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    sources={str(e):{name:file_sha256(histories.out/'sources'/str(e)/name) for name in ('case.json','history.json')} for e in sorted({k[0] for k in anchors})}
    identity=dict(parent=model_hash.hexdigest(),anchors=[list(k) for k in anchors],histories=sources,
                  implementation={p.relative_to(Path(__file__).parents[2]).as_posix():file_sha256(p)
                                  for p in (Path(__file__),*(Path(__file__).parents[1]/'core'/name
                                            for name in ('model.py','spatial.py','training_graphs.py')))},torch=torch.__version__,
                  gpu=torch.cuda.get_device_name() if next(parent.parameters()).is_cuda else None,
                  modes=['deterministic_fp32_no_tf32','mixed_bf16_spatial_fp32_gru'],temperature=.5,block=32,
                  attention_backend=parent.attention_backend,history_batch=HISTORY_BATCH)
    if path.exists():
        saved=torch.load(path,map_location='cpu',weights_only=False)
        if saved.get('identity')==identity:
            result=saved['distributions']
            result['_timing']=dict(cache_hit=True,load_seconds=perf_counter()-started,cold_seconds=None)
            return result
    result={}
    with evaluation_precision():
        for mixed in (False,True):
            parent.enable_amp(mixed,spatial_batch_size=32 if mixed else 0)
            result['mixed' if mixed else 'fp32']=reference_distributions(parent,histories,anchors)
    seconds=perf_counter()-started
    atomic_torch_save(dict(identity=identity,distributions=result),path)
    result['_timing']=dict(cache_hit=False,load_seconds=None,cold_seconds=seconds)
    return result


def logical_step(policy,optimizer,histories,pairs,anchor_keys,references):
    optimizer.zero_grad(set_to_none=True)
    pair_total=torch.zeros((),device=next(policy.parameters()).device)
    kl_total=torch.zeros_like(pair_total)
    pair_rows={};anchor_counts={}
    for row in pairs:pair_rows.setdefault((row['episode'],row['index']),[]).append(row)
    for key in map(tuple,anchor_keys):anchor_counts[key]=anchor_counts.get(key,0)+1
    keys=list(dict.fromkeys([*pair_rows,*anchor_counts]))
    for entries in _history_groups(histories,keys):
        logits=batched_target_logits(policy,entries)
        loss=pair_total.new_zeros(())
        for (key,_),z in zip(entries,logits):
            for row in pair_rows.get(key,()):
                value=pair_loss(z,row['better'],row['worse'],row['weight'])/len(pairs)
                pair_total+=value.detach();loss=loss+value
            if key in anchor_counts:
                value=categorical_kl(references[key].to(z.device),z)*anchor_counts[key]/len(anchor_keys)
                kl_total+=value.detach();loss=loss+value
        loss.backward()
    norm=torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad],.5,error_if_nonfinite=True)
    optimizer.step();require_finite_state(policy,optimizer)
    measured=torch.stack((pair_total,kl_total,norm)).detach().cpu().tolist()
    return dict(zip(('pair_loss','anchor_kl','gradient_norm'),measured))


def fit(cfg,out,dataset,identity,budget=None):
    if budget is not None:return fit_with_storage(cfg,out,dataset,identity,budget)
    from civ5rl.engine.phase_budget import PhaseBudget
    with PhaseBudget(out,"training",cfg["training_seconds"]) as budget:
        return fit_with_storage(cfg,out,dataset,identity,budget)


def fit_with_storage(cfg,out,dataset,identity,budget):
    from civ5rl.reporting.comparison_storage import restore_tree,archive_tree
    out=Path(out)
    state=read_json(out/'training.json') if (out/'training.json').exists() else {}
    if state.get('status') in ('complete','no_accepted_pairs'):
        return _fit(cfg,out,dataset,identity,budget)
    result=_fit(cfg,out,dataset,identity,budget)
    if (out/'sources').exists():archive_tree(out,'sources')
    state=read_json(out/'training.json');state.update(seconds=budget.elapsed(),overshoot_seconds=max(0,budget.elapsed()-cfg['training_seconds']))
    write_json(out/'training.json',state)
    return result


def publish_candidate(cfg,out,state,identity):
    from civ5rl.provenance.common import file_sha256
    original=torch.load(cfg['parents'][0],map_location='cpu',weights_only=False)
    changed=any(not torch.equal(value.cpu(),original['model'][name]) for name,value in state['model'].items())
    atomic_torch_save(dict(variant='C2P',model=state['model'],optimizer=state['optimizer'],state={k:v for k,v in state.items() if k not in ('model','optimizer')},config=original['config'],identity=identity),out/'candidate.pt')
    write_json(out/'training.json',dict(status='complete',training_steps=state['steps'],epoch=state['epoch'],cursor=state['cursor'],seconds=state['seconds'],overshoot_seconds=max(0,state['seconds']-1800),stop_reason=state['stop_reason'],changed=changed,attempted_final_kl=state['attempted_final_kl'],candidate_sha256=file_sha256(out/'candidate.pt')))
    return out/'candidate.pt' if changed else None


def _fit(cfg,out,dataset,identity,budget):
    out=Path(out);state_path=out/'training.pt'
    state=torch.load(state_path,map_location='cpu',weights_only=False) if state_path.exists() else dict(epoch=0,cursor=0,steps=0,seconds=0.,order=[],anchor_order=[],anchor_cursor=0)
    pairs=dataset['pairs'];anchors=dataset['anchors']
    from civ5sim.engine import digest
    identity=dict(passport=identity,dataset_hash=digest(dataset))
    if 'identity' in state and state['identity']!=identity:raise ValueError('Training dataset/identity changed')
    completed=read_json(out/'training.json') if (out/'training.json').exists() else {}
    if state.get('stop_reason') and completed.get('status')=='complete':
        from civ5rl.provenance.common import file_sha256
        if file_sha256(out/'candidate.pt')!=completed['candidate_sha256']:raise ValueError('Candidate checkpoint changed')
        return out/'candidate.pt' if completed['changed'] else None
    if state.get('stop_reason'):
        return publish_candidate(cfg,out,state,identity)
    if not pairs:
        write_json(out/'training.json',dict(status='no_accepted_pairs',training_steps=0,**{k:dataset.get(k) for k in ('selected_forks','completed_forks','accepted_pairs','rejections')}));return None
    from civ5rl.reporting.comparison_storage import restore_tree
    restore_tree(out,'sources')
    started=perf_counter();base_seconds=state['seconds']
    policy=configure_policy(cfg['parents'][0]);optimizer=torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=1e-5,betas=(.9,.999),eps=1e-5,weight_decay=0)
    shuffle=random.Random(cfg['seed']+101)
    histories=Histories(out,source_hashes=dataset.get('source_hashes'))
    # Policy still has parent weights. Reuse it for frozen references before
    # installing the student's resumed state; a second CUDA model is unnecessary.
    references=prepare_references(policy,histories,anchors,out/'references.pt')
    if 'model' in state:
        policy.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);shuffle.setstate(state['shuffle']);restore_rng(state['rng'])
    if 'rng' in state:restore_rng(state['rng'])
    def save(path):
        state.update(model=policy.state_dict(),optimizer=optimizer.state_dict(),shuffle=shuffle.getstate(),rng=capture_rng(),seconds=budget.elapsed(),identity=identity)
        atomic_torch_save(state,path)
        write_json(out/'training.json',{k:state[k] for k in ('epoch','cursor','steps','seconds')})
    if not (out/'verified.pt').exists():save(out/'verified.pt')
    stop=False
    while state['epoch']<2 and not stop:
        if state['cursor']>=len(pairs):
            kl=full_kl(policy,histories,anchors,references['fp32'])
            if kl>.015:stop=True;break
            state.update(epoch=state['epoch']+1,cursor=0,order=[],last_kl=kl)
            save(out/'verified.pt');save(state_path)
            continue
        if budget.expired():break
        if not state['order']:
            state['order']=list(range(len(pairs)));shuffle.shuffle(state['order'])
        selected=state['order'][state['cursor']:state['cursor']+32]
        anchor_batch=[]
        for _ in range(2*len(selected)):
            if state['anchor_cursor']>=len(state['anchor_order']):
                state['anchor_order']=list(range(len(anchors)));shuffle.shuffle(state['anchor_order']);state['anchor_cursor']=0
            anchor_batch.append(anchors[state['anchor_order'][state['anchor_cursor']]])
            state['anchor_cursor']+=1
        step_started=perf_counter()
        metrics=logical_step(policy,optimizer,histories,[pairs[i] for i in selected],anchor_batch,references['mixed'])
        state['cursor']+=len(selected);state['steps']+=1
        save(state_path)
        with (out/'steps.jsonl').open('a',encoding='utf-8') as f:f.write(__import__('json').dumps(dict(step=state['steps'],seconds=state['seconds'],step_seconds=perf_counter()-step_started,**metrics))+'\n')
    if not stop:
        kl=state['last_kl'] if state['cursor']==0 and 'last_kl' in state else full_kl(policy,histories,anchors,references['fp32'])
    elapsed=budget.elapsed()
    if stop or kl>.015:
        state=torch.load(out/'verified.pt',map_location='cpu',weights_only=False)
        policy.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);shuffle.setstate(state['shuffle']);restore_rng(state['rng'])
        state['stop_reason']='kl_limit'
    else:state['stop_reason']='epochs_completed' if state['epoch']==2 else 'time_limit'
    state['attempted_final_kl']=kl
    base_seconds=elapsed;started=perf_counter()
    save(state_path)
    return publish_candidate(cfg,out,state,identity)
