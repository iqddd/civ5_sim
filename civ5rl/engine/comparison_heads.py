"""Frozen-history FP32 inputs for action-head-only preference fitting.

Cache packets keep cell embeddings and one hidden vector per position, rather
than expanding either across legal actions. Packed action tensors retain every
legal command. CPU mmap packets bound RAM; only a microbatch is copied to CUDA.
Packets are derived data: their source/runtime identity and SHA-256 are checked
before reuse. No output of the trainable preview projection is cached.
"""
from collections import defaultdict
from pathlib import Path
from time import perf_counter
import torch
from torch.nn import functional as F

from civ5rl.core.model import batch_inputs
from civ5rl.engine.comparison_training import (
    _encode_histories, atomic_torch_save, configure_policy, categorical_kl,
)
from civ5rl.engine.optimizer_transaction import finite_state_flags
from civ5rl.provenance.common import file_sha256, read_json, write_json
from civ5sim.engine import digest

MODE = 'frozen_heads_step_kl'


def configure_heads(checkpoint, device='cuda'):
    policy = configure_policy(checkpoint, device=device, mixed=False)
    for name, parameter in policy.named_parameters():
        parameter.requires_grad_(name.startswith(('action.', 'attack_preview.')))
    return policy


def learning_metrics(pairs, original, baseline, current, anchors):
    """Summarize fixed-position post-update logits; ties are not preferences.

Each entry is (argmax, pair_margin, weighted_softplus). Pair margins/losses are
stored per pair, not per position, so repeated targets retain multiplicity.
Anchor changes describe spread, not correctness on those positions.
"""
    groups = {}
    for label, indices in (
        ('all', list(range(len(pairs)))),
        ('alternative', [i for i,p in enumerate(pairs) if p['better'] != original[(p['episode'],p['index'])]]),
        ('parent_confirmation', [i for i,p in enumerate(pairs) if p['better'] == original[(p['episode'],p['index'])]]),
    ):
        rows = [(baseline['pairs'][i], current['pairs'][i]) for i in indices]
        n = len(rows)
        groups[label] = dict(count=n, pair_loss=sum(b[2] for _,b in rows)/n if n else 0.,
            preferred=sum(b[1]>0 for _,b in rows), ties=sum(b[1]==0 for _,b in rows),
            better_argmax=sum(b[0]==pairs[i]['better'] for i,(_,b) in zip(indices,rows)),
            preference_gained=sum(a[1]<=0<b[1] for a,b in rows),
            preference_lost=sum(a[1]>0>=b[1] for a,b in rows),
            argmax_gained=sum(a[0]!=pairs[i]['better'] and b[0]==pairs[i]['better'] for i,(a,b) in zip(indices,rows)),
            argmax_lost=sum(a[0]==pairs[i]['better'] and b[0]!=pairs[i]['better'] for i,(a,b) in zip(indices,rows)))
    pair_keys = {(p['episode'],p['index']) for p in pairs}
    changes = {}
    for label, keys in (('all',list(map(tuple,anchors))),
                        ('outside_pairs',[tuple(k) for k in anchors if tuple(k) not in pair_keys])):
        count = sum(baseline['choices'][k] != current['choices'][k] for k in keys)
        changes[label] = dict(count=len(keys), changed=count, fraction=count/len(keys) if keys else 0.)
    return dict(pairs=groups, anchors=changes)


class HeadFeatures:
    def __init__(self, policy, histories, dataset):
        self.policy, self.histories, self.dataset = policy, histories, dataset
        self.device = next(policy.parameters()).device
        self.directory = histories.directory.parent/'head-features-fp32'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.records, self.locations, self.original = {}, {}, {}
        self.keys = sorted({(p['episode'],p['index']) for p in dataset['pairs']} |
                           set(map(tuple,dataset['anchors'])))
        self.statistics = dict(positions=len(self.keys), cache_hits=0, cache_builds=0,
                               bytes=0, precision='deterministic_fp32_no_tf32')
        self.baseline = None

    @torch.no_grad()
    def prepare(self, checkpoint, budget=None):
        grouped = defaultdict(list)
        for episode,index in self.keys: grouped[episode].append(index)
        common = dict(parent=file_sha256(checkpoint), implementation=file_sha256(__file__),
            encoder=self.histories.encoder_identity, torch=str(torch.__version__),
            gpu=torch.cuda.get_device_name(self.device) if self.device.type=='cuda' else 'cpu',
            attention_backend=self.policy.attention_backend, precision='fp32_no_amp_no_tf32',
            temperature=.5, spatial_block=32, history_batch=1)
        started = perf_counter()
        for episode, indices in grouped.items():
            identity = dict(common=common, source=self.histories.source_identity(episode), indices=indices)
            path = self.directory/f'{episode}-{digest(identity)}.pt'
            receipt = path.with_suffix('.json')
            if path.exists() and receipt.exists():
                saved = read_json(receipt)
                if saved['identity'] != identity or saved['sha256'] != file_sha256(path):
                    raise ValueError('Frozen-head feature cache changed')
                self.statistics['cache_hits'] += 1
            else:
                history,_ = self.histories.get(episode)
                sequence = history[:max(indices)+1]
                x, context = _encode_histories(self.policy,[sequence])
                h,_ = self.policy.sequence_memory(context,[len(sequence)])
                selected = torch.tensor(indices,device=self.device)
                items = [history[i] for i in indices]
                counts = [len(item.sources) for item in items]
                packed = {name:torch.cat([torch.from_numpy(getattr(item,name)) for item in items])
                          for name in ('sources','targets','action_features','attack_preview')}
                offsets = [0]
                for count in counts: offsets.append(offsets[-1]+count)
                raw = self.histories.raw(episode)
                atomic_torch_save(dict(identity=identity, x=x[selected], h=h[selected],
                    actions=packed, offsets=offsets, chosen=[raw[i]['chosen'] for i in indices]),path)
                write_json(receipt,dict(identity=identity,sha256=file_sha256(path)))
                self.statistics['cache_builds'] += 1
            record = torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            if record['identity'] != identity: raise ValueError('Frozen-head cache identity mismatch')
            self.records[episode] = record
            self.statistics['bytes'] += path.stat().st_size
            for row,index in enumerate(indices):
                self.locations[(episode,index)] = (episode,row)
                self.original[(episode,index)] = record['chosen'][row]
            if budget is not None and budget.expired():
                # A partial cache is reusable, but an incomplete run cannot train.
                raise TimeoutError('Training budget expired during frozen-head feature preparation')
        self.statistics['preparation_seconds'] = perf_counter()-started
        self.references = {}
        for keys, logits in self.logits(self.keys):
            logp = (logits/.5).log_softmax(-1).cpu()
            for row,key in enumerate(keys):
                count = self.count(key)
                self.references[key] = logp[row,:count].clone()
        _,self.baseline = self.measure(raw=True)
        self.parity = self.check_parity()
        print(f"Frozen FP32 features: {len(self.keys)} positions, {self.statistics['bytes']/1024**3:.2f} GiB",flush=True)

    def count(self,key):
        episode,row=self.locations[tuple(key)];offsets=self.records[episode]['offsets']
        return offsets[row+1]-offsets[row]

    def groups(self, keys):
        groups = defaultdict(list)
        for key in dict.fromkeys(map(tuple,keys)):
            episode,row = self.locations[key]
            groups[self.records[episode]['x'].shape[1]].append(key)
        for keys in groups.values():
            batch=[];width=0
            for key in keys:
                count=self.count(key)
                if batch and (len(batch)>=32 or max(width,count)*(len(batch)+1)>8192):
                    yield batch;batch=[];width=0
                batch.append(key);width=max(width,count)
            if batch:yield batch

    def inputs(self,keys):
        width=max(self.count(key) for key in keys);b=len(keys)
        batch=dict(sources=torch.full((b,width),-1,dtype=torch.long),
                   targets=torch.full((b,width),-1,dtype=torch.long),
                   action_features=torch.zeros(b,width,17),attack_preview=torch.zeros(b,width,5),
                   action_mask=torch.zeros(b,width,dtype=torch.bool))
        xs=[];hs=[]
        for i,key in enumerate(keys):
            episode,row=self.locations[key];record=self.records[episode]
            start,stop=record['offsets'][row:row+2];count=stop-start
            for name in record['actions']:batch[name][i,:count].copy_(record['actions'][name][start:stop])
            batch['action_mask'][i,:count]=True
            xs.append(record['x'][row]);hs.append(record['h'][row])
        return ({k:v.to(self.device) for k,v in batch.items()},
                torch.stack(xs).to(self.device),torch.stack(hs).to(self.device))

    def logits(self,keys):
        for group in self.groups(keys):
            batch,x,h=self.inputs(group)
            logits,_=self.policy.score_heads(batch,x,h)
            yield group,logits

    def parent_batch(self,keys,width):
        result=torch.zeros(len(keys),width,device=self.device)
        for row,key in enumerate(keys):
            ref=self.references[key]
            result[row,:len(ref)].copy_(ref)
        return result

    @torch.no_grad()
    def check_parity(self):
        """Check independent ordinary FP32 history execution, including forced states."""
        from civ5rl.engine.comparison_training import target_logits
        selected=[]
        for key in self.keys:
            if not selected or self.count(key)==1 or len(selected)<3 and key[0]!=selected[-1][0]:
                selected.append(key)
            if len(selected)>=4:break
        maximum=0.;argmax_changes=0
        for key in selected:
            history,_=self.histories.get(key[0])
            ordinary=target_logits(self.policy,history,key[1])
            cached=next(self.logits([key]))[1][0,:len(ordinary)]
            maximum=max(maximum,float((cached-ordinary).abs().max()))
            argmax_changes+=int(cached.argmax()!=ordinary.argmax())
            torch.testing.assert_close(cached,ordinary,atol=1e-4,rtol=1e-4)
        if argmax_changes:raise ValueError('Cached FP32 heads changed parity argmax')
        return dict(positions=len(selected),max_logit_absolute_error=maximum,argmax_changes=argmax_changes)

    @torch.no_grad()
    def measure(self,raw=False):
        total=torch.zeros((),dtype=torch.float64,device=self.device)
        anchors=set(map(tuple,self.dataset['anchors']))
        pair_lookup=defaultdict(list)
        for i,p in enumerate(self.dataset['pairs']):pair_lookup[(p['episode'],p['index'])].append((i,p))
        current=dict(choices={},pairs=[None]*len(self.dataset['pairs']))
        for keys,logits in self.logits(self.keys):
            parent=self.parent_batch(keys,logits.shape[1])
            logq=(logits/.5).log_softmax(-1)
            delta=(parent.double()-logq.double()).masked_fill(torch.isneginf(logits),0.)
            values=(parent.double().exp()*delta).masked_fill(torch.isneginf(logits),0.).sum(-1)
            weights=torch.tensor([int(key in anchors) for key in keys],device=self.device)
            total+=(values*weights).sum()
            # One compact readback per batch, not one synchronization per position.
            choices=logits.argmax(-1).cpu().tolist()
            positions=[];better=[];worse=[];indices=[];pair_weights=[]
            for row,key in enumerate(keys):
                current['choices'][key]=choices[row]
                for i,p in pair_lookup[key]:
                    positions.append(row);better.append(p['better']);worse.append(p['worse'])
                    indices.append(i);pair_weights.append(p['weight'])
            if positions:
                rows=torch.tensor(positions,device=self.device)
                margins=logits[rows,torch.tensor(better,device=self.device)]-logits[rows,torch.tensor(worse,device=self.device)]
                losses=F.softplus(-margins/.5)*torch.tensor(pair_weights,device=self.device)
                measured=torch.stack((margins,losses),-1).cpu().tolist()
                for i,row,(margin,loss) in zip(indices,positions,measured):
                    current['pairs'][i]=(choices[row],margin,loss)
        kl=float(total/len(anchors))
        if raw:return kl,current
        return kl,learning_metrics(self.dataset['pairs'],self.original,self.baseline,current,self.dataset['anchors'])

    def update(self,optimizer,pairs,anchor_keys):
        optimizer.zero_grad(set_to_none=True)
        pair_lookup=defaultdict(list)
        for p in pairs:pair_lookup[(p['episode'],p['index'])].append(p)
        anchor_set=set(map(tuple,anchor_keys))
        keys=list(dict.fromkeys([*pair_lookup,*map(tuple,anchor_keys)]))
        pair_total=torch.zeros((),device=self.device);kl_total=torch.zeros_like(pair_total)
        for keys,logits in self.logits(keys):
            pair=logits.new_zeros(());kl=logits.new_zeros(())
            # Batched index gathers keep the loss free of per-pair kernel launches.
            positions=[];better=[];worse=[];weights=[]
            for row,key in enumerate(keys):
                for p in pair_lookup[key]:
                    positions.append(row);better.append(p['better']);worse.append(p['worse']);weights.append(p['weight']/len(pairs))
            if positions:
                rows=torch.tensor(positions,device=self.device)
                diff=logits[rows,torch.tensor(worse,device=self.device)]-logits[rows,torch.tensor(better,device=self.device)]
                pair=(F.softplus(diff/.5)*torch.tensor(weights,device=self.device)).sum()
            parent=self.parent_batch(keys,logits.shape[1]);logq=(logits/.5).log_softmax(-1)
            delta=(parent-logq).masked_fill(torch.isneginf(logits),0.)
            factors=torch.tensor([int(k in anchor_set)/len(anchor_keys) for k in keys],device=self.device)
            kl=((parent.exp()*delta)*factors[:,None]).sum()
            (pair+kl).backward();pair_total+=pair.detach();kl_total+=kl.detach()
        parameters=[p for p in self.policy.parameters() if p.requires_grad]
        norm=torch.nn.utils.clip_grad_norm_(parameters,.5,error_if_nonfinite=self.device.type!='cuda')
        if self.device.type=='cuda':torch._assert_async(torch.isfinite(norm),'Non-finite head gradient')
        optimizer.step()
        flags=finite_state_flags(self.policy,optimizer)
        if not all(bool(flag) for flag in flags.values()):raise FloatingPointError('Non-finite head model/Adam')
        result=torch.stack((pair_total,kl_total,norm)).detach().cpu().tolist()
        return dict(zip(('pair_loss','anchor_kl','gradient_norm'),result))
