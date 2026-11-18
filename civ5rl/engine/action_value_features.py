"""Read frozen head packets without policy-head diagnostics or eager model load.

Use the producer's feature-format version and input identity. Missing packets
are rebuilt from verified public histories; module paths are not cache keys.
"""
from collections import defaultdict
import torch
from civ5sim.engine import digest
from civ5rl.provenance.common import file_sha256,read_json,write_json
from civ5rl.engine.comparison_training import atomic_torch_save,_encode_histories,target_logits
from civ5rl.engine.action_value import public_action_vector


class FrozenActionFeatures:
    def __init__(self,histories,dataset,checkpoint):
        self.histories,self.dataset,self.checkpoint=histories,dataset,checkpoint
        self.directory=histories.directory.parent/'head-features-fp32'
        self.directory.mkdir(parents=True,exist_ok=True)
        self.records,self.locations={},{}
        self.statistics=dict(cache_hits=0,cache_builds=0,bytes=0,policy_loads=0,parity_checks=0)
        self.policy=None

    @torch.no_grad()
    def prepare(self):
        from civ5rl.engine import comparison_heads
        grouped=defaultdict(list)
        keys=sorted({(p['episode'],p['index']) for p in self.dataset['pairs']} |
                    set(map(tuple,self.dataset['anchors'])))
        for episode,index in keys:grouped[episode].append(index)
        common=dict(parent=file_sha256(self.checkpoint),implementation=comparison_heads.FEATURE_CACHE_VERSION,
            encoder=self.histories.encoder_identity,torch=str(torch.__version__),
            gpu=torch.cuda.get_device_name(),attention_backend='compact_fp32',precision='fp32_no_amp_no_tf32',
            temperature=.5,spatial_block=32,history_batch=1)
        for episode,indices in grouped.items():
            identity=dict(common=common,source=self.histories.source_identity(episode),indices=indices)
            path=self.directory/f'{episode}-{digest(identity)}.pt';receipt=path.with_suffix('.json')
            if path.exists() and receipt.exists():
                saved=read_json(receipt)
                if saved['identity']!=identity or saved['sha256']!=file_sha256(path):
                    raise ValueError('Frozen action feature cache changed')
                self.statistics['cache_hits']+=1
            else:
                self.build(episode,indices,identity,path,receipt)
            record=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
            if record['identity']!=identity:raise ValueError('Frozen action identity changed')
            self.records[episode]=record
            self.statistics['bytes']+=path.stat().st_size
            for row,index in enumerate(indices):self.locations[(episode,index)]=(episode,row)
        return self

    def build(self,episode,indices,identity,path,receipt):
        from civ5rl.engine.comparison_heads import configure_heads
        if self.policy is None:
            self.policy=configure_heads(self.checkpoint);self.policy.requires_grad_(False)
            self.statistics['policy_loads']+=1
        history,_=self.histories.get(episode);sequence=history[:max(indices)+1]
        x,context=_encode_histories(self.policy,[sequence])
        h,_=self.policy.sequence_memory(context,[len(sequence)])
        selected=torch.tensor(indices,device='cuda');items=[history[i] for i in indices]
        packed={name:torch.cat([torch.from_numpy(getattr(item,name)) for item in items])
                for name in ('sources','targets','action_features','attack_preview')}
        offsets=[0]
        for item in items:offsets.append(offsets[-1]+len(item.sources))
        raw=self.histories.raw(episode)
        record=atomic_torch_save(dict(identity=identity,x=x[selected],h=h[selected],actions=packed,
            offsets=offsets,chosen=[raw[i]['chosen'] for i in indices]),path)
        if not self.statistics['parity_checks']:
            vectors=torch.stack([public_action_vector(record,0,a) for a in range(offsets[1])]).cuda()
            expected=target_logits(self.policy,history,indices[0])
            actual=self.policy.action[2](self.policy.action[1](self.policy.action[0](vectors[:,:529])+
                self.policy.attack_preview(vectors[:,529:]))).flatten()
            torch.testing.assert_close(actual,expected,atol=1e-4,rtol=1e-4)
            self.statistics['parity_checks']+=1
        write_json(receipt,dict(identity=identity,sha256=file_sha256(path)))
        self.statistics['cache_builds']+=1
