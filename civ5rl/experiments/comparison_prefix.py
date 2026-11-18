"""Boundary GRU states for immutable public source histories.

A source captures u171 hidden after each own observation, before its action can
influence the next input. A continuation installs the state after its boundary
observation at reset. The worker still reconstructs its public encoder and CP
prefix locally, including the alternative action, but sends no observe RPC.
Saved histories without captures are prefetched once per source, not per branch.
"""
from collections import OrderedDict
from pathlib import Path
import json
import numpy as np
import torch

from civ5rl.core.context import context_from_spec
from civ5rl.core.observation import ObservationEncoder
from civ5rl.core.model import batch_inputs
from civ5rl.provenance.common import ROOT, read_json, file_sha256
from civ5sim import Action


class SourcePrefixes:
    def __init__(self, policy, checkpoint, runtime, max_bytes=64*1024**2):
        self.policy, self.device = policy, next(policy.parameters()).device
        self.identity = dict(checkpoint=file_sha256(checkpoint),runtime=runtime.identity(),
                             torch=str(torch.__version__),prefix_encoding_version=1)
        self.cache = OrderedDict()
        self.max_bytes = max_bytes
        self.hits = self.prefills = self.restored_observations = 0

    def _put(self, source, history, frames):
        # Retain only boundary indices, not parsed multi-megabyte observations.
        commands=tuple(row['command'] for row in history)
        self.cache[str(source)] = (self._stamp(source),commands,frames)
        self.cache.move_to_end(str(source))
        while len(self.cache)>1 and sum(v[2].numel()*v[2].element_size() for v in self.cache.values())>self.max_bytes:
            self.cache.popitem(last=False)

    @staticmethod
    def _stamp(source):
        stats=[(Path(source)/name).stat() for name in ('case.json','history.json')]
        return tuple((s.st_size,s.st_mtime_ns) for s in stats)

    def capture(self, source, frames):
        history = read_json(Path(source)/'history.json')
        if len(history)!=len(frames):
            raise ValueError('Source hidden/history count differs')
        values = (torch.cat(frames,1)[0].detach().cpu() if frames else
                  torch.empty((0,self.policy.hidden_size),dtype=torch.float32))
        payload = dict(hidden=values.numpy(),identity=json.dumps(self.identity,sort_keys=True),
                       case_sha256=file_sha256(Path(source)/'case.json'),
                       history_sha256=file_sha256(Path(source)/'history.json'))
        path=Path(source)/'prefix-states.npz';temporary=path.with_suffix('.tmp')
        with temporary.open('wb') as stream:np.savez(stream,**payload)
        temporary.replace(path)
        self._put(source,history,values)

    @torch.inference_mode()
    def restore(self, source, command):
        source=str(source);path=Path(source)/'history.json';stamp=self._stamp(source)
        saved=self.cache.get(source)
        if saved is not None:
            if saved[0]!=stamp:
                raise ValueError('Source public history changed during collection')
            commands,frames=saved[1:];self.hits+=1;self.cache.move_to_end(source)
        else:
            history=read_json(path);captured=Path(source)/'prefix-states.npz';frames=None
            if captured.exists():
                with np.load(captured,allow_pickle=False) as data:
                    if json.loads(str(data['identity']))==self.identity:
                        if str(data['history_sha256'])!=file_sha256(path):
                            raise ValueError('Captured prefix public history changed')
                        if str(data['case_sha256'])!=file_sha256(Path(source)/'case.json'):
                            raise ValueError('Captured prefix context changed')
                        frames=torch.from_numpy(data['hidden'].copy())
            if frames is None:
                case=read_json(Path(source)/'case.json')
                encoder=ObservationEncoder(context_from_spec(case,seed=case['agent_seeds'][case['learner_side']]),self.policy.variant)
                encoded=[]
                for row in history:
                    legal=[Action(**a) for a in row['actions']]
                    encoded.append(encoder.encode(row['observation'],legal))
                    encoder.remember_action(legal[row['chosen']],row['observation'])
                hidden=None;outputs=[]
                for start in range(0,len(encoded),32):
                    batch=batch_inputs(encoded[start:start+32],self.device,spatial_only=True,
                                       compact_units=self.policy.attention_backend=='compact_fp32')
                    _,context=self.policy.encode_spatial(batch)
                    output,hidden=self.policy.memory(context[:,None].float(),hidden)
                    outputs.append(output[:,0].cpu())
                frames=torch.cat(outputs);self.prefills+=1
            if frames.shape!=(len(history),self.policy.hidden_size) or frames.dtype!=torch.float32 or not torch.isfinite(frames).all():
                raise ValueError('Invalid captured recurrent prefix')
            self._put(source,history,frames)
            commands=tuple(row['command'] for row in history)
        indices=[i for i,value in enumerate(commands) if value==command]
        if len(indices)!=1:raise ValueError('Continuation boundary missing from source history')
        count=indices[0]+1;self.restored_observations+=count
        return frames[count-1].to(self.device).reshape(1,1,-1).clone(),count
