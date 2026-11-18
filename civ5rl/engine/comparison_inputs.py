"""Verified public histories, encoded once; mmap backing and bounded tensor residency.

Source case/history SHA-256 values are checked once per process. The derived
tensor cache is keyed by those hashes, the last required position and encoder/
packing implementation identities. Raw parsed JSON rows are not retained.

Spatial tensors reuse SequenceBuffer gathering: up to 2 GiB reside on CUDA,
then up to 1 GiB in pinned RAM, with remaining tensors backed by mmap. CUDA
placement uses allocation_with_reserve; failed admission discards partial copies
and changes only cache placement, never the mathematical path. Unit-index
padding is derived from public batch inputs and never truncates legal keys.
"""
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
import os

import numpy as np
import torch

from civ5rl.core.context import context_from_spec
from civ5rl.core.model import batch_inputs
from civ5rl.core.observation import Encoded, ObservationEncoder
from civ5rl.engine.sequence import SequenceBuffer, allocation_with_reserve
from civ5rl.engine.constants import DEFAULT_GPU_RESERVE_BYTES, DEFAULT_PINNED_REPLAY_BYTES
from civ5rl.provenance.common import ROOT, file_sha256, read_json, write_json
from civ5sim import Action
from civ5sim.engine import digest


class History(Sequence):
    """A cheap span into immutable tensors; individual Encoded views are diagnostic."""
    def __init__(self, owner, episode, start, stop):
        self.owner, self.episode, self.start, self.stop = owner, episode, start, stop

    def __len__(self):
        return self.stop-self.start

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            if step != 1:
                return [self[i] for i in range(start, stop, step)]
            return History(self.owner, self.episode, self.start+start, self.start+stop)
        if index < 0:index += len(self)
        if not 0 <= index < len(self):raise IndexError(index)
        return self.owner.item(self.episode, self.start+index)


class Histories:
    def __init__(self, out, source_hashes=None, *, limits=None, device='cpu',
                 gpu_bytes=2*1024**3, pinned_bytes=DEFAULT_PINNED_REPLAY_BYTES, cache_dir=None):
        self.out = Path(out)
        scratch = os.environ.get('CIV5RL_SCRATCH_DIR')
        cache_root = (Path(scratch)/self.out.resolve().name if scratch else
                      ROOT/'.tmp'/'comparison-inputs'/digest(str(self.out.resolve())))
        self.directory = Path(cache_dir) if cache_dir is not None else cache_root/'prepared-histories'
        self.source_hashes, self.limits = source_hashes, limits or {}
        self.device, self.gpu_limit, self.pinned_limit = str(device), gpu_bytes, pinned_bytes
        self.records, self.sources, self.buffers = {}, {}, {}
        self.counters = Counter()
        # Bump encoding_version when tensor semantics change, not for file moves.
        self.encoder_identity = dict(encoding_version=1,torch_version=str(torch.__version__),
                                     numpy_version=np.__version__)
        self._gpu_closed = self._pinned_closed = False

    def source_identity(self, episode):
        if episode not in self.sources:
            folder = self.out/'sources'/str(episode)
            hashes = {name:file_sha256(folder/name) for name in ('case.json','history.json')}
            if self.source_hashes is not None and hashes != self.source_hashes[str(episode)]:
                raise ValueError('Recorded public history changed')
            self.sources[episode] = hashes
            self.counters['source_verifications'] += 1
        return self.sources[episode]

    def raw(self, episode):
        """Explicit, uncached diagnostic access; never retained by the training store."""
        self.source_identity(episode)
        return read_json(self.out/'sources'/str(episode)/'history.json')

    def _build(self, episode, identity, path):
        started = perf_counter()
        folder = self.out/'sources'/str(episode)
        case = read_json(folder/'case.json')
        rows = read_json(folder/'history.json')
        limit = self.limits.get(episode, len(rows))
        if not 0 < limit <= len(rows):raise ValueError('Target outside recorded history')
        encoder = ObservationEncoder(context_from_spec(case,seed=case['agent_seeds'][case['learner_side']]),'C2P')
        encoded = []
        for row in rows[:limit]:
            legal = [Action(**a) for a in row['actions']]
            encoded.append(encoder.encode(row['observation'],legal))
            encoder.remember_action(legal[row['chosen']],row['observation'])
        spatial = batch_inputs(encoded,'cpu',spatial_only=True,compact_units=True)
        counts = torch.tensor([len(x.sources) for x in encoded])
        actions = {key:torch.from_numpy(np.concatenate([getattr(x,key) for x in encoded]))
                   for key in ('sources','targets','action_features','attack_preview')}
        offsets = torch.cat((counts.new_zeros(1),counts.cumsum(0)))
        units = (spatial['unit_indices']>=0).sum(-1)
        path.parent.mkdir(parents=True,exist_ok=True)
        temp = path.with_suffix('.tmp')
        torch.save(dict(identity=identity,spatial=spatial,actions=actions,counts=counts,
                        offsets=offsets,units=units),temp)
        temp.replace(path)
        write_json(path.with_suffix('.json'),dict(identity=identity,sha256=file_sha256(path)))
        self.counters['encoded_episodes'] += 1
        self.counters['encoded_decisions'] += limit
        self.counters['preparation_seconds'] += perf_counter()-started

    def get(self, episode):
        if episode not in self.records:
            identity = dict(schema=1,sources=self.source_identity(episode),encoder=self.encoder_identity,
                            limit=self.limits.get(episode))
            path = self.directory/f'{episode}-{digest(identity)}.pt'
            receipt = path.with_suffix('.json')
            if not path.exists() or not receipt.exists():self._build(episode,identity,path)
            saved = read_json(receipt)
            if saved['identity'] != identity or saved['sha256'] != file_sha256(path):
                raise ValueError('Prepared public history changed')
            record = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
            if record['identity'] != identity:raise ValueError('Prepared public history identity changed')
            self.records[episode] = record
            self.counters['prepared_loads'] += 1
        else:self.counters['prepared_hits'] += 1
        return History(self,episode,0,len(self.records[episode]['counts'])),None

    def item(self, episode, index):
        record = self.records[episode]
        start,stop = map(int,record['offsets'][index:index+2])
        fields = {key:value[index].numpy() for key,value in record['spatial'].items()
                  if key in ('cells','neighbors','globals','previous')}
        fields.update({key:value[start:stop].numpy() for key,value in record['actions'].items()})
        return Encoded(**fields)

    def prepare(self, keys, *, device=None, budget=None):
        """Finish immutable preparation before optimizer timing; cache placement is optional."""
        if device is not None:self.device=str(device)
        for episode in dict.fromkeys(e for e,_ in keys):
            self.get(episode)
            if budget is None or not budget.expired():self._place(episode)

    def _place(self, episode):
        existing = self.buffers.get(episode)
        if existing is not None:
            tensor = existing.data['cells']
            if tensor.device.type=='cuda' or tensor.is_pinned() or not self.device.startswith('cuda'):
                return
        spatial = self.records[episode]['spatial']
        size = sum(t.numel()*t.element_size() for t in spatial.values())
        data = spatial
        if self.device.startswith('cuda') and not self._gpu_closed and self.counters['gpu_bytes']+size <= self.gpu_limit:
            copied = {}
            try:
                with allocation_with_reserve(self.device,DEFAULT_GPU_RESERVE_BYTES):
                    for key,value in spatial.items():copied[key] = value.to(self.device)
                data = copied
                self.counters['gpu_bytes'] += size
            except torch.cuda.OutOfMemoryError:
                copied.clear()
                self._gpu_closed = True
                self.counters['gpu_admission_fallbacks'] += 1
        if data is spatial and self.device.startswith('cuda') and not self._pinned_closed and self.counters['pinned_bytes']+size <= self.pinned_limit:
            copied = {}
            try:
                for key,value in spatial.items():copied[key] = value.pin_memory()
                data = copied
                self.counters['pinned_bytes'] += size
            except RuntimeError:
                copied.clear();self._pinned_closed = True
                self.counters['pinned_admission_fallbacks'] += 1
        self.buffers[episode] = SequenceBuffer.from_spatial_tensors(data,str(data['cells'].device))

    def spatial_inputs(self, sequences, device):
        parts = []; maximum = 2
        for seq in sequences:
            record = self.records[seq.episode]
            maximum = max(maximum,int(record['units'][seq.start:seq.stop].max()))
            buffer = self.buffers.get(seq.episode)
            if buffer is None:buffer = SequenceBuffer.from_spatial_tensors(record['spatial'],'cpu')
            part = buffer.gather_spatial_inputs([(0,seq.start,seq.stop)])
            parts.append(part)
        capacity = 1 << (maximum-1).bit_length()
        output = {}
        for key in parts[0]:
            values = [(p[key][:,:capacity] if key=='unit_indices' else p[key]) for p in parts]
            if key=='unit_indices':
                values = [torch.nn.functional.pad(v,(0,capacity-v.shape[1]),value=-1)
                          if v.shape[1]<capacity else v for v in values]
            values = [v.to(device,non_blocking=v.is_pinned()) for v in values]
            output[key] = torch.cat(values,0)
        self.counters['packed_spatial_batches'] += 1
        return output

    def statistics(self):
        return dict(self.counters,episodes=len(self.records),gpu_limit_bytes=self.gpu_limit,
                    pinned_limit_bytes=self.pinned_limit,host_storage='mmap',raw_rows_retained=0)
