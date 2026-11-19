"""Shared encoded prefixes plus immutable branch tails; no duplicated battle tensors."""

from copy import deepcopy
from collections import OrderedDict

import torch

from civ5rl.core.context import context_from_spec
from civ5rl.core.observation import ObservationEncoder
from civ5rl.engine.comparison.inputs import Histories, History
from civ5rl.provenance.common import read_json, file_sha256
from civ5sim import Action
from civ5sim.engine import digest


class TurnStorage(Histories):
    def __init__(self, out, sources, roots, branches, anchors, device):
        limits = {}
        for episode, index in anchors:
            limits[episode] = max(limits.get(episode, 0), index+1)
        for b in branches.values():
            limits[b['episode']] = max(limits.get(b['episode'], 0), b['prefix'])
        super().__init__(out, limits=limits, device=device)
        self.source_root, self.roots, self.branches = sources, roots, branches
        self.prefix_encoder = None
        self.prefix_key = None
        self.history_rows = OrderedDict()

    def source_folder(self, episode):
        if episode in self.branches:
            b = self.branches[episode]
            return self.roots[b['packet']] / b['folder']
        return self.source_root / str(episode)

    def source_identity(self, episode):
        if episode not in self.branches:
            return super().source_identity(episode)
        if episode not in self.sources:
            b = self.branches[episode]
            self.sources[episode] = dict(prefix=super().source_identity(b['episode']),
                prefix_length=b['prefix'], tail=file_sha256(self.source_folder(episode)/'turn-history.json'),
                encoding_version=1)
        return self.sources[episode]

    def source_rows(self, episode):
        if episode not in self.history_rows:
            self.history_rows[episode]=read_json(self.source_root / str(episode) / 'history.json')
            if len(self.history_rows)>2:self.history_rows.popitem(last=False)
        self.history_rows.move_to_end(episode)
        return self.history_rows[episode]

    def turn_record(self, episode):
        """Load and validate the public turn and its recorded outcome together."""
        b = self.branches[episode]
        folder = self.source_folder(episode)
        rows = read_json(folder/'turn-history.json')
        completed, branch = read_json(folder/'completed.json'), read_json(folder/'branch.json')
        if branch['position_key'] != b['key'] or branch['command'] != b['command'] or branch['turn_variant'] != b['variant'] or branch['repetition'] != b['repetition']:
            raise ValueError('Branch does not match fixed training manifest')
        if not completed['prefix_verified'] or completed['position_key'] != b['key']:
            raise ValueError('Unverified branch prefix')
        original = self.source_rows(b['episode'])
        if original[b['prefix']]['command'] != b['command'] or rows[0]['command'] != b['command']:
            raise ValueError('Own-observation prefix boundary differs')
        if digest(rows[0]['observation']) != digest(original[b['prefix']]['observation']):
            raise ValueError('Branch has a different public initial observation')
        return rows, int(completed['winner'] == completed['side'])

    def tail_rows(self, episode):
        return self.turn_record(episode)[0]

    def encoded_rows(self, episode):
        if episode not in self.branches:
            return super().encoded_rows(episode)
        b = self.branches[episode]
        key = (b['episode'], b['prefix'])
        if key != self.prefix_key:
            case = read_json(self.source_root / str(b['episode']) / 'case.json')
            encoder = ObservationEncoder(context_from_spec(case,
                seed=case['agent_seeds'][case['learner_side']]), 'C2P')
            for row in self.source_rows(b['episode'])[:b['prefix']]:
                legal = [Action(**a) for a in row['actions']]
                encoder.encode(row['observation'], legal)
                encoder.remember_action(legal[row['chosen']], row['observation'])
            self.prefix_key, self.prefix_encoder = key, encoder
        encoder = deepcopy(self.prefix_encoder)
        encoded = []
        for row in self.tail_rows(episode):
            legal = [Action(**a) for a in row['actions']]
            encoded.append(encoder.encode(row['observation'], legal))
            encoder.remember_action(legal[row['chosen']], row['observation'])
        return encoded


class TurnHistories:
    """Expose complete branch histories while sharing stored source-prefix tensors."""
    def __init__(self, out, sources, roots, branches, anchors, device='cuda'):
        self.storage = TurnStorage(out, sources, roots, branches, anchors, device)
        self.branches, self.records, self.targets = branches, {}, {}

    def get(self, episode):
        if episode not in self.records:
            if episode in self.branches:
                b = self.branches[episode]
                self.storage.get(episode)
                tail = self.storage.records[episode]
                if b['prefix']:
                    self.storage.get(b['episode'])
                    prefix = self.storage.records[b['episode']]
                    self.records[episode] = {k: torch.cat((prefix[k][:b['prefix']], tail[k])) for k in ('counts','units')}
                else:self.records[episode] = {k:tail[k] for k in ('counts','units')}
            else:
                self.storage.get(episode)
                self.records[episode] = self.storage.records[episode]
        return History(self, episode, 0, len(self.records[episode]['counts'])), None

    def item(self, episode, index):
        if episode in self.branches:
            b = self.branches[episode]
            if index < b['prefix']:
                return self.storage.item(b['episode'], index)
            index -= b['prefix']
        return self.storage.item(episode, index)

    def action_count(self, episode, index):
        self.get(episode)
        return int(self.records[episode]['counts'][index])

    def source_identity(self, episode):
        return self.storage.source_identity(episode)

    def spatial_inputs(self, sequences, device):
        spans = []
        for seq in sequences:
            b = self.branches.get(seq.episode)
            if b is None:
                spans.append(History(self.storage,seq.episode,seq.start,seq.stop))
            else:
                boundary = b['prefix']
                if seq.start < boundary:
                    spans.append(History(self.storage,b['episode'],seq.start,min(seq.stop,boundary)))
                if seq.stop > boundary:
                    spans.append(History(self.storage,seq.episode,max(0,seq.start-boundary),seq.stop-boundary))
        return self.storage.spatial_inputs(spans, device)

    def prepare(self, keys, *, device=None, budget=None):
        for episode in dict.fromkeys(e for e,_ in keys):
            if budget is not None and budget.expired():
                raise TimeoutError('Whole-turn preparation budget exhausted')
            self.get(episode)
            self.storage._place(episode)
            if episode in self.branches and self.branches[episode]['prefix']:
                self.storage._place(self.branches[episode]['episode'])

    def statistics(self):
        return self.storage.statistics()

    def comparison_targets(self, a, b, ra, rb):
        """Exact common commands carry no direct preference, even when outcomes differ."""
        common=0
        for x,y in zip(ra,rb):
            if x['actions'][x['chosen']] != y['actions'][y['chosen']]:
                break
            if digest(x['observation']) != digest(y['observation']):
                raise ValueError('Equal prefix commands have different public observations')
            common+=1
        result=[]
        for episode,rows in ((a,ra),(b,rb)):
            keys=[]
            for i,row in enumerate(rows[common:],start=common):
                if len(row['actions']) > 1:
                    key=(episode,self.branches[episode]['prefix']+i)
                    self.targets[key]=row['chosen']
                    keys.append(key)
            result.append(keys)
        return result
