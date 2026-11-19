"""Fixed paired-damage comparisons and training-only anchors for whole turns."""

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from itertools import combinations
from pathlib import Path

from civ5rl.provenance.common import ROOT, read_json, write_json, file_sha256
from civ5rl.reporting.comparison_storage import readable_members


def verify_inputs(cfg):
    for name, sha in cfg['input_sha256'].items():
        if file_sha256(ROOT / name) != sha:
            raise ValueError(f'Whole-turn input changed: {name}')


def build_dataset(cfg, *, position_indices=None, anchor_limit=None):
    """Keep zero comparisons in denominators; screening-selected pairs use fresh futures only."""
    verify_inputs(cfg)
    original = ROOT / cfg['screening_run']
    fresh = ROOT / cfg['confirmation_run']
    manifest = read_json(original / 'positions.json')
    review = {tuple(p['pos']): p for p in
              (read_json_lines(ROOT / cfg['review']))}
    selected = {tuple(p['position_key']): p['variant'] for p in
                read_json(original / 'derived/turn-review/candidates.json')['rows']}
    confirmed = defaultdict(dict)
    for row in read_json(fresh / 'continuations-summary.json'):
        confirmed[tuple(row['position_key'])][row['turn_variant'], row['repetition']] = row
    positions, branches = [], {}
    indices = range(len(manifest['positions'])) if position_indices is None else position_indices
    for i in indices:
        p = manifest['positions'][i]
        key = tuple(p['key'])
        old = {(r['v'], r['r']): r for r in review[key]['rows']}
        comparisons = []
        for a, b in combinations(range(4), 2):
            new = key in selected and (a, b) == (0, selected[key])
            repeats = range(4, 12) if new else range(4)
            for r in repeats:
                pair = []
                outcomes = []
                for v in (a, b):
                    identity = 1_000_000_000 + i * 100 + v * 12 + r
                    row = confirmed[key][v, r] if new else old[v, r]
                    outcomes.append(row['outcome'] == 'W' if new else row['wdl'] == 'W')
                    # JSON object keys are strings. Use that representation before hashing too.
                    branches[str(identity)] = dict(position=i, episode=p['episode'], world=p['world'],
                        prefix=p['history_index'], command=p['command'], key=p['key'],
                        variant=v, repetition=r, packet='confirmation' if new else 'screening',
                        folder=f'{p["episode"]}-{p["key"][1]}/{v}-{r}')
                    pair.append(identity)
                comparisons.append(dict(a=pair[0], b=pair[1], sign=int(outcomes[0])-int(outcomes[1]),
                                        repeats=len(repeats)))
        positions.append(dict(index=i, key=p['key'], world=p['world'], comparisons=comparisons))
    all_anchors = read_json(ROOT / cfg['anchor_input'])['positions']
    worlds = set(read_json(ROOT / cfg['split'])['training'])
    source_packet = original / 'sources.storage'
    episodes = sorted({e for e, _ in all_anchors})
    with readable_members(source_packet, [f'{e}/case.json' for e in episodes]) as source:
        allowed = {e for e in episodes if read_json(source / str(e) / 'case.json')['parent_world'] in worlds}
    anchors = [p for p in all_anchors if p[0] in allowed]
    if len(anchors) != 7000 or len(worlds) != 33:
        raise ValueError('Expected exactly 7000 anchors from the 33 training worlds')
    if anchor_limit is not None:
        anchors = anchors[:anchor_limit]
    counts = dict(positions=len(positions), variant_pairs=len(positions)*6,
        comparisons=sum(len(p['comparisons']) for p in positions),
        nonzero=sum(c['sign'] != 0 for p in positions for c in p['comparisons']),
        positions_with_signal=sum(any(c['sign'] for c in p['comparisons']) for p in positions),
        anchors=len(anchors))
    if position_indices is None and counts != dict(positions=512, variant_pairs=3072,
            comparisons=12572, nonzero=1123, positions_with_signal=152, anchors=7000):
        raise ValueError(f'Unexpected whole-turn inventory: {counts}')
    return dict(schema=1, positions=positions, branches=branches, anchors=anchors,
                counts=counts, input_sha256=cfg['input_sha256'])


def read_json_lines(path):
    import json
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


@contextmanager
def history_inputs(cfg, dataset):
    """Read only needed public histories; extraction can be rebuilt after any interruption."""
    active = {c[k] for p in dataset['positions'] for c in p['comparisons'] if c['sign'] for k in ('a', 'b')}
    branches = {b: dataset['branches'][str(b)] for b in sorted(active)}
    episodes = {branch['episode'] for branch in branches.values()}
    episodes.update(e for e, _ in dataset['anchors'])
    with ExitStack() as stack:
        sources = stack.enter_context(readable_members(ROOT / cfg['screening_run'] / 'sources.storage',
            [f'{e}/{name}.json' for e in episodes for name in ('case', 'history')]))
        roots = {}
        for packet, run in [('screening', cfg['screening_run']), ('confirmation', cfg['confirmation_run'])]:
            names = [f'{b["folder"]}/{name}.json' for b in branches.values() if b['packet']==packet
                     for name in ('turn-history', 'completed', 'branch')]
            if names:
                roots[packet] = stack.enter_context(readable_members(ROOT / run / 'continuations.storage', names))
        yield sources, roots, branches
