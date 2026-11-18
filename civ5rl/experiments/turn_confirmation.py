"""Independent damage futures for fixed, previously selected turn variants."""

from collections import Counter
import shutil

import numpy as np

from civ5rl.provenance.common import ROOT, read_json, write_json, file_sha256
from civ5rl.provenance import checkpoint


INTERPRETATION = dict(
    primary='Equal-position mean paired win difference over all fixed candidates, including identical boundary states.',
    positive='CI95 lower bound > 0 supports average usefulness of selection on these positions and a subsequent policy-learning experiment; no training is started automatically.',
    uncertain='CI95 containing zero leaves the average advantage unresolved.',
    negative='CI95 upper bound < 0 supports an average disadvantage relative to argmax on these positions.',
    scope='Conditional on the selected positions; not a test of choosing variants from public information or of a learned policy in complete battles.',
    individual='Position signs are descriptive: eight repeats give a 12.5 percentage-point resolution.',
    boundary='Boundary hashes describe simulator state, excluding agent and GRU memory.')


def prepare_confirmation(cfg, out):
    """Freeze candidates and task identities before any confirmation outcomes."""
    from civ5rl.experiments.turn_comparison import branch_spec

    spec = cfg['confirmation']
    source = ROOT / spec['run']
    for name, sha in cfg['collection_input']['sha256'].items():
        if file_sha256(ROOT / cfg['collection_input']['run'] / name) != sha:
            raise ValueError(f'Match configuration changed: {name}')
    for name, sha in spec['sha256'].items():
        if file_sha256(source / name) != sha:
            raise ValueError(f'Confirmation input changed: {name}')
    if file_sha256(ROOT / cfg['parent']) != cfg['parent_sha256']:
        raise ValueError('Parent changed')
    original = read_json(source / 'positions.json')
    from civ5rl.experiments.turn_comparison import NAMESPACE
    if original['namespace'] != NAMESPACE:
        raise ValueError('Screening RNG namespace differs')
    if original['parent_sha256'] != cfg['parent_sha256']:
        raise ValueError('Confirmation parent differs from screening parent')
    candidates = read_json(source / spec['candidates'])['rows']
    positions_by_key = {tuple(p['key']): p for p in original['positions']}
    if len(candidates) != cfg['positions'] or len({tuple(c['position_key']) for c in candidates}) != len(candidates):
        raise ValueError('Candidate count or uniqueness differs')
    positions, tasks = [], []
    for candidate in candidates:
        position = positions_by_key[tuple(candidate['position_key'])]
        variant = candidate['variant']
        wins = candidate['screening_wins']
        expected = max(range(4), key=lambda v: (wins[v], -v))
        if position['world'] != candidate['world'] or variant not in (1, 2, 3) or variant != expected or wins[variant] <= wins[0]:
            raise ValueError('Fixed candidate is inconsistent with screening selection')
        positions.append({**position, 'selected_variant': variant})
        old = {branch_spec(position, 0, r, '')['seed'] for r in range(4)}
        fresh = [branch_spec(position, 0, r, '')['seed'] for r in spec['repetitions']]
        if len(set(fresh)) != 8 or old.intersection(fresh):
            raise ValueError('Confirmation damage seeds are not new and distinct')
        for v in (0, variant):
            for r in spec['repetitions']:
                branch = branch_spec(position, v, r, '')
                tasks.append(dict(position_key=position['key'], variant=v, repetition=r,
                                  damage_seed=branch['seed'], sampling_seed=branch['sampling_seed']))
    lineage = dict(confirmation=spec, parent_sha256=cfg['parent_sha256'])
    manifest = dict(schema=1, namespace=original['namespace'], positions=positions,
                    parent_sha256=cfg['parent_sha256'], tasks=tasks,
                    repetitions=spec['repetitions'], interpretation=INTERPRETATION)
    if (out / 'passport.json').exists():
        checkpoint.verify_execution_passport(out, cfg, lineage)
    else:
        identity = checkpoint.passport(out, cfg, lineage)
        write_json(out / 'run.json', dict(kind=cfg['kind'], config=cfg, identity=identity))
    if (out / 'positions.json').exists():
        if read_json(out / 'positions.json') != manifest:
            raise ValueError('Frozen confirmation manifest differs')
    else:
        write_json(out / 'positions.json', manifest)
    if not (out / 'sources.storage').exists():
        shutil.copytree(source / 'sources.storage', out / 'sources.storage')
    write_json(out / 'selection.json', dict(selected=len(positions),
        target_continuations=len(tasks), sha256=file_sha256(out / 'positions.json'),
        world_counts=dict(Counter(p['world'] for p in positions))))
    write_json(out / 'analysis-plan.json', dict(interpretation=INTERPRETATION,
        bootstrap=spec['bootstrap'], repetitions=spec['repetitions'],
        candidate_sha256=spec['sha256'][spec['candidates']]))
    if not (out / 'collection.json').exists():
        write_json(out / 'status.json', dict(status='prepared', phase='collection'))


def paired_statistics(worlds, differences, *, samples=10000, seed=20261103):
    """Resample whole worlds while retaining equal weights for positions.

    A world drawn twice contributes every one of its positions twice. Dividing
    by the resulting position count avoids silently changing the estimand to
    an equal-world mean when cluster sizes differ.
    """
    worlds = np.asarray(worlds)
    differences = np.asarray(differences, dtype=np.float64)
    unique = np.unique(worlds)
    sums = np.array([differences[worlds == w].sum() for w in unique])
    counts = np.array([(worlds == w).sum() for w in unique])
    draws = np.random.default_rng(seed).integers(len(unique), size=(samples, len(unique)))
    values = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    low, high = np.quantile(values, [.025, .975]).tolist()
    return dict(mean_win_delta=float(differences.mean()), ci95=[low, high],
                worlds=len(unique), samples=samples, seed=seed,
                conclusion='positive' if low > 0 else 'negative' if high < 0 else 'uncertain')


def summarize_confirmation(cfg, out, manifest, rows, seconds):
    indexed = {(tuple(r['position_key']), r['turn_variant'], r['repetition']): r for r in rows}
    if len(indexed) != len(manifest['tasks']) or len(rows) != len(indexed):
        raise ValueError('Confirmation is incomplete or contains duplicate results')
    positions, compact = [], []
    totals = {role: Counter() for role in ('argmax', 'selected')}
    for p in manifest['positions']:
        wins = Counter()
        same = 0
        for r in manifest['repetitions']:
            pair = [indexed[(tuple(p['key']), v, r)] for v in (0, p['selected_variant'])]
            same += pair[0]['boundary_state_sha256'] == pair[1]['boundary_state_sha256']
            for role, row in zip(('argmax', 'selected'), pair):
                outcome = 'D' if row['winner'] is None else 'W' if row['winner'] == row['side'] else 'L'
                wins[role] += outcome == 'W'
                totals[role][outcome] += 1
                compact.append({**row, 'world': p['world'], 'role': role, 'outcome': outcome})
        positions.append(dict(position_key=p['key'], world=p['world'], selected_variant=p['selected_variant'],
            wins=dict(wins), paired_win_delta=(wins['selected'] - wins['argmax']) / 8,
            identical_boundary_states=same))
    stats = paired_statistics([p['world'] for p in positions], [p['paired_win_delta'] for p in positions],
                              **cfg['confirmation']['bootstrap'])
    result = dict(status='complete', positions=len(positions), continuations=len(rows), seconds=seconds,
        variants={role: {k: totals[role][k] for k in 'WDL'} for role in totals}, primary=stats,
        position_signs=dict(positive=sum(p['paired_win_delta'] > 0 for p in positions),
            zero=sum(p['paired_win_delta'] == 0 for p in positions),
            negative=sum(p['paired_win_delta'] < 0 for p in positions)),
        position_resolution_percentage_points=12.5, interpretation=INTERPRETATION)
    write_json(out / 'continuations-summary.json', compact)
    write_json(out / 'position-results.json', positions)
    write_json(out / 'result.json', result)
    return result
