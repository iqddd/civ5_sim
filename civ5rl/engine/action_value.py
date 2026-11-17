"""Frozen-feature outcome regression and world-balanced counterfactual utility.

Only public action vectors enter the head. Outcome counts supervise independent
Bernoulli probabilities, not a two-action softmax. Evaluation always uses the
first four paired futures, avoiding the collection's outcome-dependent extension.
"""
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def split_worlds(worlds):
    if len(set(worlds)) != 57:
        raise ValueError('Require 57 distinct training worlds')
    ordered = sorted(worlds, key=lambda w: hashlib.sha256(
        f'c2p_action_value_split_v1|17|{w}'.encode()).hexdigest())
    return dict(training=ordered[:33], selection=ordered[33:45], confirmation=ordered[45:])


def make_head(device='cuda'):
    return nn.Sequential(nn.Linear(534, 128), nn.GELU(), nn.Linear(128, 1)).to(device)


def make_optimizer(head):
    return torch.optim.AdamW([
        dict(params=[p for p in head.parameters() if p.ndim > 1], weight_decay=1e-4),
        dict(params=[p for p in head.parameters() if p.ndim == 1], weight_decay=0.),
    ], lr=3e-4, betas=(.9, .999), eps=1e-8)


def outcome_loss(logits, wins, counts):
    return (F.binary_cross_entropy_with_logits(logits, wins/counts, reduction='none')*counts).sum()/counts.sum()


def fork_targets(fork):
    """Keep adaptive extra futures for fitting, first-group futures for scoring."""
    n=fork['executed_repetitions']
    if n not in (4,8) or set(fork['winners'])!={f'{r}-{a}' for r in range(n) for a in range(2)}:
        raise ValueError('Unpaired continuation outcomes')
    if any(w not in (None,0,1) for w in fork['winners'].values()):
        raise ValueError('Invalid winner')
    won=[[int(fork['winners'][f'{r}-{a}']==fork['side']) for r in range(n)] for a in range(2)]
    return [sum(v) for v in won],[n,n],[sum(v[:4]) for v in won]


def public_action_vector(record, location, action):
    """Match the policy's source/target sentinel handling before learned layers."""
    k=record['offsets'][location]+action
    embeddings=[record['x'][location,int(record['actions'][name][k])]
                if int(record['actions'][name][k])>=0 else torch.zeros(128)
                for name in ('sources','targets')]
    return torch.cat((record['h'][location],*embeddings,
        record['actions']['action_features'][k],record['actions']['attack_preview'][k]))


def world_scores(logits, first_wins, worlds):
    """Return [head, parent, alternative] means for each equally weighted world."""
    logits = np.asarray(logits)
    if not np.isfinite(logits).all():
        raise FloatingPointError('Non-finite action-value predictions')
    rates = np.asarray(first_wins, dtype=np.float64)/4
    choose = (logits[:, 1] > logits[:, 0]).astype(np.int64)
    values = np.column_stack((rates[np.arange(len(rates)), choose], rates))
    return np.stack([values[np.asarray(worlds) == w].mean(0) for w in sorted(set(worlds))])


def utility(scores):
    means = np.asarray(scores).mean(0)
    return float(means[0]-max(means[1], means[2]))


def summarize(logits, first_wins, worlds):
    scores = world_scores(logits, first_wins, worlds)
    means = scores.mean(0)
    z = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(first_wins)/4, dtype=torch.float64)
    return dict(G=utility(scores), head=float(means[0]), parent=float(means[1]),
        alternative=float(means[2]), bce=float(F.binary_cross_entropy_with_logits(z, y)),
        brier=float((z.sigmoid().square()-2*z.sigmoid()*y+y).mean()),
        alternative_fraction=float((z[:, 1] > z[:, 0]).double().mean()),
        worlds={str(w): row.tolist() for w, row in zip(sorted(set(worlds)), scores)})


def bootstrap(scores, samples=10000, seed=20261103):
    scores = np.asarray(scores, dtype=np.float64)
    indices = np.random.Generator(np.random.PCG64(seed)).integers(0, len(scores), (samples, len(scores)))
    means = scores[indices].mean(1)
    draws = means[:, 0]-np.maximum(means[:, 1], means[:, 2])
    low, high = np.quantile(draws, [.025, .975])
    g = utility(scores)
    return dict(G=g, ci95=[float(low), float(high)], samples=samples, seed=seed,
                clusters=len(scores), supported=bool(g >= .01 and low > 0))


@torch.no_grad()
def predict(head, inputs):
    return head(inputs.flatten(0, 1)).reshape(-1, 2).cpu().numpy()
