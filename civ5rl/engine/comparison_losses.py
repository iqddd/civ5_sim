"""Batched preference and parent-KL reductions, preserving separate sample means."""
from functools import lru_cache
import torch
from torch.nn import functional as F


def preference_and_kl(logits,pair_indices,pair_weights,reference,anchor_weights):
    # Rows with no pair are omitted; duplicates retain every original weight.
    pair_logits=logits.index_select(0,pair_indices[:,0])
    better=pair_logits.gather(1,pair_indices[:,1:2]).squeeze(1)
    worse=pair_logits.gather(1,pair_indices[:,2:3]).squeeze(1)
    pair=(pair_weights*F.softplus((worse-better)/.5)).sum()
    logq=(logits/.5).log_softmax(-1)
    delta=(reference-logq).masked_fill(torch.isneginf(logits),0.)
    kl=((reference.exp()*delta).sum(-1)*anchor_weights).sum()
    return pair,kl


def full_kl_rows(logits,reference,probabilities):
    # Retain FP32 log-softmax, followed by the established FP64 subtraction/exp.
    # Padding is -inf in logits, but must contribute zero without 0*inf NaNs.
    logq=(logits/.5).log_softmax(-1).double()
    delta=(reference-logq).masked_fill(torch.isneginf(logits),0.)
    return (probabilities*delta).sum(-1)


@lru_cache(maxsize=2)
def loss_runner(cuda):
    return (torch.compile(preference_and_kl,dynamic=True,fullgraph=True,mode='default')
            if cuda else preference_and_kl)


@lru_cache(maxsize=2)
def control_runner(cuda):
    return (torch.compile(full_kl_rows,dynamic=True,fullgraph=True,mode='default')
            if cuda else full_kl_rows)
