"""Whole-turn fit orchestration using the shared transactional update controller."""

import random
from pathlib import Path

import torch

from civ5rl.engine.comparison.control import ComparisonTrainingState, run_updates
from civ5rl.engine.comparison import training as shared
from civ5rl.engine.comparison.turn_training import (
    control_kl, current_chosen, hidden_states_at, logical_step, pack_batch, prepare_anchor_subset,
    prepare_preferences, reference_distributions,
)
from civ5rl.engine.runtime.transactions import cpu_snapshot, atomic_torch_save
from civ5rl.provenance.common import ROOT, read_json, write_json, seed_for
from civ5rl.provenance.rng import capture_rng, restore_rng
from civ5sim.engine import digest


class TurnTrainingState(ComparisonTrainingState):
    def record_accepted(self,pending,count,kl):
        super().record_accepted(pending,count,kl)
        per_epoch=self.metadata['steps_per_epoch']
        self.epoch=self.steps//per_epoch
        self.cursor=(self.steps%per_epoch)*self.metadata['positions_per_step']


def configure_policy(parent, *, compile_model=True):
    # Keep the inherited model and all scientific FP32 operators. Compilation is
    # an execution option; no CUDA Graph pools or mixed-precision paths are used.
    if compile_model:
        return shared.configure_policy(parent,mixed=False)
    from civ5rl.core.agent import RLAgent
    policy=RLAgent.from_checkpoint(parent,'cpu','argmax').policy.to('cuda').train()
    policy.enable_amp(False)
    policy.enable_inverse_routing();policy.enable_action_routing();policy.enable_maximum_routing()
    policy.configure_execution(activation_checkpointing=False,attention_backend=shared.ATTENTION_BACKEND)
    for p in policy.value.parameters():p.requires_grad_(False)
    return policy


def parent_references(policy, histories, dataset, keys, out, cfg, *, branch_steps=None):
    identity = dict(
        version=2,
        dataset=digest(dataset),
        parent=cfg['parent_sha256'][0],
        keys=keys,
        precision='deterministic_cuda_fp32_no_amp_no_tf32',
        temperature=0.5,
        attention_backend=policy.attention_backend,
        torch=str(torch.__version__),
        gpu=torch.cuda.get_device_name(),
    )
    path=Path(out)/'parent-references.pt'
    saved=torch.load(path,map_location='cpu',weights_only=False) if path.exists() else None
    if saved is None or saved['identity']!=identity:
        with torch.no_grad(), shared.evaluation_precision():
            chosen = current_chosen(policy, histories, keys, branch_steps=branch_steps)
            anchors = reference_distributions(policy, histories, dataset['anchors'])
        saved=cpu_snapshot(dict(identity=identity,chosen=chosen,anchors=anchors))
        atomic_torch_save(saved,path)
    device=next(policy.parameters()).device
    chosen=saved['chosen'].to(device)
    return ({key:chosen[i] for i,key in enumerate(keys)},
            shared.DeviceReferences(saved['anchors'],device),
            shared.DeviceReferences(saved['anchors'],device,control=True))


def train(cfg,out,dataset,histories,budget,*,microbatch=2,compile_model=True,monitor=None):
    out=Path(out);device='cuda';options=cfg['training']
    identity=dict(dataset=digest(dataset),config=digest(cfg))
    path=out/'training.pt'
    if path.exists():
        payload=torch.load(path,map_location='cpu',weights_only=False)
        if payload['identity']!=identity:raise ValueError('Whole-turn fit input changed')
        state=TurnTrainingState.from_checkpoint_dict(payload)
    else:
        orders=[]
        for epoch in range(options['max_epochs']):
            order=list(range(len(dataset['positions'])))
            random.Random(seed_for(cfg['seed'],epoch,'positions',namespace=cfg['namespace'])).shuffle(order)
            orders.append(order)
        anchor_order=list(range(len(dataset['anchors'])))
        random.Random(seed_for(cfg['seed'],0,'anchors',namespace=cfg['namespace'])).shuffle(anchor_order)
        state=TurnTrainingState(mode='turn_preferences_step_kl',lr=options['initial_lr'],anchor_order=anchor_order)
        state.metadata.update(position_orders=orders,positions_per_step=options['positions_per_step'],
                              steps_per_epoch=len(dataset['positions'])//options['positions_per_step'])
    if state.stop_reason:
        return publish(cfg,out,state.to_checkpoint_dict(),identity)
    policy=configure_policy(str(ROOT/cfg['parents'][0]),compile_model=compile_model)
    optimizer=torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=options['initial_lr'],
                                betas=(.9,.999),eps=1e-5,weight_decay=0)
    preferences = prepare_preferences(histories, dataset['positions'], budget=budget)
    chosen_keys = sorted(histories.targets)
    anchor_subset = prepare_anchor_subset(histories, dataset['anchors'], preferences.training_observations,
                                          budget=budget)
    keys=list(dict.fromkeys([*chosen_keys,*map(tuple,dataset['anchors'])]))
    histories.prepare(keys,device=device,budget=budget)
    parent, training_refs, control_refs = parent_references(
        policy, histories, dataset, chosen_keys, out, cfg, branch_steps=preferences.branch_steps
    )
    if 'model' in state.metadata:
        policy.load_state_dict(state.metadata['model']);optimizer.load_state_dict(state.metadata['optimizer'])
        restore_rng(state.metadata['rng'])

    def save():
        state.seconds=budget.elapsed()
        state.metadata.update(identity=identity,model=policy.state_dict(),optimizer=optimizer.state_dict(),rng=capture_rng())
        payload=cpu_snapshot(state.to_checkpoint_dict())
        atomic_torch_save(payload,path)
        write_json(out/'training.json',dict(status='training',training_steps=state.steps,
            attempts=state.attempts,rejected_attempts=state.rejected_attempts,epoch=state.epoch,
            cursor=state.cursor,seconds=state.seconds,lr=state.lr,last_full_kl=state.last_full_kl,
            stop_reason=state.stop_reason,learning=state.metadata.get('learning')))
        from civ5rl.provenance.common import atomic_write_jsonl
        atomic_write_jsonl(out/'attempts.jsonl',state.history)
        atomic_write_jsonl(out/'steps.jsonl',[r for r in state.history if r['accepted']])

    current_step = None
    current_batch = None
    position_ids = None
    reference = None
    current_branch_prefixes = None
    accepted_anchor_step = None
    accepted_anchor_keys = None
    accepted_anchor_prefixes = None
    pending_anchor_prefixes = None

    controller={**options,'cycle_anchors':True,'max_accepted_steps':min(options['max_accepted_steps'],
        options['max_epochs']*state.metadata['steps_per_epoch'])}

    def update(anchors):
        nonlocal current_step, current_batch, position_ids, reference, current_branch_prefixes
        nonlocal accepted_anchor_step, accepted_anchor_keys, accepted_anchor_prefixes, pending_anchor_prefixes
        anchor_tuples = tuple(map(tuple, anchors))
        # Retries keep the accepted-step counter and reuse the same immutable inputs and verified prefixes.
        if current_step != state.steps:
            epoch, step = divmod(state.steps, state.metadata['steps_per_epoch'])
            size = options['positions_per_step']
            position_ids = state.metadata['position_orders'][epoch][step * size:(step + 1) * size]
            current_batch = pack_batch(
                [preferences.rows[i] for i in position_ids],
                device,
                histories=histories,
                branch_steps=preferences.branch_steps,
            )
            reference = (torch.stack([parent[k] for k in current_batch.keys]) if current_batch.keys
                         else torch.empty(0, device=device))
            needed_bp = [
                bk for pos in current_batch.positions for bk in pos.boundary_keys if bk is not None
            ]
            current_branch_prefixes = hidden_states_at(
                policy, histories, needed_bp, branch_steps=preferences.branch_steps
            )
            current_step = state.steps

        if accepted_anchor_step != state.steps or accepted_anchor_keys != anchor_tuples:
            if (
                pending_anchor_prefixes is not None
                and pending_anchor_prefixes[0] == state.steps
                and pending_anchor_prefixes[1] == anchor_tuples
            ):
                accepted_anchor_prefixes = pending_anchor_prefixes[2]
            else:
                needed_anchor_bp = [(ep, idx - 128) for ep, idx in set(anchor_tuples) if idx >= 128]
                bp_h = hidden_states_at(policy, histories, needed_anchor_bp)
                accepted_anchor_prefixes = {
                    (ep, idx): bp_h[(ep, idx - 128)] for ep, idx in set(anchor_tuples) if idx >= 128
                }
            accepted_anchor_step = state.steps
            accepted_anchor_keys = anchor_tuples
        pending_anchor_prefixes = None

        metrics = logical_step(
            policy,
            optimizer,
            histories,
            current_batch,
            anchors,
            reference,
            training_refs,
            microbatch=microbatch,
            anchor_prefixes=accepted_anchor_prefixes,
            branch_prefixes=current_branch_prefixes,
        )
        return {**metrics,'positions':position_ids}

    def measure():
        nonlocal pending_anchor_prefixes
        next_anchors = None
        collected = None
        if state.steps + 1 < controller['max_accepted_steps']:
            next_start = state.anchor_cursor + options['anchors_per_step']
            next_anchors = tuple(
                tuple(dataset['anchors'][state.anchor_order[(next_start + i) % len(dataset['anchors'])]])
                for i in range(options['anchors_per_step'])
            )
            collected = {}
        kl, metrics = control_kl(
            policy,
            histories,
            dataset['anchors'],
            control_refs,
            anchor_subset=anchor_subset,
            collect_prefixes=next_anchors,
            collected_prefixes=collected,
        )
        if kl <= options['kl_limit']:
            if next_anchors is not None:
                pending_anchor_prefixes = (state.steps + 1, next_anchors, collected)
            if current_batch is not None and current_batch.keys:
                values = current_chosen(
                    policy, histories, current_batch.keys, branch_steps=preferences.branch_steps
                )
                scores = ((values - reference)[current_batch.indices] * current_batch.mask).sum(-1) / current_batch.mask.sum(-1).clamp_min(1)
                margins = current_batch.signs * (scores[:, 0] - scores[:, 1])
                metrics.update(
                    preference_margin_mean=float(margins.mean()),
                    winning_execution_preferred=int((margins > 0).sum()),
                    informative_comparisons=len(margins),
                )
        else:
            pending_anchor_prefixes = None
        return kl, metrics

    try:
        if monitor is not None:monitor.mark('training')
        run_updates(policy,optimizer,state,dataset['anchors'],controller,budget,save,update,measure)
    finally:
        save()
    return publish(cfg,out,torch.load(path,map_location='cpu',weights_only=False),identity)


def publish(cfg,out,payload,identity):
    if payload.get('identity',identity)!=identity:raise ValueError('Candidate identity differs')
    if 'model' not in payload:
        parent=torch.load(ROOT/cfg['parents'][0],map_location='cpu',weights_only=False)
        payload.update(model=parent['model'],optimizer={},identity=identity)
    candidate=shared.publish_candidate(cfg,out,payload,identity)
    report=read_json(Path(out)/'training.json')
    report.update(attempts=payload.get('attempts',0),lr=payload.get('lr'),
                  rejected_attempts=payload.get('rejected_attempts',0),
                  last_full_kl=payload.get('last_full_kl',0),learning=payload.get('learning'),
                  automatic_development=False)
    write_json(Path(out)/'training.json',report)
    return candidate
