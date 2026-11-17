"""Short action-value smoke/profiling and human-readable offline results."""
from pathlib import Path
from time import perf_counter
import torch

from civ5rl.engine.action_value import make_head,make_optimizer,outcome_loss
from civ5rl.engine.comparison_training import atomic_torch_save
from civ5rl.provenance.common import read_json,write_json


def smoke(cfg,out):
    from torch.profiler import profile,ProfilerActivity
    from civ5rl.engine.resources import ResourceMonitor
    out=Path(out)
    if any(p.name!='run.lock' and p.name!='derived' for p in out.iterdir()):
        raise ValueError('Use an empty smoke directory')
    head=make_head();opt=make_optimizer(head)
    x=torch.randn(256,2,534,device='cuda')
    n=torch.full((256,2),4.,device='cuda');w=torch.randint(0,5,(256,2),device='cuda').float()
    def step():
        opt.zero_grad(set_to_none=True)
        loss=outcome_loss(head(x.flatten(0,1)).reshape(-1,2),w,n)
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
        return loss
    times=[]
    with ResourceMonitor(out/'telemetry') as monitor:
        monitor.mark('head_smoke')
        for i in range(8):
            torch.cuda.synchronize();start=perf_counter();step();torch.cuda.synchronize()
            if i>=3:times.append(perf_counter()-start)
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as trace:
            for _ in range(3):step()
        trace.export_chrome_trace(str(out/'trace.json'))
    atomic_torch_save(dict(model=head.state_dict(),optimizer=opt.state_dict()),out/'resume.pt')
    original=step().detach().clone();weights={k:v.detach().clone() for k,v in head.state_dict().items()}
    saved=torch.load(out/'resume.pt',weights_only=False,map_location='cpu')
    head.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer'])
    torch.testing.assert_close(step(),original,atol=0,rtol=0)
    for k,v in head.state_dict().items():torch.testing.assert_close(v,weights[k],atol=0,rtol=0)
    controller=controller_smoke(cfg,out/'controller')
    cache=cached_vector_smoke(cfg)
    result=dict(status='passed',parameters=sum(p.numel() for p in head.parameters()),
        steady_step_seconds=sum(times)/len(times),resume_exact=True,controller=controller,public_cache=cache,
        top_kernels=sorted([dict(name=e.key,cuda_us=e.device_time_total,count=e.count)
            for e in trace.key_averages() if e.device_time_total],key=lambda r:r['cuda_us'],reverse=True)[:15])
    write_json(out/'precheck.json',result)
    print(result,flush=True)
    return result


def controller_smoke(cfg,out):
    """Discardable synthetic fit: interrupted steps must equal an uninterrupted fit."""
    import numpy as np
    from unittest.mock import patch
    from civ5rl.engine.action_value import split_worlds
    from civ5rl.experiments.action_value import train,evaluate
    from civ5rl.provenance.common import configure
    out=Path(out);split=split_worlds(list(range(10000,10057)))
    worlds=np.repeat(np.arange(10000,10057),2)
    configure(17,'cuda')
    data=dict(inputs=torch.randn(len(worlds),2,534),wins=torch.randint(0,5,(len(worlds),2)).float(),
        counts=torch.full((len(worlds),2),4.),first_wins=np.zeros((len(worlds),2),dtype=int),worlds=worlds,
        keys=[[i,0] for i in range(len(worlds))])
    data['first_wins']=data['wins'].numpy().astype(int)
    options={**cfg,'epochs':2,'batch_forks':16}
    reference=out/'reference';resumed=out/'resumed'
    for directory in (reference,resumed):
        write_json(directory/'split.json',split);write_json(directory/'data.json',dict(preparation_seconds=0))
    configure(17,'cuda');train(options,reference,data)
    original_step=torch.optim.AdamW.step
    calls=0
    def interrupted(optimizer,*args,**kwargs):
        nonlocal calls
        result=original_step(optimizer,*args,**kwargs);calls+=1
        if calls==7:raise InterruptedError('Smoke interruption in an uncommitted epoch')
        return result
    configure(17,'cuda')
    with patch.object(torch.optim.AdamW,'step',interrupted):
        try:train(options,resumed,data)
        except InterruptedError:pass
        else:raise AssertionError('Smoke interruption did not occur')
    train(options,resumed,data)
    a=torch.load(reference/'training.pt',weights_only=False,map_location='cpu')
    b=torch.load(resumed/'training.pt',weights_only=False,map_location='cpu')
    for key in ('epoch','steps','best','cursor','order'):
        assert a[key]==b[key],key
    for name in a['model']:torch.testing.assert_close(a['model'][name],b['model'][name],atol=0,rtol=0)
    import random
    configure(17,'cuda');legacy=make_head();legacy_opt=make_optimizer(legacy)
    shuffle=random.Random('c2p_action_value_shuffle_v1|17')
    training=np.flatnonzero(np.isin(data['worlds'],split['training'])).tolist()
    x,w,n=[data[key].cuda() for key in ('inputs','wins','counts')]
    for _ in range(2):
        order=training[:];shuffle.shuffle(order)
        for start in range(0,len(order),16):
            ids=torch.tensor(order[start:start+16],device='cuda')
            legacy_opt.zero_grad(set_to_none=True)
            outcome_loss(legacy(x[ids].flatten(0,1)).reshape(-1,2),w[ids],n[ids]).backward()
            torch.nn.utils.clip_grad_norm_(legacy.parameters(),1.);legacy_opt.step()
    for name,value in legacy.state_dict().items():
        torch.testing.assert_close(value.cpu(),a['model'][name],atol=0,rtol=0)

    for key,row in a['optimizer']['state'].items():
        for name,value in row.items():torch.testing.assert_close(value,b['optimizer']['state'][key][name],atol=0,rtol=0)
    train(options,resumed,data)
    result=evaluate(options,resumed,data)
    receipt=(resumed/'selected.json').read_bytes()
    assert evaluate(options,resumed,data)==result
    assert (resumed/'selected.json').read_bytes()==receipt
    predictions=torch.load(resumed/'confirmation-predictions.pt',weights_only=False)
    probs=torch.as_tensor(predictions['logits'],dtype=torch.float64).sigmoid().numpy()
    np.testing.assert_array_equal(predictions['probs'],probs)
    np.testing.assert_array_equal(predictions['delta_prob'],probs[:,1]-probs[:,0])
    assert b['best']['epoch']==1 or b['best']['G']>a['history'][0]['selection']['G']
    return dict(resume_exact=True,steps=b['steps'],terminal_fit_stable=True,confirmation_idempotent=True,
        epoch_boundary_resume=True,probabilities_exported=True,original_update_math_exact=True)


def cached_vector_smoke(cfg):
    """Compare the vector assembler to the frozen policy on real cached actions."""
    from civ5rl.engine.comparison_heads import configure_heads
    from civ5rl.engine.comparison_inputs import Histories
    from civ5rl.engine.action_value import public_action_vector
    from civ5rl.provenance.common import ROOT,file_sha256
    directory=Histories(ROOT/cfg['feature_cache_run']).directory.parent/'head-features-fp32'
    receipt=next(directory.glob('111068-*.json'),None)
    if receipt is None:
        return dict(status='cache_absent',note='Full preparation rebuilds features and checks ordinary-history parity')
    metadata=read_json(receipt);path=receipt.with_suffix('.pt')
    assert file_sha256(path)==metadata['sha256']
    assert metadata['identity']['common']['parent']==cfg['parent_sha256']
    record=torch.load(path,weights_only=False,map_location='cpu',mmap=True)
    assert record['identity']==metadata['identity']
    from civ5rl.engine.action_value_features import FrozenActionFeatures
    dataset=read_json(ROOT/cfg['collection_input']['run']/'dataset.json')
    subset={**dataset,'pairs':[p for p in dataset['pairs'] if p['episode']==111068],
        'anchors':[k for k in dataset['anchors'] if k[0]==111068]}
    histories=Histories(ROOT/cfg['feature_cache_run'])
    # The normal preparation verifies these bytes against the source tree;
    # this bounded smoke has the already recorded hashes and existing packet.
    histories.sources[111068]=dataset['source_hashes']['111068']
    fast=FrozenActionFeatures(histories,subset,ROOT/cfg['parent']).prepare()
    assert fast.statistics['cache_hits']==1 and fast.statistics['policy_loads']==0
    torch.testing.assert_close(fast.records[111068]['x'],record['x'],atol=0,rtol=0)
    torch.testing.assert_close(fast.records[111068]['h'],record['h'],atol=0,rtol=0)
    policy=configure_heads(ROOT/cfg['parent']);policy.requires_grad_(False)
    start,stop=record['offsets'][:2]
    batch={name:value[start:stop].unsqueeze(0).cuda() for name,value in record['actions'].items()}
    batch['action_mask']=torch.ones(1,stop-start,dtype=torch.bool,device='cuda')
    x=record['x'][:1].cuda();h=record['h'][:1].cuda()
    with torch.no_grad():
        logits,_=policy.heads(batch,x,h)
        vectors=torch.stack([public_action_vector(record,0,a) for a in range(stop-start)]).cuda()
        manual=policy.action[2](policy.action[1](policy.action[0](vectors[:,:529])+policy.attack_preview(vectors[:,529:]))).flatten()
    torch.testing.assert_close(manual,logits[0],atol=1e-4,rtol=1e-4)
    return dict(episode=111068,actions=stop-start,max_logit_error=float((manual-logits[0]).abs().max()),
        cache_policy_loads=fast.statistics['policy_loads'])


def report(out):
    out=Path(out)
    if not (out/'training.json').exists():return
    training=read_json(out/'training.json')
    lines=['# Action-value fitting','',f"Completed epochs: {training['epoch']}; steps: {training['steps']}; training: {training['seconds']:.2f}s.",
        '',f"Best selection: {training['best']}."]
    if (out/'result.json').exists():
        result=read_json(out/'result.json');c=result['confirmation'];b=result['bootstrap']
        lines+=['','| Confirmation rule | Mean win fraction |','|---|---:|']
        lines += [f'| {k} | {c[k]:.6f} |' for k in ('head','parent','alternative')]
        lines+=['',f"G: {b['G']:+.6f}; CI95: {b['ci95']}; support for a limited policy experiment: {b['supported']}.",
            '',f"Preparation: {result['preparation_seconds']:.2f}s; confirmation: {result['confirmation_seconds']:.2f}s.",
            '', 'This assesses the new head on held-out training worlds; u171 has used these worlds. It does not establish full-battle policy improvement. Working policy: u171.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
