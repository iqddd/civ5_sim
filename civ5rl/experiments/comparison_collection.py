"""Public histories and paired counterfactual matches with exact CP prefix replay."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
import random
import torch
from civ5rl.core.agent import RLAgent
from civ5rl.core.model import batch_inputs
from civ5rl.core.observation import ordered_actions
from civ5rl.engine.evaluation import EvaluationExecutor, BatchedInference, evaluation_agent
from civ5rl.protocols.imitation import search_config
from civ5rl.provenance.common import read_json, write_json, seed_for
from civ5sim import Action, MatchConfig, run_match
from civ5sim.cp_agent import CPAgent
from civ5sim.engine import digest


def select_forks(history, episode_id, namespace, maximum=4):
    rounds = {}
    for i, row in enumerate(history):
        if row['chosen'] != row['alternative']:
            rounds.setdefault(row['observation']['turn'], []).append(i)
    rng = random.Random(seed_for(17, episode_id, 'selection', namespace=namespace))
    chosen = rng.sample(sorted(rounds), min(maximum, len(rounds)))
    return sorted(rng.choice(rounds[r]) for r in chosen)


def preference(winners, side):
    # Two alternatives, eight paired futures. Rewards remain exactly 1/0/0.
    wins = [[int(w == side) for w in row] for row in winners]
    deltas = [sum(wins[1][i:i+4])-sum(wins[0][i:i+4]) for i in (0,4)]
    d1,d2 = deltas
    reason = ('both_zero' if d1 == d2 == 0 else 'one_zero' if not d1 or not d2
              else 'opposite_signs' if d1*d2 < 0 else 'accepted')
    return dict(deltas=deltas, accepted=reason=='accepted', reason=reason,
                better=1 if d1>0 else 0, weight=abs(d1+d2)/8 if reason=='accepted' else 0.,
                wdl=[{k:sum(('D' if w is None else 'W' if w==side else 'L')==k for w in row) for k in 'WDL'} for row in winners])


class ProposalInference(BatchedInference):
    def __init__(self, policies, device):
        super().__init__(policies[0],device)
        self.policies=policies

    @torch.inference_mode()
    def infer(self, requests):
        start=perf_counter()
        legal=[ordered_actions(r[3]) for r in requests]
        choices=[]
        for member,policy in enumerate(self.policies):
            agents=[r[0].members[member] for r in requests]
            encoded=[a.encoder.encode(r[2],acts) for a,r,acts in zip(agents,requests,legal)]
            hidden=torch.cat([a.hidden if a.hidden is not None else torch.zeros(1,1,policy.hidden_size,device=self.device) for a in agents],1)
            logits,_,hidden=policy(batch_inputs(encoded,self.device),hidden)
            choices.append(logits.argmax(-1).cpu().tolist())
            for i,a in enumerate(agents): a.hidden=hidden[:,i:i+1].clone()
        outputs=[]
        for i,(agent,command,obs,_,forced) in enumerate(requests):
            action=forced if command=='observe' else legal[i][choices[0][i]]
            if action not in legal[i]: raise ValueError('Illegal recorded command')
            for a in agent.members: a.encoder.remember_action(action,obs)
            outputs.append(None if command=='observe' else (action,[dict(kind='proposal',alternative=choices[1][i])]))
        self.forward_seconds+=perf_counter()-start
        self.batches+=1;self.items+=len(requests)
        return outputs


class ProposalExecutor(EvaluationExecutor):
    def __init__(self, checkpoints, runtime):
        super().__init__(str(checkpoints[0]),runtime)
        second=RLAgent.from_checkpoint(checkpoints[1],'cpu','argmax').policy.to(self.runtime.device).eval()
        self.inference=ProposalInference([self.inference.policy,second],self.runtime.device)

    def _receive(self, connection):
        result=super()._receive(connection)
        a=self.connections.get(connection)
        if a is not None and not hasattr(a,'members'):
            second=RLAgent(self.inference.policies[1],self.runtime.device,'argmax')
            second.encoder=deepcopy(a.encoder);second.hidden=None
            a.members=[a,second]
        return result


def _verify(directory):
    # Continuation replays start at the reseeded snapshot, but retain the original horizon.
    from civ5rl.experiments.evaluation import verify_replay
    verify_replay(directory)
    write_json(directory/'checked.json',dict(replay_verified=True))


def match_task(task):
    cfg,case,folder,checkpoint,branch=task
    folder=Path(folder)
    command=0
    history=[]
    source=read_json(Path(branch['source'])/'replay.json') if branch else None
    boundary=branch['command'] if branch else None
    prefix_verified=False
    def hook(battle,index):
        nonlocal command,prefix_verified
        command=index
        if source is None: return
        if index <= boundary:
            expected=source['initial'] if index==0 else None
            actual=battle.snapshot()
            expected_hash=digest(expected) if index==0 else source['steps'][index-1]['state_sha256']
            if digest(actual)!=expected_hash: raise ValueError(f'Prefix state differs at {index}')
        if index==boundary:
            saved=battle.snapshot()
            saved['state']['rng']=branch['seed']
            battle.reset(saved)
            prefix_verified=True
    class Learner:
        def __init__(self):
            self.inner=evaluation_agent(checkpoint,'argmax',temperature=.5)
        def reset(self,context):
            self.inner.reset(context)
            self.requires_attack_preview=self.inner.requires_attack_preview
        def act(self,obs,actions):
            legal=ordered_actions(actions)
            if branch and command<=boundary:
                action=Action(**(source['steps'][command]['action'] if command<boundary else branch['action']))
                if action not in legal: raise ValueError('Illegal prefix/alternative')
                self.inner.observe_step(obs,legal,action)
                return action
            action=self.inner.act(obs,legal)
            if not branch:
                events=self.inner.drain_diagnostics()
                alternative=next(e['alternative'] for e in events if e['kind']=='proposal')
                history.append(dict(command=command,observation=deepcopy(obs),actions=[asdict(a) for a in legal],chosen=legal.index(action),alternative=alternative))
            return action
    class Opponent:
        def __init__(self):self.inner=CPAgent(search_config(cfg))
        def reset(self,context):self.inner.reset(context);self.planner=self.inner.planner
        def act_with_planning(self,obs,actions,services):
            action=self.inner.act_with_planning(obs,actions,services)
            if branch and command<boundary and asdict(action)!=source['steps'][command]['action']:
                raise ValueError('CP prefix command differs')
            return action
        def on_action_result(self,*args):return self.inner.on_action_result(*args)
    learner,cp=Learner(),Opponent()
    agents=[None,None];side=case['learner_side'];agents[side]=learner;agents[1-side]=cp
    scenario=case['scenario']
    config=MatchConfig(scenario['map'],case['armies'],60,fog=scenario['fog'],random_damage=scenario['random_damage'],max_calls_per_turn=case['max_calls_per_turn'],turn_reset_mode='end_turn')
    try:
        match=run_match(config,agents,scenario=scenario,agent_seeds=case['agent_seeds'],before_decision=hook)
    finally:cp.inner.close()
    folder.mkdir(parents=True,exist_ok=True)
    match.save(folder)
    write_json(folder/'case.json',case)
    if match.result['status']!='completed': raise RuntimeError(match.result['error'])
    if branch and not prefix_verified: raise ValueError('Branch boundary not reached')
    _verify(folder)
    if not branch: write_json(folder/'history.json',history)
    row=dict(winner=match.result['winner'],side=side,commands=match.result['commands'],rounds=match.result['completed_rounds'],prefix_verified=prefix_verified if branch else None)
    write_json(folder/'completed.json',row)
    return str(folder)
