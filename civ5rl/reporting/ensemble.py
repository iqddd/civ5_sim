"""Frozen probability ensembles, using the ordinary match and RPC evaluator."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
from time import perf_counter
import torch
from civ5rl.core.agent import RLAgent
from civ5rl.core.model import batch_inputs
from civ5rl.core.observation import ordered_actions
from civ5rl.engine.evaluation import EvaluationExecutor, EvaluationRuntime, BatchedInference, evaluation_precision
from civ5rl.provenance.common import read_json, write_json, file_sha256, configure
from civ5sim.engine import digest


def mean_probabilities(logits):
    return torch.stack([(value / .5).softmax(-1) for value in logits]).mean(0)


class EnsembleInference(BatchedInference):
    def __init__(self, policies, device, *, hidden_capacity=32):
        super().__init__(policies[0], device, hidden_capacity=hidden_capacity)
        self.policies = policies

    def combine_encoded_logits(self, logits_all, requests):
        choices=mean_probabilities(logits_all).argmax(-1).cpu().tolist()
        return [None if command=='observe' else dict(index=choices[i])
                for i,(_,command,_) in enumerate(requests)]

    @torch.inference_mode()
    def infer(self, requests):
        started = perf_counter()
        actions = [ordered_actions(r[3]) for r in requests]
        logits_all = []
        for member, policy in enumerate(self.policies):
            agents = [r[0].members[member] for r in requests]
            encoded = [a.encoder.encode(r[2], legal) for a, r, legal in zip(agents, requests, actions)]
            batch = batch_inputs(encoded, self.device, compact_units=policy.attention_backend=='compact_fp32')
            hidden = self.gather_hidden(policy, agents, member)
            logits, _, hidden = policy(batch, hidden)
            logits_all.append(logits)
            self.store_hidden(agents, hidden, member)
        choices = mean_probabilities(logits_all).argmax(-1).cpu().tolist()
        outputs = []
        for i, (agent, command, observation, _, forced) in enumerate(requests):
            chosen = forced if command == 'observe' else actions[i][choices[i]]
            if chosen not in actions[i]:
                raise ValueError('Forced action is not legal')
            for member in agent.members:
                member.encoder.remember_action(chosen, observation)
            outputs.append(None if command == 'observe' else (chosen, []))
        self.forward_seconds += perf_counter()-started
        self.batches += 1
        self.items += len(requests)
        return outputs


class EnsembleExecutor(EvaluationExecutor):
    def __init__(self, checkpoints, runtime):
        super().__init__(str(checkpoints[0]), runtime)
        policies = [self.inference.policy]
        from civ5rl.engine.evaluation import prepare_evaluation_policy
        policies += [prepare_evaluation_policy(RLAgent.from_checkpoint(p, 'cpu', 'argmax').policy,self.runtime.device) for p in checkpoints[1:]]
        if any(p.variant != policies[0].variant for p in policies):
            raise ValueError('Ensemble variants differ')
        self.inference = EnsembleInference(policies, self.runtime.device, hidden_capacity=self.runtime.workers)

def checked_match(task):
    from civ5rl.experiments.ppo_target_scale import evaluation_task
    from civ5rl.experiments.evaluation import verify_replay
    row = evaluation_task(task)
    verify_replay(Path(task[3]))
    write_json(Path(task[3])/'checked.json', {'replay_verified':True})
    return row


def evaluate_ensemble(run, checkpoints, out):
    if len(checkpoints) != 2:
        raise ValueError('A 50/50 ensemble requires exactly two checkpoints')
    from civ5rl.reporting.evaluation_storage import readable_evaluations, archive_evaluations, restore_evaluations
    from civ5rl.reporting.evaluation_audit import load_eval_rows, clustered_paired_bootstrap, eval_wdl_summary
    from civ5rl.engine.control import RunLock
    from civ5sim.combat_ai import LIBRARY_PATH
    run, out = Path(run).resolve(), Path(out).resolve()
    checkpoints = [Path(p).resolve() for p in checkpoints]
    config = read_json(run/'run.json')['config']
    plan = read_json(run/'manifest.json')
    if plan.get('turn_reset_mode') != 'end_turn':
        raise ValueError('Requires end_turn source run')
    configure(17, 'cuda')
    runtime = EvaluationRuntime('cuda',8,32,.001).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/'evaluation.storage').exists() and not (out/'evaluation').exists():
        restore_evaluations(out)
    with evaluation_precision(), RunLock(out):
        identity = dict(schema=1, checkpoints=[dict(path=str(p),sha256=file_sha256(p)) for p in checkpoints],
                        weights=[.5,.5],temperature=.5, runtime=runtime.identity(),
                        evaluator=file_sha256(Path(__file__)), planner=file_sha256(LIBRARY_PATH),
                        manifest=file_sha256(run/'manifest.json'), source_config=digest(config),
                        execution_sources={n:file_sha256(Path(n)) for n in ['civ5rl/core/model.py','civ5rl/core/observation.py','civ5sim/match.py','build/combat.dll']})
        if (out/'identity.json').exists() and read_json(out/'identity.json') != identity:
            raise ValueError('Evaluation identity changed')
        write_json(out/'identity.json',identity)
        def evaluate(purpose, label, paths):
            cases = plan[purpose]
            expected = 512 if purpose == 'development_512' else 256
            if len(cases) != expected: raise ValueError('Wrong case count')
            folder = out/'evaluation'/purpose/label
            tasks = []
            for case in cases:
                if case['max_rounds'] != 60 or case['scenario'].get('turn_reset_mode') != 'end_turn':
                    raise ValueError('Wrong match physics')
                directory = folder/f"case-{case['case_id']:03}-side-{case['learner_side']}"
                if (directory/'checked.json').exists():
                    if read_json(directory/'case.json') != case: raise ValueError('Changed case')
                    continue
                tasks.append((config,case,str(paths[0]),str(directory)))
            started = perf_counter()
            if tasks:
                executor = EnsembleExecutor(paths,runtime) if len(paths)==2 else EvaluationExecutor(str(paths[0]),runtime)
                for count,_ in enumerate(executor.run(checked_match,tasks),1):
                    if count % 8 == 0 or count == len(tasks): print(f'{purpose}/{label}: {count}/{len(tasks)}',flush=True)
                write_json(out/f'{purpose}-{label}-timing.json',dict(seconds=perf_counter()-started,inference=executor.statistics()))
            return load_eval_rows(folder,expected_n=expected,plan=cases,require_artifacts=True)
        ensemble = evaluate('development_512','ensemble',checkpoints)
        cache = read_json(run/'evaluations.json')
        baseline = next(e for e in cache.values() if e['checkpoint']=='checkpoints/initial.pt' and e['purpose']=='development_512')
        with RunLock(run), readable_evaluations(run):
            parent = load_eval_rows(run/baseline['runs']['argmax']['cp']['directory'],512,plan=plan['development_512'],require_artifacts=True)
            if eval_wdl_summary(parent)['W'] != 361: raise ValueError('Baseline differs from fixed 361 wins')
            result = dict(development=eval_wdl_summary(ensemble),comparison=clustered_paired_bootstrap(parent,ensemble),historical_gpu_comparison=True)
        write_json(out/'summary.json',result)
        if result['development']['W'] > 361:
            parent = evaluate('confirmation','parent',[checkpoints[0]])
            candidate = evaluate('confirmation','ensemble',checkpoints)
            result['additional'] = dict(parent=eval_wdl_summary(parent),ensemble=eval_wdl_summary(candidate),comparison=clustered_paired_bootstrap(parent,candidate))
        else:
            result['additional'] = {'status':'not_run_no_development_gain'}
        for item,p in zip(identity['checkpoints'],checkpoints):
            if file_sha256(p)!=item['sha256']: raise ValueError('Checkpoint changed')
        write_json(out/'summary.json',result)
        (out/'report.md').write_text('# Fixed 50/50 probability ensemble\n\nDevelopment: '+str(result['development'])+'\n\nPaired comparison: '+str(result['comparison'])+'\n\nAdditional: '+str(result['additional'])+'\n\nT=0.5; end_turn, 60 rounds, CUDA FP32. Historical parent Development used RTX 5090; local GPU may differ numerically. All match replays verified; no training or closed test.\n',encoding='utf-8')
    archive_evaluations(out)
    return result
