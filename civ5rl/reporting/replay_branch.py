"""Verified command-sequence branches of saved policy-vs-CP matches (read-only sources)."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path
from dataclasses import asdict
import json
from civ5sim import Battle, Action, MatchConfig, run_match
from civ5sim.engine import digest
from civ5sim.cp_agent import CPAgent
from civ5rl.core.agent import RLAgent
from civ5rl.core.observation import ordered_actions
from civ5rl.engine.evaluation import evaluation_precision
from civ5rl.protocols.imitation import search_config
from civ5rl.provenance.common import configure, file_sha256, write_json


def branch_replay(source, command, action, *, checkpoint=None, out=None, device="cuda", check_only=False, forbidden_move_targets=(), continue_to_round=None):
    source = Path(source).resolve()
    if source.is_dir():
        source = source / "replay.json"
    rep = json.loads(source.read_text(encoding="utf-8"))
    case = json.loads(source.with_name("case.json").read_text(encoding="utf-8"))
    if continue_to_round is not None:
        if type(continue_to_round) is not int or continue_to_round <= case["max_rounds"]:
            raise ValueError("Continuation must extend the original round limit")
        command = len(rep["steps"]) + 1
    if not 1 <= command <= len(rep["steps"]) + int(continue_to_round is not None):
        raise ValueError("command is a one-based existing command number")
    boundary = command - 1
    supplied = action if isinstance(action, list) else [action]
    if not supplied and not forbidden_move_targets and continue_to_round is None:
        raise ValueError("Provide replacements or forbidden_move_targets")
    replacements = [a if isinstance(a, Action) else Action(**a) for a in supplied]
    for a in replacements:
        a.native()
    replacement = replacements[0] if replacements else None
    with Battle(rep["initial"]) as battle:
        for i, step in enumerate(rep["steps"][:boundary], 1):
            result = battle.step(step["action"])
            if asdict(result) != step["result"] or digest(battle.snapshot()) != step["state_sha256"]:
                raise ValueError(f"Source prefix differs at command {i}")
        if continue_to_round is None and battle.snapshot()["state"]["side"] != case["learner_side"]:
            raise ValueError("Replacement must be on the model's side")
        restriction_turn = battle.snapshot()["state"]["turn"]
        if any(type(t) is not int or not 0 <= t < len(battle._map["tiles"]) for t in forbidden_move_targets):
            raise ValueError("Forbidden move target is outside the map")
        checks = []
        for offset, a in enumerate(replacements):
            active = battle.snapshot()["state"]["side"]
            available = [asdict(v) for v in battle.available_actions() if v.unit == a.unit]
            result = battle.step(a) if active == case["learner_side"] else None
            checks.append(dict(command=command+offset, action=asdict(a),
                               legal=result is not None and result.ok,
                               error=result.error if result is not None else "not_model_turn",
                               available_for_unit=available))
            if not checks[-1]['legal']:
                break
        legal = all(c['legal'] for c in checks) and len(checks) == len(replacements)
        error = next((c['error'] for c in checks if not c['legal']), None)
    inspection = dict(command=command, original=rep["steps"][boundary]["action"] if boundary < len(rep["steps"]) else None, replacement=asdict(replacement) if replacement else None,
                      forbidden_move_targets=list(forbidden_move_targets), restriction_turn=restriction_turn,
                      replacements=[asdict(a) for a in replacements],checks=checks,
                      legal=legal, error=error, prefix_verified=boundary,
                      source=str(source), source_sha256=file_sha256(source))
    if check_only:
        return inspection
    if not legal:
        raise ValueError(f"Replacement at {command:03d} is illegal: {error}; use --check-only for legal alternatives")
    if out is None or checkpoint is None:
        raise ValueError("checkpoint and out are required for execution")
    out, checkpoint = Path(out).resolve(), Path(checkpoint).resolve()
    if out.exists():
        raise ValueError("Use a new output directory")
    if out == source.parent or source.is_relative_to(out):
        raise ValueError("Output must not contain the source")
    sha = file_sha256(checkpoint)
    configure(17, device)
    class Shared:
        index = 0
        selected = None
        masked_decisions = 0

    class CP:
        def __init__(self):
            self.inner = CPAgent(search_config(dict(teacher_search=case['opponent_search'])))

        def reset(self, context):
            self.inner.reset({**context, "max_rounds": case["max_rounds"]})
            self.planner = self.inner.planner

        def act_with_planning(self, observation, actions, services):
            action = self.inner.act_with_planning(observation, actions, services)
            if Shared.index < boundary and asdict(action) != rep['steps'][Shared.index]['action']:
                raise ValueError(f"CP decision differs at command {Shared.index + 1}")
            Shared.index += 1
            return action

        def on_action_result(self, *args):
            return self.inner.on_action_result(*args)

        def drain_diagnostics(self):
            return self.inner.drain_diagnostics()

    class Model:
        def __init__(self):
            self.inner = RLAgent.from_checkpoint(checkpoint, 'cpu', 'argmax')
            self.inner.policy.to(device)
            self.inner.device = device
            self.requires_attack_preview = self.inner.requires_attack_preview

        def reset(self, context):
            self.inner.reset({**context, "max_rounds": case["max_rounds"]})

        def act(self, observation, actions):
            index = Shared.index
            Shared.index += 1
            if index < boundary + len(replacements):
                actions = ordered_actions(actions)
                # Forward the actual history once, then remember the executed command,
                # including prior manual replacements, rather than the policy's argmax.
                import torch
                with torch.inference_mode():
                    logits = self.inner._infer(observation, actions)
                action = Action(**rep['steps'][index]['action']) if index < boundary else replacements[index-boundary]
                if action not in actions:
                    raise ValueError(f"Recorded model command is illegal at {index + 1}")
                self.inner.encoder.remember_action(action, observation)
                if index == boundary:
                    Shared.selected = asdict(actions[int(logits[0].argmax())])
                return action
            if forbidden_move_targets and observation['turn'] == restriction_turn:
                import torch
                actions = ordered_actions(actions)
                with torch.inference_mode():
                    logits = self.inner._infer(observation, actions).clone()
                    for i, a in enumerate(actions):
                        if a.kind in ('move', 'swap') and a.target in forbidden_move_targets:
                            logits[0, i] = -torch.inf
                    if not torch.isfinite(logits).any():
                        raise ValueError('Movement restriction leaves no action')
                    Shared.masked_decisions += 1
                    return self.inner.action_from_logits(logits, observation, actions)
            return self.inner.act(observation, actions)

        def drain_diagnostics(self):
            return self.inner.drain_diagnostics()

    cp = CP()
    with evaluation_precision():
        model = Model()
        try:
            scenario = case['scenario']
            config = MatchConfig(scenario['map'],case['armies'],continue_to_round or case['max_rounds'],
                                 seed=case['streams']['battle'],fog=scenario['fog'],
                                 random_damage=scenario['random_damage'],
                                 max_calls_per_turn=case['max_calls_per_turn'],
                                 turn_reset_mode=rep['initial']['state'].get('turn_reset_mode','legacy'))
            agents = [cp, model] if case['learner_side'] == 1 else [model, cp]
            match = run_match(config, agents, scenario=scenario, agent_seeds=case['agent_seeds'])
        finally:
            cp.inner.close()
    if match.result['status'] != 'completed':
        raise RuntimeError(f"Branch failed: {match.result.get('error')}")
    if match.replay['steps'][:boundary] != rep['steps'][:boundary]:
        raise ValueError('Reconstructed model/CP prefix differs physically or in RNG')
    if [s['action'] for s in match.replay['steps'][boundary:boundary+len(replacements)]] != [asdict(a) for a in replacements]:
        raise ValueError('Replacement was not executed')
    match.replay['experiment'] = dict(kind='counterfactual_action',prefix_commands=boundary,
        description=f'С команды {command:03d}: замены {[asdict(a) for a in replacements]}; запрещённые цели перемещений {list(forbidden_move_targets)} до конца хода {restriction_turn}. Память и RNG префикса сохранены.',
        source=str(source),source_sha256=inspection['source_sha256'],replacement=asdict(replacement) if replacement else None,replacements=[asdict(a) for a in replacements],
        forbidden_move_targets=list(forbidden_move_targets),restriction_turn=restriction_turn,masked_decisions=Shared.masked_decisions,
        continue_to_round=continue_to_round,model_time_horizon=case['max_rounds'],
        original_action=inspection['original'],checkpoint=str(checkpoint),checkpoint_sha256=sha,
        original_model_argmax_at_branch=Shared.selected,cp_prefix_decisions_verified=True)
    if file_sha256(checkpoint) != sha or file_sha256(source) != inspection['source_sha256']:
        raise ValueError('An input changed during branching')
    if continue_to_round is not None:
        match.replay['experiment'].update(kind='continuation', resume_after=boundary,
            description=f'Продолжение до {continue_to_round} раундов. Все {boundary} исходных команд сохранены; временные признаки модели ограничены исходным горизонтом {case["max_rounds"]}.')
    match.save(out)
    write_json(out/'case.json',case)
    from tools.view_combat import prepare_replay
    if not prepare_replay(out/'replay.json')['verified']:
        raise ValueError('Branch replay verification failed')
    write_json(out/'checked.json',dict(replay_verified=True,prefix_verified=boundary,
                                     replay_sha256=file_sha256(out/'replay.json')))
    return dict(directory=str(out), **{k:match.result[k] for k in
                ('status','winner','completed_rounds','alive','remaining_hp','commands')})
