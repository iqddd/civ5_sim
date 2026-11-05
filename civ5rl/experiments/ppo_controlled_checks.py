"""Focused acceptance and reporting for the controlled PPO experiment."""

from collections import Counter
from copy import deepcopy
from time import perf_counter

import torch

from civ5rl.engine.optimizer_transaction import AdamSnapshot, require_finite_state
from civ5rl.engine.ppo_fast import BatchedBackend
from civ5rl.experiments.imitation_helpers import assert_equal
from civ5rl.provenance.common import ROOT, configure, read_json, write_json


def preflight(config, plan, out):
    # This checks live optimized CUDA graphs without another collection campaign.
    from civ5rl.experiments.ppo_after_imitation import create_policy, optimizer_for
    from civ5rl.experiments.ppo_after_imitation_checks import public_episodes
    configure(17, "cuda")
    policy = create_policy(config, "cuda")
    optimizer = optimizer_for(policy, config, "cuda")
    episodes = public_episodes(policy, "cuda")
    backend = BatchedBackend(policy, episodes, config, "cuda")
    backend.precheck()
    spans = [(i, 1, len(e["chosen"])) for i, e in enumerate(episodes)]
    count = sum(b-a for _, a, b in spans)
    checks = []
    for initialized in (False, True):
        print(f"Controlled CUDA rollback: initialized Adam={initialized}", flush=True)
        optimizer.zero_grad(set_to_none=True)
        backend.backward(spans, count, 0)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), .5, error_if_nonfinite=True)
        snapshot = AdamSnapshot(policy, optimizer)
        expected_model = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        expected_adam = deepcopy(optimizer.state_dict())
        before = backend.kl()
        optimizer.step()
        require_finite_state(policy, optimizer)
        attempted = backend.kl()
        snapshot.restore()
        restored = backend.kl()
        assert_equal(policy.state_dict(), expected_model)
        assert_equal(optimizer.state_dict(), expected_adam)
        if before is None:
            assert restored is None
        else:
            torch.testing.assert_close(restored, before, rtol=1e-5, atol=1e-7)
        checks.append({"initialized_adam": initialized, "before_kl": before,
                       "attempted_kl": attempted, "restored_kl": restored,
                       "bitwise_model_adam_equal": True})
        # Keep one real accepted initialization only in this disposable fixture.
        optimizer.step()
    torch.cuda.synchronize()
    write_json(out / "preflight.json", {"passed": True, "CUDA_BF16_rollback": checks,
        "research_updates": 0, "CP_collection_code_unchanged_in_this_change": True,
        "full_suite_and_resume_tests": False})
    del policy, optimizer, backend, episodes
    torch.cuda.empty_cache()


def extend_report(config, out, state, rows, final):
    lines = ["", "## Контроль принятия шагов", "",
        f"Rollouts: {state['updates']}; с принятыми шагами: {state['accepted_updates']}; "
        f"Adam attempted/accepted/rejected: {state['attempted_steps']}/{state['optimizer_steps']}/{state['rejected_steps']}.",
        "LR=1e-5. Приём по полному categorical KL(old || current) на свободных состояниях; предел 0,015. Sampled KL — диагностика.",
        "Те же 57 training-worlds, 228 crops и streams seed 17; физические расстановки не объявляются независимыми от прежнего опыта.",
        "", "| Rollout | Попытки/принятые/отклонённые | Max attempted KL | Final accepted KL | Loss policy/value | Critic mean/MSE/EV | Gradient mean/max | Weight delta | Epochs started/completed |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p = r['ppo']
        lines.append(f"| {r['update']} | {p.get('attempted_optimizer_steps', 0)}/{p['optimizer_steps']}/{p.get('rejected_optimizer_steps', 0)} | {p.get('max_attempted_kl')} | {p.get('final_accepted_kl')} | {p.get('policy_loss')}/{p.get('value_loss')} | {p.get('critic_mean')}/{p.get('critic_mse')}/{p.get('critic_explained_variance')} | {p.get('gradient_norm_mean')}/{p.get('gradient_norm_max')} | {r['weight_delta_l2']} | {p.get('epochs_started')}/{p.get('epochs_completed')} |")
    parent = read_json(ROOT / config['experiment']['parent_run'] / 'state.json')
    previous = parent['evaluated'].get('104')
    lines += ["", "## Сопоставление с прежним PPO update 104", ""]
    if previous:
        for mode, enemies in previous['runs'].items():
            for enemy, result in enemies.items():
                lines.append(f"- Прежний update 104, {mode}/{enemy}: {result['summary']}.")
    if final and final.get('baseline_selected'):
        lines += ["", "**PPO-кандидат не прошёл development-отбор.** Дополнительное сравнение одинаковой политики не является проверкой эквивалентности; CI не рассчитывался."]
    for name, field in [('Collection', None), ('PPO including KL', 'ppo_seconds'), ('KL', 'kl_seconds'), ('Snapshots/rollback', 'transaction_seconds'), ('Runtime compilation', 'runtime_compilation_seconds')]:
        total = sum(r['collection'].get('collection_seconds', 0) if field is None else r['ppo'].get(field, 0) for r in rows)
        lines.append(f"{name}: {total:.3f} с.")
    lines += ["", "Loss/градиенты и обработанные позиции включают отвергнутые попытки; Adam counter checkpoint включает только принятые. Детали каждой попытки находятся в updates.jsonl."]
    if rows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
        evaluated = [(int(k), v) for k, v in state['evaluated'].items()]
        decisions = {0: 0, **{r['update']: r['total_decisions'] for r in rows}}
        for mode in ('argmax', 'sample'):
            axes[0].plot([decisions.get(n, 0) for n, e in evaluated],
                         [e['runs'][mode]['cp']['summary']['win_rate'] for n, e in evaluated],
                         marker='o', label=f'Development CP {mode}')
            if previous:
                axes[0].axhline(previous['runs'][mode]['cp']['summary']['win_rate'], linestyle=':', label=f'Previous update104 {mode}')
        axes[1].plot([r['total_decisions'] for r in rows], [r['ppo'].get('max_attempted_kl') or 0 for r in rows], label='Max attempted full KL')
        axes[1].plot([r['total_decisions'] for r in rows], [r['ppo'].get('final_accepted_kl') or 0 for r in rows], label='Final accepted full KL')
        axes[1].axhline(.015, linestyle=':', label='Acceptance threshold')
        for ax in axes:
            ax.grid(alpha=.25)
            ax.legend()
            ax.set_xlabel('Learner decisions')
        fig.savefig(out / 'controlled-learning-curves.png', dpi=150)
        plt.close(fig)
    return lines
