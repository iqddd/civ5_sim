"""Focused acceptance and reporting for the controlled PPO experiment."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch

from civ5rl.engine.ppo.backend import BatchedBackend
from civ5rl.engine.runtime.transactions import AdamSnapshot, require_finite_state
from civ5rl.experiments._shared.markdown import MarkdownReport
from civ5rl.experiments.imitation_helpers import assert_equal
from civ5rl.experiments.ppo_after_imitation import create_policy, optimizer_for
from civ5rl.experiments.ppo_after_imitation_checks import public_episodes
from civ5rl.provenance.common import ROOT, configure, read_json, write_json


def preflight(config: dict[str, Any], plan: dict[str, Any], out: Path) -> None:
    # This checks live optimized CUDA graphs without another collection campaign.
    configure(17, "cuda")
    policy = create_policy(config, "cuda")
    optimizer = optimizer_for(policy, config, "cuda")
    episodes = public_episodes(policy, "cuda")
    backend = BatchedBackend(policy, episodes, config, "cuda")
    backend.precheck()
    spans = [(index, 1, len(episode["chosen"])) for index, episode in enumerate(episodes)]
    count = sum(stop - start for _, start, stop in spans)
    checks = []
    for adam_initialized in (False, True):
        print(f"Controlled CUDA rollback: initialized Adam={adam_initialized}", flush=True)
        optimizer.zero_grad(set_to_none=True)
        backend.backward(spans, count, 0)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5, error_if_nonfinite=True)
        snapshot = AdamSnapshot(policy, optimizer)
        expected_model = {name: value.detach().clone() for name, value in policy.state_dict().items()}
        expected_adam = deepcopy(optimizer.state_dict())
        baseline_kl = backend.kl()
        optimizer.step()
        require_finite_state(policy, optimizer)
        attempted_kl = backend.kl()
        snapshot.restore()
        restored_kl = backend.kl()
        assert_equal(policy.state_dict(), expected_model)
        assert_equal(optimizer.state_dict(), expected_adam)
        if baseline_kl is None:
            assert restored_kl is None
        else:
            torch.testing.assert_close(restored_kl, baseline_kl, rtol=1e-5, atol=1e-7)
        checks.append(
            {
                "initialized_adam": adam_initialized,
                "before_kl": baseline_kl,
                "attempted_kl": attempted_kl,
                "restored_kl": restored_kl,
                "bitwise_model_adam_equal": True,
            }
        )
        # Keep one real accepted initialization only in this disposable fixture.
        optimizer.step()
    torch.cuda.synchronize()
    write_json(
        out / "preflight.json",
        {
            "passed": True,
            "CUDA_BF16_rollback": checks,
            "research_updates": 0,
            "CP_collection_code_unchanged_in_this_change": True,
            "full_suite_and_resume_tests": False,
        },
    )
    del policy, optimizer, backend, episodes
    torch.cuda.empty_cache()


def _acceptance_rows(rows: Sequence[dict[str, Any]]) -> Iterable[tuple[object, ...]]:
    for update in rows:
        metrics = update["ppo"]
        yield (
            update["update"],
            f"{metrics.get('attempted_optimizer_steps', 0)}/{metrics['optimizer_steps']}/{metrics.get('rejected_optimizer_steps', 0)}",
            metrics.get("max_attempted_kl"),
            metrics.get("final_accepted_kl"),
            f"{metrics.get('policy_loss')}/{metrics.get('value_loss')}",
            f"{metrics.get('critic_mean')}/{metrics.get('critic_mse')}/{metrics.get('critic_explained_variance')}",
            f"{metrics.get('gradient_norm_mean')}/{metrics.get('gradient_norm_max')}",
            update["weight_delta_l2"],
            f"{metrics.get('epochs_started')}/{metrics.get('epochs_completed')}",
        )


def _render_acceptance_section(state: dict[str, Any], rows: Sequence[dict[str, Any]]) -> MarkdownReport:
    lines = MarkdownReport(
        [
            "",
            "## Контроль принятия шагов",
            "",
            (
                f"Rollouts: {state['updates']}; с принятыми шагами: {state['accepted_updates']}; "
                f"Adam attempted/accepted/rejected: {state['attempted_steps']}/{state['optimizer_steps']}/{state['rejected_steps']}."
            ),
            "LR=1e-5. Приём по полному categorical KL(old || current) на свободных состояниях; предел 0,015. Sampled KL — диагностика.",
            "Те же 57 training-worlds, 228 crops и streams seed 17; физические расстановки не объявляются независимыми от прежнего опыта.",
            "",
        ]
    )
    headers = (
        "Rollout",
        "Попытки/принятые/отклонённые",
        "Max attempted KL",
        "Final accepted KL",
        "Loss policy/value",
        "Critic mean/MSE/EV",
        "Gradient mean/max",
        "Weight delta",
        "Epochs started/completed",
    )
    lines.table(headers, ["left"] * len(headers), _acceptance_rows(rows))
    return lines


def _append_parent_comparison(
    lines: MarkdownReport, previous: dict[str, Any] | None, final: dict[str, Any] | None
) -> None:
    lines += ["", "## Сопоставление с прежним PPO update 104", ""]
    if previous:
        for mode, enemies in previous["runs"].items():
            for enemy, result in enemies.items():
                lines.append(f"- Прежний update 104, {mode}/{enemy}: {result['summary']}.")
    if final and final.get("baseline_selected"):
        lines += [
            "",
            "**PPO-кандидат не прошёл development-отбор.** Дополнительное сравнение одинаковой политики не является проверкой эквивалентности; CI не рассчитывался.",
        ]


def _append_timings(lines: MarkdownReport, rows: Sequence[dict[str, Any]]) -> None:
    for name, field in [
        ("Collection", None),
        ("PPO including KL", "ppo_seconds"),
        ("KL", "kl_seconds"),
        ("Snapshots/rollback", "transaction_seconds"),
        ("Runtime compilation", "runtime_compilation_seconds"),
    ]:
        total = sum(
            row["collection"].get("collection_seconds", 0) if field is None else row["ppo"].get(field, 0)
            for row in rows
        )
        lines.append(f"{name}: {total:.3f} с.")
    lines += [
        "",
        "Loss/градиенты и обработанные позиции включают отвергнутые попытки; Adam counter checkpoint включает только принятые. Детали каждой попытки находятся в updates.jsonl.",
    ]


def _render_controlled_learning_curves(
    out: Path,
    state: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    previous: dict[str, Any] | None,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    evaluated = [(int(update), result) for update, result in state["evaluated"].items()]
    decisions = {0: 0, **{row["update"]: row["total_decisions"] for row in rows}}
    for mode in ("argmax", "sample"):
        axes[0].plot(
            [decisions.get(update, 0) for update, _ in evaluated],
            [result["runs"][mode]["cp"]["summary"]["win_rate"] for _, result in evaluated],
            marker="o",
            label=f"Development CP {mode}",
        )
        if previous:
            axes[0].axhline(
                previous["runs"][mode]["cp"]["summary"]["win_rate"],
                linestyle=":",
                label=f"Previous update104 {mode}",
            )
    axes[1].plot(
        [row["total_decisions"] for row in rows],
        [row["ppo"].get("max_attempted_kl") or 0 for row in rows],
        label="Max attempted full KL",
    )
    axes[1].plot(
        [row["total_decisions"] for row in rows],
        [row["ppo"].get("final_accepted_kl") or 0 for row in rows],
        label="Final accepted full KL",
    )
    axes[1].axhline(0.015, linestyle=":", label="Acceptance threshold")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
        axis.set_xlabel("Learner decisions")
    fig.savefig(out / "controlled-learning-curves.png", dpi=150)
    plt.close(fig)


def extend_report(
    config: dict[str, Any],
    out: Path,
    state: dict[str, Any],
    rows: Sequence[dict[str, Any]],
    final: dict[str, Any] | None,
) -> list[str]:
    lines = _render_acceptance_section(state, rows)
    parent = read_json(ROOT / config["experiment"]["parent_run"] / "state.json")
    previous = parent["evaluated"].get("104")
    _append_parent_comparison(lines, previous, final)
    _append_timings(lines, rows)
    if rows:
        _render_controlled_learning_curves(out, state, rows, previous)
    return list(lines)
