# Архив документации

История сохранена для происхождения решений и воспроизводимости; это не
действующий контракт и не разрешение запускать прежние «следующие шаги».
Формулировки proposal/не запущено и поздние результаты могут стоять в одном
документе: статус сверяется по [журналу](../experiment-ledger.md) и артефактам,
а не определяется первой фразой протокола.

Основные руководства: [RL](../rl-architecture.md),
[операции](../rl-operations.md), [Combat AI](../combat-ai.md),
[ядро](../combat-engine.md). Замороженные паспорта и sources.zip при ревизии
не менялись. Относительные Markdown-ссылки перенесённых документов пересчитаны.

## Combat AI: перенос и аудиты

- [combat-ai-contract-history.md](combat-ai/combat-ai-contract-history.md) — Historical contract snapshot containing outdated and overlapping descriptions predating the audit.
- [combat-ai-experiments.md](combat-ai/combat-ai-experiments.md) — Protocol and results for experiments with search variants and tactical evaluation methods.
- [combat-ai-planner-invariants.md](combat-ai/combat-ai-planner-invariants.md) — Planner fixes, their provenance, verified invariants, and supporting tactical search scenarios.
- [combat-ai-port-plan.md](combat-ai/combat-ai-port-plan.md) — Original Combat AI porting plan, completed stages, and historical verification results.
- [combat-ai-portability.md](combat-ai/combat-ai-portability.md) — Initial analysis of Community Patch portability, architectural constraints, and proposed implementation order.
- [combat-ai-risk-audit.md](combat-ai/combat-ai-risk-audit.md) — Audit of end-of-action risk, hopeless positions, and alignment with the original Community Patch.
- [combat-ai-search-audit.md](combat-ai/combat-ai-search-audit.md) — Investigation of CP search cost, optimization hypotheses, and links to supporting measurements.
- [combat-ai-striking-distance-audit.md](combat-ai/combat-ai-striking-distance-audit.md) — Analysis of attacking groups and ranged units, the integration defect, and its correction.

## RL: протоколы и направления

- [rl-cp-imitation-continuation.md](rl-protocols/rl-cp-imitation-continuation.md) — Proposal to continue BC and subsequently recorded training and evaluation results.
- [rl-cp-imitation-more-data.md](rl-protocols/rl-cp-imitation-more-data.md) — Proposal to expand CP imitation data and subsequently recorded experiment results.
- [rl-cp-imitation-next-step.md](rl-protocols/rl-cp-imitation-next-step.md) — Historical proposal to finish the first CP imitation experiment and evaluate battles.
- [rl-cp-imitation-own-states.md](rl-protocols/rl-cp-imitation-own-states.md) — Training on CP advice for policy-generated states, including data collection and results.
- [rl-cp-imitation-target-scale.md](rl-protocols/rl-cp-imitation-target-scale.md) — CP imitation protocol for target map and army sizes, including evaluation conditions.
- [rl-cp-imitation.md](rl-protocols/rl-cp-imitation.md) — Initial CP imitation preparation and cost benchmark protocol, including passport notes.
- [rl-next-stage-direction.md](rl-protocols/rl-next-stage-direction.md) — Historical proposal for the first full-battle CP imitation experiment and bounded training.
- [rl-ppo-target-scale.md](rl-protocols/rl-ppo-target-scale.md) — PPO conditions for target maps and armies, parent model, and remote transfer.
- [rl-target-bc-sampling-direction.md](rl-protocols/rl-target-bc-sampling-direction.md) — Comparison of argmax and sampling temperatures for the original BC on fixed cases.

## Операции: прежние справочники и инвентаризации

- [local-artifacts.md](operations/local-artifacts.md) — Historical inventory of restored local artifacts, archives, and experiment reproducibility limitations.
- [rl-audit.md](operations/rl-audit.md) — Auditing saved experiments, comparing paired outcomes, and interpreting available integrity checks.
- [rl-git-baseline-benchmark.md](operations/rl-git-baseline-benchmark.md) — Comparing replay against a Git revision using ABBA ordering and isolated state.
- [rl-passport-source-roles.md](operations/rl-passport-source-roles.md) — Source roles in experiment passports, frozen archive verification, and conditions for resuming runs.
- [rl-performance-telemetry.md](operations/rl-performance-telemetry.md) — PPO phase timings, GPU utilization, telemetry collection, and measurement averaging rules.
- [rl-remote-operations.md](operations/rl-remote-operations.md) — Packaging, launching, inspecting, and retrieving experiments on an existing remote instance.
- [rl-run-operations.md](operations/rl-run-operations.md) — Pausing and closing runs, generating intermediate reports, and diagnosing replay behavior.
- [rl-runtime-profiling.md](operations/rl-runtime-profiling.md) — Collection profiling, worker and environment selection, comparability criteria, and measured results.
- [rl-source-copy-cleanup.md](operations/rl-source-copy-cleanup.md) — Inventory of removed source copies, preserved patches, and reconstruction of historical implementations.

## Архитектура: прежнее описание

- [rl-module-architecture.md](architecture/rl-module-architecture.md) — RL module architecture, dependency directions, and an older description of experiment passports.

## История, сохранённая по прежним путям

Эти документы читаются конфигурациями, валидаторами или исследовательским
кодом. Они оставлены целиком; перенос потребовал бы правок вне docs.
Для rl-v1 путь учитывают также именованные migration policies.
Редактирование Markdown не блокирует обычную execution identity, но
миграционная проверка имеет отдельные правила. Нельзя обобщать одно на другое.

- [rl-v1-spec.md](../rl-v1-spec.md) — RL agent specification covering observations, memory, architecture, training, evaluation, and constraints.
- [rl-diagnostic-protocol.md](../rl-diagnostic-protocol.md) — Original D0 and D1 diagnostic criteria, budgets, evaluation, and limits on conclusions.
- [rl-d0-one-decision-protocol.md](../rl-d0-one-decision-protocol.md) — Frozen one-decision D0 diagnostic protocol covering rewards, training, and evaluation constraints.
- [rl-d1-transfer-protocol.md](../rl-d1-transfer-protocol.md) — D1 transfer protocol covering initialization, optimizer, fixed cases, limits, and acceptance criteria.
- [rl-multishot-protocol.md](../rl-multishot-protocol.md) — Frozen D0-M multistep diagnostic specification covering data, PPO, evaluation, and constraints.
- [rl-ppo-after-imitation.md](../rl-ppo-after-imitation.md) — PPO protocol after CP imitation, including parent, rewards, budgets, evaluation, and constraints.
- [rl-ppo-controlled-updates.md](../rl-ppo-controlled-updates.md) — PPO protocol with controlled policy updates, optimization conditions, evaluation, and stopping criteria.
- [rl-ppo-scale-up.md](../rl-ppo-scale-up.md) — Historical proposal to increase the PPO budget, with fixed data and evaluation criteria.
- [rl-ppo-scale-up-continuation.md](../rl-ppo-scale-up-continuation.md) — Conditions for continuing PPO from a saved checkpoint, with budgets and evaluation.
- [rl-ppo-after-target-bc-direction.md](../rl-ppo-after-target-bc-direction.md) — PPO direction from target BC, critic preparation, budget, and early stopping criteria.
- [rl-ppo-target-bc-t05-direction.md](../rl-ppo-target-bc-t05-direction.md) — PPO protocol from original BC at temperature 0.5 with consistent policy probabilities.
- [cp-reset-study.md](../cp-reset-study.md) — Protocol comparing standard CP against plan-reset behavior on a fixed set of cases.
