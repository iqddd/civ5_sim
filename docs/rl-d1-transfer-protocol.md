# D1 transfer — optimization revision 1

> Исторический протокол / исходная спецификация. Путь сохранён: документ используется конфигурацией или кодом.
> Это условия отдельного этапа, а не актуальный общий справочник; наличие плана не доказывает запуск.
> Проверенные статусы и ограничения: [журнал](experiment-ledger.md); текущее устройство: [справочник](rl-architecture.md).

Frozen before training. Seed 17 only. Results: `artifacts/rl/diagnostic-d1/`.

## Initialization and cases

Reuse original D1 train/dev manifests exactly, including states and fixed dev
seeds. No generation or test access. Dev: 64 layouts × both sides, 128 games.
Evaluate the original random seed-17 model and D0-one-decision update 6 before
training. Only the random model is the gate baseline. Baseline evaluations do
not count towards the two-success streak.

Load all policy/value weights from update 6 bitwise; create fresh Adam and
training RNG, with zero recurrent state per episode. No one-decision wrapper.
Keep C, BF16, spatial block 32, 128 environments, complete episodes, ≥8192
decisions per update, terminal +1/-1/0, gamma=lambda=1, PPO clip 0.2,
value coefficient 0.5, four epochs, gradient norm 0.5 and all other original
architecture/objective settings.

## Optimizer revision

Shuffle recurrent fragments of at most 128 decisions. Form exactly 1024 valid
tokens per Adam step, splitting a fragment when needed and reconstructing its
prefix with current weights without gradients. Never join independent episodes.
Microbatch: at most eight fragments. Normalize accumulated loss over the entire
minibatch's valid tokens, excluding padding; clip once before Adam.step(). The
last incomplete minibatch is normalized by its actual size.

After every Adam step measure the existing sampled KL estimator over the whole
rollout against its behavior policy. Stop remaining minibatches/epochs if KL
exceeds 0.015. Retain the exceeding step; no rollback or adaptation. Log each
step's token count, KL, pre-clip norm and duration, plus started/completed epochs.

LR and entropy restart from zero D1 decisions: 0.0003→0.0001 and 0.01→0.002,
linearly over 2,000,000 D1 decisions, then constant. Snapshot schedules before
collection and keep fixed through PPO. Evaluations/epochs/Adam steps do not
advance this counter. Previous D0 cost (49,152 decisions) is reported separately.

## Gate, limits and recovery

After every update evaluate the same 128 dev games. Pass after two consecutive
post-update evaluations each with ≥80% wins, ≥20 percentage points over the
initial random network, and a strictly positive lower bound of the paired 95%
bootstrap improvement interval. Bootstrap: 10,000 samples, seed 20261103,
cluster by layout preserving both sides and both policies. Failure resets streak.
Incomplete evaluation is a technical failure and does not advance the gate.
Check mathematical attainability before training.

At most eight updates and 1800 accumulated seconds of collection, PPO, KL and
update servicing. Finish and evaluate an update already started. Evaluate/pause
time is separate. No automatic D0 rerun, seed 42, tuning or extension.

Resume only this experiment with exact config/source/corpus/environment identity.
Restore its Adam, RNG, counters and schedules; never reset them on resume.
Commit pending evaluation with the completed-update checkpoint. Resume completes
it before another update, recovers metrics idempotently, and reruns only work
that never reached a completed-update checkpoint. Historical checkpoints are
weights-only inputs, not a migration bypass. Preserve original artifacts.

## Acceptance

Test exact token packing/splits/remainders, loss normalization, recurrent prefixes,
reference/batched agreement, per-step KL stop and optimizer counters. With sources
frozen, test Windows file-entrypoint interrupt/resume including pending/partial
evaluation, metrics recovery, graceful stop, next update, CPU deterministic
equivalence and CUDA/BF16 replay. Verify RNG isolation, gates and limits. Run
regression tests before bounded GPU training; replay and hash-audit its artifacts.

Report random baseline, transfer, every update, outcomes/attacks/damage/losses,
optimizer diagnostics, true budget and NVML telemetry. An initial ranged attack
is not a guaranteed winning shot in D1. Conclusions do not extend to ordinary
maps, memory usefulness or strength against greedy.
