# CP imitation v1 — preparation and cost benchmark

The first execution stops after eight CP–greedy training matches and RLAgent
latency measurements. It does not start the long collection, research training,
or 320 validation matches. Direction: `rl-next-stage-direction.md`.

```powershell
.venv\Scripts\python.exe tools/run_cp_imitation.py prepare
.venv\Scripts\python.exe tools/run_cp_imitation.py benchmark --workers 4
.venv\Scripts\python.exe tools/run_cp_imitation.py report
```

Results: `artifacts/rl/cp-imitation/report.md`. Other phases are explicit commands
(`collect`, `train`, `evaluate`), to be invoked after reviewing benchmark cost.
`--workers N` sets independent match processes; task IDs/seeds do not depend on N.
`--device cpu|cuda` controls the training/evaluation network, not CP generation.

## Frozen experiment

Seed 17, fresh model C, 10×10 existing maps, warrior+archer on each side, 100 HP,
fog/random damage enabled, 30 rounds, 2000 commands per side turn as technical
guard. Half contact/half separated starts, CP side alternates independently.
SearchConfig is serialized in `configs/rl-cp-imitation.json`; its privileged
queries remain enabled. The network receives no CP scores/plans/hidden positions.

Shuffle the sorted 64 train-world IDs with Python Random(17): first seven whole
worlds become offline holdout, remaining 57 training. Validation uses the existing
16 validation worlds, two cases each (contact/separated), played on both sides.
Test maps are not loaded. Manifest includes source hashes and fixed case formulas.

CP recording subclasses CPAgent so run_match retains its planning services and
action-result callbacks. Store all completed trajectories, including forced
commands and losses/draws. Record public observations, ordered legal lists and
the index of the actual CP-selected action. Privileged match artifacts live in
separate replay/scenario files and are not read by the supervised encoder.

The benchmark produces exactly training match IDs 0..7: four contact, four
separated, four CP-side-0, four CP-side-1. CPU pool wall time includes startup.
RLAgent CPU/FP32 and CUDA/BF16 measurements use fixed public teacher histories,
with teacher previous actions remembered, warmup measured separately. Eager
RLAgent inference is measured; no CP call occurs in latency probes. Mean, median
and p95 cover encode+forward+selection, separately for forced/free decisions.
CPU/GPU contention at larger N is not inferred from a single-agent latency probe.

## Future explicitly invoked phases

Collect ≥20000 free training decisions and ≥2457 holdout decisions; finish each
submitted N-match wave and report overflow. Collection wall budget is one hour
per invocation, including both splits; existing benchmark episodes count.
On a technical failure preserve replay/prefix and stop. Never train a silently
reduced sample: training refuses an incomplete corpus. Sources/config/corpus/
environment must match the immutable passport before a later phase proceeds.

BC: three epochs, Adam 1e−4, betas .9/.999, eps 1e−5, weight decay 0; categorical
CE only, no entropy or value loss. Value head parameters are frozen; shared
encoder and GRU learn. Minibatch counts exactly 1024 non-forced labels, excluding
padding; final remainder uses its actual count. Fragments contain up to 128
consecutive decisions including forced states, microbatches up to eight fragments.
Current-weight prefixes are reconstructed without gradients; recurrent gradients
flow through forced states inside a fragment. Zero-label fragments still enter
later prefixes. GRU and encoder state reset at every episode. Clip once at .5.
BF16 spatial dense operations, FP32 GRU/heads/loss/weights, spatial block 32.

Thirty-minute optimizer-work budget; finish a started minibatch. Save each
completed epoch (model/Adam/RNG), plus a partial checkpoint on budget exhaustion.
Use the last completed epoch as final; never select based on battle outcomes.
Holdout CE/accuracy by teacher-command kind excludes forced commands.

Evaluation: initial/final against CP and greedy (256 games) plus teacher versus
greedy (64), all on the fixed 32×2 paired validation cases. Sampling temperature 1,
matched battle/policy/opponent seeds, same CP config. Report W/D/L, win-rate change,
movement/attacks and round-limit draws; no artificial pass gate. Save full replays.
Evaluation budget must be decided after cost review; it is not auto-launched.

## Focused validation

Two tests target silent mistakes: public teacher-action indexing/world leakage,
and CE normalization/gradients under forced states and microbatch accumulation,
including episode-local encoder history and frozen value weights. Reuse existing
numeric/memory checks when affected. No full-suite or resume-test runs.
Replay several benchmark games and compare every recorded public learner input,
legal list, command result and state hash. Record data hashes and actual costs.

## Идентичность и документация

Markdown сохраняется с хешами в паспорте и `sources.zip` для истории опыта,
но добавление, удаление или изменение `.md` не блокирует выполнение. Изменения
исполняемых исходников, ядра, конфигурации, корпуса и окружения по-прежнему
блокируют запуск. Для данного продолжения исходный паспорт benchmark сохранён
в `benchmark-passport/`; авторизованное изменение проверки зафиксировано в
`identity-transition.json`. Начальные веса и обучающие условия не менялись.

### Паспорта после рефакторинга engine

Паспорта и `sources.zip` прежних CP imitation запусков неизменяемы. Перенос сценариев и изменение исполняемых файлов меняют execution identity: продолжение или повторный запуск по старому паспорту и дочерние сценарии от старых родителей завершаются ошибкой проверки происхождения. Чтобы проводить новые опыты на этой версии кода, создайте новый базовый запуск в отдельном output-каталоге и заново выполните необходимые подготовку, сбор, обучение и оценки; для последующих дочерних опытов явно зафиксируйте его паспорт, хеши и согласованный протокол. Не перезаписывайте прежние паспорта и не расширяйте списки допустимых изменений ради обхода проверки.
