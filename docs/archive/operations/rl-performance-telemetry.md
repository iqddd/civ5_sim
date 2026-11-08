# PPO performance telemetry

> Архив: исторический протокол, исследование или прежний справочник.
> Числа, команды, версии и формулировки «следующий шаг» относятся к описанному этапу.
> Проверенные статусы и ограничения: [журнал](../../experiment-ledger.md); текущее устройство: [справочник](../../rl-architecture.md).

Run the standard read-only reporter on a run directory, including a running or
paused run. It needs no Torch, GPU, plotting packages, or experiment-specific code:

```powershell
python tools/ctl_rl.py performance --run artifacts/rl/ppo-target-bc-t05 --last 10
python tools/ctl_rl.py performance --run artifacts/rl/ppo-target-bc-t05 --last 10 --json > performance.json
```

Omit `--last` for the whole run. A cycle counts only after its complete row is
written to `updates.jsonl`; the current incomplete rollout is excluded. The
report includes selected rollout IDs, mean/median/min/max timers, per-phase
duration and GPU Core Load, missing-data coverage, uninterrupted 60-second NVML
windows, and observed continuous GPU intervals of at least ten minutes.

For remote runs, invoke the same command over SSH in the remote repository:

```text
ssh -p PORT USER@HOST 'cd /workspace/civ5-t05/civ5_sim && python3 tools/ctl_rl.py performance --run artifacts/rl/ppo-target-bc-t05 --last 10'
```

The after-target-BC runner also writes `performance.json` / `performance.md` and
`performance-last10.json` / `performance-last10.md` during its normal report phase
(including pause/final reports). No analysis scripts belong in run artifacts.
On-demand analysis is read-only unless `--output` is supplied.

## Measurement definitions

- `collection`: the collector's recorded wall time.
- `ppo_recorded`: existing optimizer timer. It starts after backend creation and
  precheck; compilation intervals must not be blindly subtracted from it.
- `ppo_operation`: the entire `update()` call, including backend setup/precheck.
- `ppo_operation_without_compilation`: that same interval minus its intersection
  with TorchDynamo callbacks. These two timers are available for new runs only.
- `dynamo_callbacks`: tracing/backend callback intervals, not exclusively GPU
  code generation. They do not prove that a new CUDA kernel was compiled.
- Phase means use time-weighted, whole-device NVML GPU Core Load. Compilation,
  collection, critic, evaluation and checkpoint phases are reported separately.
  Missing NVML values remain missing rather than becoming zero.

`ResourceMonitor` now appends `telemetry/phases.jsonl` on phase transitions,
including a unique session ID and monotonic clock. Samples carry the same ID.
The reporter splits sampled GPU intervals at those boundaries. GPU measurements
still have their original sampling resolution; exact phase boundaries do not
make NVML instantaneous. It excludes gaps longer than max(5s, 3 sampling intervals),
session changes and monotonic-clock resets rather than interpolating across them.

Old runs without phase events remain readable: durations/boundaries are estimated
from sample labels, marked `sampled_or_mixed`. No exact timing is invented for
their initial/final unsampled intervals. A live last partial JSONL record is
ignored; corrupt interior records are errors. Per-phase means use only cycles
where that phase was observed. NVML coverage and sample counts show limitations.

The 60-second windows require contiguous, fully sampled GPU intervals in one
phase and cycle; collection, compilation, missing data and pauses break them.
Short PPO phases are not joined across CPU collection to satisfy the GPU policy.
