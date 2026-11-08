# Run operations and derived snapshots

All paths below are examples. Run from the repository root. `RUN` denotes the
local run directory; nothing here connects to or deletes a remote machine.

```powershell
.venv/Scripts/python.exe tools/ctl_rl.py request-pause --run RUN --after-update 12
.venv/Scripts/python.exe tools/ctl_rl.py request-pause --run RUN --after-epoch 10
.venv/Scripts/python.exe tools/ctl_rl.py snapshot --run RUN --out C:/temp/rl-snapshot --block-size 5 --last 20
.venv/Scripts/python.exe tools/ctl_rl.py wait-and-finalize --run RUN --out C:/temp/rl-final --timeout 600
.venv/Scripts/python.exe tools/ctl_rl.py close --run RUN --reason "Research stopped at saved boundary"
```

The request is a JSON threshold in `pause.request`, not a process signal. The
PPO-after-target-BC runner supports update thresholds and target-scale imitation
supports epoch thresholds. Other runner/boundary combinations are rejected.
PPO checks the request only at a saved update boundary, and BC only at a fully
saved epoch. A scheduled milestone/epoch evaluation is completed first; no
extra final evaluation is launched to satisfy a pause. PPO's existing `resume`
removes the request. The imitation runner has no general safe pause-resume
implementation: it preserves the boundary but refuses automatic continuation;
its old input-fix restart is specialized and must not be assumed equivalent.
A runner already running older source does not acquire these new hooks.

`wait-and-finalize` waits on the run lock and saved status with a finite timeout,
then writes only a derived snapshot. It reports the original status, including
paused, technical_error and closed_by_user; it does not evaluate, complete, or
resume a run. `close` takes the run lock, refuses incomplete PPO/epoch or pending
evaluation, checks the update ledger, and writes `closure.json` and `status.json`.
It leaves state and checkpoint bytes untouched. Closed runners refuse ordinary
training/resume; reopening requires an explicit separate decision.

Snapshots require explicit `--out`. Use an external directory or
`RUN/derived/`, `RUN/snapshots/`, or `RUN/interim-snapshot/`. Only the three
owned output files are replaced; unrelated historical neighbors are preserved. They contain JSON, Markdown and
`learning-curves-detailed.png`. Training win rate is pooled by episode, and
PPO metrics are means over updates in each block. Development uses saved fixed
argmax-vs-CP points; duplicate checkpoints are deduplicated. Snapshot generation
reuses the performance summary for phase/NVML accounting; it does not run
matches or alter experiment files. Replaces one-off reporting portions of
`interim-snapshot/build_report.py`, `ppo-target-scale/finalize.py` and similar
post-analysis scripts; pause replaces PID polling in `stop_at_epoch.py` for the
supported runner versions.

## Bounded benchmark and replay diagnostic

For a fixed-weight replay ABBA comparison with a Git baseline, see
[Git baseline benchmark](rl-git-baseline-benchmark.md). This avoids the old
`*_before.py` copies and `importlib` loading from artifact directories.

```powershell
.venv/Scripts/python.exe tools/bench_rl.py --config CONFIG --checkpoint PPO_CHECKPOINT --rollout ROLLOUT --out C:/temp/rl-bench --repeats 3 --device cuda
.venv/Scripts/python.exe tools/diagnose_cp_replay.py --rollout ROLLOUT --episode 0 --trace-from 100 --out C:/temp/cp-trace.json
```

The benchmark output directory must be empty and separate from its inputs. The
checkpoint must be a compatible **PPO boundary checkpoint containing PPO Adam**;
BC Adam parameter groups may differ and are not silently reset. It runs one
warmup plus N fixed-case trials using the existing collector, rollout writer and
PPO update, restores model, optimizer, gradients and RNG between trials, checks
command/outcome hashes and protects input hashes. It performs real discarded
updates and collection, so execute it only when that work is intended. Its API
`assert_update_parity` runs callbacks in ABBA order; callers provide a reset for
model/Adam and independent snapshots of model, optimizer, gradients, sampled KL
and commands. Pass `shuffle_rng` if minibatch shuffling uses one.

The diagnostic replays an existing episode. `--trace-from` enables detailed CP
planner tracing from a command index and requires a trace-enabled native library;
omit it for replay and root-state checks alone. The report includes `matched`,
first mismatches, outcome, and root checks. These replace the repeated full-cycle
scripts in `collection-pipeline`, `collector-optimization`, `ppo-utilization`,
`vast-4090/capsule`, plus `collection-vast-live/diagnose_replay.py`,
`capture_last_plan.py` and `collector-optimization/check_roots.py` for their
common diagnostic purposes. GPU full-cycle benchmarking was not run while these
tools were introduced.

`integrate_gpu_interval` accepts elapsed seconds relative to one
`ResourceMonitor.start()` session, not absolute `perf_counter` or UTC. Its
coverage fraction denominator is the entire requested interval, including any
filtered phases; unavailable NVML returns null load/busy/idle, not zero.
