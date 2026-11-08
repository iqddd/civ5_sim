# Project tools

Inventory checked on November 8, 2026: the root contains 33 Python files:
24 enabled public CLIs, one disabled CLI, and eight internal helper modules.
`combat_viewer.html` is the viewer resource. The 16 historical files are
listed separately in [archive/README.md](archive/README.md).

Run commands from the repository root using the project environment:

```powershell
.venv/Scripts/python.exe tools/run_ppo.py --help
.venv/Scripts/python.exe -m tools.run_ppo --help
```

Enabled public CLIs support both invocation forms. The disabled machine
benchmark raises `NotImplementedError`, including when invoked with `--help`.
A reusable tool accepts explicit inputs; an `--out` option alone does not make
a script with a fixed experimental plan reusable.

## Build and source verification

| Tool | Purpose |
|---|---|
| [build_native.py](build_native.py) | Extract the SDK fractal implementation and compile the standalone map-generation library without modifying SDK sources. |
| [build_combat.py](build_combat.py) | Extract CP rules and geometry, record provenance, and compile the standalone combat kernel and damage oracle. |
| [build_combat_ai.py](build_combat_ai.py) | Compile the CP planner library and native checks; optionally build only the library or a separate profiling variant. |
| [check_combat_ai_sources.py](check_combat_ai_sources.py) | Verify the CP checkout against the pinned source inventory and SHA-256 hashes without changing either checkout. |

Build scripts use MSVC on Windows and C++17 compilation on Linux. Generated
sources, binaries, and compiler scripts stay in `build/`. Linux fractal output
is `libfractal.so`; Windows output is `fractal.dll`. Build the combat kernel
before the planner to prepare the shared rules and geometry inputs.

## Matches, diagnostics, and viewing

| Tool | Purpose |
|---|---|
| [view_combat.py](view_combat.py) | Browse saved matches locally, verify current replays, and display hash-linked frame caches for historical rules. |
| [diagnose_combat.py](diagnose_combat.py) | Explain recorded commands or planner queries, repeat computations, and examine legal alternatives without editing the source replay. |
| [resume_combat_replay.py](resume_combat_replay.py) | Continue a verified replay at a turn boundary with restored physical state/RNG and a fresh CP coordinator. |
| [tactical_corpus.py](tactical_corpus.py) | Generate fresh tactical requests and verify returned plans through physical execution rather than historical assignment expectations. |
| [check_planner_refactor.py](check_planner_refactor.py) | Compare compatible planner libraries on fresh or supplied requests, checking search results and physical plan replay. |
| [check_combat_viewer.py](check_combat_viewer.py) | Check viewer navigation, playback, visibility, and layout at a supplied URL using Edge and Playwright. |
| [check_combat_refresh.py](check_combat_refresh.py) | Check replay version indicators and explicit synchronization of an already open replay at a supplied viewer URL. |
| [diagnose_cp_replay.py](diagnose_cp_replay.py) | Replay one saved RL episode, locate the first divergence, and optionally record detailed CP planner traces. |

Ordinary matches use `python -m civ5sim match`; there is no duplicate match
launcher. `combat_viewer.html` remains beside `view_combat.py` as its resource.
Diagnostics and replay continuation accept `--out` or `--output`. Continuation
also accepts the positional output directory; provide exactly one output form.

## Run operations and source evidence

| Tool | Purpose |
|---|---|
| [ctl_rl.py](ctl_rl.py) | Control supported runs and analyze saved artifacts through snapshot, performance, audit, compare, request-pause, close, and wait-and-finalize. |
| [remote_rl.py](remote_rl.py) | Package, launch, inspect, and retrieve a frozen deployment on an existing instance through package, launch, status, and pull. |
| [rl_source_evidence.py](rl_source_evidence.py) | Verify historical source evidence or restore exact source bytes from Git blobs and patches using verify and restore. |

Instance rental stays outside `remote_rl.py`; use the Vast.ai workflow for
machine selection. Source restoration requires a new output path and accepts
`--out` or `--output`. Editing tools does not authorize resuming frozen runs on
changed execution sources, and saved passports/source archives remain immutable.

## Corpus preparation, training, and evaluation

| Tool | Purpose |
|---|---|
| [run_cp_imitation.py](run_cp_imitation.py) | Prepare a map corpus or dispatch configured BC preparation, collection, training, evaluation, reporting, and supported complete runs. |
| [run_ppo.py](run_ppo.py) | Dispatch PPO after imitation, controlled updates, scale-up, target-scale, and after-target-BC phases using an explicit required configuration. |

BC training and collection implementations live in `civ5rl/experiments/`,
including `imitation_base.py`. Initial BC benchmark and latency measurements
remain in `tools/_imitation_benchmark.py`. PPO pause/resume applies only to
supported target protocols; runtime overrides and bounded update/output options
apply only to PPO after target BC.

Corpus preparation does not train a model:

```powershell
.venv/Scripts/python.exe tools/run_cp_imitation.py prepare-corpus --config configs/rl-v1.json --out artifacts/rl/corpus --game "E:/civ5/Sid Meier's Civilization V" --workers 4
```

Here `rl-v1.json` supplies map-generation settings. The original general RL-01
pipeline and duplicate PPO launchers were deleted without compatibility wrappers.
Configured BC/PPO runners execute training and match evaluation; `ctl_rl.py`
provides saved-run reports and supported run control.

## Performance measurements

| Tool | Purpose |
|---|---|
| [bench_rl.py](bench_rl.py) | Benchmark a complete collection/PPO cycle or fixed-weight replay, optionally comparing replay against one Git-versioned backend module. |
| [benchmark_combat.py](benchmark_combat.py) | Measure kernel execution, Python/native overhead, action generation, observations, batching, and memory without starting RL training. |
| [benchmark_combat_ai.py](benchmark_combat_ai.py) | Run isolated repeated CP matches from a selected suite, enforce process timeouts, collect telemetry, and verify replays. |
| [benchmark_decision_diagnostics.py](benchmark_decision_diagnostics.py) | Measure off/compact/full diagnostic overhead on saved queries, checking result parity without playing new matches. |
| [benchmark_ppo_replay.py](benchmark_ppo_replay.py) | Compare cached and uncached GPU PPO replay/KL throughput with fixed weights, telemetry, and no Adam steps. |
| [profile_rl_runtime.py](profile_rl_runtime.py) | Profile fixed-episode CPU replay or fixed-policy collection, with optional cProfile, native counters, queue diagnostics, and quota counters. |
| [tune_rl_collection.py](tune_rl_collection.py) | Select streaming collection workers, environment capacity, and reply size on fixed cases, then write recommended runtime settings. |
| [benchmark_rl_machine.py](benchmark_rl_machine.py) | Disabled pending rework: every module-defined function and the CLI raises NotImplementedError; the v1 comparison series is closed. |

Use a separate output for each new measurement. The tuner starts with a large
environment pool, normally 80 when the selected workload permits, and probes
workers before checking capacity/reply changes. The archived A/B/C and
B0–B3/W80 scripts retain fixed historical plans, including their WPR passes.
`benchmark_decision_diagnostics.py` accepts `--out` and `--output` equally.

## Internal modules and output policy

These modules have no standalone CLI. Public tools import them through `tools.*`.

| Module | Responsibility |
|---|---|
| [_build_common.py](_build_common.py) | C++ function extraction, Visual Studio discovery, and shared-library/executable compilation for Windows and Linux build CLIs. |
| [_collection_tuning.py](_collection_tuning.py) | Collection candidate representation, probe construction, and ranking shared by measurement tools. |
| [_common.py](_common.py) | Atomic JSON output and streaming SHA-256 file fingerprints using the standard library. |
| [_imitation_benchmark.py](_imitation_benchmark.py) | Initial BC collection/latency measurements, including demonstrations later reused by the training-data collection phase. |
| [_process.py](_process.py) | Bounded subprocess execution and process-tree cleanup, with fatal errors when cleanup cannot be confirmed. |
| [_resources.py](_resources.py) | Aggregate saved resource samples for CLI measurements, including phase-specific utilization and peak memory. |
| [rl_collection_hardware.py](rl_collection_hardware.py) | Read CPU topology, affinity, available memory, and Linux cgroup limits without importing Torch. |
| [rl_cpu_replay.py](rl_cpu_replay.py) | Replay fixed learner commands through the simulator and opponent using reusable CPU workers without Torch imports. |

Build and BC benchmark helpers remain pinned in execution provenance because
they affect compiled code or training data. Pure measurement helpers are excluded
from training identity. Add reusable audit, benchmark, deployment, and reporting
logic under `tools/` and expose it through existing CLI subcommands. `civ5rl`
records low-level performance data for those tools.

`artifacts/` holds data, checkpoints, logs, JSON metrics, plots, and Markdown
reports. Do not create Python or other executable helpers there. Narrow one-off
experiment scripts may use disposable `.tmp/`; durable workflows must not depend
on its contents. Historical tool descriptions are in the [archive index](archive/README.md).
