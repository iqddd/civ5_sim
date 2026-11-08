# Replay baseline directly from Git

> Архив: исторический протокол, исследование или прежний справочник.
> Числа, команды, версии и формулировки «следующий шаг» относятся к описанному этапу.
> Проверенные статусы и ограничения: [журнал](../../experiment-ledger.md); текущее устройство: [справочник](../../rl-architecture.md).

Edit `civ5rl/engine/ppo_fast.py`, then compare working replay against the version
committed at HEAD. No `*_before.py`, source extraction or import replacement is
needed:

```powershell
.venv/Scripts/python.exe tools/bench_rl.py replay --run artifacts/rl/ppo-after-imitation --compare-git-rev HEAD --out artifacts/rl/ppo-utilization/git-replay.json
```

`--run` uses the saved resolved config (or `run.json`'s config), `state.json`'s
latest checkpoint and the saved latest/first rollout. For a matched historical
checkpoint/rollout or a run whose recorded paths have moved, name them explicitly:

```powershell
.venv/Scripts/python.exe tools/bench_rl.py replay --config artifacts/rl/ppo-after-imitation/resolved-config.json --checkpoint artifacts/rl/ppo-after-imitation/checkpoints/update-0001.pt --rollout artifacts/rl/ppo-after-imitation/rollouts/update-0001.pt --compare-git-rev HEAD --out artifacts/rl/ppo-utilization/git-replay-first.json
```

An output ending in `.json` names the report file; otherwise the report is
`OUT/benchmark.json`. Resource samples and phase events go into a sibling
`REPORT_STEM-telemetry` directory. Existing reports/telemetry are not overwritten.
Omitting `--compare-git-rev` measures the working implementation only, without
claiming comparative parity. Input paths or `--run` are required; the output
directory does not select a scientific experiment implicitly.

The default is one warmup per implementation and three measured ABBA blocks
(12 measured replay passes). `--warmup` and `--repeats` bound that work. The
optional `--episode-limit N` selects the first N saved episodes and records their
indices; it does not collect new cases. The default device is CUDA; use
`--device cpu --episode-limit 1 --warmup 1 --repeats 1` for a short CPU check.

## What is compared

The A implementation is **only** `ppo_fast.py` from the resolved Git commit; B is
the working `ppo_fast.py`. Model, simulator, native libraries and imported helper
modules are the shared current implementation. This is not an old checkout or a
comparison of whole historical environments. The report records the resolved
commit, source blob SHA-256, working source SHA-256, shared dependency hashes,
input hashes, ordered episode selection, temperature, timings and replay metrics.

Each side has a separate RolloutCache and a private copy of saved episodes.
Before each pass, model, gradients and Python/NumPy/Torch/CUDA RNG are restored.
No optimizer is constructed, no Adam step runs, and no new match is played.
The historical checkpoint's optimizer remains an unchanged input file. Model
weights must stay bitwise unchanged, and all replay diagnostic metrics must
match in `rtol=3e-5, atol=2e-7`. Different metric schemas or missing temperature
support fail rather than silently claiming equivalence. In particular an old
T=1-only backend cannot serve as the baseline for a T=0.5 rollout.

Warmup is excluded. Replay compilation time is reported separately but included
in the measured wall time when it occurs. NVML labels separate preparation,
compilation, replay and validation/hash work. A short CPU check establishes
functional comparison, not a claim of representative GPU throughput.

For an incompatible old API or changes spanning several dependent modules, use
an isolated checkout and compare explicit results instead. This command does
not patch dependencies or relax mathematical checks to make an old version run.
The existing `full-cycle` mode retains its complete-collection/PPO behavior;
`--compare-git-rev` is supported for `replay` only.

## In-memory loader

```python
from civ5rl.provenance.common import load_module_from_git

baseline = load_module_from_git(
    "civ5rl.engine._my_git_baseline", "civ5rl/engine/ppo_fast.py", rev="HEAD"
)
```

The loader executes trusted repository code. It resolves the revision to an
immutable commit, decodes the source, registers an unused module name (needed by
dataclasses), and compiles entirely in memory. It rejects existing module names
and removes the temporary registration if compilation/import fails. The virtual
`__file__`/traceback and linecache refer to Git source, not today's disk file.
Modules using relative imports need a module name under their real package;
modules loading resources relative to `__file__` need a separate checkout.
The helper does not load dependency versions from Git.
