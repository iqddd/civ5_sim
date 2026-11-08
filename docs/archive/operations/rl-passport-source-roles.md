# Frozen experiment provenance and editable tooling

> Архив: исторический протокол, исследование или прежний справочник.
> Числа, команды, версии и формулировки «следующий шаг» относятся к описанному этапу.
> Проверенные статусы и ограничения: [журнал](../../experiment-ledger.md); текущее устройство: [справочник](../../rl-architecture.md).

Source identity has two separate uses:

| Operation | What is checked |
|---|---|
| Analyze saved artifacts, rebuild plots, recount wins or bootstrap | The saved passport, resolved config, corpus manifest, and every member of the saved source archive. No comparison to working sources, installed packages, current GPU, or current Git commit. |
| Train/resume, generate data, or execute new evaluation matches | Frozen evidence first, then the current execution source/config/corpus/environment identity. Read-only tooling is excluded from source comparison. |

The archive check alone does not verify all checkpoints, dataset files or match
outcomes. Use the corresponding ledger/checkpoint/evaluation audits for those.
It also does not authorize resuming on changed training code.

## Source roles

`artifact_sources(config)` and old/new identity comparisons use the same policy
in `civ5rl/provenance/identity.py`. They exclude:

- `civ5rl/reporting/` and `civ5rl/deploy/`;
- `civ5rl/provenance/audit.py` (the existing shared audit module) and the proposed
  `civ5rl/evaluation/audit.py` path;
- the pure `experiments/report_cp_imitation.py` module;
- tests and non-execution tools, including report/audit/inspect/benchmark/profiler
  CLIs and archived analysis helpers.

Core, engine, protocols, simulator/map/native code, provenance implementation and
mixed experiment runners remain pinned. `experiments/evaluation.py` remains
pinned because it executes validation cases and replays. Its pure saved-result
loading, W/D/L analysis and paired-bootstrap functions now live in
`reporting/evaluation_audit.py`, with their existing imports re-exported; editing
those analysis functions no longer changes training source identity. Training/data/match
and build entrypoints under `tools/run_*`, `train_*`, `prepare_*`, `evaluate_*`,
`build_*`, plus `tools/tactical_corpus.py`, remain pinned too. Configuration and
dependency inputs retain their existing checks. Adding plotting functions inside
a mixed training runner still changes execution code: put new analysis in the
reporting or audit modules instead.

Read-only exclusions also apply to `passport_artifacts` glob matches. Such a glob
does not silently reintroduce a report or test into execution identity. Scientific
helpers must live in the pinned package rather than be imported from an unpinned
benchmark/reporting CLI.

## Existing runs

Historical passports and `sources.zip` are not rewritten. Old reporting/test
entries are excluded when comparing an old execution identity with a current
one, but their original archived bytes are still verified against every recorded
SHA-256. Editing or deleting today's report file therefore does not invalidate
that archived evidence; corrupting its archived copy does.

```powershell
.venv/Scripts/python.exe tools/ctl_rl.py audit --run artifacts/rl/ppo-controlled-updates --out C:/temp/controlled-audit
```

This command is read-only and explicitly reports `working_tree_compared: false`.
It needs a passport-based run with `passport.json`, `sources.zip`,
`resolved-config.json` and `corpus-manifest.json`. Other run formats are not
silently certified by this command.
For historical runs missing `corpus-manifest.json`, the saved `manifest.json`
is accepted only when its full content matches the passport's corpus hash.
New passports also hash the resolved configuration. Old passports lacking that
hash are left unchanged; the CLI reports `config_hash_verified: false` for them.
Callers can still provide an expected `config` for exact comparison. Archive and
corpus integrity checks remain mandatory in both formats.

For Python callers:

```python
from civ5rl.provenance.checkpoint import verify_frozen_passport
identity = verify_frozen_passport(out, config=config, manifest=manifest)
```

`ppo_after_imitation.check_identity(config, out)` now checks frozen artifacts for
analysis by default. Its execution callers explicitly pass `for_execution=True`;
that path checks live training code and the existing parent/map conditions.
The continuation, more-data and own-state `report` phases also validate saved
evidence directly instead of calling their training initialization/lineage code.
Parent lineage and checkpoint resume use the shared execution-role policy.

This separation permits editing tooling while a run is paused or complete.
It does not hot-reload running processes, relax training mathematics, change
budgets, or make old training code compatible with today's implementation.
