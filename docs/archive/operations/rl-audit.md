# Saved-run audit

> Архив: исторический протокол, исследование или прежний справочник.
> Числа, команды, версии и формулировки «следующий шаг» относятся к описанному этапу.
> Проверенные статусы и ограничения: [журнал](../../experiment-ledger.md); текущее устройство: [справочник](../../rl-architecture.md).

```powershell
python tools/ctl_rl.py audit --run RUN --out C:/temp/run-audit
python tools/ctl_rl.py compare --before BEFORE --after AFTER --expected-n 128 --out C:/temp/comparison --clusters 16 --samples 10000 --seed 20261103
```

`audit` writes `audit.json` and `audit.md`, with a status and missing-check reason
for each section: frozen passport/source archive, transactional PPO ledger,
CPU checkpoints/Adam/RNG, complete saved evaluations and W/D/L, paired confirmation
bootstrap. It uses saved identities and archives, never current working sources.
New remote deployments retain a separate deployment passport and source archive;
this is identified explicitly instead of pretending the runner produced a passport.

An existing writer lock is held throughout the stable audit. A live writer or
missing lock yields `incomplete`; use `snapshot` for a live intermediate view.
Unavailable historical formats are not passed. Audit status distinguishes failed
integrity from missing coverage. A passed ledger is accounting evidence, not
proof of convergence or completion. CUDA RNG receives structural checks;
Python, NumPy, CPU Torch and shuffle generators are validated in isolation.
Inherited Adam offsets come from saved run/config metadata. Specialized historical
analyses and independently established ledger offsets remain available through
`civ5rl.provenance.audit`; the aggregate audit does not invent missing history.

`compare` accepts complete opponent/mode directories, checks case hashes, both
sides, metadata and W/D/L files, and clusters paired differences by parent world.
Defaults are 16 clusters, 10,000 samples and seed 20261103. Parameters are saved
in `comparison.json`. No models or battles are executed.

Outputs must be external or under `derived/`, `snapshots/`, `interim-snapshot/`.
Scientific state, checkpoints, RNG and evaluation files are never rewritten.
The former audit, snapshot/performance wrappers and run-control CLI were replaced
by `ctl_rl.py`.
