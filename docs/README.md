# Documentation

Read this index before starting work, then consult the relevant references.
Command paths are relative to the repository root. The main references were
checked against the code on November 11, 2026; consult the ledger for individual
experiment conditions and results. Historical documents do not authorize new runs.

## Main references

- [combat-engine.md](combat-engine.md) — Combat engine, Battle interface, save formats, visibility rules, match execution, and replay verification.
- [combat-ai.md](combat-ai.md) — Current Combat AI interfaces, coordinator rules, group recovery, porting limitations, and diagnostic tools.
- [rl-architecture.md](rl-architecture.md) — RL modules, information boundaries, observation tensors, network architecture, training, passports, and run resumption requirements.
- [rl-operations.md](rl-operations.md) — Run control, result auditing, reporting, performance measurement, collection tuning, and remote experiment packaging.
- [vast-cpu-ratings.json](../configs/vast-cpu-ratings.json) — Machine-readable CPU Single Thread ratings and sources used for Vast.ai admission filtering.
- [experiment-ledger.md](experiment-ledger.md) — Experiment lineage, configurations, parent models, verified results, conflicting evidence, and links to saved reports.

## Detailed contracts and provenance

- [combat/combat.md](combat/combat.md) — Detailed combat rules, scenarios, commands, Python API, match orchestration, and supported profile limitations.
- [combat/combat-viewer.md](combat/combat-viewer.md) — Launching the local viewer, navigating commands, selecting visibility modes, and verifying saved frame compatibility.
- [combat-ai/combat-ai-contract.md](combat-ai/combat-ai-contract.md) — Complete current contract for search, memory, coordination, support, and errors; stable anchors serve code and tests.
- [combat-ai/combat-ai-cp-transfer.md](combat-ai/combat-ai-cp-transfer.md) — Mapping mechanisms to Community Patch, pinned sources, adaptation limitations, and historical verification evidence.
- [combat-ai/combat-ai-diagnostics.md](combat-ai/combat-ai-diagnostics.md) — Recording planner requests, interpreting traces, repeating computations, checking alternatives, and measuring diagnostic overhead.
- [combat-ai-sources.json](combat-ai-sources.json) — Machine-readable Community Patch source inventory with a pinned commit and checksums for provenance verification.

## History

- [archive/README.md](archive/README.md) — Index of archived investigations, former references, and historical protocols retained at paths required by configurations.

Update general rules in the existing reference rather than appending conflicting
historical updates. Keep detailed run reports in `artifacts/` and link them from
the ledger. Retain archival protocols only when needed for reproducibility or
provenance; disposable experiment scripts belong in `.tmp/`, never `artifacts/`.

Before moving documents, check configuration, code, test, and Markdown references.
Some historical documents remain in the root of `docs/` because they are file
inputs. Detailed Combat contracts retain paths and anchors used by code; their
contents do not replace the experiment status recorded in the ledger.
