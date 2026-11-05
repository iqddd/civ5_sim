# RL module architecture

The package separates policy inputs and simulation adapters, reusable learning algorithms, experiment protocols, run orchestration, and reproducibility checks:

| Package | Responsibility |
| --- | --- |
| `civ5rl.core` | Public input contracts, feature schema and encoding, model, agent, and environment. |
| `civ5rl.engine` | Collection, shared slots, PPO, behavior cloning, sequence buffers, optimizer helpers, run control, and resource monitoring. |
| `civ5rl.protocols` | Dataset and case construction, protocol evaluations, memory probes, diagnostics, and declarative CP imitation invariants. |
| `civ5rl.experiments` | CP imitation continuation, expanded-data and own-state run scenarios, shared experiment setup and evaluation, and Markdown reports. |
| `civ5rl.provenance` | Case manifests, common utilities, checkpoints, parent lineage, migration validation, and configuration validation. |

Core does not depend on engine or protocols. Engine does not import CLI modules from `tools/` or experiment scenarios. Collector workers import without loading PyTorch. `tools/run_cp_imitation.py` dispatches the three child protocols to `experiments`; shared policy/optimizer factories, worker pools, and demonstration loading live in `experiments.imitation_helpers`, rather than in the CLI. Fixed protocol constraints are declared in `protocols.imitation_invariants` and checked by shared validators.

The lightweight `core.schema` declares named feature layouts and computes their widths. Encoding order and network parameter shapes remain compatible with existing checkpoint tensors. The typed contracts in `core.contracts` describe scenario, worker, and episode dictionary boundaries. Collector methods share a `_WaveState`; shared-memory slots declare each field's cell, action, or fixed axis in their layout.

PPO uses one update loop in `engine.ppo` and one parameterized `BatchedBackend` in `engine.ppo_fast`. With batched PPO disabled, the backend uses nonresident inputs and one sequence per microbatch. Batched mode supports resident rollout inputs and configured microbatches. Both use the shared recurrent prefix reconstruction and behavior replay validation; the separate `ReferenceBackend` has been removed.

`engine.sequence.SequenceBuffer` supplies episode offsets, action counts, chunks, and input gathering for `DemonstrationCache`, `ResidentDemonstrations`, and `RolloutCache`. Resident allocation checks available CUDA memory and keeps a reserve. Cache subclasses supply BC or PPO labels. Shared span counting serves BC label minibatches, PPO token minibatches, and the resumable own-state `LabelStream`. `engine.training` provides the BC optimizer step and frozen-parameter verification. `CompilationClock` in `experiments.imitation_helpers` registers Dynamo callbacks as a context manager and removes them on exceptional exits.

`experiments.evaluation` runs the paired validation cases, verifies saved replays and executed learner histories, and compares evaluation directories using a bootstrap clustered by parent world. Pairing checks include case metadata, duplicate keys, and placement/side coverage. `engine.resources.analyze_telemetry` aggregates the selected phase, converts memory metrics to GiB, and computes time-weighted windows, with 60 seconds as the default. Phase changes, unavailable GPU samples, timestamp resets, and sampling gaps break windows; GPU metrics come from NVIDIA NVML.

Historical flat imports such as `civ5rl.observation` resolve to the same module objects as their canonical paths. This preserves private imports, patches, and old serialized `Encoded` references. New code should use canonical paths.

Passports recursively capture implementation files, RL configuration and documentation, native sources, tools, and engine build artifacts. Add experiment-specific workspace-relative paths or glob patterns through the optional `passport_artifacts` config list. Snapshot archive entries and source hashes are deduplicated. Execution identity compares non-Markdown source hashes and the recorded environment and corpus, excluding the Git commit. Markdown remains in the snapshot for historical context, but documentation changes alone do not block resume. Moving or changing executable sources prevents a strict resume even when checkpoint tensor shapes remain compatible.

`provenance.lineage` verifies parent source archives against their passport hashes, rejects source changes outside the protocol's explicit allowlist, and checks execution identity fields. A child protocol can permit a different corpus manifest while retaining environment checks. Existing output directories must still match their resolved configuration and execution identity. Named migration policies in `configs/rl-migration-policies.json` remain narrowly scoped; disallowed model, simulator, configuration, or policy-data changes are rejected.

The engine refactor changes executable source identity. Historical passports and `sources.zip` archives stay immutable, and their allowlists are not widened to bypass the refactor. New experiments require a new baseline run in a separate output directory; subsequent child runs must explicitly record its passport, checkpoint/data hashes, and agreed protocol. See [CP imitation passport guidance](rl-cp-imitation.md#паспорта-после-рефакторинга-engine).
