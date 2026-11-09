## Required reading

- Read `docs/README.md` before starting work, then read the relevant documents linked there. It is the entry point for current contracts, operating procedures, and experiment history.

## Progress Updates in Long-Running Tasks

1. **Cadence / Triggers:** Emit a visible message in chat at logical phase boundaries, whenever changing your plan or hypothesis, or whenever **more than 2–3 minutes** have elapsed since the last visible message (based on tool output timestamps `Created At` / `Completed At`) or **10–13 tool calls** have accumulated without visible text.
2. **Content (1–2 short paragraphs):**
   - What intermediate result or conclusion was reached in the current step;
   - Key decisions or assumptions made (especially at branching points or trade-offs where the approach could drift off course);
   - The immediate next step and why it is needed.
3. **Surface Key Reasoning:** Do not keep pivotal trade-offs, blockers, or course corrections purely inside internal `Thinking` blocks — surface them in the visible response as work progresses so the tool execution log is broken up by clear status checkpoints.

## Documentation policy

- `docs/` is the durable reference for how the project works and how to operate it: current contracts, architecture, reusable procedures, and a compact experiment ledger. It is not a workspace for generated reports, run logs, debugging diaries, or scratch plans.
- Update the relevant existing document when behavior changes. Rewrite the affected rule in place; do not accumulate dated updates or version appendices that contradict earlier text.
- Before creating a document, check whether its content belongs in an existing reference. Add a separate document only for a distinct, durable purpose; avoid duplicate descriptions and one document per run or investigation.
- Keep experiment results and detailed run reports in `artifacts/`. Add concise status, configuration, parent, result, and report links to `docs/experiment-ledger.md` instead of copying reports into `docs/`.
- Use `docs/archive/` only for historical protocols or investigations worth retaining for reproducibility or provenance. Archiving is not a reason to preserve every scratch note. Clearly distinguish proposals, completed runs, and unverified historical claims.
- Before moving documentation, check configuration, code, test, and Markdown references to its path and anchors. Preserve required input paths unless their consumers are deliberately updated; do not rewrite frozen passports or source archives to accommodate a move.
- Keep `docs/README.md` current when adding, moving, or retiring documents. Preserve its list format: filename/link followed by a concise 10–20-word abstract. Keep the index in English.

## Artifacts, tooling roles, and temporary scripts

- `artifacts/` is only for data, checkpoints, logs, JSON metrics, plots, and Markdown reports. It is an output/evidence directory, not a source-code directory.
- Do not create `.py` files or other executable helpers under `artifacts/`, including experiment-specific audit, benchmark, packaging, or reporting scripts. Existing historical scripts do not justify adding more; do not alter frozen evidence merely to enforce this layout.
- Keep execution and measurement/operational code separated by passport role (`is_readonly_source` in `civ5rl/provenance/identity.py`). Execution modules (`civ5rl/core`, `civ5rl/engine`, `civ5rl/protocols`, `civ5rl/experiments`, simulator code, and training/collection/evaluation/build entrypoints) are hashed into execution passports: editing them changes `execution_identity` and blocks normal `resume` and lineage checks. Execution code should record only low-level runtime facts and telemetry, while shared logic for auditing, benchmarking, Vast.ai packaging, and reporting belongs in non-execution locations excluded from execution passports (`civ5rl/reporting/`, `civ5rl/deploy/`, non-execution modules in `tools/`, or a dedicated subpackage registered in `is_readonly_source`) so operational and measurement tooling can evolve without invalidating training passports.
- Exploratory diagnostics, trace parsers, ad-hoc benchmarks, and one-off experiment helpers may live freely in `.tmp/`. Assume `.tmp/` can be cleared at any time: no durable workflow, resume path, or reproducibility claim may depend on its contents, and final data or reports belong in `artifacts/`.
- When a diagnostic, benchmark, audit, packaging, or reporting workflow becomes repeatable across runs, promote it to a subcommand of an appropriate CLI in `tools/` (extending an existing CLI rather than creating per-run scripts) backed by passport-excluded modules instead of accumulating permanent scripts in `.tmp/`.

## Testing policy

This is a single-developer research project. Prioritize experiment progress and clear implementation over comprehensive test coverage. Accept reasonable risk; do not apply production or enterprise testing standards by default.

- For routine changes, run the affected workflow or an existing relevant test. Do not automatically add new tests.
- Add tests primarily for silent correctness failures that could invalidate experiments: incorrect learning math, hidden-information leakage, cross-episode state contamination, or broken reproducibility.
- A failure that immediately produces a clear traceback usually does not need a separate test.
- Add regression tests for bugs that were costly to diagnose and are likely to recur.
- Test meaningful outcomes, not internal call sequences, module layout, thin wrappers, or implementation details.
- Do not preserve legacy compatibility unless explicitly required.
- Prefer a few focused checks over extensive fixtures, mocks, edge-case matrices, or duplicated coverage.
- Run checks relevant to the change. Reserve full-suite runs for broad changes; do not repeat successful checks without a concrete reason.
- Do not expand the task into test infrastructure work or unrelated test cleanup.

Before adding a test, identify the specific failure it catches and why a normal run or existing check would miss it. If the benefit is marginal, skip it.

## GPU utilization policy

- During sustained iterative data processing on the GPU, target GPU Core Load of at least 80%, averaged over 60 seconds within that iterative phase. Use NVIDIA NVML or an equivalent hardware metric; do not use Windows WDDM GPU D3D Usage as a substitute.
- Treat NVML GPU Core Load >= 80% as necessary, not sufficient: a single-SM or serialized kernel (notably PyTorch deterministic sorted indexing_backward on high-fan-in gathers under 	orch.use_deterministic_algorithms(True)) can hold NVML Core Load near 100% while leaving most SMs idle and GPU power (% TDP) anomalously low. Cross-check Core Load against GPU power (% TDP), step wall time relative to model arithmetic, and a short Kineto kernel trace (	op_kernels) whenever verifying new or modified GPU code.
- Identify bottlenecks from profiler/kernel traces, never from guesswork or component counts alone.
- If the 60-second average is below 80%, investigate bottlenecks and optimization options. Continuing below this target requires an explicit, evidence-based explanation.
- Evaluate each sustained iterative GPU phase separately if a run has several. The requirement does not apply to an iterative phase shorter than 10 minutes; total run wall time is not the criterion. Do not extend a short phase just to measure or raise utilization.
- Exclude initial preparation, compilation, post-processing, CPU-only collection/evaluation, pauses, and other work outside iterative GPU data processing from both phase duration and utilization averages. Preserve the agreed experimental conditions when optimizing; do not silently change the model, precision, batch size, or training budget to raise utilization.

## Vast.ai machine selection

- Unless explicitly agreed otherwise, search and prepare two parallel candidates (`--parallel 2`) when creating a Vast.ai instance. Keep the requested GPU profile and its other defaults unchanged.

## Experiment planning and pre-launch checks

- When proposing a new model component, state its expected computational cost and simpler alternatives.
- Keep local pre-launch benchmarks short: measure warmup plus a handful of steps and a short Kineto trace (e.g., --warmup 3 --steps 5 --trace-steps 3) rather than hundreds of steps.
- Before starting a full local or remote run, calculate steps_per_epoch * target_gpu_step_seconds * planned_epochs (or total planned updates) and compare it against 	raining_seconds. If the planned training volume does not fit within the time limit, report the numbers before launching rather than after.
