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
- If the 60-second average is below 80%, investigate bottlenecks and optimization options. Continuing below this target requires an explicit, evidence-based explanation.
- Evaluate each sustained iterative GPU phase separately if a run has several. The requirement does not apply to an iterative phase shorter than 10 minutes; total run wall time is not the criterion. Do not extend a short phase just to measure or raise utilization.
- Exclude initial preparation, compilation, post-processing, CPU-only collection/evaluation, pauses, and other work outside iterative GPU data processing from both phase duration and utilization averages. Preserve the agreed experimental conditions when optimizing; do not silently change the model, precision, batch size, or training budget to raise utilization.

## Vast.ai machine selection

- Unless explicitly agreed otherwise, search and prepare two parallel candidates (`--parallel 2`) when creating a Vast.ai instance. Keep the requested GPU profile and its other defaults unchanged.
