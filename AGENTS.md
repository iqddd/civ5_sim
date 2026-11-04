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