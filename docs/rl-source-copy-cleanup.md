# Historical source copies in artifacts

Accepted implementations belong to the main package. The saved copies below
were checked against Git history and today's implementation; no missing accepted
feature was found and no old module was copied over the current package.

| Historical copy | Main implementation and retained behavior |
|---|---|
| `collection-pipeline/collector_accepted.py` | `civ5rl/engine/collector.py`: ready-worker replies, legal-action validation, slot generations, per-slot GRU ownership, one result transfer per inference batch. Accepted source equals commit `8e37dab` after CRLF normalization; that commit is an ancestor of HEAD. Later streaming and temperature changes remain. |
| `collection-pipeline/ppo_artifacts_accepted.py` | `civ5rl/experiments/ppo_artifacts.py`: interned actions, packed ragged behavior distributions, omitted reconstructible caches, pickle protocol 5, atomic saves. Current source is identical after newline normalization. |
| `collection-pipeline/stream_worker_accepted.py` | `civ5rl/engine/stream_worker.py` and `worker_runtime.py`: bounded ready replies and wave/generation validation. CPU environment/encoding lifecycle was extracted into the shared runtime. Original is in commit `8e37dab`. |
| `collector-optimization/combat_ai_optimized.py` | `civ5sim/combat_ai.py`: immutable terrain tuple caching, ctypes tile cache by identity, cached dataclass field names, cache cleanup. Original is in commit `8e37dab`; later per-call packed-state and tracing remain. |
| `ppo-utilization/ppo_fast_measured.py` | `civ5rl/engine/ppo_fast.py`: replay groups/input plans, compiled replay reductions, carried recurrent state, combined diagnostic host transfer. Current code additionally retains free-decision KL and consistent temperature. Historical measurement differences are retained as a patch. |

The seven GRU `source_snapshot` files are exact Git versions (with original line
endings recorded). The production GRU graph backend remains in `training_graphs`
and `model`; the MLP/Identity quality hypothesis was not merged. The reusable
ablation entrypoint is `tools/archive/benchmark_gru_ablation.py`.

Each affected run has `source-evidence.json` and `source-evidence/README.md`.
They record original paths, SHA-256, byte counts, immutable Git bases and compact
patches. The patches are historical evidence and are not applied to today's
working tree. One previously untracked 126-line inference candidate is retained
as an add-file patch because it has no Git base. Temporary probe/benchmark
helpers are retired with their hashes and replacement commands; their exact
bytes are not claimed to be reconstructible.

```powershell
python tools/rl_source_evidence.py verify artifacts/rl/collection-pipeline/source-evidence.json
python tools/rl_source_evidence.py restore artifacts/rl/collection-pipeline/source-evidence.json artifacts/rl/collection-pipeline/collector_graphs_rejected.py --output C:/temp/collector-graphs.py
```

All 24 source variants were reconstructed byte-for-byte before deletion. Raw
result JSON and logs are verified against the migration inventory. The GRU
original artifact-hash manifest remains under `source-evidence/`; its historical
source entries are checked through reconstruction rather than missing files.
The small GRU summary now verifies saved results without rewriting them or
recreating `source_snapshot`.

CUDA Graph inference was rejected after a small full-collection slowdown. The
coalescing measurement did **not** establish worse latency: its short ABBA wall
times favored the delayed candidate, without a larger batch or per-observation
latency evidence. It was not adopted as the default. Neither rejected hypothesis
is enabled by this cleanup.

## Verification of this cleanup

- Removed 40 Python files (9,434 lines, 403,379 bytes): 24 historical source
  variants and 16 retired helpers. Source patches total 41,715 bytes.
- Reconstructed all 24 variants before and after deletion, matching original
  SHA-256 and byte counts, including CRLF. Verified 149 unchanged raw JSON/log
  files and the updated 42-entry GRU artifact manifest.
- Existing CP planner tests (15), package/import tests (2), PPO control test (1),
  and four PPO-after-imitation checks passed. The old free-KL test has a stale
  fake Cache exposing `batch` while the implementation already in HEAD calls
  `replay_batch`; it fails before testing KL. Neither production code nor that
  test fixture was changed by this cleanup.
- The GRU summary runs successfully using reconstructed historical sources.
  Ruff and `git diff --check` passed. No training or new evaluation was run.
