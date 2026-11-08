# Historical tools

These 16 files preserve completed experiments and investigations. They are for
inspection or deliberate historical reproduction, not recommended commands for
new experiments. Inputs and old native libraries may need restoration from
local archives. Current public tools are described in [../README.md](../README.md).

## Historical CLI entrypoints

| Tool | Historical purpose |
|---|---|
| [benchmark_gru_ablation.py](benchmark_gru_ablation.py) | Compare GRU/MLP/Identity on 512 fixed episodes with eight minibatches, a pinned parent checkpoint, and dated output. |
| [study_cp_reset.py](study_cp_reset.py) | Execute the completed CP reset study with its fixed corpus and 372-match plan. |
| [check_viewer.py](check_viewer.py) | Check one historical map viewer with fixed map counts, dimensions, crop size, and download expectations. |
| [compare_target_selection.py](compare_target_selection.py) | Compare a completed target-selection change against CP 0.9.3 using its historical input and output directories. |
| [measure_tactical_storage.py](measure_tactical_storage.py) | Reproduce the ABI5-to-ABI6 storage investigation, including its experiment-specific conversion of old requests. |
| [publish_greedy_replays.py](publish_greedy_replays.py) | Publish and archive the historical CP/greedy match series using its original viewer URLs. |
| [summarize_coordinator_acceptance.py](summarize_coordinator_acceptance.py) | Summarize before/after acceptance evidence for one specific CP-07 coordinator revision. |
| [migrate_rl_audit.py](migrate_rl_audit.py) | Perform the narrowly defined migration for completed audit/pipeline revisions, rather than general checkpoint resumption. |
| [run_rl_diagnostic.py](run_rl_diagnostic.py) | Run historical D0/D1/one-decision diagnostic protocols with their fixed criteria, budgets, and continuation rules. |
| [compare_ready_streaming.py](compare_ready_streaming.py) | Compare fixed 200-case A/B/C/C/B/A or W/80 runs against historical inputs, warmups, and parity checks. |
| [diagnose_rl_collection.py](diagnose_rl_collection.py) | Investigate fixed B0–B3/W80 configurations, with separate unranked Windows WPR/ETW diagnostic passes. |
| [evaluate_target_bc_sampling.py](evaluate_target_bc_sampling.py) | Evaluate original target BC epoch 10 using pinned hashes, 128 development cases, and three action-selection modes. |

Invoke these CLIs from the repository root:
`python tools/archive/<name>.py ...`. Moving them does not make their fixed
protocols reusable or authorize additional training, evaluations, or rentals.

## Runner dependencies without standalone CLIs

| Module | Historical purpose |
|---|---|
| [run_rl_multishot.py](run_rl_multishot.py) | Implement D0-M training and evaluation for the diagnostic runner, preserving the frozen multishot protocol. |
| [report_rl_one.py](report_rl_one.py) | Render saved results for the D0-one-decision protocol only. |
| [report_rl_d1.py](report_rl_d1.py) | Render saved results for the bounded D1 transfer protocol only. |
| [report_rl_multishot.py](report_rl_multishot.py) | Render saved results for the D0-M multishot protocol only. |

The general RL-01 pipeline and `run_rl01.py` were deleted. The diagnostic
protocols above have their own runners; the former `control_rl.py` is unavailable.
Controlled updates use the current shared PPO CLI:
`python tools/run_ppo.py <phase> --config configs/rl-ppo-controlled-updates.json`.

Source snapshots, manifests, and historical migration policies retain their
original recorded paths and bytes. A source move can legitimately invalidate
strict execution checks; no bypass was added. A newly executed sampling
comparison needs a new output when its evaluator identity differs.
`compare_ready_streaming.py` records its entrypoint path and SHA-256 in
`summary.json` instead of copying itself into output as `run_series.py`.
