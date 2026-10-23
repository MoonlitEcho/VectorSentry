# VectorSentry results index (sub-phase 2E)

Every chart, table, and number from sub-phases 1C-1D and 2A-2C, in one
place, so the formal paperwork can be assembled from real, already-computed
results later without re-running experiments under time pressure. Not a
report itself -- for the full reasoning behind every number here, see
`DEVLOG.md` at the repo root (dated, one entry per sub-phase).

All results use the **primary benchmark**: BEIR SciFact, 5,000 documents,
seed 42, Generation A = `all-MiniLM-L6-v2`, Generation B = `all-MiniLM-L12-v2`
(`phase1_baseline/configs/default.yaml` / `phase2_proposed/configs/default.yaml`
-- identical on purpose, see either file's header comment).

## From Phase 1 (`from_phase1_baseline/`)

Copied from `phase1_baseline/results/` so everything Phase 2's paperwork
needs is in one folder; `phase1_baseline/results/` remains the canonical
copy for Phase 1's own review.

| File | Sub-phase | Contents |
|---|---|---|
| `detection_metrics.csv`, `detection_f1_by_ratio.png` | 1C | precision/recall/F1 for KNN-OOD, LOF, Isolation Forest x 5 ratios. KNN-OOD is the strongest baseline (F1 ~0.51-0.57), and is the detector both phases' remediation/scheduling build on. |
| `remediation_metrics.csv`, `remediation_quality_by_budget_{unweighted,traffic_weighted}.png` | 1D | nDCG@10/Recall@10 for FIFO full re-embed, uniform patch, no-remediation, oracle-full-recovery, x 5 ratios x 6 budget levels. This is the reference point Phase 2 is compared against in 2C. |
| `aniso_stress_bge/*` | 1C (bonus finding) | Same corpus/seed, Generation B = bge-small-en-v1.5 instead of MiniLM-L12: demonstrates a highly anisotropic embedding model can invert or nullify every baseline detector's signal. Documented separately, not blended into the primary comparison. |

## Phase 2's own results (this folder, top level)

| File | Sub-phase | Contents |
|---|---|---|
| `scheduler_greedy_full_scale.csv`, `scheduler_action_allocation.png` | 2A | Production (efficient-frontier) greedy schedule x 5 ratios x 6 budget levels: action mix (full re-embed / patch / defer) and total benefit. Patch is almost never chosen in the primary config -- correct given `recovery_patch=0.064` is empirically dominated in cost-efficiency by full re-embed. |
| `scheduler_optimality_gap.csv`, `scheduler_optimality_gap.png` | 2A | Naive-greedy vs. frontier-greedy vs. exact-solver benefit on a tractable 300-doc subsample. Primary config: ~0% gap for both (degenerate to uniform-cost knapsack). See `balanced_actions_stress_test/` for the non-trivial case. |
| `migration_metrics.csv`, `migration_quality_by_budget.png` | 2B | Full migrate+verify loop: for every ratio x budget level, the scheduler's plan, whether verification passed or aborted, and the resulting FULL-corpus retrieval quality on whatever actually ended up live (not an idealized unconstrained number). 10 of 30 combinations aborted -- see DEVLOG for the two diagnosed reasons. |
| `comparison_quality_by_budget_{unweighted,traffic_weighted}.png`, `comparison_improvement.csv` | 2C | Head-to-head: Phase 1's two baselines vs. Phase 2's scheduler, same chart. Phase 2 beats both Phase 1 baselines in 18/30 ratio x budget combinations (loses in exactly the 10 that 2B aborted). Wins are decisive at low budgets: up to 10x less budget needed to reach 50% of the oracle-vs-no-remediation headroom at some ratios; up to +45% of headroom recovered at 10% budget. |

## Supplementary stress test (`balanced_actions_stress_test/`)

Same corpus/seed as the primary config, but `scheduler.recovery_patch=0.12`
(tuned near cost-efficiency parity with full re-embed, instead of the
empirically-dominated 0.064) -- specifically to exercise a genuinely
non-trivial three-way MCKP tradeoff. Exposed a real 88% optimality gap in
the naive flat-ratio-sort greedy; the production frontier greedy closes it
to ~0% (max 0.14%) on the identical instance. This is the evidence behind
switching the production scheduler from naive to frontier greedy.

| File | Contents |
|---|---|
| `scheduler_greedy_full_scale.csv`, `scheduler_action_allocation.png` | Production greedy's schedule under the stress-test recovery factor (patch now used substantially, unlike the primary config). |
| `scheduler_optimality_gap.csv`, `scheduler_optimality_gap.png` | The naive-vs-frontier-vs-exact comparison that motivated the algorithm switch: 88.00% mean gap (naive) vs. 0.00% (frontier). |

## Headline numbers, for quick reference

- **Detection (1C):** KNN-OOD F1 ~0.51-0.57 across ratios (primary config).
- **Baseline remediation ceiling (1D):** even FIFO at 100% budget never
  reaches the oracle line -- the gap is entirely attributable to imperfect
  detection recall (~52-58%), not insufficient budget.
- **Scheduler algorithm choice (2A):** naive greedy 88.00% mean optimality
  gap on a stress-test instance vs. frontier greedy's 0.00% (max 0.14%).
- **Migration safety (2B):** verification gate fires for real -- 10/30
  ratio x budget combinations aborted, all traced to either (a) fixing an
  unrelated document shifting ranking for a different query, or (b) the
  gate conservatively protecting a small canary set from even a minor
  regression while a large aggregate improvement (+0.117 nDCG@10 in one
  case) goes undeployed.
- **Head-to-head (2C):** Phase 2 beats Phase 1's best baseline in 18/30
  combinations; up to 10x less budget needed for the same quality target
  at some ratios; loses in exactly the 10 combinations the safety gate
  aborted -- an honest, two-sided result, not a uniform win.
- **Live Sentry (2D):** the full detect -> alert -> schedule -> migrate ->
  verify -> clear loop runs autonomously end to end on the real 5000-doc
  benchmark after one manual trigger, including a "give up and defer for
  manual review" path when verification can't be satisfied even at maximum
  budget -- disclosed honestly in the UI, not hidden.

## Reproducing everything from scratch

```bash
# Phase 1 (produces phase1_baseline/results/*)
python common/scripts/build_benchmark.py
python phase1_baseline/scripts/run_detection.py
python phase1_baseline/scripts/run_remediation.py

# Phase 2 (produces phase2_proposed/results/*)
python phase2_proposed/scripts/run_scheduler.py
python phase2_proposed/scripts/run_migration.py
python phase2_proposed/scripts/run_comparison.py

# Stress tests (both phases)
python common/scripts/build_benchmark.py --config phase1_baseline/configs/aniso_stress_bge.yaml
python phase1_baseline/scripts/run_detection.py --config phase1_baseline/configs/aniso_stress_bge.yaml
python phase2_proposed/scripts/run_scheduler.py --config phase2_proposed/configs/balanced_actions_stress_test.yaml

# Live demo (not a batch script -- interactive)
streamlit run phase2_proposed/live_sentry_dashboard.py
```

After regenerating Phase 1 or Phase 2's primary results, re-copy the
Phase 1 files into `from_phase1_baseline/` (or just re-run this sub-phase's
copy step) so this index stays in sync.
