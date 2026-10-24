# VectorSentry results index

Every chart, table, and number from Phase 1 and Phase 2, in one place, so
the formal paperwork can be assembled from real, already-computed results.
Not a report itself -- for the reasoning behind every number, see
`DEVLOG.md` at the repo root (the 2026-10-24 "Redesign" entry covers
everything regenerated in this folder).

All results use the **primary benchmark**: BEIR SciFact, 5,000 documents,
seed 42, Generation A = `all-MiniLM-L6-v2`, Generation B = `all-MiniLM-L12-v2`,
Zipf query traffic (s = 1.0) with impression-based document traffic
(`phase1_baseline/configs/default.yaml` / `phase2_proposed/configs/default.yaml`
-- identical benchmark sections on purpose). Budgets are in full-re-embed
units, `budget_level x round(ratio x N)`, the same absolute budget for every
strategy in both phases.

## From Phase 1 (`from_phase1_baseline/`)

Copied from `phase1_baseline/results/`; that folder stays the canonical copy
for Phase 1's own review.

| File | Contents |
|---|---|
| `detection_metrics.csv`, `detection_f1_by_ratio.png` | KNN-OOD / LOF / Isolation Forest precision, recall, F1 x 5 ratios. KNN-OOD is the strongest baseline (F1 0.51-0.57). |
| `remediation_metrics.csv`, `remediation_quality_by_budget_{unweighted,traffic_weighted}.png` | FIFO full re-embed and uniform patch on KNN-OOD's flagged set, plus no-remediation and oracle, x 5 ratios x 6 budgets. The reference Phase 2 is compared against. |
| `aniso_stress_bge/*` | Same corpus, Generation B = bge-small-en-v1.5: every Phase 1 detector inverts or collapses (F1 ~0.00-0.53). |

## Phase 2 (this folder)

| File | Contents |
|---|---|
| `detection_comparison.csv`, `detection_comparison_f1.png` | Phase 2's generation-fingerprint detector vs. the three Phase 1 detectors, on BOTH pairings, probe documents excluded. Phase 1 detectors are given the true ratio; the fingerprint is given none. |
| `scheduler_greedy_full_scale.csv`, `scheduler_action_allocation.png` | Expected-benefit MCKP schedule (efficient-frontier greedy) x 5 ratios x 6 budgets: the action mix. At low budgets most flagged documents are patched and the most-shown ones fully re-embedded; at full budget almost everything is re-embedded. |
| `scheduler_optimality_gap.csv`, `scheduler_optimality_gap.png` | Naive greedy vs. frontier greedy vs. exact ILP on a 300-document subsample. |
| `migration_metrics.csv`, `migration_quality_by_budget.png` | The full pipeline (detect -> MCKP -> staged, verified rollout with budget recycling) and three ablations, x 5 ratios x 6 budgets, quality measured on whatever actually ended up live. Also one negative-control row per ratio (`negative_control_patch_clean`). |
| `comparison_quality_by_budget_{unweighted,traffic_weighted}.png`, `comparison_improvement.csv` | Head-to-head vs. the better Phase 1 baseline at every matched ratio x budget, unweighted and traffic-weighted. |

## Supplementary stress test (`balanced_actions_stress_test/`)

The original 2A solver stress test (`recovery_source: config`,
recovery_patch = 0.12 on KNN-OOD severity): naive greedy 87.2% mean gap,
frontier greedy 0.00%. Kept because it isolates the solver; the primary
config now exercises the same three-way tradeoff on its own.

## Headline numbers

- **Detection:** fingerprint mean F1 **0.996** (primary) and **0.998**
  (bge-small stress test) with no assumed ratio, vs. KNN-OOD 0.548 / 0.003,
  LOF 0.328 / 0.243, Isolation Forest 0.279 / 0.003 given the true ratio.
- **Action effects, estimated label-free by query-log replay on the probe
  batch (ridge adapter):** a patch restores 65% of a contaminated document's
  lost impressions (`recovery_patch` = 0.650) and damages a clean document by
  about as much (`harm_patch` = 0.657).
- **Scheduler:** the primary MCKP is now a real three-way tradeoff; naive
  greedy loses **40.3%** of the optimum on average, the frontier greedy
  **0.01%** (max 0.04%).
- **Head-to-head:** Phase 2 beats the better Phase 1 baseline in **30 / 30**
  ratio x budget combinations; mean **+0.065 nDCG@10**, **+52.2%** of the
  oracle-vs-no-remediation headroom (traffic-weighted **+47.2%**). It reaches
  50% of the headroom at budget 0.1-0.2 at every ratio; FIFO needs 0.75-1.0,
  or never gets there (ratios 0.35 and 0.50).
- **Ablations** (mean % of headroom over the better Phase 1 baseline):
  fingerprint + FIFO +29.5%, KNN-OOD + MCKP +7.1%, both together +52.2%. At
  budget 0.1 neither alone does much (-3% to +26%); together +26% to +64%.
- **Safety gate:** on the real plans it costs 0.003 nDCG on average (worst
  0.020) versus shipping ungated. On the negative control (patching the
  most-shown clean documents, which would drop nDCG@10 by up to 0.126) it
  rolls back **5,884 of 6,021** harmful patches, and live quality ends within
  0.004 of untouched at every ratio.
- **Live Sentry:** a 500-document incident on the full benchmark is flagged
  (503 flags vs. 4 on the clean control), remediated in 3 rounds (~12 s), and
  nDCG@10 recovers 0.593 -> 0.645 (clean: 0.652).

## Reproducing everything

```bash
# Benchmark + Phase 1 (phase1_baseline/results/*)
python common/scripts/build_benchmark.py
python phase1_baseline/scripts/run_detection.py
python phase1_baseline/scripts/run_remediation.py

# Phase 2 (phase2_proposed/results/*)
python phase2_proposed/scripts/run_detection.py
python phase2_proposed/scripts/run_scheduler.py
python phase2_proposed/scripts/run_migration.py
python phase2_proposed/scripts/run_comparison.py

# Stress tests
python common/scripts/build_benchmark.py --config phase1_baseline/configs/aniso_stress_bge.yaml
python phase1_baseline/scripts/run_detection.py --config phase1_baseline/configs/aniso_stress_bge.yaml
python phase2_proposed/scripts/run_scheduler.py --config phase2_proposed/configs/balanced_actions_stress_test.yaml

# Live demo (interactive)
python -m streamlit run phase2_proposed/live_sentry_dashboard.py
```

After regenerating Phase 1's results, re-copy them into
`from_phase1_baseline/` so this folder stays in sync.
