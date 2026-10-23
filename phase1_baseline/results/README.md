# Phase 1 results index

Not a report -- an index of what's here, how it was produced, and where to
read the full reasoning. See `DEVLOG.md` at the repo root for the complete,
dated write-up of every decision and diagnosis behind these numbers.

All results below use the **primary benchmark**: BEIR SciFact, 5,000
documents, seed 42, Generation A = `all-MiniLM-L6-v2`, Generation B =
`all-MiniLM-L12-v2` (`phase1_baseline/configs/default.yaml`).

## Detection (sub-phase 1C)

| File | Produced by | Contents |
|---|---|---|
| `detection_metrics.csv` | `phase1_baseline/scripts/run_detection.py` | precision/recall/F1 for KNN-OOD, LOF, Isolation Forest x 5 contamination ratios |
| `detection_f1_by_ratio.png` | same | F1 vs. ratio, all three detectors |

KNN-OOD is the strongest baseline (F1 ~0.51-0.57, roughly flat across
ratios) and is the detector `remediation` and the dashboard build on
(`remediation.detector` in the config).

## Remediation (sub-phase 1D)

| File | Produced by | Contents |
|---|---|---|
| `remediation_metrics.csv` | `phase1_baseline/scripts/run_remediation.py` | nDCG@10/Recall@10 (unweighted + traffic-weighted), for FIFO full re-embed, uniform patch, no-remediation, and oracle-full-recovery, x 5 ratios x 6 budget levels |
| `remediation_quality_by_budget_unweighted.png` | same | primary, statistically robust comparison chart |
| `remediation_quality_by_budget_traffic_weighted.png` | same | secondary chart -- thematically the right metric, but noisy at this `zipf_s` (effective sample size ~6.5 of ~300 eval queries; see DEVLOG) |

## Anisotropy stress test (sub-phase 1C, documented bonus finding)

`aniso_stress_bge/` -- same corpus/seed, but Generation B =
`bge-small-en-v1.5` instead of MiniLM-L12
(`phase1_baseline/configs/aniso_stress_bge.yaml`). Demonstrates that a
highly anisotropic embedding model can invert or nullify every baseline
detector's anomaly signal -- kept separate from the primary comparison
above, not blended into it. Full diagnosis in DEVLOG's sub-phase 1C entry.

## Reproducing all of the above

```bash
python common/scripts/build_benchmark.py                                              # ~10-20 min, one-time
python phase1_baseline/scripts/run_detection.py
python phase1_baseline/scripts/run_remediation.py
python common/scripts/build_benchmark.py --config phase1_baseline/configs/aniso_stress_bge.yaml
python phase1_baseline/scripts/run_detection.py --config phase1_baseline/configs/aniso_stress_bge.yaml
```
