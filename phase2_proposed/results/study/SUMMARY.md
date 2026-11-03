# Multi-dataset study: summary

Seeds per dataset: SciFact 5, NFCorpus 5, FiQA 5, Quora 5. Values are mean ± std over seeds.

## Benchmarks

| Dataset | Mean headroom (nDCG@10, over ratios) | nDCG@10 clean index |
|---|---|---|
| SciFact | 0.123 ± 0.012 | 0.649 ± 0.002 |
| NFCorpus | 0.030 ± 0.001 | 0.318 ± 0.000 |
| FiQA | 0.067 ± 0.003 | 0.417 ± 0.001 |
| Quora | 0.119 ± 0.001 | 0.971 ± 0.000 |

## Detection F1 (mean over contamination ratios, probe documents excluded)

| Dataset | Fingerprint (ours) | KNN-OOD | LOF | Isolation Forest |
|---|---|---|---|---|
| SciFact | 0.996 ± 0.002 | 0.545 ± 0.003 | 0.327 ± 0.007 | 0.326 ± 0.032 |
| NFCorpus | 1.000 ± 0.000 | 0.483 ± 0.012 | 0.283 ± 0.009 | 0.353 ± 0.018 |
| FiQA | 0.988 ± 0.002 | 0.389 ± 0.003 | 0.216 ± 0.003 | 0.294 ± 0.018 |
| Quora | 0.966 ± 0.004 | 0.341 ± 0.001 | 0.224 ± 0.001 | 0.244 ± 0.023 |

## Gain over the better baseline at matched budget (% of headroom, nDCG@10)

| Dataset | Wins | VectorSentry (full) | Without the gate | Fingerprint + FIFO | KNN-OOD + MCKP | Traffic-weighted (full): median, wins | Abs. gain (full) |
|---|---|---|---|---|---|---|---|
| SciFact | 146/150 | 47.7 ± 4.7 | 49.6 ± 7.1 | 24.8 ± 4.3 | 7.4 ± 1.7 | 54.2, 125/150 | 0.0629 ± 0.0063 |
| NFCorpus | 145/150 | 38.8 ± 7.0 | 38.9 ± 7.5 | 16.0 ± 3.2 | 6.8 ± 3.5 | 24.9, 93/150 | 0.0115 ± 0.0009 |
| FiQA | 141/150 | 36.2 ± 4.9 | 39.9 ± 7.5 | 29.9 ± 5.9 | 11.3 ± 3.2 | 54.0, 136/150 | 0.0225 ± 0.0023 |
| Quora | 145/150 | 39.3 ± 4.1 | 39.3 ± 4.1 | 22.0 ± 0.3 | 0.8 ± 0.3 | 68.9, 150/150 | 0.0438 ± 0.0050 |

Overall: VectorSentry beats the better baseline in 577 of 600 (dataset, seed, ratio, budget) settings; mean gain 40.5% of headroom. One-sided Wilcoxon signed-rank test, VectorSentry > best baseline: p = 2.33e-96.

Losses: 23, at contamination ratios 5%, 10%, 35%, 50%; largest absolute loss 0.0243 nDCG@10 (headroom there: 0.0633 on average).

By contamination ratio (all datasets): 5%: 105/120 wins, 37.4%; 10%: 118/120 wins, 45.8%; 20%: 120/120 wins, 41.1%; 35%: 119/120 wins, 39.9%; 50%: 115/120 wins, 38.2%

## Budget level needed to recover half the headroom (median over seeds and ratios; '—' = not reached)

| Dataset | FIFO re-embed | Uniform patch | VectorSentry | Reached by VectorSentry / FIFO |
|---|---|---|---|---|
| SciFact | 0.75 | — | 0.10 | 25/25 vs 14/25 |
| NFCorpus | 1.00 | — | 0.50 | 25/25 vs 14/25 |
| FiQA | — | — | 0.50 | 25/25 vs 12/25 |
| Quora | — | — | 0.50 | 25/25 vs 0/25 |

## Verification gate

Negative control = patch the most-shown documents the detector calls clean. 'Harmful' settings are those where shipping it unchecked lowers nDCG@10.

| Dataset | Harmful settings | Patches blocked there | Loss there: unchecked → with gate | Non-harmful settings: patches blocked, loss with gate | Waves passed / tried (real plans) |
|---|---|---|---|---|---|
| SciFact | 25/25 | 95.1% (28,448 / 29,899) | 0.096 → 0.003 | — | 1,392 / 3,563 |
| NFCorpus | 25/25 | 91.9% (19,926 / 21,689) | 0.031 → 0.002 | — | 999 / 1,393 |
| FiQA | 25/25 | 93.0% (111,886 / 120,321) | 0.046 → 0.005 | — | 1,226 / 3,096 |
| Quora | 15/25 | 98.9% (46,286 / 46,792) | 0.029 → 0.000 | 68.5%, 0.021 max | 750 / 750 |

All harmful settings: 94.4% of harmful patches blocked; mean loss 0.053 unchecked vs 0.003 with the gate (worst 0.022).
