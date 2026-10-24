"""CLI: runs Phase 1's baseline remediation strategies (FIFO full re-embed,
uniform patch) against the shared benchmark, sweeping every configured
contamination ratio x budget level, and saves quality-vs-budget metrics +
a comparison chart to phase1_baseline/results/.

This is the reference point Phase 2's prioritized scheduler must beat in
2C, using the exact same benchmark, budgets, and contamination ratios.

Usage:
    python phase1_baseline/scripts/run_remediation.py [--config PATH]

Pipeline per contamination ratio:
  1. Take the (already-built) mixed-generation index as the "before" state.
  2. Run the configured 1C detector (default: KNN-OOD) to get a FLAGGED
     mask -- remediation only ever sees what detection actually found, not
     oracle ground truth.
  3. Fit one patch transform (common.patch) from a fixed calibration sample,
     reused across every ratio and budget level (it doesn't depend on
     either).
  4. For each budget level: run both strategies via the SAME FIFO ingestion
     order (phase1.remediation.fifo_remediation_plan), differing only in
     cost-per-fix and what "fixed" means for that document's embedding.
  5. Evaluate nDCG@10/Recall@10 (unweighted and traffic-weighted) on the
     resulting index for every strategy x budget x ratio combination.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _src in ("common/src", "phase1_baseline/src", "phase2_proposed/src"):
    sys.path.insert(0, str(_REPO_ROOT / _src))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from common.benchmark import build_benchmark, build_contaminated_ratio, mixed_embeddings  # noqa: E402
from common.config import Config, load_config  # noqa: E402
from common.embeddings import embed_texts  # noqa: E402
from common.metrics import evaluate_retrieval  # noqa: E402
from common.patch import fit_adapter, sample_calibration_indices  # noqa: E402
from common.seed import set_seed  # noqa: E402
from phase1.detection import detect_isolation_forest, detect_knn_ood, detect_lof  # noqa: E402
from phase1.remediation import fifo_remediation_plan  # noqa: E402

DETECTORS = {
    "knn_ood": lambda emb, ratio, cfg: detect_knn_ood(emb, ratio=ratio, k=cfg.detection.knn_k),
    "lof": lambda emb, ratio, cfg: detect_lof(emb, ratio=ratio, k=cfg.detection.lof_k),
    "isolation_forest": lambda emb, ratio, cfg: detect_isolation_forest(emb, ratio=ratio, seed=cfg.seed),
}


def run_remediation_sweep(cfg: Config) -> pd.DataFrame:
    benchmark = build_benchmark(cfg)
    doc_ids = [d.doc_id for d in benchmark.documents]
    eval_queries = [q for q in benchmark.queries if q.query_id in benchmark.qrels_eval]
    query_embeddings = embed_texts([q.text for q in eval_queries], cfg.embedding_models.generation_a)
    n_docs = len(benchmark.documents)

    # Patch transform is ratio-independent (fixed calibration sample, see
    # common/patch.py) -- fit once, reuse for every ratio and budget level.
    calibration_indices = sample_calibration_indices(
        len(benchmark.documents), cfg.remediation.calibration_sample_size, cfg.seed
    )
    # The patch is applied to whatever vector is STORED for a document (see
    # common.patch.Adapter): for a false positive that is its correct
    # generation-A vector, which the patch then damages. The original
    # version patched every flagged document with its true generation-B
    # vector, information no operator has.
    adapter = fit_adapter(
        benchmark.emb_a, benchmark.emb_b, calibration_indices, cfg.remediation.adapter, cfg.remediation.ridge_alpha
    )

    detect_fn = DETECTORS[cfg.remediation.detector]
    rows: list[dict] = []

    for ratio in cfg.contamination.ratios:
        contaminated = build_contaminated_ratio(cfg, benchmark, ratio)
        mixed = mixed_embeddings(benchmark, contaminated)

        flagged = detect_fn(mixed, ratio, cfg).predicted
        n_flagged = int(flagged.sum())

        no_remediation = evaluate_retrieval(
            mixed, doc_ids, eval_queries, query_embeddings, benchmark.qrels_eval, benchmark.query_traffic
        )
        rows.append(
            {"ratio": ratio, "strategy": "no_remediation", "budget_level": 0.0, "cost_spent": 0.0, "n_fixed": 0, **no_remediation}
        )

        # Oracle ceiling: every TRULY contaminated doc (not just flagged
        # ones) fully re-embedded -- the best any budget could achieve, and
        # the gap between this and FIFO's budget=1.0 result is the cost of
        # imperfect detection specifically, separate from imperfect budget.
        oracle = mixed.copy()
        oracle[contaminated.mask] = benchmark.emb_a[contaminated.mask]
        oracle_metrics = evaluate_retrieval(
            oracle, doc_ids, eval_queries, query_embeddings, benchmark.qrels_eval, benchmark.query_traffic
        )
        rows.append(
            {
                "ratio": ratio,
                "strategy": "oracle_full_recovery",
                "budget_level": None,
                "cost_spent": None,
                "n_fixed": int(contaminated.mask.sum()),
                **oracle_metrics,
            }
        )

        for budget_level in cfg.budget.levels:
            # Sized from the operator's prevalence estimate round(ratio * N),
            # the same absolute budget Phase 2 gets. For KNN-OOD this equals
            # the old n_flagged-based sizing exactly (it flags round(ratio * N)).
            budget_units = budget_level * round(ratio * n_docs) * cfg.remediation.cost_full_reembed
            t0 = time.time()

            fixed_full, spent_full = fifo_remediation_plan(flagged, budget_units, cfg.remediation.cost_full_reembed)
            emb_fifo = mixed.copy()
            emb_fifo[fixed_full] = benchmark.emb_a[fixed_full]
            metrics_fifo = evaluate_retrieval(
                emb_fifo, doc_ids, eval_queries, query_embeddings, benchmark.qrels_eval, benchmark.query_traffic
            )
            rows.append(
                {
                    "ratio": ratio,
                    "strategy": "fifo_full_reembed",
                    "budget_level": budget_level,
                    "cost_spent": spent_full,
                    "n_fixed": int(fixed_full.sum()),
                    **metrics_fifo,
                }
            )

            fixed_patch, spent_patch = fifo_remediation_plan(flagged, budget_units, cfg.remediation.cost_patch)
            emb_patch = mixed.copy()
            emb_patch[fixed_patch] = adapter(mixed[fixed_patch])
            metrics_patch = evaluate_retrieval(
                emb_patch, doc_ids, eval_queries, query_embeddings, benchmark.qrels_eval, benchmark.query_traffic
            )
            rows.append(
                {
                    "ratio": ratio,
                    "strategy": "uniform_patch",
                    "budget_level": budget_level,
                    "cost_spent": spent_patch,
                    "n_fixed": int(fixed_patch.sum()),
                    **metrics_patch,
                }
            )

            elapsed = time.time() - t0
            print(
                f"ratio={ratio:.2f} budget={budget_level:.2f}: "
                f"fifo n_fixed={fixed_full.sum():4d} ndcg@10_tw={metrics_fifo['ndcg@10_traffic_weighted']:.3f} | "
                f"patch n_fixed={fixed_patch.sum():4d} ndcg@10_tw={metrics_patch['ndcg@10_traffic_weighted']:.3f} "
                f"({elapsed:.1f}s)"
            )

    return pd.DataFrame(rows)


def plot_quality_vs_budget(df: pd.DataFrame, out_path: Path, metric: str = "ndcg@10_traffic_weighted") -> None:
    ratios = sorted(df["ratio"].unique())
    fig, axes = plt.subplots(1, len(ratios), figsize=(4 * len(ratios), 4), sharey=True)
    if len(ratios) == 1:
        axes = [axes]

    for ax, ratio in zip(axes, ratios):
        sub = df[df["ratio"] == ratio]
        no_rem = sub[sub["strategy"] == "no_remediation"][metric].iloc[0]
        oracle = sub[sub["strategy"] == "oracle_full_recovery"][metric].iloc[0]

        for strategy, label in [("fifo_full_reembed", "FIFO full re-embed"), ("uniform_patch", "Uniform patch")]:
            curve = sub[sub["strategy"] == strategy].sort_values("budget_level")
            ax.plot(curve["budget_level"], curve[metric], marker="o", label=label)

        ax.axhline(no_rem, color="gray", linestyle=":", label="No remediation" if ratio == ratios[0] else None)
        ax.axhline(oracle, color="green", linestyle="--", label="Oracle full recovery" if ratio == ratios[0] else None)
        ax.set_title(f"ratio={ratio:.2f}")
        ax.set_xlabel("Budget level")
        ax.set_ylim(0, 1.05)

    axes[0].set_ylabel(metric)
    fig.suptitle(f"Phase 1 baseline remediation: {metric} vs. budget")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="phase1_baseline/configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    df = run_remediation_sweep(cfg)

    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "remediation_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")

    # Two views on purpose (see DEVLOG.md's sub-phase 1D entry): the
    # unweighted nDCG@10 is the statistically robust primary comparison
    # (every eval query contributes comparably) -- the traffic-weighted
    # variant is thematically what this project cares about, but with
    # zipf_s=1.2 over a ~300-query eval set its effective sample size is
    # only ~6.5 queries, so treat it as a noisier, secondary view.
    unweighted_path = results_dir / "remediation_quality_by_budget_unweighted.png"
    plot_quality_vs_budget(df, unweighted_path, metric="ndcg@10")
    print(f"Saved plot to {unweighted_path}")

    weighted_path = results_dir / "remediation_quality_by_budget_traffic_weighted.png"
    plot_quality_vs_budget(df, weighted_path, metric="ndcg@10_traffic_weighted")
    print(f"Saved plot to {weighted_path}")


if __name__ == "__main__":
    main()
