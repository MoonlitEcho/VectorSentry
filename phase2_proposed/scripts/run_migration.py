"""CLI: runs Phase 2's full migrate-and-verify loop (2B) -- for every
contamination ratio x budget level, solves the MCKP schedule (2A's
production frontier greedy), executes it as a shadow-build + golden-query
verification + atomic swap-or-abort, and records both the verification
outcome and the resulting FULL-corpus retrieval quality (on whatever ends
up live -- the shadow if verification passed, the unchanged original if it
was aborted).

The saved CSV is the artifact 2C's head-to-head comparison reads Phase 2's
numbers from, in the same shape as phase1_baseline/results/
remediation_metrics.csv, so both can be plotted on one chart.

Usage:
    python phase2_proposed/scripts/run_migration.py [--config PATH]
"""

from __future__ import annotations

import argparse
import sys
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
from common.patch import apply_patch, fit_patch_transform, sample_calibration_indices  # noqa: E402
from common.seed import set_seed  # noqa: E402
from phase1.detection import detect_knn_ood  # noqa: E402
from phase2.migrate import execute_migration, select_golden_query_indices  # noqa: E402
from phase2.scheduler import build_efficient_frontier_segments, compute_severity, solve_greedy  # noqa: E402


def run_migration_sweep(cfg: Config) -> pd.DataFrame:
    benchmark = build_benchmark(cfg)
    doc_ids = [d.doc_id for d in benchmark.documents]
    eval_queries = [q for q in benchmark.queries if q.query_id in benchmark.qrels_eval]
    query_embeddings = embed_texts([q.text for q in eval_queries], cfg.embedding_models.generation_a)

    calibration_indices = sample_calibration_indices(
        len(benchmark.documents), cfg.remediation.calibration_sample_size, cfg.seed
    )
    transform = fit_patch_transform(benchmark.emb_a, benchmark.emb_b, calibration_indices)
    patched_emb_b = apply_patch(benchmark.emb_b, transform)

    golden_indices = select_golden_query_indices(eval_queries, benchmark.query_traffic, cfg.migration.golden_query_count)
    golden_queries = [eval_queries[i] for i in golden_indices]
    golden_query_embeddings = query_embeddings[golden_indices]

    rows: list[dict] = []

    for ratio in cfg.contamination.ratios:
        contaminated = build_contaminated_ratio(cfg, benchmark, ratio)
        mixed = mixed_embeddings(benchmark, contaminated)
        detection_result = detect_knn_ood(mixed, ratio=ratio, k=cfg.detection.knn_k)
        flagged_indices = np.flatnonzero(detection_result.predicted)
        n_flagged = len(flagged_indices)

        severity = compute_severity(detection_result.scores[flagged_indices])
        traffic = benchmark.doc_traffic[flagged_indices]
        segments = build_efficient_frontier_segments(
            severity,
            traffic,
            cfg.remediation.cost_full_reembed,
            cfg.remediation.cost_patch,
            cfg.scheduler.recovery_full_reembed,
            cfg.scheduler.recovery_patch,
        )

        for budget_level in cfg.budget.levels:
            budget_units = budget_level * n_flagged * cfg.remediation.cost_full_reembed
            plan = solve_greedy(segments, n_items=n_flagged, budget=budget_units)

            report = execute_migration(
                mixed,
                benchmark.emb_a,
                patched_emb_b,
                flagged_indices,
                plan.actions,
                doc_ids,
                golden_queries,
                golden_query_embeddings,
                benchmark.qrels_eval,
                benchmark.query_traffic,
                regression_tolerance=cfg.migration.regression_tolerance,
            )

            full_quality = evaluate_retrieval(
                report.live_embeddings,
                doc_ids,
                eval_queries,
                query_embeddings,
                benchmark.qrels_eval,
                benchmark.query_traffic,
                k=10,
            )

            rows.append(
                {
                    "ratio": ratio,
                    "budget_level": budget_level,
                    "n_flagged": n_flagged,
                    "n_full_reembed": report.n_full_reembed,
                    "n_patch": report.n_patch,
                    "n_defer": report.n_defer,
                    "verification_passed": report.verification_passed,
                    "golden_ndcg_before": report.golden_ndcg_before,
                    "golden_ndcg_after": report.golden_ndcg_after,
                    **full_quality,
                }
            )
            print(
                f"ratio={ratio:.2f} budget={budget_level:.2f}: "
                f"{'PASSED' if report.verification_passed else 'ABORTED'} "
                f"golden_ndcg {report.golden_ndcg_before:.3f}->{report.golden_ndcg_after:.3f}  "
                f"full ndcg@10={full_quality['ndcg@10']:.3f}"
            )

    return pd.DataFrame(rows)


def plot_migration_quality(df: pd.DataFrame, out_path: Path) -> None:
    ratios = sorted(df["ratio"].unique())
    fig, axes = plt.subplots(1, len(ratios), figsize=(4 * len(ratios), 4), sharey=True)
    if len(ratios) == 1:
        axes = [axes]

    for ax, ratio in zip(axes, ratios):
        sub = df[df["ratio"] == ratio].sort_values("budget_level")
        ax.plot(sub["budget_level"], sub["ndcg@10"], marker="o", label="Phase 2 (MCKP, verified)")
        aborted = sub[~sub["verification_passed"]]
        if len(aborted) > 0:
            ax.scatter(aborted["budget_level"], aborted["ndcg@10"], color="red", zorder=5, label="Aborted")
        ax.set_title(f"ratio={ratio:.2f}")
        ax.set_xlabel("Budget level")
        ax.set_ylim(0, 1.0)

    axes[0].set_ylabel("nDCG@10 (unweighted, live embeddings)")
    fig.suptitle("Phase 2: verified migration quality vs. budget")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="phase2_proposed/configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    df = run_migration_sweep(cfg)

    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "migration_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    plot_path = results_dir / "migration_quality_by_budget.png"
    plot_migration_quality(df, plot_path)
    print(f"Saved {plot_path}")

    n_aborted = int((~df["verification_passed"]).sum())
    print(f"\n{n_aborted} of {len(df)} ratio x budget combinations aborted verification.")


if __name__ == "__main__":
    main()
