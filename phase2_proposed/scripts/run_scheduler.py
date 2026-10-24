"""CLI: runs Phase 2's MCKP prioritization scheduler against the shared
benchmark, across every contamination ratio x budget level, and saves both
the full-scale schedule (production frontier greedy) and a three-way
optimality comparison (naive greedy vs. frontier greedy vs. exact solver,
on a tractable subsample) to phase2_proposed/results/.

Uses the identical benchmark/detector as Phase 1 (see
phase2_proposed/configs/default.yaml's header comment) so results are
directly comparable in 2C without any re-embedding.

Usage:
    python phase2_proposed/scripts/run_scheduler.py [--config PATH]
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
from common.seed import set_seed  # noqa: E402
from phase1.detection import detect_knn_ood  # noqa: E402
from phase2.pipeline import build_phase2_context  # noqa: E402
from phase2.scheduler import (  # noqa: E402
    build_frontier_segments_from_benefits,
    build_mckp_items_from_benefits,
    compute_severity,
    expected_benefits,
    solve_exact,
    solve_greedy,
    solve_greedy_naive,
)


def _scheduling_instance(cfg: Config, benchmark, ctx, ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(candidate document indices, benefit_full, benefit_patch) for one
    ratio, under whichever benefit model the config selects:

    - recovery_source "replay" (primary): fingerprint-flagged documents,
      expected benefits from P(contaminated), impression traffic and the
      replay-estimated patch recovery/harm (phase2.pipeline).
    - recovery_source "config" (the balanced-actions stress test and the
      original 2A model): KNN-OOD-flagged documents, min-max severity x
      traffic x the configured recovery constants.
    """
    contaminated = build_contaminated_ratio(cfg, benchmark, ratio)
    mixed = mixed_embeddings(benchmark, contaminated)
    if cfg.scheduler.recovery_source == "replay":
        p_contaminated, flagged = ctx.detector.flag(mixed, cfg.fingerprint.threshold)
        indices = np.flatnonzero(flagged)
        benefit_full, benefit_patch = expected_benefits(
            p_contaminated[indices], ctx.traffic[indices], cfg.scheduler.recovery_full_reembed,
            ctx.recovery_patch, ctx.harm_patch,
        )
    else:
        detection_result = detect_knn_ood(mixed, ratio=ratio, k=cfg.detection.knn_k)
        indices = np.flatnonzero(detection_result.predicted)
        base = compute_severity(detection_result.scores[indices]) * benchmark.doc_traffic[indices]
        benefit_full = base * cfg.scheduler.recovery_full_reembed
        benefit_patch = base * cfg.scheduler.recovery_patch
    return indices, benefit_full, benefit_patch


def run_full_scale_sweep(cfg: Config) -> pd.DataFrame:
    """Runs the PRODUCTION scheduler (efficient-frontier greedy) at FULL
    SCALE (every flagged document) for every ratio x budget level -- the
    plan a real deployment would execute (2B) and 2C compares against
    Phase 1's baselines."""
    benchmark = build_benchmark(cfg)
    ctx = build_phase2_context(cfg, benchmark)
    rows: list[dict] = []

    for ratio in cfg.contamination.ratios:
        indices, benefit_full, benefit_patch = _scheduling_instance(cfg, benchmark, ctx, ratio)
        n_flagged = len(indices)
        segments = build_frontier_segments_from_benefits(
            benefit_full, benefit_patch, cfg.remediation.cost_full_reembed, cfg.remediation.cost_patch
        )

        for budget_level in cfg.budget.levels:
            # Same absolute budget sizing as Phase 1 and run_migration.py.
            budget_units = budget_level * round(ratio * len(benchmark.documents)) * cfg.remediation.cost_full_reembed
            plan = solve_greedy(segments, n_items=n_flagged, budget=budget_units)
            n_full = int(np.sum(plan.actions == "full_reembed"))
            n_patch = int(np.sum(plan.actions == "patch"))
            n_defer = n_flagged - n_full - n_patch
            rows.append(
                {
                    "ratio": ratio,
                    "budget_level": budget_level,
                    "n_flagged": n_flagged,
                    "n_full_reembed": n_full,
                    "n_patch": n_patch,
                    "n_defer": n_defer,
                    "total_benefit": plan.total_benefit,
                    "total_cost": plan.total_cost,
                }
            )
            print(
                f"ratio={ratio:.2f} budget={budget_level:.2f}: "
                f"full_reembed={n_full:4d} patch={n_patch:4d} defer={n_defer:4d} "
                f"benefit={plan.total_benefit:.4f}"
            )

    return pd.DataFrame(rows)


def run_optimality_gap_sweep(cfg: Config) -> pd.DataFrame:
    """For every ratio, subsamples `scheduler.exact_solver_sample_size`
    flagged documents and runs ALL THREE solvers on that identical,
    tractable instance across every budget level (proportionally scaled to
    the subsample size), reporting both the naive and the frontier
    greedy's optimality gap against the exact solver -- the comparison
    that motivated switching the production scheduler from the naive to
    the frontier greedy (see DEVLOG.md's sub-phase 2A entry)."""
    benchmark = build_benchmark(cfg)
    ctx = build_phase2_context(cfg, benchmark)
    rows: list[dict] = []

    for ratio in cfg.contamination.ratios:
        indices, benefit_full, benefit_patch = _scheduling_instance(cfg, benchmark, ctx, ratio)

        rng = np.random.default_rng(cfg.seed)
        sample_size = min(cfg.scheduler.exact_solver_sample_size, len(indices))
        sample = rng.choice(len(indices), size=sample_size, replace=False)
        bf, bp = benefit_full[sample], benefit_patch[sample]
        items = build_mckp_items_from_benefits(bf, bp, cfg.remediation.cost_full_reembed, cfg.remediation.cost_patch)
        segments = build_frontier_segments_from_benefits(
            bf, bp, cfg.remediation.cost_full_reembed, cfg.remediation.cost_patch
        )

        for budget_level in cfg.budget.levels:
            budget_units = budget_level * sample_size * cfg.remediation.cost_full_reembed
            naive_plan = solve_greedy_naive(items, n_items=sample_size, budget=budget_units)
            frontier_plan = solve_greedy(segments, n_items=sample_size, budget=budget_units)
            exact_plan = solve_exact(items, n_items=sample_size, budget=budget_units)

            def _gap_pct(candidate_benefit: float) -> float:
                if exact_plan.total_benefit <= 0:
                    return 0.0
                return (exact_plan.total_benefit - candidate_benefit) / exact_plan.total_benefit * 100

            rows.append(
                {
                    "ratio": ratio,
                    "budget_level": budget_level,
                    "sample_size": sample_size,
                    "naive_greedy_benefit": naive_plan.total_benefit,
                    "frontier_greedy_benefit": frontier_plan.total_benefit,
                    "exact_benefit": exact_plan.total_benefit,
                    "naive_gap_pct": _gap_pct(naive_plan.total_benefit),
                    "frontier_gap_pct": _gap_pct(frontier_plan.total_benefit),
                }
            )
            print(
                f"[gap] ratio={ratio:.2f} budget={budget_level:.2f} n={sample_size}: "
                f"naive={naive_plan.total_benefit:.4f} ({_gap_pct(naive_plan.total_benefit):.2f}% gap)  "
                f"frontier={frontier_plan.total_benefit:.4f} ({_gap_pct(frontier_plan.total_benefit):.2f}% gap)  "
                f"exact={exact_plan.total_benefit:.4f}"
            )

    return pd.DataFrame(rows)


def plot_action_allocation(df: pd.DataFrame, out_path: Path) -> None:
    ratios = sorted(df["ratio"].unique())
    fig, axes = plt.subplots(1, len(ratios), figsize=(4 * len(ratios), 4), sharey=True)
    if len(ratios) == 1:
        axes = [axes]

    for ax, ratio in zip(axes, ratios):
        sub = df[df["ratio"] == ratio].sort_values("budget_level")
        ax.stackplot(
            sub["budget_level"],
            sub["n_full_reembed"],
            sub["n_patch"],
            sub["n_defer"],
            labels=["Full re-embed", "Patch", "Defer"],
        )
        ax.set_title(f"ratio={ratio:.2f}")
        ax.set_xlabel("Budget level")

    axes[0].set_ylabel("Flagged documents")
    fig.suptitle("Phase 2 scheduler: chosen action mix vs. budget")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_optimality_gap(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart: mean optimality gap by ratio, naive vs. frontier
    greedy. Grouped rather than a line-by-budget chart because the gap is
    essentially constant across budget levels within a ratio (see
    DEVLOG.md's sub-phase 2A entry for why) -- a bar comparison makes the
    naive-vs-frontier improvement the visual point, not a flat line."""
    summary = df.groupby("ratio")[["naive_gap_pct", "frontier_gap_pct"]].mean().reset_index()

    x = np.arange(len(summary))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, summary["naive_gap_pct"], width, label="Naive greedy")
    ax.bar(x + width / 2, summary["frontier_gap_pct"], width, label="Frontier greedy (production)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r:.2f}" for r in summary["ratio"]])
    ax.set_xlabel("Contamination ratio")
    ax.set_ylabel("Mean optimality gap vs. exact solver (%)")
    ax.set_title(f"Greedy heuristic optimality gap (n={df['sample_size'].iloc[0]} subsample)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="phase2_proposed/configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Full-scale schedule (production frontier greedy) ===")
    full_scale_df = run_full_scale_sweep(cfg)
    full_scale_path = results_dir / "scheduler_greedy_full_scale.csv"
    full_scale_df.to_csv(full_scale_path, index=False)
    print(f"Saved {full_scale_path}")
    plot_action_allocation(full_scale_df, results_dir / "scheduler_action_allocation.png")

    print("\n=== Optimality gap sweep (naive vs. frontier vs. exact, subsampled) ===")
    gap_df = run_optimality_gap_sweep(cfg)
    gap_path = results_dir / "scheduler_optimality_gap.csv"
    gap_df.to_csv(gap_path, index=False)
    print(f"Saved {gap_path}")
    plot_optimality_gap(gap_df, results_dir / "scheduler_optimality_gap.png")

    print(f"\nMean naive greedy gap:    {gap_df['naive_gap_pct'].mean():.2f}%  (max {gap_df['naive_gap_pct'].max():.2f}%)")
    print(f"Mean frontier greedy gap: {gap_df['frontier_gap_pct'].mean():.2f}%  (max {gap_df['frontier_gap_pct'].max():.2f}%)")


if __name__ == "__main__":
    main()
