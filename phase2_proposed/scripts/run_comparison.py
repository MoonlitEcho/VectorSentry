"""CLI: sub-phase 2C -- head-to-head comparison of Phase 2's prioritized
scheduler against Phase 1's baselines (FIFO full re-embed, uniform patch),
using the identical benchmark, budgets, and contamination ratios saved in
1F. Combines phase1_baseline/results/remediation_metrics.csv and
phase2_proposed/results/migration_metrics.csv into one chart and a
quantified improvement summary.

Phase 2's numbers are read as-is from migration_metrics.csv -- i.e. the
VERIFIED, actually-deployed quality (including any budget/ratio
combinations where 2B's golden-query gate aborted and rolled back), not an
idealized unconstrained-scheduler number. The comparison is honest about
where Phase 2 does and does not win.

Usage:
    python phase2_proposed/scripts/run_comparison.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _src in ("common/src", "phase1_baseline/src", "phase2_proposed/src"):
    sys.path.insert(0, str(_REPO_ROOT / _src))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

PHASE1_CSV = _REPO_ROOT / "phase1_baseline" / "results" / "remediation_metrics.csv"
PHASE2_CSV = _REPO_ROOT / "phase2_proposed" / "results" / "migration_metrics.csv"
RESULTS_DIR = _REPO_ROOT / "phase2_proposed" / "results"

STRATEGY_LABELS = {
    "no_remediation": "No remediation",
    "oracle_full_recovery": "Oracle full recovery",
    "fifo_full_reembed": "Phase 1: FIFO full re-embed",
    "uniform_patch": "Phase 1: Uniform patch",
    "phase2_mckp": "Phase 2: MCKP scheduler (verified)",
}


def load_combined() -> pd.DataFrame:
    phase1_df = pd.read_csv(PHASE1_CSV)
    phase2_df = pd.read_csv(PHASE2_CSV).copy()
    phase2_df["strategy"] = "phase2_mckp"

    keep_cols = ["ratio", "strategy", "budget_level", "ndcg@10", "recall@10", "ndcg@10_traffic_weighted", "recall@10_traffic_weighted"]
    combined = pd.concat([phase1_df[keep_cols], phase2_df[keep_cols]], ignore_index=True)
    return combined


def plot_comparison(df: pd.DataFrame, out_path: Path, metric: str) -> None:
    ratios = sorted(df["ratio"].unique())
    fig, axes = plt.subplots(1, len(ratios), figsize=(4.2 * len(ratios), 4.2), sharey=True)
    if len(ratios) == 1:
        axes = [axes]

    real_strategies = ["fifo_full_reembed", "uniform_patch", "phase2_mckp"]
    colors = {"fifo_full_reembed": "tab:blue", "uniform_patch": "tab:orange", "phase2_mckp": "tab:green"}

    for ax, ratio in zip(axes, ratios):
        sub = df[df["ratio"] == ratio]
        no_rem = sub[sub["strategy"] == "no_remediation"][metric].iloc[0]
        oracle = sub[sub["strategy"] == "oracle_full_recovery"][metric].iloc[0]

        for strategy in real_strategies:
            curve = sub[sub["strategy"] == strategy].sort_values("budget_level")
            if len(curve) == 0:
                continue
            ax.plot(
                curve["budget_level"], curve[metric], marker="o", label=STRATEGY_LABELS[strategy],
                color=colors[strategy],
            )

        ax.axhline(no_rem, color="gray", linestyle=":", label="No remediation" if ratio == ratios[0] else None)
        ax.axhline(oracle, color="black", linestyle="--", label="Oracle full recovery" if ratio == ratios[0] else None)
        ax.set_title(f"ratio={ratio:.2f}")
        ax.set_xlabel("Budget level")
        ax.set_ylim(0, 1.0)

    axes[0].set_ylabel(metric)
    fig.suptitle(f"Phase 1 vs. Phase 2: {metric} vs. budget (head-to-head)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def quantify_improvement(df: pd.DataFrame, metric: str = "ndcg@10") -> pd.DataFrame:
    """For each ratio x budget level, compares Phase 2 against the BETTER
    of Phase 1's two baselines at that exact budget -- both absolute and
    relative (as a fraction of the available oracle-vs-no-remediation
    headroom) improvement."""
    rows = []
    for ratio in sorted(df["ratio"].unique()):
        sub = df[df["ratio"] == ratio]
        no_rem = sub[sub["strategy"] == "no_remediation"][metric].iloc[0]
        oracle = sub[sub["strategy"] == "oracle_full_recovery"][metric].iloc[0]
        headroom = oracle - no_rem

        for budget_level in sorted(sub[sub["strategy"] == "phase2_mckp"]["budget_level"].unique()):
            fifo_row = sub[(sub["strategy"] == "fifo_full_reembed") & (sub["budget_level"] == budget_level)]
            patch_row = sub[(sub["strategy"] == "uniform_patch") & (sub["budget_level"] == budget_level)]
            p2_row = sub[(sub["strategy"] == "phase2_mckp") & (sub["budget_level"] == budget_level)]
            if len(fifo_row) == 0 or len(patch_row) == 0 or len(p2_row) == 0:
                continue

            best_phase1 = max(fifo_row[metric].iloc[0], patch_row[metric].iloc[0])
            phase2_value = p2_row[metric].iloc[0]
            abs_improvement = phase2_value - best_phase1
            pct_of_headroom = abs_improvement / headroom * 100 if headroom > 0 else 0.0

            rows.append(
                {
                    "ratio": ratio,
                    "budget_level": budget_level,
                    "best_phase1": best_phase1,
                    "phase2": phase2_value,
                    "abs_improvement": abs_improvement,
                    "pct_of_headroom": pct_of_headroom,
                }
            )

    return pd.DataFrame(rows)


def budget_to_reach_target(df: pd.DataFrame, strategy: str, ratio: float, target: float, metric: str = "ndcg@10"):
    """Smallest budget_level at which `strategy` reaches `target` for this
    ratio, or None if it never does within the swept budget levels."""
    sub = df[(df["ratio"] == ratio) & (df["strategy"] == strategy)].sort_values("budget_level")
    hit = sub[sub[metric] >= target]
    return float(hit["budget_level"].iloc[0]) if len(hit) > 0 else None


def main() -> None:
    df = load_combined()

    for metric in ("ndcg@10", "ndcg@10_traffic_weighted"):
        suffix = "unweighted" if metric == "ndcg@10" else "traffic_weighted"
        plot_comparison(df, RESULTS_DIR / f"comparison_quality_by_budget_{suffix}.png", metric)

    improvement_df = quantify_improvement(df)
    improvement_df.to_csv(RESULTS_DIR / "comparison_improvement.csv", index=False)

    print("=== Phase 2 vs. best-of-Phase-1, at matched budget (unweighted nDCG@10) ===")
    print(improvement_df.to_string(index=False))
    print()
    print(f"Mean improvement across all ratio x budget combinations: {improvement_df['abs_improvement'].mean():+.4f} nDCG@10")
    print(f"  ({improvement_df['pct_of_headroom'].mean():+.1f}% of available oracle-vs-no-remediation headroom)")
    print(f"Combinations where Phase 2 beat both Phase 1 baselines: "
          f"{(improvement_df['abs_improvement'] > 0).sum()} / {len(improvement_df)}")
    print(f"Combinations where Phase 2 underperformed both (incl. verification aborts): "
          f"{(improvement_df['abs_improvement'] < 0).sum()} / {len(improvement_df)}")

    print("\n=== Budget needed to reach a fixed quality target, per ratio ===")
    for ratio in sorted(df["ratio"].unique()):
        oracle = df[(df["ratio"] == ratio) & (df["strategy"] == "oracle_full_recovery")]["ndcg@10"].iloc[0]
        no_rem = df[(df["ratio"] == ratio) & (df["strategy"] == "no_remediation")]["ndcg@10"].iloc[0]
        target = no_rem + 0.5 * (oracle - no_rem)  # halfway to the oracle ceiling
        b_fifo = budget_to_reach_target(df, "fifo_full_reembed", ratio, target)
        b_p2 = budget_to_reach_target(df, "phase2_mckp", ratio, target)
        print(f"ratio={ratio:.2f} target(50% of headroom)={target:.3f}: "
              f"FIFO needs budget={b_fifo}, Phase 2 needs budget={b_p2}")


if __name__ == "__main__":
    main()
