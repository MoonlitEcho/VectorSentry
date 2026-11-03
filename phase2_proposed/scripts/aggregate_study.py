"""CLI: aggregates the multi-dataset, multi-seed study written by
run_study.py into tables (CSV + a Markdown summary) and figures.

Every comparison uses the definitions in run_comparison.py: VectorSentry
against the better of the two baselines at the same budget, as a share of
the oracle-vs-no-remediation headroom, and the smallest budget level that
recovers half that headroom. Numbers are averaged over ratios and budgets
within a run, then reported as mean +/- std over seeds.

Usage:
    python phase2_proposed/scripts/aggregate_study.py [--study phase2_proposed/results/study]
"""

from __future__ import annotations

import argparse
import importlib.util
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
from scipy.stats import wilcoxon  # noqa: E402

_spec = importlib.util.spec_from_file_location("run_comparison", _REPO_ROOT / "phase2_proposed/scripts/run_comparison.py")
comparison = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(comparison)

DATASET_LABELS = {"scifact": "SciFact", "nfcorpus": "NFCorpus", "fiqa": "FiQA", "quora": "Quora"}
METHOD_LABELS = {"fingerprint": "Fingerprint (ours)", "knn_ood": "KNN-OOD", "lof": "LOF", "isolation_forest": "Isolation Forest"}
KEEP = ["ratio", "strategy", "budget_level", "ndcg@10", "ndcg@10_traffic_weighted"]
# Strategies scored against the best baseline, in the order they are reported.
VARIANTS = {
    "phase2_verified": "VectorSentry (full)",
    "phase2_unverified": "Without the gate",
    "ablation_fingerprint_fifo": "Fingerprint + FIFO",
    "ablation_knn_mckp": "KNN-OOD + MCKP",
}


def load(study: Path, name: str) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in sorted(study.glob(f"*/seed*/{name}"))]
    if not frames:
        raise FileNotFoundError(f"No {name} under {study}; run run_study.py first.")
    return pd.concat(frames, ignore_index=True)


def gains(baseline: pd.DataFrame, phase2: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, seed, ratio, budget, variant): gain over the better baseline."""
    rows = []
    for (dataset, seed), b in baseline.groupby(["dataset", "seed"]):
        p = phase2[(phase2["dataset"] == dataset) & (phase2["seed"] == seed)]
        for variant in VARIANTS:
            # quantify_improvement scores the strategy named PHASE2, so rename the variant to it.
            combined = pd.concat([b[KEEP], p[p["strategy"] == variant][KEEP]], ignore_index=True)
            combined["strategy"] = combined["strategy"].replace({variant: comparison.PHASE2})
            for metric, suffix in (("ndcg@10", ""), ("ndcg@10_traffic_weighted", "_tw")):
                imp = comparison.quantify_improvement(combined, metric)
                for _, r in imp.iterrows():
                    rows.append({"dataset": dataset, "seed": seed, "variant": variant, "metric": "ndcg" + suffix,
                                 "ratio": r["ratio"], "budget_level": r["budget_level"],
                                 "best_baseline": r["best_phase1"], "value": r["phase2"],
                                 "abs_gain": r["abs_improvement"], "pct_headroom": r["pct_of_headroom"]})
    return pd.DataFrame(rows)


def half_headroom_budgets(baseline: pd.DataFrame, phase2: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, seed), b in baseline.groupby(["dataset", "seed"]):
        p = phase2[(phase2["dataset"] == dataset) & (phase2["seed"] == seed)]
        df = pd.concat([b[KEEP], p[KEEP]], ignore_index=True)
        for ratio in sorted(df["ratio"].unique()):
            oracle = df[(df["ratio"] == ratio) & (df["strategy"] == "oracle_full_recovery")]["ndcg@10"].iloc[0]
            no_rem = df[(df["ratio"] == ratio) & (df["strategy"] == "no_remediation")]["ndcg@10"].iloc[0]
            target = no_rem + 0.5 * (oracle - no_rem)
            row = {"dataset": dataset, "seed": seed, "ratio": ratio, "no_remediation": no_rem, "oracle": oracle,
                   "headroom": oracle - no_rem}
            for strategy in ("fifo_full_reembed", "uniform_patch", "phase2_verified"):
                row[f"beta50_{strategy}"] = comparison.budget_to_reach_target(df, strategy, ratio, target)
            rows.append(row)
    return pd.DataFrame(rows)


def pm(series: pd.Series, digits: int) -> str:
    return f"{series.mean():.{digits}f} ± {series.std(ddof=1) if len(series) > 1 else 0:.{digits}f}"


def plot_detection(det: pd.DataFrame, out: Path) -> None:
    per_seed = det.groupby(["dataset", "seed", "method"])["f1"].mean().reset_index()
    datasets = [d for d in DATASET_LABELS if d in per_seed["dataset"].unique()]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    width = 0.2
    for k, (method, label) in enumerate(METHOD_LABELS.items()):
        stats = [per_seed[(per_seed["dataset"] == d) & (per_seed["method"] == method)]["f1"] for d in datasets]
        ax.bar(np.arange(len(datasets)) + (k - 1.5) * width, [s.mean() for s in stats], width,
               yerr=[s.std(ddof=1) if len(s) > 1 else 0 for s in stats], capsize=2, label=label)
    ax.set_xticks(np.arange(len(datasets)), [DATASET_LABELS[d] for d in datasets])
    ax.set_ylabel("Detection F1 (mean over ratios)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_quality(baseline: pd.DataFrame, phase2: pd.DataFrame, out: Path, ratio: float = 0.2) -> None:
    df = pd.concat([baseline, phase2], ignore_index=True)
    df = df[np.isclose(df["ratio"], ratio)]
    datasets = [d for d in DATASET_LABELS if d in df["dataset"].unique()]
    fig, axes = plt.subplots(1, len(datasets), figsize=(3.2 * len(datasets), 3.0), squeeze=False)
    styles = {"fifo_full_reembed": ("FIFO re-embed", "tab:blue"), "uniform_patch": ("Uniform patch", "tab:orange"),
              "phase2_verified": ("VectorSentry", "tab:green")}
    for ax, dataset in zip(axes[0], datasets):
        sub = df[df["dataset"] == dataset]
        for strategy, (label, color) in styles.items():
            g = sub[sub["strategy"] == strategy].groupby("budget_level")["ndcg@10"]
            mean, std = g.mean(), g.std(ddof=1).fillna(0)
            ax.plot(mean.index, mean.values, marker="o", ms=3, color=color, label=label,
                    linewidth=2.2 if strategy == "phase2_verified" else 1.4)
            ax.fill_between(mean.index, mean - std, mean + std, color=color, alpha=0.15)
        for strategy, ls, label in (("no_remediation", ":", "No remediation"), ("oracle_full_recovery", "--", "Oracle")):
            ax.axhline(sub[sub["strategy"] == strategy]["ndcg@10"].mean(), color="gray" if ls == ":" else "black",
                       linestyle=ls, linewidth=1, label=label)
        ax.set_title(DATASET_LABELS[dataset], fontsize=10)
        ax.set_xlabel("Budget level β")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("nDCG@10")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_ablation(g: pd.DataFrame, out: Path) -> None:
    per_seed = g[g["metric"] == "ndcg"].groupby(["dataset", "seed", "variant"])["pct_headroom"].mean().reset_index()
    datasets = [d for d in DATASET_LABELS if d in per_seed["dataset"].unique()]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    width = 0.2
    for k, (variant, label) in enumerate(VARIANTS.items()):
        stats = [per_seed[(per_seed["dataset"] == d) & (per_seed["variant"] == variant)]["pct_headroom"] for d in datasets]
        ax.bar(np.arange(len(datasets)) + (k - 1.5) * width, [s.mean() for s in stats], width,
               yerr=[s.std(ddof=1) if len(s) > 1 else 0 for s in stats], capsize=2, label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(datasets)), [DATASET_LABELS[d] for d in datasets])
    ax.set_ylabel("Gain over best baseline\n(% of headroom)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", default="phase2_proposed/results/study")
    args = parser.parse_args()
    study = Path(args.study)

    det, baseline, phase2 = load(study, "detection.csv"), load(study, "baseline.csv"), load(study, "phase2.csv")
    g = gains(baseline, phase2)
    beta = half_headroom_budgets(baseline, phase2)
    g.to_csv(study / "summary_gains.csv", index=False)
    beta.to_csv(study / "summary_beta50.csv", index=False)

    datasets = [d for d in DATASET_LABELS if d in det["dataset"].unique()]
    lines = ["# Multi-dataset study: summary", "",
             f"Seeds per dataset: {', '.join(f'{DATASET_LABELS[d]} {det[det.dataset == d].seed.nunique()}' for d in datasets)}. "
             "Values are mean ± std over seeds.", ""]

    # Corpus and benchmark characteristics.
    lines += ["## Benchmarks", "", "| Dataset | Mean headroom (nDCG@10, over ratios) | nDCG@10 clean index |", "|---|---|---|"]
    for d in datasets:
        per_seed = beta[beta.dataset == d].groupby("seed")
        lines.append(f"| {DATASET_LABELS[d]} | {pm(per_seed['headroom'].mean(), 3)} | {pm(per_seed['oracle'].mean(), 3)} |")

    # Detection.
    per_seed_det = det.groupby(["dataset", "seed", "method"])["f1"].mean().reset_index()
    lines += ["", "## Detection F1 (mean over contamination ratios, probe documents excluded)", "",
              "| Dataset | " + " | ".join(METHOD_LABELS.values()) + " |", "|---" * (len(METHOD_LABELS) + 1) + "|"]
    for d in datasets:
        cells = [pm(per_seed_det[(per_seed_det.dataset == d) & (per_seed_det.method == m)]["f1"], 3) for m in METHOD_LABELS]
        lines.append(f"| {DATASET_LABELS[d]} | " + " | ".join(cells) + " |")

    # Head-to-head and ablations.
    lines += ["", "## Gain over the better baseline at matched budget (% of headroom, nDCG@10)", "",
              "| Dataset | Wins | " + " | ".join(VARIANTS.values()) + " | Traffic-weighted (full): median, wins | Abs. gain (full) |",
              "|---" * (len(VARIANTS) + 4) + "|"]
    for d in datasets:
        sub = g[(g.dataset == d) & (g.metric == "ndcg")]
        full = sub[sub.variant == "phase2_verified"]
        wins = f"{int((full.abs_gain > 0).sum())}/{len(full)}"
        cells = [pm(sub[sub.variant == v].groupby("seed")["pct_headroom"].mean(), 1) for v in VARIANTS]
        # A few Zipf-heavy queries dominate the traffic-weighted metric, so its mean swings
        # by tens of points between seeds; the median and win count are the stable summary.
        tw = g[(g.dataset == d) & (g.metric == "ndcg_tw") & (g.variant == "phase2_verified")]
        lines.append(f"| {DATASET_LABELS[d]} | {wins} | " + " | ".join(cells) +
                     f" | {tw['pct_headroom'].median():.1f}, {int((tw.abs_gain > 0).sum())}/{len(tw)} | "
                     f"{pm(full.groupby('seed')['abs_gain'].mean(), 4)} |")
    full_all = g[(g.metric == "ndcg") & (g.variant == "phase2_verified")]
    stat = wilcoxon(full_all["value"], full_all["best_baseline"], alternative="greater")
    lines += ["", f"Overall: VectorSentry beats the better baseline in {int((full_all.abs_gain > 0).sum())} of "
              f"{len(full_all)} (dataset, seed, ratio, budget) settings; mean gain {full_all.pct_headroom.mean():.1f}% "
              f"of headroom. One-sided Wilcoxon signed-rank test, VectorSentry > best baseline: p = {stat.pvalue:.2e}."]
    losses = full_all[full_all.abs_gain <= 0]
    if len(losses):
        lines += ["", f"Losses: {len(losses)}, at contamination ratios {', '.join(f'{r:.0%}' for r in sorted(losses.ratio.unique()))}; "
                  f"largest absolute loss {-losses.abs_gain.min():.4f} nDCG@10 "
                  f"(headroom there: {beta.set_index(['dataset', 'seed', 'ratio']).loc[list(zip(losses.dataset, losses.seed, losses.ratio)), 'headroom'].mean():.4f} on average)."]
    by_ratio = full_all.groupby("ratio").agg(wins=("abs_gain", lambda s: int((s > 0).sum())), n=("abs_gain", "size"),
                                              gain=("pct_headroom", "mean"))
    lines += ["", "By contamination ratio (all datasets): " + "; ".join(
        f"{r:.0%}: {int(row.wins)}/{int(row.n)} wins, {row.gain:.1f}%" for r, row in by_ratio.iterrows())]

    # Budget to recover half the headroom.
    lines += ["", "## Budget level needed to recover half the headroom (median over seeds and ratios; '—' = not reached)", "",
              "| Dataset | FIFO re-embed | Uniform patch | VectorSentry | Reached by VectorSentry / FIFO |", "|---|---|---|---|---|"]
    for d in datasets:
        sub = beta[beta.dataset == d]
        cells = []
        for s in ("fifo_full_reembed", "uniform_patch", "phase2_verified"):
            col = sub[f"beta50_{s}"].astype(float)
            cells.append(f"{col.median():.2f}" if col.notna().mean() >= 0.5 else "—")
        reached = f"{sub['beta50_phase2_verified'].notna().sum()}/{len(sub)} vs {sub['beta50_fifo_full_reembed'].notna().sum()}/{len(sub)}"
        lines.append(f"| {DATASET_LABELS[d]} | " + " | ".join(cells) + f" | {reached} |")

    # Gate.
    control = phase2[phase2.strategy == "negative_control_patch_clean"]
    verified = phase2[phase2.strategy == "phase2_verified"]
    # The control patches clean documents, which is meant to hurt; on some corpora at high
    # contamination it does not (patching clean vectors can move them closer to the mixed
    # index). The gate is only judged on settings where shipping the control really hurts.
    control = control.assign(unchecked_loss=control["ndcg@10_no_remediation"] - control["ndcg@10_if_shipped_unchecked"],
                             gated_loss=control["ndcg@10_no_remediation"] - control["ndcg@10"])
    lines += ["", "## Verification gate", "",
              "Negative control = patch the most-shown documents the detector calls clean. "
              "'Harmful' settings are those where shipping it unchecked lowers nDCG@10.", "",
              "| Dataset | Harmful settings | Patches blocked there | Loss there: unchecked → with gate | "
              "Non-harmful settings: patches blocked, loss with gate | Waves passed / tried (real plans) |",
              "|---|---|---|---|---|---|"]
    for d in datasets:
        c, v = control[control.dataset == d], verified[verified.dataset == d]
        h, nh = c[c.unchecked_loss > 0], c[c.unchecked_loss <= 0]
        nh_cell = (f"{nh.n_rolled_back.sum() / nh.n_flagged.sum():.1%}, {nh.gated_loss.max():.3f} max"
                   if len(nh) else "—")
        lines.append(f"| {DATASET_LABELS[d]} | {len(h)}/{len(c)} | "
                     f"{h.n_rolled_back.sum() / h.n_flagged.sum():.1%} ({int(h.n_rolled_back.sum()):,} / {int(h.n_flagged.sum()):,}) | "
                     f"{h.unchecked_loss.mean():.3f} → {h.gated_loss.mean():.3f} | {nh_cell} | "
                     f"{int(v.waves_passed.sum()):,} / {int(v.waves_tried.sum()):,} |")
    h = control[control.unchecked_loss > 0]
    lines += ["", f"All harmful settings: {h.n_rolled_back.sum() / h.n_flagged.sum():.1%} of harmful patches blocked; "
              f"mean loss {h.unchecked_loss.mean():.3f} unchecked vs {h.gated_loss.mean():.3f} with the gate "
              f"(worst {h.gated_loss.max():.3f})."]

    (study / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    plot_detection(det, study / "fig_study_detection.png")
    plot_quality(baseline, phase2, study / "fig_study_quality_r20.png")
    plot_ablation(g, study / "fig_study_ablation.png")
    # The summary contains non-ASCII symbols; don't crash on a cp1252 Windows console.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
