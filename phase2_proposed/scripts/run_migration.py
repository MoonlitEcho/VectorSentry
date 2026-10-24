"""CLI: runs Phase 2's full pipeline -- fingerprint detection ->
expected-benefit MCKP -> staged, verified rollout with budget recycling --
for every contamination ratio x budget level, and records the resulting
FULL-corpus retrieval quality on whatever actually ends up live.

Alongside the headline strategy it runs three ablations on the identical
benchmark and budget, so the gain can be attributed rather than asserted:

  phase2_verified            fingerprint + MCKP + staged gate   (the proposed system)
  phase2_unverified          first-round plan, shipped ungated  (what the gate costs/saves)
  ablation_fingerprint_fifo  fingerprint flags, FIFO re-embed   (detection gain alone)
  ablation_knn_mckp          KNN-OOD flags, MCKP + staged gate  (prioritization gain alone)

plus, once per ratio, a NEGATIVE CONTROL for the gate: a deliberately
harmful plan that patches the most-shown documents the detector says are
clean. A gate that lets that through is not protecting anything.

The saved CSV has the same ratio/strategy/budget_level/metric columns as
phase1_baseline/results/remediation_metrics.csv, so run_comparison.py can
put every strategy on one chart.

Usage:
    python phase2_proposed/scripts/run_migration.py [--config PATH]
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
from common.metrics import evaluate_retrieval  # noqa: E402
from common.seed import set_seed  # noqa: E402
from phase1.detection import detect_knn_ood  # noqa: E402
from phase1.remediation import fifo_remediation_plan  # noqa: E402
from phase2.migrate import build_shadow_embeddings, execute_staged_migration  # noqa: E402
from phase2.pipeline import build_canary, build_phase2_context, plan_remediation, remediate  # noqa: E402
from phase2.scheduler import compute_severity  # noqa: E402


def budget_units_for(cfg: Config, n_documents: int, ratio: float, budget_level: float) -> float:
    """Budget in full-re-embed units, sized from the operator's prevalence
    estimate round(ratio * N) -- the SAME absolute budget for every strategy
    in both phases. (The original sizing, budget_level * n_flagged, would
    hand a detector that flags more documents a bigger budget.)"""
    return budget_level * round(ratio * n_documents) * cfg.remediation.cost_full_reembed


def _outcome_row(outcome) -> dict:
    waves = [w for _, report in outcome.rounds for w in report.waves]
    return {
        "cost_spent": outcome.cost_spent,
        "n_fixed": outcome.n_full_reembed + outcome.n_patch,
        "n_full_reembed": outcome.n_full_reembed,
        "n_patch": outcome.n_patch,
        "n_rolled_back": outcome.n_rolled_back,
        "rounds": len(outcome.rounds),
        "waves_passed": sum(w.passed for w in waves),
        "waves_tried": len(waves),
        "canary_before": outcome.rounds[0][1].canary_before if outcome.rounds else np.nan,
        "canary_after": outcome.rounds[-1][1].canary_after if outcome.rounds else np.nan,
    }


def run_migration_sweep(cfg: Config) -> pd.DataFrame:
    benchmark = build_benchmark(cfg)
    n_docs = len(benchmark.documents)
    doc_ids = [d.doc_id for d in benchmark.documents]
    eval_rows = [i for i, q in enumerate(benchmark.queries) if q.query_id in benchmark.qrels_eval]
    eval_queries = [benchmark.queries[i] for i in eval_rows]
    query_embeddings = benchmark.query_emb_a[eval_rows]

    ctx = build_phase2_context(cfg, benchmark)
    canary, canary_description = build_canary(cfg, benchmark)
    if ctx.effects is not None:
        e = ctx.effects
        print(
            f"Replay calibration ({cfg.remediation.adapter} adapter, {e.n_probe_documents_with_impressions} probe docs "
            f"with impressions): retention raw={e.retention_raw:.3f} patched={e.retention_patched:.3f} "
            f"clean-patched={e.retention_clean_patched:.3f} -> recovery_patch={e.recovery_patch:.3f}, "
            f"harm_patch={e.harm_patch:.3f}"
        )
    print(f"Gate: {canary_description}; margin {cfg.migration.regression_tolerance}, "
          f"confidence {cfg.migration.gate_confidence}")

    def quality(embeddings: np.ndarray) -> dict[str, float]:
        return evaluate_retrieval(
            embeddings, doc_ids, eval_queries, query_embeddings, benchmark.qrels_eval, benchmark.query_traffic
        )

    rows: list[dict] = []
    for ratio in cfg.contamination.ratios:
        contaminated = build_contaminated_ratio(cfg, benchmark, ratio)
        mixed = mixed_embeddings(benchmark, contaminated)

        p_contaminated, fp_flagged = ctx.detector.flag(mixed, cfg.fingerprint.threshold)
        fp_indices = np.flatnonzero(fp_flagged)
        knn = detect_knn_ood(mixed, ratio=ratio, k=cfg.detection.knn_k)
        knn_indices = np.flatnonzero(knn.predicted)
        knn_severity = compute_severity(knn.scores[knn_indices])  # the original 2A "severity" for the ablation

        for budget_level in cfg.budget.levels:
            t0 = time.time()
            budget = budget_units_for(cfg, n_docs, ratio, budget_level)
            base = {"ratio": ratio, "budget_level": budget_level, "budget_units": budget}

            outcome = remediate(
                cfg, ctx, mixed, benchmark.emb_a, fp_indices, p_contaminated[fp_indices], budget, canary, seed=cfg.seed
            )
            rows.append(
                {**base, "strategy": "phase2_verified", "n_flagged": len(fp_indices), **_outcome_row(outcome),
                 **quality(outcome.live_embeddings)}
            )
            p2 = rows[-1]

            first_plan = plan_remediation(cfg, ctx, fp_indices, p_contaminated[fp_indices], budget)
            actions = first_plan.plan.actions
            unverified = build_shadow_embeddings(mixed, benchmark.emb_a, ctx.adapter, fp_indices, actions)
            rows.append(
                {
                    **base,
                    "strategy": "phase2_unverified",
                    "n_flagged": len(fp_indices),
                    "cost_spent": first_plan.plan.total_cost,
                    "n_fixed": int(np.sum(actions != "defer")),
                    "n_full_reembed": int(np.sum(actions == "full_reembed")),
                    "n_patch": int(np.sum(actions == "patch")),
                    **quality(unverified),
                }
            )

            fixed, spent = fifo_remediation_plan(fp_flagged, budget, cfg.remediation.cost_full_reembed)
            fifo_fp = mixed.copy()
            fifo_fp[fixed] = benchmark.emb_a[fixed]
            rows.append(
                {
                    **base,
                    "strategy": "ablation_fingerprint_fifo",
                    "n_flagged": len(fp_indices),
                    "cost_spent": spent,
                    "n_fixed": int(fixed.sum()),
                    "n_full_reembed": int(fixed.sum()),
                    "n_patch": 0,
                    **quality(fifo_fp),
                }
            )

            knn_outcome = remediate(
                cfg, ctx, mixed, benchmark.emb_a, knn_indices, knn_severity, budget, canary, seed=cfg.seed
            )
            rows.append(
                {**base, "strategy": "ablation_knn_mckp", "n_flagged": len(knn_indices), **_outcome_row(knn_outcome),
                 **quality(knn_outcome.live_embeddings)}
            )

            print(
                f"ratio={ratio:.2f} budget={budget_level:.2f}: phase2 ndcg@10={p2['ndcg@10']:.3f} "
                f"(full={p2['n_full_reembed']}, patch={p2['n_patch']}, rolled back={p2['n_rolled_back']}, "
                f"rounds={p2['rounds']}, waves {p2['waves_passed']}/{p2['waves_tried']}) | "
                f"unverified={rows[-3]['ndcg@10']:.3f} fp+fifo={rows[-2]['ndcg@10']:.3f} "
                f"knn+mckp={rows[-1]['ndcg@10']:.3f} ({time.time() - t0:.1f}s)"
            )

        # Negative control for the gate (see module docstring). Every patch
        # here damages a correct vector, so a working gate must roll back
        # (nearly) all of them.
        harmful = np.array([i for i in np.argsort(-ctx.traffic) if not fp_flagged[i]][: max(1, len(fp_indices))])
        harmful_actions = np.full(len(harmful), "patch", dtype=object)
        control = execute_staged_migration(
            mixed, benchmark.emb_a, ctx.adapter, harmful, harmful_actions, np.arange(len(harmful)), canary,
            n_waves=cfg.migration.n_waves,
            regression_tolerance=cfg.migration.regression_tolerance,
            gate_confidence=cfg.migration.gate_confidence,
            bootstrap_samples=cfg.migration.bootstrap_samples,
            bisect_depth=cfg.migration.bisect_depth,
            seed=cfg.seed,
        )
        unchecked = build_shadow_embeddings(mixed, benchmark.emb_a, ctx.adapter, harmful, harmful_actions)
        rows.append(
            {
                "ratio": ratio,
                "budget_level": None,
                "strategy": "negative_control_patch_clean",
                "n_flagged": len(harmful),
                "n_fixed": control.n_patch,
                "n_patch": control.n_patch,
                "n_rolled_back": control.n_rolled_back,
                "waves_passed": sum(w.passed for w in control.waves),
                "waves_tried": len(control.waves),
                "ndcg@10_no_remediation": quality(mixed)["ndcg@10"],
                "ndcg@10_if_shipped_unchecked": quality(unchecked)["ndcg@10"],
                **quality(control.live_embeddings),
            }
        )
        r = rows[-1]
        print(
            f"ratio={ratio:.2f} NEGATIVE CONTROL (patch {len(harmful)} clean high-traffic docs): gate rolled back "
            f"{control.n_rolled_back}/{len(harmful)}; ndcg@10 {r['ndcg@10']:.3f} with gate, "
            f"{r['ndcg@10_if_shipped_unchecked']:.3f} if shipped unchecked, {r['ndcg@10_no_remediation']:.3f} untouched"
        )

    return pd.DataFrame(rows)


def plot_migration_quality(df: pd.DataFrame, out_path: Path) -> None:
    ratios = sorted(df["ratio"].unique())
    fig, axes = plt.subplots(1, len(ratios), figsize=(4 * len(ratios), 4), sharey=True)
    if len(ratios) == 1:
        axes = [axes]

    styles = {
        "phase2_verified": ("Phase 2 (verified, staged)", "tab:green", "-"),
        "phase2_unverified": ("Phase 2 plan, no gate", "tab:green", ":"),
        "ablation_fingerprint_fifo": ("Ablation: fingerprint + FIFO", "tab:purple", "--"),
        "ablation_knn_mckp": ("Ablation: KNN-OOD + MCKP", "tab:brown", "--"),
    }
    for ax, ratio in zip(axes, ratios):
        sub = df[df["ratio"] == ratio]
        for strategy, (label, color, ls) in styles.items():
            curve = sub[sub["strategy"] == strategy].sort_values("budget_level")
            ax.plot(curve["budget_level"], curve["ndcg@10"], marker="o", color=color, linestyle=ls, label=label)
        ax.set_title(f"ratio={ratio:.2f}")
        ax.set_xlabel("Budget level")
        ax.set_ylim(0.3, 0.7)

    axes[0].set_ylabel("nDCG@10 (unweighted, live embeddings)")
    fig.suptitle("Phase 2: verified quality vs. budget, with ablations")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))
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

    verified = df[df["strategy"] == "phase2_verified"]
    control = df[df["strategy"] == "negative_control_patch_clean"]
    print(
        f"\nStaged rollout: {int(verified['waves_passed'].sum())} of {int(verified['waves_tried'].sum())} wave attempts "
        f"passed; {int(verified['n_rolled_back'].sum())} candidate fixes rolled back and not replaced, across "
        f"{len(verified)} ratio x budget runs."
    )
    print(
        f"Negative control: gate rolled back {int(control['n_rolled_back'].sum())} of {int(control['n_flagged'].sum())} "
        f"harmful patches."
    )


if __name__ == "__main__":
    main()
