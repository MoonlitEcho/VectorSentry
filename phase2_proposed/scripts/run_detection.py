"""CLI: head-to-head detection comparison -- Phase 2's generation-
fingerprint detector vs. Phase 1's three baselines (KNN-OOD, LOF,
Isolation Forest) -- on BOTH model pairings: the primary MiniLM-L6 ->
MiniLM-L12 benchmark and the bge-small anisotropy stress test, where every
Phase 1 detector fails outright (DEVLOG.md 1C).

Fairness notes:
- Phase 1 detectors are given the TRUE contamination ratio as their
  assumed prevalence (their most favourable setting); the fingerprint
  detector is given no ratio at all (fixed threshold P(gen B) > 0.5).
- Probe-batch documents are excluded from every detector's metrics: their
  stored vector is literally one of the fingerprint's reference vectors.

Usage:
    python phase2_proposed/scripts/run_detection.py
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
from common.config import load_config  # noqa: E402
from common.patch import sample_calibration_indices  # noqa: E402
from common.seed import set_seed  # noqa: E402
from phase1.detection import (  # noqa: E402
    DetectionResult,
    detect_isolation_forest,
    detect_knn_ood,
    detect_lof,
    evaluate_detection,
)
from phase2.fingerprint import FingerprintDetector  # noqa: E402

PAIRINGS = {
    "MiniLM-L6 -> MiniLM-L12 (primary)": "phase2_proposed/configs/default.yaml",
    "MiniLM-L6 -> bge-small (stress test)": "phase1_baseline/configs/aniso_stress_bge.yaml",
}
METHOD_LABELS = {
    "knn_ood": "KNN-OOD (Phase 1)",
    "lof": "LOF (Phase 1)",
    "isolation_forest": "Isolation Forest (Phase 1)",
    "fingerprint": "Generation fingerprint (Phase 2)",
}


def run(pairing: str, config_path: str) -> list[dict]:
    cfg = load_config(config_path)
    set_seed(cfg.seed)
    benchmark = build_benchmark(cfg)
    n = len(benchmark.documents)
    probe = sample_calibration_indices(n, cfg.remediation.calibration_sample_size, cfg.seed)
    keep = np.ones(n, dtype=bool)
    keep[probe] = False
    detector = FingerprintDetector.fit(benchmark.emb_a, benchmark.emb_b, probe, k=cfg.fingerprint.k)

    rows = []
    for ratio in cfg.contamination.ratios:
        contaminated = build_contaminated_ratio(cfg, benchmark, ratio)
        mixed = mixed_embeddings(benchmark, contaminated)
        prob, flagged = detector.flag(mixed, cfg.fingerprint.threshold)
        results = [
            detect_knn_ood(mixed, ratio=ratio, k=cfg.detection.knn_k),
            detect_lof(mixed, ratio=ratio, k=cfg.detection.lof_k),
            detect_isolation_forest(mixed, ratio=ratio, seed=cfg.seed),
            DetectionResult("fingerprint", prob, flagged),
        ]
        for result in results:
            scored = DetectionResult(result.method, result.scores[keep], result.predicted[keep])
            metrics = evaluate_detection(scored, contaminated.mask[keep])
            rows.append({"pairing": pairing, "ratio": ratio, "method": result.method, **metrics})
            print(
                f"{pairing:38s} ratio={ratio:.2f} {result.method:17s} "
                f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f}"
            )
    return rows


def plot(df: pd.DataFrame, out_path: Path) -> None:
    pairings = list(df["pairing"].unique())
    fig, axes = plt.subplots(1, len(pairings), figsize=(6.5 * len(pairings), 4.5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, pairing in zip(axes, pairings):
        sub = df[df["pairing"] == pairing]
        for method, label in METHOD_LABELS.items():
            curve = sub[sub["method"] == method].sort_values("ratio")
            ax.plot(
                curve["ratio"], curve["f1"], marker="o", label=label,
                linewidth=2.5 if method == "fingerprint" else 1.5,
            )
        ax.set_title(pairing)
        ax.set_xlabel("Contamination ratio")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Detection F1 (probe docs excluded)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Detection: Phase 1 baselines vs. Phase 2 generation fingerprint")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="phase2_proposed/results")
    args = parser.parse_args()

    rows = []
    for pairing, config_path in PAIRINGS.items():
        rows += run(pairing, config_path)
    df = pd.DataFrame(rows)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "detection_comparison.csv", index=False)
    plot(df, out_dir / "detection_comparison_f1.png")
    print(f"\nSaved {out_dir / 'detection_comparison.csv'} and detection_comparison_f1.png")
    print("\nMean F1 by pairing and method:")
    print(df.groupby(["pairing", "method"])["f1"].mean().unstack().round(3).to_string())


if __name__ == "__main__":
    main()
