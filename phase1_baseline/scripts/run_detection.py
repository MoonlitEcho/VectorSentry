"""CLI: runs Phase 1's baseline detectors (KNN-OOD, LOF, Isolation Forest)
against the shared benchmark, across every contamination ratio in the
config, and saves precision/recall/F1 metrics + a comparison chart to
phase1_baseline/results/.

Usage:
    python phase1_baseline/scripts/run_detection.py [--config PATH]

Assumes the benchmark has already been built (common/scripts/build_benchmark.py)
-- if not, this script builds it itself via the same cache-aware functions.
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

matplotlib.use("Agg")  # headless: this script saves PNGs, it doesn't open windows
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from common.benchmark import build_benchmark, build_contaminated_ratio, mixed_embeddings  # noqa: E402
from common.config import Config, load_config  # noqa: E402
from common.seed import set_seed  # noqa: E402
from phase1.detection import (  # noqa: E402
    detect_isolation_forest,
    detect_knn_ood,
    detect_lof,
    evaluate_detection,
)


def run_all_detectors(cfg: Config) -> pd.DataFrame:
    benchmark = build_benchmark(cfg)
    rows: list[dict] = []

    for ratio in cfg.contamination.ratios:
        contaminated = build_contaminated_ratio(cfg, benchmark, ratio)
        embeddings = mixed_embeddings(benchmark, contaminated)

        detector_calls = {
            "knn_ood": lambda: detect_knn_ood(embeddings, ratio=ratio, k=cfg.detection.knn_k),
            "lof": lambda: detect_lof(embeddings, ratio=ratio, k=cfg.detection.lof_k),
            "isolation_forest": lambda: detect_isolation_forest(embeddings, ratio=ratio, seed=cfg.seed),
        }

        for name, call in detector_calls.items():
            t0 = time.time()
            result = call()
            elapsed = time.time() - t0
            metrics = evaluate_detection(result, contaminated.mask)
            rows.append({"ratio": ratio, "method": name, "seconds": elapsed, **metrics})
            print(
                f"ratio={ratio:.2f} {name:>17s}: precision={metrics['precision']:.3f} "
                f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} ({elapsed:.1f}s)"
            )

    return pd.DataFrame(rows)


def plot_f1_by_ratio(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, group in df.groupby("method"):
        group = group.sort_values("ratio")
        ax.plot(group["ratio"], group["f1"], marker="o", label=name)
    ax.set_xlabel("Contamination ratio")
    ax.set_ylabel("F1 score")
    ax.set_title("Baseline detector F1 vs. contamination ratio (Phase 1)")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="phase1_baseline/configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    df = run_all_detectors(cfg)

    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = results_dir / "detection_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics to {csv_path}")

    plot_path = results_dir / "detection_f1_by_ratio.png"
    plot_f1_by_ratio(df, plot_path)
    print(f"Saved plot to {plot_path}")


if __name__ == "__main__":
    main()
