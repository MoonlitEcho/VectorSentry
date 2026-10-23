"""CLI entry point: builds the full shared contaminated benchmark that both
Phase 1 and Phase 2 evaluate against.

Usage:
    python common/scripts/build_benchmark.py [--config PATH] [--force]

Running this once populates the on-disk cache (see common/src/common/benchmark.py
for the exact layout and caching rules). Every later script -- detectors,
remediation strategies, the dashboard -- calls build_benchmark() /
build_contaminated_ratio() and gets a cache hit instead of recomputing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Defensive path setup: lets this script run via `python common/scripts/...`
# even in an environment where `pip install -e .` wasn't run first (e.g. a
# fresh Colab session cloning just the repo).
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _src in ("common/src", "phase1_baseline/src", "phase2_proposed/src"):
    sys.path.insert(0, str(_REPO_ROOT / _src))

from common.benchmark import build_benchmark, build_contaminated_ratio  # noqa: E402
from common.config import load_config  # noqa: E402
from common.seed import set_seed  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="phase1_baseline/configs/default.yaml")
    parser.add_argument("--force", action="store_true", help="Recompute even if cached artifacts exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    t0 = time.time()
    print(
        f"Loading/building shared benchmark: corpus={cfg.corpus.name} "
        f"max_docs={cfg.corpus.max_docs} seed={cfg.seed}"
    )
    benchmark = build_benchmark(cfg, force=args.force)
    print(
        f"  {len(benchmark.documents)} documents, {len(benchmark.queries)} queries "
        f"with qrels ({len(benchmark.qrels_eval)} have test-split qrels for eval)"
    )
    print(f"  embeddings: generation_a {benchmark.emb_a.shape}, generation_b {benchmark.emb_b.shape}")
    print(
        f"  doc traffic: min={benchmark.doc_traffic.min():.5f} "
        f"max={benchmark.doc_traffic.max():.5f} "
        f"docs_with_zero_traffic={(benchmark.doc_traffic == 0).sum()}/{len(benchmark.doc_traffic)}"
    )
    print(f"  done in {time.time() - t0:.1f}s")

    for ratio in cfg.contamination.ratios:
        t0 = time.time()
        contaminated = build_contaminated_ratio(cfg, benchmark, ratio, force=args.force)
        n_contaminated = int(contaminated.mask.sum())
        print(
            f"  ratio={ratio:.2f}: {n_contaminated}/{len(contaminated.mask)} docs contaminated "
            f"({time.time() - t0:.1f}s)"
        )

    print(f"Benchmark cached at: {Path(cfg.paths.cache_dir).resolve()}")
    print("Benchmark build complete.")


if __name__ == "__main__":
    main()
