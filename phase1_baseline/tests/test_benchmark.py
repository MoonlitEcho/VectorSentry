"""Integration test for the real, end-to-end benchmark pipeline: downloads
(or reuses the local HuggingFace cache for) real BEIR SciFact data, embeds
it with both real generation-A/B models, and builds a mixed-generation
FAISS index. Slower than test_data.py's synthetic unit tests by design --
it is the only place in the test suite that verifies the actual artifacts
Phase 1 and Phase 2 depend on are shaped correctly.

Marked `integration` and excluded from the default `pytest` run (see the
`addopts` in pyproject.toml) because embedding even a few hundred documents
on CPU takes minutes, not seconds. Run explicitly with:

    pytest -m integration phase1_baseline/tests/test_benchmark.py

max_docs=800 is used (not the config default of 5000) purely to keep this
test's embedding time small; 800 comfortably exceeds the ~667 documents
SciFact's qrels require `subsample_corpus` to keep (see common/data.py).

All tests share one module-scoped `benchmark` fixture (built once) instead
of each test rebuilding it -- rebuilding per-test was measured taking
~15 minutes for 4 tests; sharing it drops that to ~1 embedding pass.
"""

import numpy as np
import pytest

from common.benchmark import build_benchmark, build_contaminated_ratio
from common.config import Config, CorpusConfig, PathsConfig, TrafficConfig

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    cache_dir = tmp_path_factory.mktemp("vectorsentry_bench_cache")
    return Config(
        seed=7,
        corpus=CorpusConfig(name="scifact", max_docs=800),
        traffic=TrafficConfig(zipf_s=1.2),
        paths=PathsConfig(cache_dir=str(cache_dir)),
    )


@pytest.fixture(scope="module")
def benchmark(cfg):
    return build_benchmark(cfg)


def test_build_benchmark_shapes_and_dimension_match(cfg, benchmark):
    n = len(benchmark.documents)
    assert n == cfg.corpus.max_docs
    assert benchmark.emb_a.shape[0] == n
    # Same output dimensionality is the premise that makes contamination
    # silent rather than a loud dimension-mismatch error -- see benchmark.py.
    assert benchmark.emb_a.shape[1] == benchmark.emb_b.shape[1]
    assert benchmark.doc_traffic.shape == (n,)
    assert len(benchmark.qrels_eval) > 0


def test_build_benchmark_uses_cache_on_second_call(cfg, benchmark):
    # `benchmark` fixture already built (and cached to disk) once; a second
    # call with the same config must hit that cache, not recompute.
    b2 = build_benchmark(cfg)

    np.testing.assert_array_equal(benchmark.emb_a, b2.emb_a)
    np.testing.assert_array_equal(benchmark.doc_traffic, b2.doc_traffic)
    assert [d.doc_id for d in benchmark.documents] == [d.doc_id for d in b2.documents]


def test_build_contaminated_ratio_mask_matches_ratio_and_index_size(cfg, benchmark):
    n = len(benchmark.documents)
    for ratio in (0.1, 0.5):
        contaminated = build_contaminated_ratio(cfg, benchmark, ratio)
        assert contaminated.mask.shape == (n,)
        assert contaminated.index.ntotal == n
        actual_ratio = contaminated.mask.mean()
        assert abs(actual_ratio - ratio) < 0.1  # loose: seeded but still random assignment


def test_build_contaminated_ratio_is_reproducible(cfg, benchmark):
    c1 = build_contaminated_ratio(cfg, benchmark, 0.2, force=True)
    c2 = build_contaminated_ratio(cfg, benchmark, 0.2, force=True)
    np.testing.assert_array_equal(c1.mask, c2.mask)
