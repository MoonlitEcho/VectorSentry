"""Fast, network-free unit tests for common.data's corpus-subsampling and
traffic-simulation logic, using small synthetic fixtures instead of real
BEIR data (that's covered separately, more slowly, in test_benchmark.py).
"""

import numpy as np
import pytest

from common.data import Document, Query, RawBeirData, compute_query_traffic, subsample_corpus


def _toy_data() -> RawBeirData:
    documents = [Document(doc_id=f"d{i}", title=f"title{i}", text=f"text{i}") for i in range(10)]
    queries = [Query(query_id=qid, text=qid) for qid in ("q0", "q1", "q2")]
    # 6 distinct documents (d0..d5) are referenced by qrels; d6..d9 are not.
    qrels_all = {"q0": {"d0", "d1"}, "q1": {"d2"}, "q2": {"d3", "d4", "d5"}}
    qrels_eval = {"q0": {"d0"}, "q2": {"d3"}}
    return RawBeirData(documents, queries, qrels_all, qrels_eval)


def test_subsample_keeps_all_qrels_docs_and_respects_max_docs():
    data = _toy_data()
    subsampled = subsample_corpus(data, max_docs=6, seed=0)

    assert len(subsampled.documents) == 6
    kept_ids = {d.doc_id for d in subsampled.documents}
    assert {"d0", "d1", "d2", "d3", "d4", "d5"} <= kept_ids


def test_subsample_preserves_original_document_order():
    data = _toy_data()
    subsampled = subsample_corpus(data, max_docs=8, seed=1)
    kept_ids = [d.doc_id for d in subsampled.documents]
    original_ids = [d.doc_id for d in data.documents]
    # kept order must be a subsequence of the original order (never reshuffled)
    assert kept_ids == [doc_id for doc_id in original_ids if doc_id in set(kept_ids)]


def test_subsample_raises_if_max_docs_too_small_for_qrels():
    data = _toy_data()  # 6 distinct docs referenced by qrels_all
    with pytest.raises(ValueError):
        subsample_corpus(data, max_docs=3, seed=0)


def test_subsample_is_noop_when_corpus_already_small_enough():
    data = _toy_data()
    subsampled = subsample_corpus(data, max_docs=100, seed=0)
    assert subsampled is data


def test_compute_query_traffic_zero_for_docs_outside_qrels():
    data = _toy_data()
    query_traffic, doc_traffic = compute_query_traffic(data, zipf_s=1.0, seed=0)

    assert doc_traffic.shape == (10,)
    for i in range(6, 10):  # d6..d9 are never a qrels target
        assert doc_traffic[i] == 0.0
    assert doc_traffic[:6].sum() > 0


def test_compute_query_traffic_query_weights_sum_to_one():
    data = _toy_data()
    query_traffic, _ = compute_query_traffic(data, zipf_s=1.0, seed=0)

    assert set(query_traffic.keys()) == {"q0", "q1", "q2"}
    assert sum(query_traffic.values()) == pytest.approx(1.0)


def test_compute_query_traffic_is_deterministic_given_seed():
    data = _toy_data()
    qt1, dt1 = compute_query_traffic(data, zipf_s=1.2, seed=42)
    qt2, dt2 = compute_query_traffic(data, zipf_s=1.2, seed=42)
    np.testing.assert_array_equal(dt1, dt2)
    assert qt1 == qt2


def test_compute_query_traffic_is_zipfian_skewed():
    # 50 queries, each mapped to exactly one distinct document.
    documents = [Document(doc_id=f"d{i}", title="", text="") for i in range(50)]
    queries = [Query(query_id=f"q{i}", text="") for i in range(50)]
    qrels_all = {f"q{i}": {f"d{i}"} for i in range(50)}
    data = RawBeirData(documents, queries, qrels_all, qrels_eval={})

    _, doc_traffic = compute_query_traffic(data, zipf_s=1.2, seed=0)
    sorted_traffic = np.sort(doc_traffic)[::-1]
    top_n = 5  # top 10% of documents
    assert sorted_traffic[:top_n].sum() > 0.5 * sorted_traffic.sum()
