"""Fast, network-free unit tests for common.metrics."""

import numpy as np
import pytest

from common.metrics import evaluate_retrieval, ndcg_at_k, recall_at_k


def test_ndcg_at_k_perfect_ranking_scores_one():
    assert ndcg_at_k(["a", "b", "c"], relevant_ids={"a", "b"}, k=3) == pytest.approx(1.0)


def test_ndcg_at_k_worst_ranking_scores_less_than_perfect():
    perfect = ndcg_at_k(["a", "b", "c"], relevant_ids={"a"}, k=3)
    worst = ndcg_at_k(["c", "b", "a"], relevant_ids={"a"}, k=3)
    assert worst < perfect
    assert worst > 0.0  # "a" is still retrieved within k, just ranked last


def test_ndcg_at_k_no_relevant_docs_retrieved_scores_zero():
    assert ndcg_at_k(["x", "y", "z"], relevant_ids={"a"}, k=3) == 0.0


def test_ndcg_at_k_no_relevant_docs_at_all_scores_zero():
    assert ndcg_at_k(["a", "b", "c"], relevant_ids=set(), k=3) == 0.0


def test_recall_at_k_all_relevant_retrieved():
    assert recall_at_k(["a", "b", "c"], relevant_ids={"a", "b"}, k=3) == 1.0


def test_recall_at_k_partial():
    assert recall_at_k(["a", "x", "y"], relevant_ids={"a", "b"}, k=3) == 0.5


def test_recall_at_k_respects_k_cutoff():
    # "a" is relevant but ranked outside the top-1 cutoff.
    assert recall_at_k(["x", "a"], relevant_ids={"a"}, k=1) == 0.0


def test_recall_at_k_no_relevant_docs_scores_zero():
    assert recall_at_k(["a", "b"], relevant_ids=set(), k=2) == 0.0


class _Query:
    def __init__(self, query_id):
        self.query_id = query_id


def test_evaluate_retrieval_result_keys_match_requested_k():
    # Three mutually well-separated 2D unit vectors -- strict, unambiguous
    # ranking for any query direction that isn't exactly equidistant.
    doc_ids = ["d0", "d1", "d2"]
    embeddings = np.array(
        [[1.0, 0.0], [-0.5, 0.866], [-0.5, -0.866]], dtype=np.float32
    )
    query_embeddings = np.array([[1.0, 0.01]], dtype=np.float32)  # near d0, slightly favors d1 over d2
    query_embeddings /= np.linalg.norm(query_embeddings, axis=1, keepdims=True)
    queries = [_Query("q0")]
    qrels_eval = {"q0": {"d0"}}
    query_traffic = {"q0": 1.0}

    result = evaluate_retrieval(embeddings, doc_ids, queries, query_embeddings, qrels_eval, query_traffic, k=1)

    assert set(result.keys()) == {"ndcg@1", "recall@1", "ndcg@1_traffic_weighted", "recall@1_traffic_weighted"}
    assert result["ndcg@1"] == pytest.approx(1.0)
    assert result["recall@1"] == pytest.approx(1.0)


def test_evaluate_retrieval_traffic_weighting_shifts_the_average():
    doc_ids = ["d0", "d1", "d2"]
    embeddings = np.array(
        [[1.0, 0.0], [-0.5, 0.866], [-0.5, -0.866]], dtype=np.float32
    )
    # q_good points at d0 and is relevant to d0 -> perfect retrieval.
    # q_bad points at d2 (drifted away) but is relevant to d0 -> total miss at k=1.
    query_embeddings = np.array([[1.0, 0.0], [-0.5, -0.866]], dtype=np.float32)
    queries = [_Query("q_good"), _Query("q_bad")]
    qrels_eval = {"q_good": {"d0"}, "q_bad": {"d0"}}

    unweighted_leaning = {"q_good": 0.5, "q_bad": 0.5}
    result_balanced = evaluate_retrieval(
        embeddings, doc_ids, queries, query_embeddings, qrels_eval, unweighted_leaning, k=1
    )
    assert result_balanced["ndcg@1"] == pytest.approx(0.5)  # (1.0 + 0.0) / 2

    # Now put almost all traffic on the query that fails -- the traffic-
    # weighted score should drop well below the unweighted 0.5, showing the
    # weighting is actually doing something (not just echoing the mean).
    skewed_to_bad_query = {"q_good": 0.05, "q_bad": 0.95}
    result_skewed = evaluate_retrieval(
        embeddings, doc_ids, queries, query_embeddings, qrels_eval, skewed_to_bad_query, k=1
    )
    assert result_skewed["ndcg@1"] == pytest.approx(0.5)  # unweighted metric unaffected by traffic
    assert result_skewed["ndcg@1_traffic_weighted"] == pytest.approx(0.05)
    assert result_skewed["ndcg@1_traffic_weighted"] < result_balanced["ndcg@1_traffic_weighted"]


def test_evaluate_retrieval_falls_back_to_unweighted_when_no_traffic():
    doc_ids = ["d0", "d1"]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    query_embeddings = np.array([[1.0, 0.0]], dtype=np.float32)
    queries = [_Query("q0")]
    qrels_eval = {"q0": {"d0"}}

    result = evaluate_retrieval(embeddings, doc_ids, queries, query_embeddings, qrels_eval, query_traffic={}, k=1)
    assert result["ndcg@1_traffic_weighted"] == result["ndcg@1"]
    assert result["recall@1_traffic_weighted"] == result["recall@1"]
