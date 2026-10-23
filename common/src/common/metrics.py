"""Standard IR retrieval-quality metrics (nDCG@k, Recall@k), implemented
directly rather than depending on `pytrec_eval` -- see DEVLOG.md's
sub-phase 1A entry for why: pytrec_eval is a compiled C binding with
uncertain Windows wheel support, and binary-relevance nDCG/Recall are
simple enough to implement directly, keeping the exact formula inspectable
for examiners.

BEIR's SciFact qrels are binary relevance (every judged pair has score 1),
so the standard binary-relevance formulas below are equivalent to the more
general graded-relevance nDCG formula (2^rel - 1) and are used directly.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from common.index import build_index, search


class _QueryLike(Protocol):
    query_id: str
    text: str


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Binary-relevance nDCG@k: relevant docs contribute 1/log2(rank+2) to
    the discounted gain, normalized by the best-possible (ideal) ordering.
    """
    dcg = sum(
        1.0 / np.log2(rank + 2)
        for rank, doc_id in enumerate(retrieved_ids[:k])
        if doc_id in relevant_ids
    )
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(rank + 2) for rank in range(ideal_hits))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the query's relevant documents that appear in the top-k
    retrieved results."""
    if not relevant_ids:
        return 0.0
    hits = len(set(retrieved_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def evaluate_retrieval(
    embeddings: np.ndarray,
    doc_ids: list[str],
    eval_queries: list[_QueryLike],
    query_embeddings: np.ndarray,
    qrels_eval: dict[str, set[str]],
    query_traffic: dict[str, float],
    k: int = 10,
) -> dict[str, float]:
    """Builds a FAISS index over `embeddings` and searches it with every
    query in `eval_queries` (aligned row-for-row with `query_embeddings`),
    reporting both the standard unweighted macro-average nDCG@k/Recall@k
    AND a traffic-weighted variant.

    The traffic-weighted variant matters for this project specifically: it
    is what makes "a high-traffic document stayed broken" show up as a
    single measurable number, rather than only being visible anecdotally in
    one before/after query example. A remediation strategy that fixes many
    low-traffic documents and few high-traffic ones can have a HIGHER
    unweighted score than one that does the reverse, while recovering LESS
    of the search experience users actually have -- the traffic-weighted
    number is the one that tracks real user-facing impact.
    """
    index = build_index(embeddings)
    doc_ids_arr = np.asarray(doc_ids)
    _, indices = search(index, query_embeddings, k)

    n = len(eval_queries)
    ndcgs = np.empty(n)
    recalls = np.empty(n)
    weights = np.empty(n)

    for row, query in enumerate(eval_queries):
        retrieved_ids = doc_ids_arr[indices[row]].tolist()
        relevant_ids = qrels_eval[query.query_id]
        ndcgs[row] = ndcg_at_k(retrieved_ids, relevant_ids, k)
        recalls[row] = recall_at_k(retrieved_ids, relevant_ids, k)
        weights[row] = query_traffic.get(query.query_id, 0.0)

    result = {
        f"ndcg@{k}": float(ndcgs.mean()),
        f"recall@{k}": float(recalls.mean()),
    }

    weight_sum = weights.sum()
    if weight_sum > 0:
        w = weights / weight_sum
        result[f"ndcg@{k}_traffic_weighted"] = float((ndcgs * w).sum())
        result[f"recall@{k}_traffic_weighted"] = float((recalls * w).sum())
    else:
        # No eval query carries any traffic weight (shouldn't happen given
        # how query_traffic is constructed, but fall back to the unweighted
        # score rather than dividing by zero).
        result[f"ndcg@{k}_traffic_weighted"] = result[f"ndcg@{k}"]
        result[f"recall@{k}_traffic_weighted"] = result[f"recall@{k}"]

    return result
