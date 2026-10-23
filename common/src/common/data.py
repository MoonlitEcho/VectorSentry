"""BEIR corpus/query/qrels loading and simulated search-traffic weighting.

Sub-phase 1B design notes (see DEVLOG.md for the full write-up):

- We load BEIR datasets from their HuggingFace mirrors (`BeIR/<name>` for
  corpus+queries, `BeIR/<name>-qrels` for relevance judgments) instead of
  the official `beir` pip package, to avoid its pinned old
  transformers/torch versions conflicting with sentence-transformers.
- "Traffic" is simulated at the QUERY level (Zipfian-distributed popularity,
  see `compute_query_traffic`) and then aggregated to a per-DOCUMENT score
  by summing the traffic of every query for which that document is a
  relevance judgment (qrels). Documents that are never a qrels target get
  zero traffic -- this is realistic: most stored documents are rarely the
  direct target of a search, while a small number of "head" documents
  dominate query volume. This is exactly the skew the project asks for, and
  it is what makes WHICH documents are contaminated (not just how many)
  matter for the Phase 1 vs Phase 2 comparison.
- We combine BOTH the train and test qrels splits when computing traffic
  (more coverage = a richer traffic signal), but keep the test-split qrels
  separate (`qrels_eval`) for retrieval-quality evaluation in 1C/1D, since
  test-only qrels is the standard BEIR practice for reporting nDCG/Recall.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from datasets import load_dataset


@dataclasses.dataclass
class Document:
    doc_id: str
    title: str
    text: str


@dataclasses.dataclass
class Query:
    query_id: str
    text: str


@dataclasses.dataclass
class RawBeirData:
    # Order of `documents` doubles as the synthetic "ingestion order" used by
    # Phase 1's FIFO remediation baseline (1D) -- BEIR's corpus ordering has
    # no relationship to query popularity, so it is a fair stand-in for an
    # arbitrary real ingestion order.
    documents: list[Document]
    queries: list[Query]
    qrels_all: dict[str, set[str]]  # query_id -> relevant doc_ids, train+test combined (traffic simulation)
    qrels_eval: dict[str, set[str]]  # query_id -> relevant doc_ids, test split only (retrieval-quality eval)


def load_beir_dataset(name: str) -> RawBeirData:
    """Loads one BEIR dataset (e.g. "scifact") from its HuggingFace mirror."""
    corpus_ds = load_dataset(f"BeIR/{name}", "corpus", split="corpus")
    queries_ds = load_dataset(f"BeIR/{name}", "queries", split="queries")
    qrels_ds = load_dataset(f"BeIR/{name}-qrels")  # DatasetDict, typically {"train", "test"} (sometimes "dev" too)

    documents = [
        Document(doc_id=str(row["_id"]), title=row["title"] or "", text=row["text"] or "")
        for row in corpus_ds
    ]
    query_text = {
        str(row["_id"]): f"{row['title']} {row['text']}".strip() if row["title"] else row["text"]
        for row in queries_ds
    }

    def _qrels_from_split(split_ds) -> dict[str, set[str]]:
        out: dict[str, set[str]] = {}
        for row in split_ds:
            if row["score"] <= 0:
                continue
            # qrels IDs come through as ints even though corpus/query IDs are
            # strings in the mirrored datasets -- normalize both to str here
            # so every later lookup by ID is consistent.
            qid, did = str(row["query-id"]), str(row["corpus-id"])
            out.setdefault(qid, set()).add(did)
        return out

    qrels_eval = _qrels_from_split(qrels_ds["test"])
    qrels_all: dict[str, set[str]] = {}
    for split_name in qrels_ds.keys():
        for qid, docids in _qrels_from_split(qrels_ds[split_name]).items():
            qrels_all.setdefault(qid, set()).update(docids)

    # Keep only queries that carry at least one relevance judgment -- a query
    # with no qrels is useless for both traffic simulation and evaluation.
    relevant_query_ids = set(qrels_all.keys())
    queries = [
        Query(query_id=qid, text=query_text[qid])
        for qid in relevant_query_ids
        if qid in query_text
    ]

    return RawBeirData(documents=documents, queries=queries, qrels_all=qrels_all, qrels_eval=qrels_eval)


def subsample_corpus(data: RawBeirData, max_docs: int, seed: int) -> RawBeirData:
    """Caps the corpus to `max_docs` documents, keeping the benchmark laptop
    / free-Colab feasible, while preserving every document referenced by
    qrels (so the traffic and retrieval-quality signal survive the cut) and
    padding the remainder with a seeded random sample of the rest -- the
    same "gold docs + random distractors" construction BEIR itself uses.

    Document order is preserved (not shuffled), maintaining a stable
    synthetic ingestion order for the FIFO baseline in 1D.
    """
    if len(data.documents) <= max_docs:
        return data

    all_doc_ids = {d.doc_id for d in data.documents}
    must_keep_ids = {did for docids in data.qrels_all.values() for did in docids} & all_doc_ids

    if len(must_keep_ids) > max_docs:
        raise ValueError(
            f"max_docs={max_docs} is smaller than the {len(must_keep_ids)} documents "
            "referenced by qrels -- raise max_docs, or the traffic/eval signal breaks."
        )

    remaining_ids = np.array([d.doc_id for d in data.documents if d.doc_id not in must_keep_ids])
    rng = np.random.default_rng(seed)
    n_fill = max_docs - len(must_keep_ids)
    filler_ids = set(rng.choice(remaining_ids, size=n_fill, replace=False))

    keep_ids = must_keep_ids | filler_ids
    kept_documents = [d for d in data.documents if d.doc_id in keep_ids]

    def _restrict(qrels: dict[str, set[str]]) -> dict[str, set[str]]:
        restricted = {qid: (docids & keep_ids) for qid, docids in qrels.items()}
        return {qid: docids for qid, docids in restricted.items() if docids}

    qrels_all = _restrict(data.qrels_all)
    qrels_eval = _restrict(data.qrels_eval)
    kept_query_ids = qrels_all.keys() | qrels_eval.keys()
    queries = [q for q in data.queries if q.query_id in kept_query_ids]

    return RawBeirData(documents=kept_documents, queries=queries, qrels_all=qrels_all, qrels_eval=qrels_eval)


def compute_query_traffic(
    data: RawBeirData, zipf_s: float, seed: int
) -> tuple[dict[str, float], np.ndarray]:
    """Simulates realistic search traffic. Returns `(query_traffic,
    doc_traffic)`: query_traffic is each query's own Zipfian popularity
    weight (needed in 1D/2A to weight retrieval-quality metrics by how much
    simulated search volume they represent); doc_traffic aggregates that to
    a per-document score aligned to `data.documents` order (needed for
    Phase 2's MCKP benefit function).

    Each query is assigned a Zipfian popularity weight: `weight(rank) ~
    1/rank^zipf_s`, with the rank-to-query assignment itself randomized
    (seeded) so popularity skew isn't an artifact of query/document ID
    ordering. Weights are normalized to sum to 1 across queries. A
    document's traffic score is the sum of the weights of every query for
    which it is a relevance judgment -- documents outside qrels_all get 0.

    doc_traffic is a relative importance score, not a probability
    distribution (it does not itself sum to 1, since one query's weight can
    spread across several relevant documents and one document can
    accumulate weight from several queries).
    """
    n_queries = len(data.queries)
    rng = np.random.default_rng(seed)
    ranks = rng.permutation(n_queries) + 1  # 1-indexed Zipf rank, randomly assigned per query
    query_weight = 1.0 / np.power(ranks, zipf_s)
    query_weight /= query_weight.sum()

    query_traffic = {q.query_id: float(w) for q, w in zip(data.queries, query_weight)}
    doc_index = {d.doc_id: i for i, d in enumerate(data.documents)}

    doc_traffic = np.zeros(len(data.documents), dtype=np.float64)
    for qid, docids in data.qrels_all.items():
        w = query_traffic.get(qid)
        if w is None:
            continue
        for did in docids:
            idx = doc_index.get(did)
            if idx is not None:
                doc_traffic[idx] += w

    return query_traffic, doc_traffic.astype(np.float32)
