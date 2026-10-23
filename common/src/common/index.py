"""FAISS ANN index construction and search helpers.

Embeddings are L2-normalized at encode time (see common/embeddings.py), so
cosine similarity search reduces to inner-product search. That lets us use
FAISS's simplest exact index, IndexFlatIP, while still matching true cosine
similarity numerically -- exact search (not an approximate index like HNSW)
is deliberate here: the corpus is small enough (thousands of docs) that
exact search costs nothing, and it removes ANN approximation error as a
confound when we later attribute retrieval-quality changes to contamination.
"""

from __future__ import annotations

import numpy as np

try:
    import faiss
except ImportError as exc:  # pragma: no cover - dependency is required, not optional
    raise ImportError(
        "faiss-cpu is required (see pyproject.toml). Install with `pip install -e .`."
    ) from exc


def build_index(embeddings: np.ndarray) -> "faiss.Index":
    """Builds an exact inner-product FAISS index over `embeddings`."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    return index


def search(index: "faiss.Index", query_embeddings: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Searches `index` for the top-k nearest neighbors of each query.

    Returns (scores, indices), each of shape (num_queries, k). `indices`
    are positions into whatever embedding matrix the index was built from.
    """
    scores, indices = index.search(np.ascontiguousarray(query_embeddings, dtype=np.float32), k)
    return scores, indices
