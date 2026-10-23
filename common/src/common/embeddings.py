"""Embedding computation with on-disk caching.

Embedding the full corpus with a given model is the expensive step and is
contamination-ratio-independent -- every ratio just picks a different subset
of documents to take from generation A vs generation B. So we embed the
FULL corpus with each generation exactly once, cache the result, and let
`common.benchmark` build each ratio's mixed-generation matrix with cheap
numpy indexing instead of re-embedding. This is what makes iterating over
five contamination ratios take seconds, not five embedding passes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from common.data import Document


def embed_texts(
    texts: list[str],
    model_name: str,
    cache_path: Path | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Encodes a list of raw strings into a normalized embedding matrix of
    shape (len(texts), model_dim).

    Embeddings are L2-normalized at encode time (`normalize_embeddings=True`)
    so that cosine similarity search reduces to a plain inner-product search
    -- this lets `common.index` use FAISS's simplest exact index
    (IndexFlatIP) while still matching true cosine similarity numerically.
    """
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path)

    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, embeddings)

    return embeddings


def embed_documents(
    documents: list[Document],
    model_name: str,
    cache_path: Path | None = None,
    batch_size: int = 64,
) -> np.ndarray:
    """Encodes `title + text` for each document -- see `embed_texts`."""
    texts = [f"{d.title} {d.text}".strip() for d in documents]
    return embed_texts(texts, model_name, cache_path, batch_size)
