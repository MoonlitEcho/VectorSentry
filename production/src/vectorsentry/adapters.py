"""Adapters between VectorSentry and the outside world.

- `VectorStore`: where the vectors live. `QdrantStore` talks to a real
  Qdrant collection (a server URL, or Qdrant's embedded local mode for tests
  and demos); `InMemoryStore` is a dependency-free stand-in for unit tests.
- `Embedder`: the model that turns document text into vectors.
  `SentenceTransformerEmbedder` runs a sentence-transformers model locally;
  an API-backed embedder (OpenAI, Cohere, ...) implements the same three
  members.

Documents are addressed by string ID everywhere; positional arrays only
exist inside one run, built from `VectorStore.fetch_all()`.
"""

from __future__ import annotations

import dataclasses
import uuid
from collections.abc import Sequence
from typing import Protocol

import numpy as np


@dataclasses.dataclass
class Snapshot:
    """Everything a run needs from the store, aligned by position."""

    ids: list[str]
    vectors: np.ndarray  # (N, d) float32, L2-normalized
    texts: list[str]  # the text each vector was embedded from (re-embedding needs it)


class VectorStore(Protocol):
    def fetch_all(self) -> Snapshot:
        """Every document's ID, current vector and source text."""
        ...

    def upsert(self, ids: Sequence[str], vectors: np.ndarray, model_id: str | None = None) -> None:
        """Overwrite the vectors of existing documents (text is unchanged)."""
        ...


class Embedder(Protocol):
    model_id: str
    cost_per_document: float  # in whatever unit the budget uses (default: full re-embeds)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """L2-normalized float32 vectors, one per text."""
        ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


# --------------------------------------------------------------------------
# Vector stores
# --------------------------------------------------------------------------

class InMemoryStore:
    """A list of (id, vector, text) rows in memory -- for unit tests."""

    def __init__(self, ids: Sequence[str], vectors: np.ndarray, texts: Sequence[str]):
        self.ids = list(ids)
        self.vectors = _normalize(vectors).copy()
        self.texts = list(texts)
        self.model_tags: dict[str, str] = {}
        self._pos = {doc_id: i for i, doc_id in enumerate(self.ids)}

    def fetch_all(self) -> Snapshot:
        return Snapshot(list(self.ids), self.vectors.copy(), list(self.texts))

    def upsert(self, ids: Sequence[str], vectors: np.ndarray, model_id: str | None = None) -> None:
        rows = [self._pos[doc_id] for doc_id in ids]
        self.vectors[rows] = _normalize(vectors)
        if model_id is not None:
            self.model_tags.update({doc_id: model_id for doc_id in ids})


class QdrantStore:
    """A Qdrant collection whose points carry the document text in their
    payload (key `text_field`) and the document ID under `id_field`.

    Point IDs in Qdrant must be integers or UUIDs, so documents are mapped to
    deterministic UUIDs derived from their string ID; the original ID stays in
    the payload. `model_id` is written to the payload on every upsert, giving
    each vector an auditable record of which model produced it.
    """

    def __init__(self, client, collection: str, text_field: str = "text", id_field: str = "doc_id", batch_size: int = 512):
        self.client = client
        self.collection = collection
        self.text_field = text_field
        self.id_field = id_field
        self.batch_size = batch_size

    @staticmethod
    def point_id(doc_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"vectorsentry:{doc_id}"))

    @classmethod
    def create(cls, client, collection: str, dim: int, **kwargs) -> QdrantStore:
        from qdrant_client import models

        if client.collection_exists(collection):
            client.delete_collection(collection)
        client.create_collection(collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))
        return cls(client, collection, **kwargs)

    def add_documents(self, ids: Sequence[str], vectors: np.ndarray, texts: Sequence[str], model_ids: Sequence[str]) -> None:
        """Bulk-load documents (used to build demo collections)."""
        from qdrant_client import models

        vectors = _normalize(vectors)
        for start in range(0, len(ids), self.batch_size):
            stop = start + self.batch_size
            self.client.upsert(self.collection, points=[
                models.PointStruct(
                    id=self.point_id(doc_id), vector=vec.tolist(),
                    payload={self.id_field: doc_id, self.text_field: text, "model_id": model},
                )
                for doc_id, vec, text, model in zip(ids[start:stop], vectors[start:stop], texts[start:stop], model_ids[start:stop])
            ])

    def fetch_all(self) -> Snapshot:
        ids, vectors, texts = [], [], []
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, limit=self.batch_size, offset=offset, with_payload=True, with_vectors=True
            )
            for p in points:
                ids.append(str(p.payload[self.id_field]))
                vectors.append(p.vector)
                texts.append(p.payload.get(self.text_field, ""))
            if offset is None:
                break
        # Stable order (by document ID) so runs are reproducible.
        order = np.argsort(np.array(ids, dtype=object))
        vec = _normalize(np.array(vectors, dtype=np.float32)) if vectors else np.zeros((0, 0), np.float32)
        return Snapshot([ids[i] for i in order], vec[order], [texts[i] for i in order])

    def upsert(self, ids: Sequence[str], vectors: np.ndarray, model_id: str | None = None) -> None:
        from qdrant_client import models

        vectors = _normalize(vectors)
        for start in range(0, len(ids), self.batch_size):
            chunk_ids = list(ids[start:start + self.batch_size])
            chunk_vec = vectors[start:start + self.batch_size]
            self.client.update_vectors(self.collection, points=[
                models.PointVectors(id=self.point_id(doc_id), vector=vec.tolist())
                for doc_id, vec in zip(chunk_ids, chunk_vec)
            ])
            if model_id is not None:
                self.client.set_payload(self.collection, payload={"model_id": model_id},
                                        points=[self.point_id(d) for d in chunk_ids])


# --------------------------------------------------------------------------
# Embedders
# --------------------------------------------------------------------------

class SentenceTransformerEmbedder:
    """A local sentence-transformers model. `cost_per_document` defaults to 1
    (budget in full-re-embed units); set it to a real price to budget in $."""

    def __init__(self, model_name: str, cost_per_document: float = 1.0, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model_id = model_name
        self.cost_per_document = cost_per_document
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)
        return self._model.encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
