"""VectorSentry -- detect, prioritize, and safely repair mixed-generation
embedding contamination in a real vector database.

This package is the production-facing layer around the research code in
`phase2_proposed/` (whose detection, calibration, scheduling and rollout
modules it reuses unchanged): adapters for real vector stores and embedding
models, real query logs instead of simulated traffic, a durable run history
with dry-run and rollback, and a write-path guard.

Names are imported lazily (PEP 562): `from vectorsentry import StateStore`
does not load PyTorch or the embedding stack, so light commands (`runs`,
`show`) start instantly.
"""

from __future__ import annotations

import importlib

_EXPORTS = {
    "Embedder": "vectorsentry.adapters",
    "InMemoryStore": "vectorsentry.adapters",
    "QdrantStore": "vectorsentry.adapters",
    "SentenceTransformerEmbedder": "vectorsentry.adapters",
    "VectorStore": "vectorsentry.adapters",
    "SentryConfig": "vectorsentry.engine",
    "VectorSentry": "vectorsentry.engine",
    "IngestGuard": "vectorsentry.guard",
    "QueryLog": "vectorsentry.querylog",
    "StateStore": "vectorsentry.state",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module 'vectorsentry' has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__
