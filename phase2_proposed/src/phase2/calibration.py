"""Phase 2 -- label-free estimation of what each remediation action is
actually worth, by REPLAYING the query log against the probe batch.

The MCKP needs, per action, "what fraction of the damage does this fix
undo?" The original 2A took recovery_patch = 0.064 from Phase 1's measured
nDCG. That number turned out to be mostly an artifact: Phase 1's uniform
patch was applied to KNN-OOD's flagged set, about half of which were clean
documents, and its gain was diluted by those wasted (and, once patch
semantics are realistic, harmful) fixes. Applied to documents that really
are contaminated, the same patch closes 35-60% of the nDCG gap.

A real operator cannot measure that on the live index (they do not know
which documents are contaminated), but they CAN measure it on the probe
batch, where both embeddings of every document are known. Shadow-traffic
replay:

1. Run the logged queries against the clean (generation-A) index and record
   every probe document's impressions: which queries showed it, at which rank.
2. Simulate three states for the probe documents and replay the same queries:
   - raw:      their stored vector is the generation-B one (contaminated);
   - patched:  generation B pushed through an adapter fitted WITHOUT them
               (k-fold, so the estimate is out-of-sample);
   - harmed:   their CLEAN generation-A vector pushed through the adapter --
               what patching a false positive does.
3. Retention = position-discounted, popularity-weighted share of the clean
   impressions that survive. Then
       recovery_patch = (ret_patched - ret_raw) / (1 - ret_raw)
       harm_patch     = (1 - ret_harmed) / (1 - ret_raw)
   both in units of "the damage contamination itself does", which is exactly
   the unit the MCKP's benefit function uses (full re-embed recovers 1.0).

Impressions rather than relevance labels: no qrels are needed, so this is
something a production team could run on its own logs.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from common.patch import fit_adapter


@dataclasses.dataclass
class ActionEffects:
    recovery_patch: float  # fraction of contamination damage a patch undoes, in [0, 1]
    harm_patch: float  # damage from patching a CLEAN document, same units, >= 0
    retention_raw: float
    retention_patched: float
    retention_clean_patched: float
    n_probe_documents_with_impressions: int


def _top_k(query_embeddings: np.ndarray, doc_embeddings: np.ndarray, depth: int) -> np.ndarray:
    scores = query_embeddings @ doc_embeddings.T
    top = np.argpartition(-scores, depth - 1, axis=1)[:, :depth]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)


def estimate_action_effects(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    probe_indices: np.ndarray,
    query_embeddings: np.ndarray,
    query_weights: np.ndarray,
    adapter_kind: str = "ridge",
    ridge_alpha: float = 1.0,
    depth: int = 10,
    n_folds: int = 5,
    seed: int = 0,
) -> ActionEffects | None:
    """Replay-based effect estimates (see module docstring). Returns None if
    no probe document ever appears in any logged query's results (nothing to
    replay -- callers fall back to the configured constant)."""
    depth = min(depth, len(emb_a))
    clean_top = _top_k(query_embeddings, emb_a, depth)
    discount = 1.0 / np.log2(np.arange(depth) + 2)

    impressions = []  # (probe position, query indices, clean ranks)
    for pos, doc in enumerate(probe_indices):
        hit = clean_top == doc
        queries = np.flatnonzero(hit.any(axis=1))
        if len(queries):
            impressions.append((pos, queries, np.argmax(hit[queries], axis=1)))
    if not impressions:
        return None

    # Out-of-fold adapters: each probe document is patched by an adapter that
    # never saw it, so the estimate is not flattered by fitting on itself.
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(len(probe_indices)), max(2, min(n_folds, len(probe_indices))))
    patched = np.empty((len(probe_indices), emb_a.shape[1]), dtype=np.float32)
    harmed = np.empty_like(patched)
    for fold in folds:
        train = np.setdiff1d(np.arange(len(probe_indices)), fold)
        adapter = fit_adapter(emb_a, emb_b, probe_indices[train], adapter_kind, ridge_alpha)
        patched[fold] = adapter(emb_b[probe_indices[fold]])
        harmed[fold] = adapter(emb_a[probe_indices[fold]])

    def retention(probe_vectors: np.ndarray) -> float:
        index = emb_a.copy()
        index[probe_indices] = probe_vectors
        kept = total = 0.0
        for pos, queries, clean_rank in impressions:
            weight = query_weights[queries]
            total += float((weight * discount[clean_rank]).sum())
            hit = _top_k(query_embeddings[queries], index, depth) == probe_indices[pos]
            new_gain = np.where(hit.any(axis=1), discount[np.argmax(hit, axis=1)], 0.0)
            kept += float((weight * new_gain).sum())
        return kept / total if total > 0 else 1.0

    ret_raw = retention(emb_b[probe_indices])
    ret_patched = retention(patched)
    ret_harmed = retention(harmed)
    damage = max(1.0 - ret_raw, 1e-9)
    return ActionEffects(
        recovery_patch=float(np.clip((ret_patched - ret_raw) / damage, 0.0, 1.0)),
        harm_patch=float(max(0.0, (1.0 - ret_harmed) / damage)),
        retention_raw=ret_raw,
        retention_patched=ret_patched,
        retention_clean_patched=ret_harmed,
        n_probe_documents_with_impressions=len(impressions),
    )
