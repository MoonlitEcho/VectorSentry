"""Drift-Adapter-style patch: a cheap, imperfect fix for a contaminated
embedding -- a single learned linear (orthogonal Procrustes) alignment from
generation-B's embedding space to generation-A's.

Shared by Phase 1 (uniform-patch baseline, 1D) and Phase 2 (the "Patch"
action in the MCKP formulation, 2A): both need the exact same underlying
fix, applied to different documents for different reasons. Living in
`common` keeps that fix identical across phases -- Phase 2 must win by
choosing WHICH documents get which action, not by silently getting a better
patch than Phase 1 had access to.

Calibration realism: fitting the transform needs (generation_a, generation_b)
embedding PAIRS for the same documents. A real operator does not have this
for contaminated documents -- the whole premise of silent contamination is
that the old embedding is gone once a document is overwritten. What a real
operator plausibly DOES have is a small "probe batch": a handful of
documents deliberately dual-embedded with both models specifically to
validate a new provider before rolling out a migration. `sample_calibration_indices`
models exactly that -- a small, FIXED random sample drawn independently of
which documents later happen to become contaminated in any given ratio.
"""

from __future__ import annotations

import numpy as np


def sample_calibration_indices(n_documents: int, sample_size: int, seed: int) -> np.ndarray:
    """Draws a small, fixed random sample of document indices to use as the
    calibration ("probe batch") set. Independent of any contamination
    ratio/mask -- see module docstring."""
    rng = np.random.default_rng(seed)
    sample_size = min(sample_size, n_documents)
    return rng.choice(n_documents, size=sample_size, replace=False)


def fit_patch_transform(emb_a: np.ndarray, emb_b: np.ndarray, calibration_indices: np.ndarray) -> np.ndarray:
    """Learns an orthogonal Procrustes rotation R minimizing
    ||A - B @ R||_F subject to R^T R = I, from the calibration subset's
    true (generation_a, generation_b) embedding pairs.

    Orthogonal (rather than an unconstrained linear map) is deliberate: it
    preserves vector norms, so applying R to a unit-normalized embedding
    keeps it unit-normalized (cosine similarity stays well-defined without
    needing to re-normalize afterward in exact arithmetic -- `apply_patch`
    still renormalizes defensively for float32 drift). It's also the
    textbook closed-form solution (via SVD) for this exact alignment
    problem, so there's no iterative optimization to second-guess.
    """
    A = emb_a[calibration_indices]
    B = emb_b[calibration_indices]
    U, _, Vt = np.linalg.svd(B.T @ A)
    R = (U @ Vt).astype(np.float32)
    return R


class Adapter:
    """A fitted generation-B -> generation-A embedding adapter: `apply(X)`
    maps stored vectors through `X @ weight + bias` and re-normalizes.

    Realism note -- the patch is applied to whatever vector is CURRENTLY
    STORED for a document, never to that document's true generation-B
    embedding looked up on the side. For a correctly flagged document the
    two are the same thing. For a false positive (a clean generation-A
    document the detector wrongly flagged) the stored vector is its
    generation-A embedding, and patching it pushes a correct vector through
    a B->A map it was never meant for -- real damage. The original pipeline
    patched false positives with their (unknowable) generation-B embedding
    instead, which both leaked information and hid this cost.
    """

    def __init__(self, kind: str, weight: np.ndarray, bias: np.ndarray):
        self.kind = kind
        self.weight = weight.astype(np.float32)
        self.bias = bias.astype(np.float32)

    def apply(self, stored: np.ndarray) -> np.ndarray:
        stored = np.atleast_2d(stored)
        mapped = stored @ self.weight + self.bias
        norms = np.linalg.norm(mapped, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # guard against a degenerate all-zero row
        return (mapped / norms).astype(np.float32)

    __call__ = apply


def fit_adapter(
    emb_a: np.ndarray, emb_b: np.ndarray, calibration_indices: np.ndarray, kind: str = "ridge", ridge_alpha: float = 1.0
) -> Adapter:
    """Fits a B -> A adapter on the probe batch's (generation_a,
    generation_b) pairs.

    - "procrustes": the orthogonal rotation from `fit_patch_transform`, no
      bias. Norm-preserving, but a rotation about the origin cannot move
      generation B's centroid onto generation A's.
    - "ridge": closed-form affine least squares with an L2 penalty,
      W = (Bc^T Bc + alpha I)^-1 Bc^T Ac on mean-centred pairs, and a bias
      that maps B's centroid onto A's. The intercept is what lets it fix
      the centroid offset a rotation cannot; the penalty keeps a 384x384 map
      fitted from ~300 pairs from overfitting.
    """
    if kind == "procrustes":
        R = fit_patch_transform(emb_a, emb_b, calibration_indices)
        return Adapter(kind, R, np.zeros(R.shape[1], dtype=np.float32))
    if kind == "ridge":
        A = emb_a[calibration_indices].astype(np.float64)
        B = emb_b[calibration_indices].astype(np.float64)
        mu_a, mu_b = A.mean(axis=0), B.mean(axis=0)
        Ac, Bc = A - mu_a, B - mu_b
        W = np.linalg.solve(Bc.T @ Bc + ridge_alpha * np.eye(B.shape[1]), Bc.T @ Ac)
        return Adapter(kind, W, mu_a - mu_b @ W)
    raise ValueError(f"Unknown adapter kind {kind!r}; expected 'ridge' or 'procrustes'.")


def apply_patch(emb_b: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Applies a fitted patch transform to (contaminated) generation-B
    embeddings. This is a PARTIAL fix by construction: a single global
    linear map cannot undo whatever nonlinear, document-specific distortion
    a different embedding model introduces -- how partial is an empirical
    question, answered by measuring retrieval quality on patched embeddings
    (see common/metrics.py), not assumed as a constant here.
    """
    patched = emb_b @ transform
    norms = np.linalg.norm(patched, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # guard against a degenerate all-zero row
    return (patched / norms).astype(np.float32)
