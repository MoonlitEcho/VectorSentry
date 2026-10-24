"""Phase 2 detection -- a generation-FINGERPRINT detector that models the
index as what it actually is: a mixture of two embedding populations.

Why Phase 1's detectors top out at F1 ~0.5 (and fail outright on the
bge-small pairing): KNN-OOD, LOF and Isolation Forest all ask "is this
vector unusual relative to the rest of the index?" But contaminated
documents are not rare outliers in one population -- they are a whole
second population, internally just as dense and well-formed as the first.
Once enough of them exist they are each other's nearest neighbours and look
perfectly normal (DEVLOG.md 1C diagnoses exactly this inversion).

The right question is "WHICH population does this vector come from?" -- and
the operator already holds a labelled sample of both: the probe batch
(common/patch.py) of documents deliberately dual-embedded with both models
to validate the new provider. Same text, both generations, known labels.
Every stored vector is compared with those probe vectors; P(generation B) is
the smoothed share of its `k` nearest probe vectors that came from
generation B. A stored vector of a given generation lies far closer to
same-generation probe vectors than to the other generation's, whatever its
topic, because each model has its own geometry (centroid, anisotropy,
per-dimension statistics).

Properties that matter for an examiner:
- No label leakage: fitting uses ONLY the probe batch's own two embeddings,
  which the operator produced on purpose. Ground-truth contamination labels
  of the index are never seen. Probe documents themselves are excluded from
  reported detection metrics (their stored vector is literally one of the
  reference vectors).
- No assumed contamination ratio: a document is flagged when
  P(generation B) > threshold (0.5 = "more likely B than A"), so the same
  detector works for a 5% incident, a 50% incident, or no incident at all
  -- which is what a continuous monitor (Live Sentry) needs.
- It outputs a probability, which the MCKP scheduler uses directly as the
  chance that a fix is needed (expected benefit), rather than a min-max
  normalized distance with no probabilistic meaning.
- Limitation, stated plainly: it needs a probe batch, and it assumes the two
  generations are geometrically distinguishable. Two checkpoints of the SAME
  model a few training steps apart could be much harder to separate; this
  benchmark pairs two different real models, as a provider upgrade does.
"""

from __future__ import annotations

import dataclasses

import numpy as np


@dataclasses.dataclass
class FingerprintDetector:
    """kNN vote against a labelled two-generation reference set."""

    reference: np.ndarray  # (2 * n_probe, d) probe vectors: generation A rows, then generation B rows
    is_generation_b: np.ndarray  # (2 * n_probe,) bool, aligned to `reference`
    k: int = 10

    @classmethod
    def fit(cls, emb_a: np.ndarray, emb_b: np.ndarray, probe_indices: np.ndarray, k: int = 10) -> "FingerprintDetector":
        reference = np.vstack([emb_a[probe_indices], emb_b[probe_indices]]).astype(np.float32)
        labels = np.r_[np.zeros(len(probe_indices), dtype=bool), np.ones(len(probe_indices), dtype=bool)]
        return cls(reference, labels, min(k, len(reference)))

    def probability_generation_b(self, stored: np.ndarray, batch_size: int = 2048) -> np.ndarray:
        """P(generation B) for every stored vector: the share of its `k`
        nearest probe vectors (cosine = inner product on unit vectors) that
        are generation B, Laplace-smoothed as (votes + 1) / (k + 2) so the
        output is never exactly 0 or 1 -- an honest probability for the
        scheduler's expected-benefit computation, not a hard verdict."""
        stored = np.atleast_2d(stored).astype(np.float32)
        out = np.empty(len(stored), dtype=np.float64)
        for start in range(0, len(stored), batch_size):
            sims = stored[start : start + batch_size] @ self.reference.T
            top = np.argpartition(-sims, self.k - 1, axis=1)[:, : self.k]
            votes = self.is_generation_b[top].sum(axis=1)
            out[start : start + batch_size] = (votes + 1.0) / (self.k + 2.0)
        return out

    def flag(self, stored: np.ndarray, threshold: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
        """Returns (probability, flagged_mask)."""
        prob = self.probability_generation_b(stored)
        return prob, prob > threshold
