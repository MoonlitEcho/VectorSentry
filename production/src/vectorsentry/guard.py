"""Write-path guard: stop wrong-model vectors at the door.

Repairing contamination after the fact costs re-embedding budget and a
period of degraded search. Preventing it is far cheaper: run the same
generation fingerprint on every vector BEFORE it is written, and reject (or
quarantine) any that look like they came from the other model. The check is
a kNN vote against a few hundred probe vectors, so it adds microseconds per
write.

    guard = sentry.guard()                       # after VectorSentry.calibrate()
    accepted_ids, accepted_vecs, report = guard.filter(ids, vectors)
    store.upsert(accepted_ids, accepted_vecs)
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import numpy as np

from phase2.fingerprint import FingerprintDetector


@dataclasses.dataclass
class GuardReport:
    p_wrong_model: np.ndarray  # (n,) P(vector came from the suspect model)
    accepted: np.ndarray  # (n,) bool
    rejected_ids: list[str]

    @property
    def n_rejected(self) -> int:
        return len(self.rejected_ids)


class IngestGuard:
    def __init__(self, detector: FingerprintDetector, threshold: float = 0.5):
        self.detector = detector
        self.threshold = threshold

    def check(self, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(P(wrong model), accepted mask) for a batch of vectors about to be written."""
        p = self.detector.probability_generation_b(vectors)
        return p, p <= self.threshold

    def filter(self, ids: Sequence[str], vectors: np.ndarray) -> tuple[list[str], np.ndarray, GuardReport]:
        p, ok = self.check(vectors)
        ids = list(ids)
        accepted_ids = [d for d, keep in zip(ids, ok) if keep]
        report = GuardReport(p, ok, [d for d, keep in zip(ids, ok) if not keep])
        return accepted_ids, np.asarray(vectors)[ok], report
