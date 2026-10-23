"""Sub-phase 1C -- baseline (existing-methodology) contamination detectors.

Three unsupervised detectors, in order of how much this project leans on
them:

- **KNN-OOD** (primary baseline): anomaly score = cosine distance to a
  document's k-th nearest neighbor. This is the strongest existing
  unsupervised detector for mixed-generation contamination specifically,
  because the failure mode we're detecting IS a distance-geometry problem
  -- a generation-B document sits far from its "true" semantic neighbors
  (which were embedded with generation A) in the shared index.
- **LOF** (Local Outlier Factor, secondary): a density-ratio variant of the
  same idea -- flags points whose local density is much lower than their
  neighbors' local density. sklearn.neighbors.LocalOutlierFactor.
- **Isolation Forest** (secondary): an ensemble of random axis-aligned
  splits; anomalies are isolated in fewer splits. Included as a weaker,
  non-distance-based comparison point -- it doesn't exploit the
  cosine-distance-geometry structure of the problem the way KNN-OOD/LOF do,
  so we expect (and later measure) it to underperform them here.

Design decision -- thresholding via an assumed contamination ratio: all
three detectors need to turn a continuous anomaly SCORE into a binary
flagged/not-flagged decision. We follow the same convention sklearn's own
LOF and IsolationForest use internally (`contamination=...`): assume the
operator has a rough prevalence estimate (e.g. "our migration touched
somewhere around 20% of documents", from a provider changelog or migration
window) and flag exactly that fraction of documents with the highest
anomaly scores. This is what makes the three detectors' precision/recall
directly comparable to each other -- they are all working from the same
assumed prevalence, not three different implicit thresholds.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


@dataclasses.dataclass
class DetectionResult:
    method: str
    scores: np.ndarray  # (N,) float, higher = more anomalous
    predicted: np.ndarray  # (N,) bool, True = flagged as contaminated


def _flag_top_ratio(scores: np.ndarray, ratio: float) -> np.ndarray:
    """Flags the `round(ratio * N)` highest-scoring documents."""
    n = len(scores)
    n_flag = round(ratio * n)
    predicted = np.zeros(n, dtype=bool)
    if n_flag > 0:
        top_indices = np.argsort(scores)[-n_flag:]
        predicted[top_indices] = True
    return predicted


def knn_ood_scores(embeddings: np.ndarray, k: int) -> np.ndarray:
    """KNN-OOD anomaly score: cosine distance from each document to its k-th
    nearest neighbor (excluding itself)."""
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(embeddings)
    distances, _ = nn.kneighbors(embeddings)
    return distances[:, -1]  # column 0 is the point itself (distance 0); take the k-th real neighbor


def detect_knn_ood(embeddings: np.ndarray, ratio: float, k: int) -> DetectionResult:
    scores = knn_ood_scores(embeddings, k=k)
    return DetectionResult("knn_ood", scores, _flag_top_ratio(scores, ratio))


def detect_lof(embeddings: np.ndarray, ratio: float, k: int) -> DetectionResult:
    lof = LocalOutlierFactor(n_neighbors=k, metric="cosine", contamination=ratio)
    labels = lof.fit_predict(embeddings)  # -1 = outlier, 1 = inlier
    scores = -lof.negative_outlier_factor_  # flip sign: higher = more anomalous, matches the other detectors
    return DetectionResult("lof", scores, labels == -1)


def detect_isolation_forest(embeddings: np.ndarray, ratio: float, seed: int) -> DetectionResult:
    iso = IsolationForest(contamination=ratio, random_state=seed)
    labels = iso.fit_predict(embeddings)  # -1 = outlier, 1 = inlier
    scores = -iso.score_samples(embeddings)  # flip sign: higher = more anomalous
    return DetectionResult("isolation_forest", scores, labels == -1)


def evaluate_detection(result: DetectionResult, true_mask: np.ndarray) -> dict[str, float]:
    """Precision/recall/F1 of `result.predicted` against ground-truth
    contamination labels `true_mask`."""
    predicted = result.predicted
    tp = int(np.sum(predicted & true_mask))
    fp = int(np.sum(predicted & ~true_mask))
    fn = int(np.sum(~predicted & true_mask))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
