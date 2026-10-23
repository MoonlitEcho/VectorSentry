"""Fast, network-free unit tests for phase1.detection, using small synthetic
well-separated clusters (not the real benchmark) so they run in the default
`pytest` suite. The real-embedding, full-scale detector run happens via
phase1_baseline/scripts/run_detection.py and is reported in DEVLOG.md, not
re-run on every `pytest` invocation.
"""

import numpy as np

from phase1.detection import (
    DetectionResult,
    detect_isolation_forest,
    detect_knn_ood,
    detect_lof,
    evaluate_detection,
)


def _synthetic_mixed_embeddings(n: int, ratio: float, seed: int):
    """A dense inlier cluster (tightly clustered UNIT vectors around one
    fixed direction, standing in for generation A) plus scattered outlier
    points (independent random unit vectors, standing in for generation-B
    "contaminated" docs) -- the classic dense-cluster-plus-scattered-outliers
    setup distance/density detectors are designed to solve.

    Important: this must be constructed in COSINE geometry, not Euclidean --
    an earlier version of this fixture used two Gaussian blobs at different
    Euclidean offsets (e.g. centered at 0 vs centered at 6) and every
    detector failed. The reason: a cluster offset far from the origin in a
    consistent direction is nearly collinear with itself, so under cosine
    distance it looks like the TIGHTEST, most "normal" cluster of all --
    inverting the intended signal. Real embeddings are L2-normalized (see
    common/embeddings.py), so the synthetic fixture must be too.
    """
    rng = np.random.default_rng(seed)
    dim = 8
    n_contaminated = round(ratio * n)
    n_majority = n - n_contaminated

    mu = rng.normal(size=dim)
    mu /= np.linalg.norm(mu)
    majority = mu + rng.normal(scale=0.05, size=(n_majority, dim))  # tight cluster around mu
    majority /= np.linalg.norm(majority, axis=1, keepdims=True)

    minority = rng.normal(size=(n_contaminated, dim))  # scattered, ~random directions
    minority /= np.linalg.norm(minority, axis=1, keepdims=True)

    embeddings = np.vstack([majority, minority]).astype(np.float32)
    mask = np.zeros(n, dtype=bool)
    mask[n_majority:] = True

    perm = rng.permutation(n)  # shuffle so contamination isn't trivially at fixed positions
    return embeddings[perm], mask[perm]


def test_knn_ood_recovers_well_separated_minority_cluster():
    embeddings, mask = _synthetic_mixed_embeddings(n=200, ratio=0.1, seed=1)
    result = detect_knn_ood(embeddings, ratio=0.1, k=5)
    metrics = evaluate_detection(result, mask)
    assert metrics["recall"] > 0.8


def test_lof_recovers_well_separated_minority_cluster():
    embeddings, mask = _synthetic_mixed_embeddings(n=200, ratio=0.1, seed=2)
    result = detect_lof(embeddings, ratio=0.1, k=10)
    metrics = evaluate_detection(result, mask)
    assert metrics["recall"] > 0.7


def test_isolation_forest_recovers_well_separated_minority_cluster():
    embeddings, mask = _synthetic_mixed_embeddings(n=200, ratio=0.1, seed=3)
    result = detect_isolation_forest(embeddings, ratio=0.1, seed=3)
    metrics = evaluate_detection(result, mask)
    # Looser bound: Isolation Forest is the weaker, non-distance-based
    # secondary baseline (see detection.py's module docstring).
    assert metrics["recall"] > 0.5


def test_predicted_count_matches_assumed_ratio():
    embeddings, mask = _synthetic_mixed_embeddings(n=200, ratio=0.2, seed=4)
    result = detect_knn_ood(embeddings, ratio=0.2, k=5)
    assert result.predicted.sum() == round(0.2 * 200)


def test_evaluate_detection_perfect_prediction():
    mask = np.array([True, False, True, False])
    result = DetectionResult("dummy", scores=np.zeros(4), predicted=mask.copy())
    metrics = evaluate_detection(result, mask)
    assert metrics == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_evaluate_detection_no_predictions_scores_zero():
    mask = np.array([True, False, True, False])
    result = DetectionResult("dummy", scores=np.zeros(4), predicted=np.zeros(4, dtype=bool))
    metrics = evaluate_detection(result, mask)
    assert metrics == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
