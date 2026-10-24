"""Fast, network-free unit tests for the redesigned pipeline components:
the ridge adapter, impression traffic, the fingerprint detector, the
replay calibration, and the expected-benefit MCKP frontier."""

import numpy as np
import pytest

from common.data import compute_impression_traffic, zipf_query_weights
from common.patch import fit_adapter
from phase2.calibration import estimate_action_effects
from phase2.fingerprint import FingerprintDetector
from phase2.scheduler import (
    build_frontier_segments_from_benefits,
    build_mckp_items_from_benefits,
    expected_benefits,
    priority_order,
    solve_exact,
    solve_greedy,
)


def _unit(x):
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def _two_generations(n=400, dim=16, seed=0):
    """Generation B = an affine distortion of generation A plus noise: a
    different centroid AND a rotation, like two real models' spaces."""
    rng = np.random.default_rng(seed)
    emb_a = _unit(rng.normal(size=(n, dim)))
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    offset = rng.normal(size=dim) * 0.8
    emb_b = _unit(emb_a @ q + offset + rng.normal(scale=0.05, size=(n, dim)))
    return emb_a, emb_b


# --- adapter -----------------------------------------------------------------


def test_ridge_adapter_beats_procrustes_when_spaces_have_different_centroids():
    emb_a, emb_b = _two_generations()
    probe, held_out = np.arange(200), np.arange(200, 400)
    ridge = fit_adapter(emb_a, emb_b, probe, "ridge", ridge_alpha=0.1)
    procrustes = fit_adapter(emb_a, emb_b, probe, "procrustes")
    cos_ridge = np.mean(np.sum(ridge(emb_b[held_out]) * emb_a[held_out], axis=1))
    cos_proc = np.mean(np.sum(procrustes(emb_b[held_out]) * emb_a[held_out], axis=1))
    assert cos_ridge > cos_proc
    assert cos_ridge > 0.9


def test_adapter_output_is_unit_normalized_and_rejects_unknown_kind():
    emb_a, emb_b = _two_generations()
    out = fit_adapter(emb_a, emb_b, np.arange(100))(emb_b[:10])
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    with pytest.raises(ValueError):
        fit_adapter(emb_a, emb_b, np.arange(100), "mlp")


# --- impression traffic --------------------------------------------------------


def test_impression_traffic_credits_shown_documents_with_a_position_discount():
    docs = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]], dtype=np.float32)
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    traffic = compute_impression_traffic(queries, np.array([1.0]), docs, depth=2)
    assert traffic[0] == pytest.approx(1.0)  # rank 0: 1/log2(2)
    assert traffic[1] == pytest.approx(1.0 / np.log2(3))  # rank 1
    assert traffic[2] == 0.0 and traffic[3] == 0.0  # never shown


def test_zipf_query_weights_sum_to_one_and_are_seeded():
    w1 = zipf_query_weights(50, 1.0, seed=3)
    np.testing.assert_allclose(w1.sum(), 1.0)
    np.testing.assert_array_equal(w1, zipf_query_weights(50, 1.0, seed=3))


# --- fingerprint detector --------------------------------------------------------


def test_fingerprint_separates_generations_without_an_assumed_ratio():
    emb_a, emb_b = _two_generations(n=600)
    probe = np.arange(100)
    detector = FingerprintDetector.fit(emb_a, emb_b, probe, k=10)
    rest = np.arange(100, 600)
    truth = np.zeros(len(rest), dtype=bool)
    truth[::3] = True  # ~33% contamination, never told to the detector
    stored = np.where(truth[:, None], emb_b[rest], emb_a[rest])
    prob, flagged = detector.flag(stored)
    assert np.mean(flagged == truth) > 0.9
    assert np.all((prob > 0) & (prob < 1))  # Laplace-smoothed, never a hard 0/1
    # ...and it beats KNN-OOD even when KNN-OOD is handed the true ratio.
    from phase1.detection import detect_knn_ood

    knn = detect_knn_ood(stored, ratio=float(truth.mean()), k=10)
    assert np.mean(flagged == truth) > np.mean(knn.predicted == truth)


# --- replay calibration ------------------------------------------------------------


def test_replay_calibration_measures_recovery_and_harm():
    emb_a, emb_b = _two_generations(n=400, dim=16)
    rng = np.random.default_rng(1)
    # Logged queries: noisy copies of documents, so each finds its neighbours.
    queries = _unit(emb_a[rng.choice(400, 300)] + rng.normal(scale=0.05, size=(300, 16)))
    weights = zipf_query_weights(300, 1.0, seed=0)
    effects = estimate_action_effects(emb_a, emb_b, np.arange(150), queries, weights, "ridge", 0.1, depth=5)
    assert effects is not None
    assert effects.retention_raw < 0.2  # contaminated probe docs vanish from results
    assert effects.recovery_patch > 0.5  # the adapter brings most of them back
    assert effects.harm_patch > 0.0  # patching a clean document is not free


# --- expected-benefit MCKP ---------------------------------------------------------


def test_expected_benefits_penalize_patching_a_probably_clean_document():
    bf, bp = expected_benefits(np.array([0.95, 0.05]), np.array([1.0, 1.0]), 1.0, 0.8, 0.7)
    assert bf[0] == pytest.approx(0.95) and bf[1] == pytest.approx(0.05)
    assert bp[0] > 0  # confident detection: patch is worth something
    assert bp[1] < 0  # likely false positive: patch does net harm


def test_frontier_from_benefits_drops_a_harmful_patch_and_keeps_a_useful_one():
    bf = np.array([1.0, 1.0])
    bp = np.array([-0.1, 0.8])  # doc 0: patch hurts; doc 1: patch is a cheap 80% fix
    segments = build_frontier_segments_from_benefits(bf, bp, 1.0, 0.1)
    by_doc = {i: [(s.from_action, s.to_action) for s in segments if s.local_index == i] for i in (0, 1)}
    assert by_doc[0] == [("defer", "full_reembed")]
    assert by_doc[1] == [("defer", "patch"), ("patch", "full_reembed")]


def test_frontier_greedy_matches_exact_on_expected_benefits():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.5, 1.0, 60)
    traffic = rng.pareto(1.5, 60)
    bf, bp = expected_benefits(p, traffic, 1.0, 0.7, 0.6)
    items = build_mckp_items_from_benefits(bf, bp, 1.0, 0.1)
    segments = build_frontier_segments_from_benefits(bf, bp, 1.0, 0.1)
    for budget in (3.0, 10.0, 30.0):
        exact = solve_exact(items, 60, budget)
        greedy = solve_greedy(segments, 60, budget)
        assert greedy.total_cost <= budget + 1e-9
        assert greedy.total_benefit >= 0.97 * exact.total_benefit


def test_priority_order_ships_the_highest_value_fixes_first():
    bf = np.array([0.1, 5.0, 1.0])
    bp = np.array([0.0, 0.0, 0.0])
    segments = build_frontier_segments_from_benefits(bf, bp, 1.0, 0.1)
    plan = solve_greedy(segments, 3, budget=3.0)
    assert priority_order(segments, plan).tolist() == [1, 2, 0]


def test_remediate_recycles_a_rejected_patch_into_a_full_reembed():
    from common.config import Config
    from common.patch import Adapter
    from phase2.migrate import labelled_canary_scorer
    from phase2.pipeline import Phase2Context, remediate

    # Ten zero-vector fillers plus one contaminated target document (index
    # 10) that its query can no longer find.
    doc_ids = [f"f{i}" for i in range(10)] + ["target"]
    emb_a = np.zeros((11, 2), dtype=np.float32)
    emb_a[10] = [1.0, 0.0]
    stored = emb_a.copy()
    stored[10] = [-1.0, 0.0]
    query = type("Q", (), {"query_id": "q", "text": ""})()
    canary = labelled_canary_scorer(doc_ids, [query], np.array([[1.0, 0.0]], dtype=np.float32), {"q": {"target"}})

    # The scheduler THINKS a patch is as good as a full re-embed at a tenth
    # of the cost (recovery_patch=1.0), so it plans a patch. But this adapter
    # leaves the vector pointing the wrong way, so the canary sees no gain
    # and (with a margin demanding improvement) rejects it.
    useless_adapter = Adapter("test", np.eye(2), np.zeros(2))
    cfg = Config()
    cfg.migration.regression_tolerance = -0.5  # demand a real improvement
    cfg.migration.recycle_rounds = 3
    ctx = Phase2Context(
        probe_indices=np.array([], dtype=int), adapter=useless_adapter, detector=None, effects=None,
        recovery_patch=1.0, harm_patch=0.0, traffic=np.ones(11),
    )
    outcome = remediate(cfg, ctx, stored, emb_a, np.array([10]), np.array([0.99]), budget_units=1.0, canary=canary)

    first_plan, first_report = outcome.rounds[0]
    assert first_plan.plan.actions.tolist() == ["patch"] and not first_report.applied.any()
    assert outcome.action_applied.tolist() == ["full_reembed"]  # recycled budget bought the exact fix
    np.testing.assert_array_equal(outcome.live_embeddings[10], emb_a[10])
    assert outcome.cost_spent == pytest.approx(1.0)
