"""Fast, network-free unit tests for phase2.migrate's shadow-build,
golden-query selection, and verify-gated cutover."""

import numpy as np

from phase2.migrate import (
    build_shadow_embeddings,
    execute_migration,
    execute_staged_migration,
    labelled_canary_scorer,
    replay_canary_scorer,
    wave_is_unsafe,
    select_golden_query_indices,
)


def _shift_patch(stored):
    """A stand-in adapter: maps any stored vector v to v + 10, so tests can
    see exactly WHICH stored vector a patch was applied to."""
    return stored + 10.0


class _Query:
    def __init__(self, query_id, text=""):
        self.query_id = query_id
        self.text = text


def test_build_shadow_embeddings_applies_actions_correctly():
    current = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    emb_a = np.array([[9.0, 9.0], [8.0, 8.0], [7.0, 7.0]], dtype=np.float32)
    flagged_indices = np.array([0, 1, 2])
    actions = np.array(["full_reembed", "patch", "defer"], dtype=object)

    shadow = build_shadow_embeddings(current, emb_a, _shift_patch, flagged_indices, actions)

    np.testing.assert_array_equal(shadow[0], emb_a[0])  # full_reembed -> true embedding
    # patch -> the adapter applied to the CURRENTLY STORED vector, not to a
    # side lookup of the document's generation-B embedding.
    np.testing.assert_array_equal(shadow[1], current[1] + 10.0)
    np.testing.assert_array_equal(shadow[2], current[2])  # defer -> unchanged


def test_build_shadow_embeddings_does_not_mutate_current():
    current = np.array([[1.0, 0.0]], dtype=np.float32)
    original = current.copy()
    emb_a = np.array([[9.0, 9.0]], dtype=np.float32)
    build_shadow_embeddings(current, emb_a, _shift_patch, np.array([0]), np.array(["patch"], dtype=object))
    np.testing.assert_array_equal(current, original)


def test_build_shadow_leaves_non_flagged_documents_untouched():
    current = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    emb_a = np.array([[9.0, 9.0], [9.0, 9.0], [9.0, 9.0]], dtype=np.float32)
    # Only document 1 is in the flagged/plan set; 0 and 2 were never candidates.
    shadow = build_shadow_embeddings(current, emb_a, _shift_patch, np.array([1]), np.array(["full_reembed"], dtype=object))
    np.testing.assert_array_equal(shadow[0], current[0])
    np.testing.assert_array_equal(shadow[2], current[2])
    np.testing.assert_array_equal(shadow[1], emb_a[1])


def test_select_golden_query_indices_picks_highest_traffic():
    queries = [_Query("q0"), _Query("q1"), _Query("q2")]
    traffic = {"q0": 0.1, "q1": 0.9, "q2": 0.3}
    indices = select_golden_query_indices(queries, traffic, n=2)
    picked_ids = {queries[i].query_id for i in indices}
    assert picked_ids == {"q1", "q2"}  # the two highest-traffic queries


def test_select_golden_query_indices_caps_at_available_queries():
    queries = [_Query("q0"), _Query("q1")]
    indices = select_golden_query_indices(queries, {"q0": 1.0, "q1": 1.0}, n=50)
    assert len(indices) == 2


def test_execute_migration_passes_and_swaps_when_shadow_improves():
    doc_ids = ["d0", "d1"]
    current = np.array([[0.5, 0.0], [0.0, -1.0]], dtype=np.float32)  # d1 currently misaligned with the query
    emb_a = np.array([[0.5, 0.0], [1.0, 0.0]], dtype=np.float32)  # true fix aligns d1 with the query
    flagged_indices = np.array([1])
    plan_actions = np.array(["full_reembed"], dtype=object)

    query = _Query("q0", "query")
    query_embedding = np.array([[1.0, 0.0]], dtype=np.float32)
    qrels_eval = {"q0": {"d1"}}
    query_traffic = {"q0": 1.0}

    report = execute_migration(
        current, emb_a, _shift_patch, flagged_indices, plan_actions, doc_ids, [query], query_embedding, qrels_eval,
        query_traffic, regression_tolerance=0.0,
    )

    assert report.verification_passed is True
    assert report.golden_ndcg_after >= report.golden_ndcg_before
    np.testing.assert_array_equal(report.live_embeddings[1], emb_a[1])  # cutover happened
    assert report.n_full_reembed == 1


def test_execute_migration_aborts_and_rolls_back_when_shadow_regresses():
    doc_ids = ["d0", "d1"]
    # d1's CURRENT (contaminated) embedding happens to coincidentally align
    # perfectly with the golden query; its "true" full-reembed target
    # (emb_a) aligns worse for THIS specific query -- a realistic edge
    # case (a global fix isn't guaranteed to help every individual query).
    current = np.array([[0.5, 0.0], [1.0, 0.0]], dtype=np.float32)
    emb_a = np.array([[0.5, 0.0], [-1.0, 0.0]], dtype=np.float32)
    flagged_indices = np.array([1])
    plan_actions = np.array(["full_reembed"], dtype=object)

    query = _Query("q0", "query")
    query_embedding = np.array([[1.0, 0.0]], dtype=np.float32)
    qrels_eval = {"q0": {"d1"}}
    query_traffic = {"q0": 1.0}

    report = execute_migration(
        current, emb_a, _shift_patch, flagged_indices, plan_actions, doc_ids, [query], query_embedding, qrels_eval,
        query_traffic, regression_tolerance=0.0,
    )

    assert report.verification_passed is False
    assert report.golden_ndcg_after < report.golden_ndcg_before
    np.testing.assert_array_equal(report.live_embeddings, current)  # rolled back, NOT the shadow


def test_gate_ships_only_waves_shown_not_to_be_worse():
    rng = np.random.default_rng(0)
    # No change at all: safe.
    assert wave_is_unsafe(np.zeros(20), 0.01, 0.95, 2000, rng)[0] is False
    # Clear improvement: safe.
    assert wave_is_unsafe(np.full(20, 0.2), 0.01, 0.95, 2000, rng)[0] is False
    # A small wobble well inside the margin: safe.
    assert wave_is_unsafe(np.array([-0.05] + [0.0] * 19), 0.01, 0.95, 2000, rng)[0] is False
    # A drop that cannot be ruled out -- not convincing, but not excluded
    # either -- is NOT shipped (the first gate version shipped it).
    assert wave_is_unsafe(np.array([-0.3, -0.3, 0.1] + [0.0] * 17), 0.01, 0.95, 2000, rng)[0] is True


def _staged_fixture(n_x_queries=1):
    # Ten zero-vector filler documents (score 0 for every query, so search
    # always has more than k=10 candidates) plus two target documents:
    #   doc 10 ("dx") is relevant to every "x" query, doc 11 ("dy") to "qy".
    # Both start contaminated (pointing away from their queries, so they rank
    # below the fillers); emb_a holds the true vectors that bring them back.
    doc_ids = [f"f{i}" for i in range(10)] + ["dx", "dy"]
    current = np.zeros((12, 2), dtype=np.float32)
    current[10], current[11] = [-1.0, 0.0], [0.0, -1.0]
    emb_a = current.copy()
    emb_a[10], emb_a[11] = [1.0, 0.0], [0.0, 1.0]
    queries = [_Query(f"qx{i}") for i in range(n_x_queries)] + [_Query("qy")]
    query_embeddings = np.array([[1.0, 0.0]] * n_x_queries + [[0.0, 1.0]], dtype=np.float32)
    qrels = {**{f"qx{i}": {"dx"} for i in range(n_x_queries)}, "qy": {"dy"}}
    return doc_ids, current, emb_a, queries, query_embeddings, qrels


def test_staged_migration_ships_good_waves_in_priority_order():
    doc_ids, current, emb_a, queries, query_embeddings, qrels = _staged_fixture()
    report = execute_staged_migration(
        current, emb_a, _shift_patch, np.array([10, 11]), np.array(["full_reembed", "full_reembed"], dtype=object),
        np.array([1, 0]), labelled_canary_scorer(doc_ids, queries, query_embeddings, qrels), n_waves=2,
    )
    assert report.applied.all()
    # Local index 1 (doc 11) was listed first, so it ships in wave 1.
    assert [(w.wave, w.n_documents) for w in report.waves] == [(1, 1), (2, 1)]
    assert report.waves[0].canary_after > report.waves[0].canary_before
    np.testing.assert_array_equal(report.live_embeddings, emb_a)
    assert report.canary_after == 1.0 and report.canary_before == 0.0
    np.testing.assert_array_equal(current[10], [-1.0, 0.0])  # never mutated in place


def test_staged_migration_rolls_back_only_the_harmful_part_of_a_wave():
    # dx starts CORRECT here and the plan's "fix" for it breaks it -- e.g. a
    # false positive. Eight golden queries depend on dx, one on dy. Shipped
    # as one wave the canary sees a convincing regression; bisection must
    # keep dy's genuine fix and roll back only dx's.
    doc_ids, current, emb_a, queries, query_embeddings, qrels = _staged_fixture(n_x_queries=8)
    current[10] = [1.0, 0.0]
    harmful = emb_a.copy()
    harmful[10] = [-1.0, 0.0]
    plan = np.array(["full_reembed", "full_reembed"], dtype=object)
    report = execute_staged_migration(
        current, harmful, _shift_patch, np.array([10, 11]), plan, np.array([1, 0]),
        labelled_canary_scorer(doc_ids, queries, query_embeddings, qrels), n_waves=1, bisect_depth=1,
    )
    assert [w.passed for w in report.waves] == [False, True, False]  # whole wave, then its two halves
    assert report.applied.tolist() == [False, True]
    assert report.n_rolled_back == 1
    np.testing.assert_array_equal(report.live_embeddings[11], emb_a[11])
    np.testing.assert_array_equal(report.live_embeddings[10], current[10])


def test_replay_canary_rewards_restoring_and_penalizes_displacing_logged_results():
    rng = np.random.default_rng(0)
    clean = rng.normal(size=(50, 8)).astype(np.float32)
    clean /= np.linalg.norm(clean, axis=1, keepdims=True)
    queries = clean[:20] + rng.normal(scale=0.05, size=(20, 8)).astype(np.float32)
    canary = replay_canary_scorer(queries, clean, k=5)
    np.testing.assert_allclose(canary(clean), 1.0)  # the pre-incident index reproduces its own logs

    broken = clean.copy()
    broken[:10] = -clean[:10]  # documents 0-9 vanish from their queries' results
    assert canary(broken).mean() < canary(clean).mean()
    restored = broken.copy()
    restored[:10] = clean[:10]
    np.testing.assert_allclose(canary(restored), 1.0)
