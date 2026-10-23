"""Fast, network-free unit tests for phase2.migrate's shadow-build,
golden-query selection, and verify-gated cutover."""

import numpy as np

from phase2.migrate import build_shadow_embeddings, execute_migration, select_golden_query_indices


class _Query:
    def __init__(self, query_id, text=""):
        self.query_id = query_id
        self.text = text


def test_build_shadow_embeddings_applies_actions_correctly():
    current = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    emb_a = np.array([[9.0, 9.0], [8.0, 8.0], [7.0, 7.0]], dtype=np.float32)
    patched = np.array([[5.0, 5.0], [4.0, 4.0], [3.0, 3.0]], dtype=np.float32)
    flagged_indices = np.array([0, 1, 2])
    actions = np.array(["full_reembed", "patch", "defer"], dtype=object)

    shadow = build_shadow_embeddings(current, emb_a, patched, flagged_indices, actions)

    np.testing.assert_array_equal(shadow[0], emb_a[0])  # full_reembed -> true embedding
    np.testing.assert_array_equal(shadow[1], patched[1])  # patch -> patched embedding
    np.testing.assert_array_equal(shadow[2], current[2])  # defer -> unchanged


def test_build_shadow_embeddings_does_not_mutate_current():
    current = np.array([[1.0, 0.0]], dtype=np.float32)
    original = current.copy()
    emb_a = np.array([[9.0, 9.0]], dtype=np.float32)
    patched = np.array([[5.0, 5.0]], dtype=np.float32)
    build_shadow_embeddings(current, emb_a, patched, np.array([0]), np.array(["full_reembed"], dtype=object))
    np.testing.assert_array_equal(current, original)


def test_build_shadow_leaves_non_flagged_documents_untouched():
    current = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    emb_a = np.array([[9.0, 9.0], [9.0, 9.0], [9.0, 9.0]], dtype=np.float32)
    patched = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]], dtype=np.float32)
    # Only document 1 is in the flagged/plan set; 0 and 2 were never candidates.
    shadow = build_shadow_embeddings(current, emb_a, patched, np.array([1]), np.array(["full_reembed"], dtype=object))
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
    patched = current.copy()  # unused in this test (full_reembed only)
    flagged_indices = np.array([1])
    plan_actions = np.array(["full_reembed"], dtype=object)

    query = _Query("q0", "query")
    query_embedding = np.array([[1.0, 0.0]], dtype=np.float32)
    qrels_eval = {"q0": {"d1"}}
    query_traffic = {"q0": 1.0}

    report = execute_migration(
        current, emb_a, patched, flagged_indices, plan_actions, doc_ids, [query], query_embedding, qrels_eval,
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
    patched = current.copy()
    flagged_indices = np.array([1])
    plan_actions = np.array(["full_reembed"], dtype=object)

    query = _Query("q0", "query")
    query_embedding = np.array([[1.0, 0.0]], dtype=np.float32)
    qrels_eval = {"q0": {"d1"}}
    query_traffic = {"q0": 1.0}

    report = execute_migration(
        current, emb_a, patched, flagged_indices, plan_actions, doc_ids, [query], query_embedding, qrels_eval,
        query_traffic, regression_tolerance=0.0,
    )

    assert report.verification_passed is False
    assert report.golden_ndcg_after < report.golden_ndcg_before
    np.testing.assert_array_equal(report.live_embeddings, current)  # rolled back, NOT the shadow
