"""Fast, network-free tests for the production package: the full calibrate ->
scan -> dry-run -> apply -> rollback flow on an in-memory store and on
Qdrant's embedded mode, plus the guard, the query log and the state store.

Two fake "embedding models" stand in for real ones: model A maps each text
to a fixed random direction (documents on related topics share a cluster);
model B is an affine distortion of model A, like a different real model."""

import numpy as np
import pytest

from vectorsentry import InMemoryStore, QdrantStore, QueryLog, SentryConfig, StateStore, VectorSentry
from vectorsentry.querylog import LoggedQuery

DIM, N_DOCS, N_TOPICS = 64, 240, 12


def _unit(x):
    return (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)


class FakeEmbedder:
    def __init__(self, model_id, table, cost=1.0):
        self.model_id, self.table, self.cost_per_document = model_id, table, cost
        self.calls = 0

    def embed(self, texts):
        self.calls += len(texts)
        return np.stack([self.table[t] for t in texts]) if texts else np.zeros((0, DIM), np.float32)


@pytest.fixture()
def world():
    rng = np.random.default_rng(0)
    topics = _unit(rng.normal(size=(N_TOPICS, DIM)))
    ids = [f"d{i:03d}" for i in range(N_DOCS)]
    texts = [f"document {i} about topic {i % N_TOPICS}" for i in range(N_DOCS)]
    vec_a = _unit(topics[np.arange(N_DOCS) % N_TOPICS] + 0.35 * rng.normal(size=(N_DOCS, DIM)))
    q, _ = np.linalg.qr(rng.normal(size=(DIM, DIM)))
    offset = 0.9 * rng.normal(size=DIM)
    to_b = lambda v: _unit(v @ q + offset + 0.03 * rng.normal(size=v.shape))  # noqa: E731

    # 60 logged queries, each near one document; logged results = top-10 on the CLEAN index.
    query_doc = rng.choice(N_DOCS, 60, replace=False)
    query_texts = [f"query about {texts[j]}" for j in query_doc]
    q_vec = _unit(vec_a[query_doc] + 0.1 * rng.normal(size=(60, DIM)))
    top = np.argsort(-(q_vec @ vec_a.T), axis=1)[:, :10]
    counts = 1000.0 / np.arange(1, 61)
    log = QueryLog([LoggedQuery(t, float(c), [ids[j] for j in top[i]]) for i, (t, c) in enumerate(zip(query_texts, counts))])

    table_a = {**dict(zip(texts, vec_a)), **dict(zip(query_texts, q_vec))}
    table_b = {**dict(zip(texts, to_b(vec_a))), **dict(zip(query_texts, to_b(q_vec)))}
    serving, suspect = FakeEmbedder("model-a", table_a), FakeEmbedder("model-b", table_b)

    contaminated = rng.random(N_DOCS) < 0.25
    stored = np.where(contaminated[:, None], np.stack([table_b[t] for t in texts]), vec_a)
    return dict(ids=ids, texts=texts, vec_a=vec_a, stored=stored, contaminated=contaminated,
                log=log, serving=serving, suspect=suspect)


def _sentry(world, store, **cfg):
    # A 60-document probe batch over 12 topics leaves ~5 same-model probe
    # vectors per topic, so the fingerprint's kNN vote must use k <= 5: with
    # the default k=10 every vote is forced to include other topics' vectors,
    # some from the other model. (300 probes on real data clear this easily.)
    config = SentryConfig(probe_size=60, fingerprint_k=5, budget=cfg.pop("budget", 1000.0), canary_queries=60,
                          bootstrap_samples=300, **cfg)
    return VectorSentry(store, world["serving"], world["suspect"], world["log"], StateStore(), config, collection="test")


def test_scan_finds_the_contaminated_documents(world):
    sentry = _sentry(world, InMemoryStore(world["ids"], world["stored"], world["texts"]))
    scan = sentry.scan()
    agreement = np.mean(scan.flagged == world["contaminated"])
    assert agreement > 0.95
    assert scan.health < 1.0  # contamination is visible to the replay canary


def test_dry_run_records_a_plan_and_changes_nothing(world):
    store = InMemoryStore(world["ids"], world["stored"], world["texts"])
    sentry = _sentry(world, store, budget=20.0)
    before = store.vectors.copy()
    calls_before = world["serving"].calls
    report = sentry.run(apply=False)

    np.testing.assert_array_equal(store.vectors, before)
    assert report.mode == "dry-run" and report.n_full + report.n_patch > 0
    assert report.cost_planned <= 20.0 + 1e-9
    # Dry-run embeds only the probe batch and canary queries, never document fixes.
    assert world["serving"].calls - calls_before <= 60 + 60 + 60
    run = sentry.state.run(report.run_id)
    assert run["mode"] == "dry-run" and run["status"] == "done"
    assert len(sentry.state.actions(report.run_id)) == report.n_flagged


def test_apply_repairs_the_index_and_rollback_restores_it_exactly(world):
    store = InMemoryStore(world["ids"], world["stored"], world["texts"])
    sentry = _sentry(world, store, budget=1000.0, tolerance=0.05)
    original = store.vectors.copy()

    report = sentry.run(apply=True)
    assert report.health_after > report.health_before
    assert report.n_full + report.n_patch > 0
    # Fully re-embedded documents now hold exactly the serving model's vector.
    reembedded = [store.ids.index(a["doc_id"]) for a in sentry.state.actions(report.run_id)
                  if a["applied"] and a["action"] == "full_reembed"]
    np.testing.assert_allclose(store.vectors[reembedded], world["vec_a"][reembedded], atol=1e-6)
    assert all(store.model_tags[world["ids"][i]] == "model-a" for i in reembedded)
    assert report.embed_cost >= report.n_full  # paid only for documents actually re-embedded

    restored = sentry.rollback(report.run_id)
    assert restored == report.n_full + report.n_patch
    np.testing.assert_allclose(store.vectors, original, atol=1e-6)
    assert sentry.state.run(report.run_id)["status"] == "rolled-back"
    with pytest.raises(ValueError):
        sentry.rollback(report.run_id)


def test_a_tight_budget_is_respected_and_spent_on_busy_documents(world):
    store = InMemoryStore(world["ids"], world["stored"], world["texts"])
    sentry = _sentry(world, store, budget=5.0)
    report = sentry.run(apply=False)
    assert report.cost_planned <= 5.0 + 1e-9
    acted = [a for a in report.actions if a["action"] != "defer"]
    deferred = [a for a in report.actions if a["action"] == "defer"]
    assert acted and deferred
    assert np.mean([a["traffic"] for a in acted]) > np.mean([a["traffic"] for a in deferred])


def test_guard_rejects_suspect_model_vectors_and_accepts_serving_ones(world):
    sentry = _sentry(world, InMemoryStore(world["ids"], world["stored"], world["texts"]))
    guard = sentry.guard()
    sample = world["texts"][:80]
    _, ok_a = guard.check(world["serving"].embed(sample))
    _, ok_b = guard.check(world["suspect"].embed(sample))
    assert ok_a.mean() > 0.95 and ok_b.mean() < 0.05
    ids, vecs, report = guard.filter(["x", "y"], np.stack([world["serving"].embed(sample[:1])[0], world["suspect"].embed(sample[:1])[0]]))
    assert ids == ["x"] and report.rejected_ids == ["y"] and len(vecs) == 1


def test_qdrant_local_mode_end_to_end(world):
    qdrant_client = pytest.importorskip("qdrant_client")
    client = qdrant_client.QdrantClient(":memory:")
    store = QdrantStore.create(client, "docs", dim=DIM)
    models = ["model-b" if c else "model-a" for c in world["contaminated"]]
    store.add_documents(world["ids"], world["stored"], world["texts"], models)

    snap = store.fetch_all()
    assert snap.ids == sorted(world["ids"]) and snap.vectors.shape == (N_DOCS, DIM)

    sentry = _sentry(world, store, tolerance=0.05)
    report = sentry.run(apply=True)
    assert report.health_after > report.health_before
    after = store.fetch_all()
    changed = [a["doc_id"] for a in sentry.state.actions(report.run_id) if a["applied"]]
    point = client.retrieve("docs", ids=[QdrantStore.point_id(changed[0])], with_payload=True)[0]
    assert point.payload["model_id"].startswith("model-a")
    assert not np.allclose(after.vectors, snap.vectors)

    sentry.rollback(report.run_id)
    np.testing.assert_allclose(store.fetch_all().vectors, snap.vectors, atol=1e-5)


def test_query_log_roundtrip_and_traffic(tmp_path, world):
    path = tmp_path / "log.jsonl"
    world["log"].save(path)
    loaded = QueryLog.load(path)
    assert len(loaded.entries) == 60 and loaded.entries[0].results == world["log"].entries[0].results
    traffic = loaded.document_traffic(world["ids"])
    first = loaded.entries[0]
    top_doc = world["ids"].index(first.results[0])
    assert traffic[top_doc] >= first.count  # rank 0 credits the full count
    assert traffic.min() >= 0


def test_state_store_rejects_unknown_fields():
    state = StateStore()
    run_id = state.start_run("scan")
    with pytest.raises(ValueError):
        state.update_run(run_id, not_a_field=1)
