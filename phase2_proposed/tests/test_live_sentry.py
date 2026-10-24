"""Fast, network-free unit tests for phase2.live_sentry's autonomous
monitor engine, using a small synthetic benchmark (not the real BEIR
pipeline) so this runs in the default `pytest` suite."""

import numpy as np
import pytest

from common.benchmark import Benchmark
from common.config import Config
from common.data import Document, Query
from phase2.live_sentry import (
    MAX_CONSECUTIVE_FAILURES,
    LiveSentryContext,
    LiveSentryState,
    STATUS_ALERT,
    STATUS_NEEDS_REVIEW,
    STATUS_REMEDIATING,
    STATUS_RESOLVED,
    STATUS_STABLE,
    run_one_tick,
)
from phase2.migrate import labelled_canary_scorer
from phase2.pipeline import build_phase2_context


def _synthetic_context(n=40, dim=8, seed=0, initial_budget_level=1.0, regression_tolerance=1.0) -> LiveSentryContext:
    rng = np.random.default_rng(seed)

    # Generation A: a tight, well-behaved cluster around one direction.
    mu = rng.normal(size=dim)
    mu /= np.linalg.norm(mu)
    emb_a = mu + rng.normal(scale=0.03, size=(n, dim))
    emb_a /= np.linalg.norm(emb_a, axis=1, keepdims=True)
    emb_a = emb_a.astype(np.float32)

    # Generation B: scattered, clearly different directions (easy to flag).
    emb_b = rng.normal(size=(n, dim))
    emb_b /= np.linalg.norm(emb_b, axis=1, keepdims=True)
    emb_b = emb_b.astype(np.float32)

    documents = [Document(doc_id=f"d{i}", title=f"title{i}", text=f"text{i}") for i in range(n)]
    query = Query(query_id="q0", text="golden query")
    doc_traffic = np.zeros(n, dtype=np.float32)
    doc_traffic[0] = 1.0  # doc0 is the traffic-heavy, golden-relevant document
    qrels_eval = {"q0": {"d0"}}
    query_traffic = {"q0": 1.0}

    benchmark = Benchmark(
        documents=documents,
        queries=[query],
        qrels_eval=qrels_eval,
        query_traffic=query_traffic,
        doc_traffic=doc_traffic,
        emb_a=emb_a,
        emb_b=emb_b,
    )

    cfg = Config(seed=seed)
    cfg.detection.knn_k = 5
    cfg.remediation.calibration_sample_size = 20
    cfg.fingerprint.k = 5
    cfg.live_sentry.initial_budget_level = initial_budget_level
    cfg.migration.golden_query_count = 1
    cfg.migration.regression_tolerance = regression_tolerance

    doc_ids = [d.doc_id for d in documents]
    # query embedding == emb_a[0]'s direction, so it retrieves d0 (its relevant doc) easily pre-incident.
    query_embedding = emb_a[0:1].copy()

    return LiveSentryContext(
        cfg=cfg,
        benchmark=benchmark,
        doc_ids=doc_ids,
        eval_queries=[query],
        query_embeddings=query_embedding,
        canary=labelled_canary_scorer(doc_ids, [query], query_embedding, qrels_eval),
        canary_description="labelled canary: 1 query",
        canary_size=1,
        # No query log in this synthetic benchmark, so the replay calibration
        # is skipped and the configured recovery constant is used instead.
        phase2=build_phase2_context(cfg, benchmark),
    )


def test_state_starts_stable_with_a_log_entry():
    ctx = _synthetic_context()
    state = LiveSentryState(ctx.benchmark)
    snap = state.snapshot()
    assert snap["status"] == STATUS_STABLE
    assert len(snap["event_log"]) == 1
    assert not snap["incident_active"]


def test_inject_contamination_marks_incident_active_and_logs():
    ctx = _synthetic_context()
    state = LiveSentryState(ctx.benchmark)
    n_injected = state.inject_contamination(ctx, batch_fraction=0.2, seed=1)

    assert n_injected == pytest.approx(8, abs=1)  # ~20% of 40
    snap = state.snapshot()
    assert snap["incident_active"] is True
    assert any("Simulated embedding provider update" in msg for _, msg in snap["event_log"])


def test_inject_contamination_only_affects_currently_clean_documents():
    ctx = _synthetic_context()
    state = LiveSentryState(ctx.benchmark)
    state.inject_contamination(ctx, batch_fraction=0.5, seed=1)
    contaminated_after_first = np.any(state.db_b_embeddings != ctx.benchmark.emb_a, axis=1)
    n_contaminated_first = int(contaminated_after_first.sum())

    # A second injection should only ever add MORE contaminated docs, never
    # "double contaminate" or reduce the count.
    state.inject_contamination(ctx, batch_fraction=0.5, seed=2)
    contaminated_after_second = np.any(state.db_b_embeddings != ctx.benchmark.emb_a, axis=1)
    assert int(contaminated_after_second.sum()) >= n_contaminated_first


def test_fingerprint_detector_flags_almost_none_of_the_clean_control():
    ctx = _synthetic_context()
    _, flagged_clean = ctx.phase2.detector.flag(ctx.benchmark.emb_a, ctx.cfg.fingerprint.threshold)
    _, flagged_b = ctx.phase2.detector.flag(ctx.benchmark.emb_b, ctx.cfg.fingerprint.threshold)
    assert float(np.mean(flagged_clean)) <= 0.05
    # This fixture's generation B is random directions in only 8 dimensions,
    # so a few land near generation A's cluster by chance; the real 384-d
    # benchmark separates far more cleanly (see run_detection.py).
    assert float(np.mean(flagged_b)) >= 0.8


def test_run_one_tick_does_nothing_when_no_incident_active():
    ctx = _synthetic_context()
    state = LiveSentryState(ctx.benchmark)
    run_one_tick(state, ctx)
    snap = state.snapshot()
    assert snap["status"] == STATUS_STABLE
    assert snap["tick_count"] == 1
    # Database B starts identical to Database A, so its flagged count
    # should reflect that (real baseline noise), not a hardcoded zero.
    assert snap["last_db_b_flagged"] == snap["last_db_a_flagged"]
    assert not snap["incident_active"]


def test_run_one_tick_detects_injected_contamination():
    ctx = _synthetic_context()
    state = LiveSentryState(ctx.benchmark)
    state.inject_contamination(ctx, batch_fraction=0.2, seed=1)
    run_one_tick(state, ctx)
    snap = state.snapshot()
    assert snap["last_db_b_flagged"] > 0
    assert snap["status"] in (STATUS_ALERT, STATUS_REMEDIATING, STATUS_RESOLVED)


def test_incident_eventually_resolves():
    ctx = _synthetic_context()
    state = LiveSentryState(ctx.benchmark)
    state.inject_contamination(ctx, batch_fraction=0.2, seed=1)

    for _ in range(5):  # generous cap; generous initial budget should resolve within 1-2 ticks
        run_one_tick(state, ctx)
        if state.snapshot()["status"] == STATUS_RESOLVED:
            break

    snap = state.snapshot()
    assert snap["status"] == STATUS_RESOLVED
    assert snap["incident_active"] is False
    # "Resolved" is relative to Database A's own baseline noise floor, not
    # literally zero -- see live_sentry.py's run_one_tick for why.
    assert snap["last_db_b_flagged"] <= snap["last_db_a_flagged"]


def test_incident_resolves_with_a_constrained_per_tick_budget():
    # Regression test for a real bug found running the full-scale (5000
    # doc) dashboard: with budget_units computed from the SHRINKING
    # remaining-candidate count each tick, the absolute per-tick budget
    # eventually dropped below the cost of even one fix, stalling forever
    # a few documents short of resolution ("Plan: 0 full re-embeds",
    # verification trivially passing on a no-op). budget_units must be
    # sized off a STABLE reference (the incident's initial flagged count)
    # instead. A constrained budget_level=0.3 (not the generous 1.0 the
    # other tests use) is what actually exercises this shrinking-pool
    # effect -- see DEVLOG.md's sub-phase 2D entry.
    ctx = _synthetic_context(n=200, initial_budget_level=0.3)
    state = LiveSentryState(ctx.benchmark)
    state.inject_contamination(ctx, batch_fraction=0.2, seed=1)

    for _ in range(30):  # generous cap; a constrained budget takes several ticks
        run_one_tick(state, ctx)
        if state.snapshot()["status"] == STATUS_RESOLVED:
            break

    snap = state.snapshot()
    assert snap["status"] == STATUS_RESOLVED
    assert snap["incident_active"] is False


def test_already_remediated_documents_are_not_rescheduled():
    # Regression test for a related bug: a document already successfully
    # remediated (verification passed) must not be handed to the
    # scheduler again on a later tick, even if it's still flagged (a
    # persistent statistical false positive re-"fixing" it again would be
    # a no-op that wins the zero-benefit tie-break every time, starving
    # genuinely-still-contaminated documents of budget indefinitely).
    ctx = _synthetic_context(n=200, initial_budget_level=0.3)
    state = LiveSentryState(ctx.benchmark)
    state.inject_contamination(ctx, batch_fraction=0.2, seed=1)

    run_one_tick(state, ctx)
    remediated_after_tick_1 = state.db_b_remediated_once.copy()
    assert remediated_after_tick_1.any()  # at least one document was actually acted on

    run_one_tick(state, ctx)
    # Once remediated, a document stays marked remediated (monotonic --
    # never un-marked) even if still flagged next tick.
    assert np.all(state.db_b_remediated_once[remediated_after_tick_1])


def test_repeated_verification_failure_defers_instead_of_looping_forever():
    # Regression test for a real bug found running the full-scale (5000
    # doc) dashboard: escalating budget on verification failure only helps
    # when budget scarcity IS the reason it failed. If the identical
    # candidate set produces the identical golden-query regression
    # regardless of budget (2B's real run showed this happens -- a fix can
    # shift ranking for an unrelated query), escalating forever at an
    # already-maxed budget is a no-op that stalls the incident permanently.
    # Force deterministic failure via an impossible regression_tolerance
    # (requires nDCG to IMPROVE by more than 1.0, which can never happen)
    # and confirm the monitor gives up after MAX_CONSECUTIVE_FAILURES
    # rather than looping without end.
    ctx = _synthetic_context(initial_budget_level=1.0, regression_tolerance=-2.0)
    state = LiveSentryState(ctx.benchmark)
    state.inject_contamination(ctx, batch_fraction=0.2, seed=1)

    for _ in range(MAX_CONSECUTIVE_FAILURES):
        run_one_tick(state, ctx)
        assert state.snapshot()["status"] != STATUS_RESOLVED  # not resolved yet -- still failing

    # One more tick should trip the give-up threshold.
    run_one_tick(state, ctx)
    assert state.db_b_deferred.any()
    assert state.consecutive_verification_failures == 0

    # Deferred candidates are excluded from further scheduling, but residual
    # anomalies must remain visibly blocked for manual review, not be called
    # resolved.
    for _ in range(3):
        if state.snapshot()["status"] == STATUS_NEEDS_REVIEW:
            break
        run_one_tick(state, ctx)
    snap = state.snapshot()
    assert snap["status"] == STATUS_NEEDS_REVIEW
    assert snap["last_db_b_flagged"] > snap["last_db_a_flagged"]
    assert snap["incident_active"] is False


def test_every_tick_records_history_and_reset_clears_it():
    ctx = _synthetic_context()
    state = LiveSentryState(ctx.benchmark)
    run_one_tick(state, ctx)
    state.inject_contamination(ctx, batch_fraction=0.2, seed=1)
    run_one_tick(state, ctx)

    history = state.snapshot()["history"]
    assert [point["tick"] for point in history] == [1, 2]
    assert history[-1]["db_b_flagged"] > history[-1]["db_a_flagged"]

    state.reset(ctx.benchmark)
    assert state.snapshot()["history"] == []
