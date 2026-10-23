"""Sub-phase 2D -- the autonomous "Live Sentry" engine: a two-database
live monitor. Database A stays untouched (the control); Database B is
where a simulated embedding-provider update gets injected. Everything
after that single manual trigger is autonomous: a background thread
re-scores both databases on a timer, raises an alert on its own when
Database B's contamination is detected, runs Phase 2's scheduler +
migrate/verify loop (2A/2B) on its own, and clears the alert once
verification passes.

This module has NO Streamlit dependency -- `LiveSentryState` and
`run_one_tick` are plain Python or synthetic embeddings, and swap in a
real background thread only in the dashboard.

Design choice -- calibrated alert threshold, not an assumed ratio: 1C/2A's
detectors threshold by an ASSUMED contamination ratio (a real operator's
rough prevalence estimate from a known migration window). A continuous
live monitor doesn't have that -- it doesn't know in advance how big the
next incident will be, if there even is one. So instead: take Database
A's (known-clean) KNN-OOD score distribution as the "normal" baseline,
and alert on any Database B document scoring above a high percentile of
it. This is standard anomaly-monitoring practice (calibrate normal, alert
on deviation) and is exactly what having a stable control database is
FOR -- see DEVLOG.md's sub-phase 2D entry.
"""

from __future__ import annotations

import dataclasses
import datetime
import threading

import numpy as np

from common.benchmark import Benchmark, build_benchmark
from common.config import Config
from common.data import Query
from common.embeddings import embed_texts
from common.patch import apply_patch, fit_patch_transform, sample_calibration_indices
from phase1.detection import knn_ood_scores
from phase2.migrate import execute_migration, select_golden_query_indices
from phase2.scheduler import build_efficient_frontier_segments, compute_severity, solve_greedy

STATUS_STABLE = "stable"
STATUS_ALERT = "alert"
STATUS_REMEDIATING = "remediating"
STATUS_RESOLVED = "resolved"

# How many consecutive verification failures AT MAXIMUM BUDGET for the same
# candidate set before the monitor gives up and defers it for manual
# review, rather than retrying an identical failing plan forever -- see
# run_one_tick's verification-failure handling.
MAX_CONSECUTIVE_FAILURES = 3


@dataclasses.dataclass
class LiveSentryContext:
    """Everything the monitor loop needs, built ONCE (expensive: loads the
    benchmark, fits the patch transform, embeds queries) and then shared
    read-only across ticks."""

    cfg: Config
    benchmark: Benchmark
    doc_ids: list[str]
    eval_queries: list[Query]
    query_embeddings: np.ndarray
    golden_queries: list[Query]
    golden_query_embeddings: np.ndarray
    patched_emb_b: np.ndarray


def build_context(cfg: Config) -> LiveSentryContext:
    benchmark = build_benchmark(cfg)
    doc_ids = [d.doc_id for d in benchmark.documents]
    eval_queries = [q for q in benchmark.queries if q.query_id in benchmark.qrels_eval]
    query_embeddings = embed_texts([q.text for q in eval_queries], cfg.embedding_models.generation_a)

    calibration_indices = sample_calibration_indices(
        len(benchmark.documents), cfg.remediation.calibration_sample_size, cfg.seed
    )
    transform = fit_patch_transform(benchmark.emb_a, benchmark.emb_b, calibration_indices)
    patched_emb_b = apply_patch(benchmark.emb_b, transform)

    golden_indices = select_golden_query_indices(eval_queries, benchmark.query_traffic, cfg.migration.golden_query_count)
    golden_queries = [eval_queries[i] for i in golden_indices]
    golden_query_embeddings = query_embeddings[golden_indices]

    return LiveSentryContext(
        cfg, benchmark, doc_ids, eval_queries, query_embeddings, golden_queries, golden_query_embeddings, patched_emb_b
    )


class LiveSentryState:
    """Thread-safe shared state: the background monitor thread writes to
    it every tick; the UI thread reads a snapshot every render and writes
    to it only via `inject_contamination()` (the one manual control)."""

    def __init__(self, benchmark: Benchmark, poll_interval_seconds: float = 4.0):
        self._lock = threading.Lock()
        self.db_a_embeddings = benchmark.emb_a.copy()  # stable control, never modified after init
        self.db_b_embeddings = benchmark.emb_a.copy()  # starts clean; incidents injected via inject_contamination()
        self.status = STATUS_STABLE
        self.event_log: list[tuple[str, str]] = []
        self.tick_count = 0
        self.last_db_a_flagged = 0
        self.last_db_b_flagged = 0
        self.incident_active = False
        self.current_budget_level = 0.0
        # Stable reference for per-tick budget sizing -- see run_one_tick's
        # budget_units computation for why this must NOT be recomputed
        # from the shrinking remaining-candidate count each tick (it would
        # eventually starve the tail end of an incident: as candidates run
        # low, budget_level * n_candidates can drop below the cost of even
        # one fix, stalling forever with a trivially-passing no-op
        # "verification"). Set once, when an incident is first detected.
        self.incident_initial_flagged_count = 0
        self.stop_requested = False
        # Tracks which Database B documents have already had a remediation
        # action APPLIED (verification passed) at least once -- realistic
        # operational history (a system knows what it already fixed), not
        # oracle ground truth. Excluding these from future scheduling
        # rounds is what stops a persistent statistical false-positive
        # (structurally always going to sit above threshold, same as some
        # fraction of Database A's own baseline) from winning the
        # scheduler's zero-benefit tie-break every single tick and
        # starving genuinely-still-contaminated documents of budget
        # indefinitely -- see DEVLOG.md's sub-phase 2D entry.
        self.db_b_remediated_once = np.zeros(len(self.db_b_embeddings), dtype=bool)
        # Documents the scheduler gave up on after repeated verification
        # failures at maximum budget -- see run_one_tick: escalating
        # budget only helps when the abort reason IS budget scarcity. If
        # the same candidate set produces the same golden-query regression
        # regardless of budget (2B found this really happens -- a fix can
        # shift ranking for an unrelated query), escalating forever is a
        # no-op that would otherwise stall the incident permanently at
        # max budget. Excluded from scheduling like db_b_remediated_once,
        # but logged distinctly as needing manual review, not as fixed.
        self.db_b_deferred = np.zeros(len(self.db_b_embeddings), dtype=bool)
        self.consecutive_verification_failures = 0
        # Mutable, lock-protected -- NOT read from Streamlit's session_state
        # by the background thread, which is a plain Python thread with no
        # Streamlit script-run context. The UI's slider calls
        # `set_poll_interval()` instead; `monitor_loop` reads it via
        # `get_poll_interval()`.
        self._poll_interval_seconds = poll_interval_seconds
        self._log(STATUS_STABLE, "Live Sentry started. Monitoring Database A (stable) and Database B (live).")

    def _log(self, status: str, message: str) -> None:
        # Caller must already hold self._lock.
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.event_log.append((timestamp, message))
        self.status = status

    def snapshot(self) -> dict:
        """A read-only copy of everything the UI needs to render one frame."""
        with self._lock:
            return {
                "status": self.status,
                "event_log": list(self.event_log),
                "tick_count": self.tick_count,
                "last_db_a_flagged": self.last_db_a_flagged,
                "last_db_b_flagged": self.last_db_b_flagged,
                "incident_active": self.incident_active,
                "current_budget_level": self.current_budget_level,
                "poll_interval_seconds": self._poll_interval_seconds,
                "n_deferred": int(self.db_b_deferred.sum()),
            }

    def get_poll_interval(self) -> float:
        with self._lock:
            return self._poll_interval_seconds

    def set_poll_interval(self, seconds: float) -> None:
        with self._lock:
            self._poll_interval_seconds = seconds

    def inject_contamination(self, ctx: LiveSentryContext, batch_fraction: float, seed: int | None = None) -> int:
        """The ONE manual control: simulates an embedding-provider update
        by overwriting a random batch of Database B's CURRENTLY-CLEAN
        documents with their real generation-B embeddings. Returns the
        number of documents injected. Detection, alerting, and
        remediation are all left to the background loop -- this method
        does nothing autonomous itself.
        """
        rng = np.random.default_rng(seed)
        with self._lock:
            n = len(self.db_b_embeddings)
            clean_mask = np.all(self.db_b_embeddings == ctx.benchmark.emb_a, axis=1)
            clean_indices = np.flatnonzero(clean_mask)
            batch_size = min(int(round(batch_fraction * n)), len(clean_indices))
            if batch_size == 0:
                self._log(self.status, "Simulated provider update requested, but no clean documents remain to affect.")
                return 0
            batch = rng.choice(clean_indices, size=batch_size, replace=False)
            self.db_b_embeddings[batch] = ctx.benchmark.emb_b[batch]
            self.incident_active = True
            self.current_budget_level = ctx.cfg.live_sentry.initial_budget_level
            self._log(
                self.status,
                f"Simulated embedding provider update: {batch_size} documents in Database B "
                f"re-embedded with a new provider.",
            )
            return batch_size

    def reset(self, benchmark: Benchmark) -> None:
        with self._lock:
            self.db_b_embeddings = benchmark.emb_a.copy()
            self.incident_active = False
            self.current_budget_level = 0.0
            self.incident_initial_flagged_count = 0
            self.last_db_b_flagged = 0
            self.db_b_remediated_once = np.zeros(len(self.db_b_embeddings), dtype=bool)
            self.db_b_deferred = np.zeros(len(self.db_b_embeddings), dtype=bool)
            self.consecutive_verification_failures = 0
            self._log(STATUS_STABLE, "Reset: Database B restored to clean state.")


def compute_alert_threshold(clean_embeddings: np.ndarray, k: int, percentile: float) -> float:
    """Calibrates the alert threshold from Database A's (known-clean) own
    KNN-OOD score distribution -- see module docstring."""
    scores = knn_ood_scores(clean_embeddings, k=k)
    return float(np.percentile(scores, percentile))


def run_one_tick(state: LiveSentryState, ctx: LiveSentryContext) -> None:
    """One iteration of the autonomous monitor loop: score both databases,
    and if Database B has an active incident, run detection -> schedule ->
    migrate/verify, applying the result only if verification passes.
    Safe to call repeatedly from a background thread; every state mutation
    goes through `state._lock`.
    """
    with state._lock:
        db_a = state.db_a_embeddings.copy()
        db_b = state.db_b_embeddings.copy()
        incident_active = state.incident_active
        budget_level = state.current_budget_level

    threshold = compute_alert_threshold(db_a, ctx.cfg.detection.knn_k, ctx.cfg.live_sentry.alert_threshold_percentile)
    scores_a = knn_ood_scores(db_a, k=ctx.cfg.detection.knn_k)
    n_flagged_a = int(np.sum(scores_a > threshold))

    # Score Database B even with no active incident -- its flagged count
    # should reflect reality (its own baseline noise, comparable to A's)
    # rather than being hardcoded, which would misrepresent a resolved
    # database as having literally zero anomalies.
    scores_b = knn_ood_scores(db_b, k=ctx.cfg.detection.knn_k)
    flagged_mask = scores_b > threshold
    flagged_indices = np.flatnonzero(flagged_mask)
    n_flagged_b = len(flagged_indices)

    if not incident_active:
        with state._lock:
            state.last_db_a_flagged = n_flagged_a
            state.last_db_b_flagged = n_flagged_b
            state.tick_count += 1
        return

    with state._lock:
        already_remediated = state.db_b_remediated_once.copy()
        deferred = state.db_b_deferred.copy()

    # Only documents flagged, never yet remediated, and not given up on
    # are real scheduling candidates:
    # - already_remediated: a document already re-embedded once (real
    #   operational history, not oracle information) that's STILL flagged
    #   is a statistical false positive -- re-fixing it again would be an
    #   exact no-op, and letting it keep winning the scheduler's
    #   zero-benefit tie-break every tick starves genuinely-still-
    #   contaminated documents of budget indefinitely.
    # - deferred: verification kept failing for this exact candidate set
    #   even at maximum budget -- see the verification-failure handling
    #   below for why retrying it forever would otherwise be a no-op.
    # See DEVLOG.md's sub-phase 2D entry for the real runs that exposed both.
    candidate_indices = np.flatnonzero(flagged_mask & ~already_remediated & ~deferred)
    n_candidates = len(candidate_indices)

    with state._lock:
        state.last_db_a_flagged = n_flagged_a
        state.last_db_b_flagged = n_flagged_b
        state.tick_count += 1

        # "Resolved" is relative to Database A's OWN baseline flagged count,
        # not literally zero: a percentile threshold calibrated from A's
        # distribution will, by construction, flag roughly (100-percentile)%
        # of ANY similarly-distributed sample -- including A's own re-scored
        # baseline every tick, and the still-clean majority of B. Requiring
        # exactly zero would mean these expected false positives keep the
        # incident "active" forever. Comparing against A's own
        # concurrently-measured noise floor is what a real anomaly monitor
        # does, and is what makes "back to nominal" well-defined here.
        # Additionally: if nothing flagged remains eligible for a FRESH
        # remediation attempt, further ticks cannot improve things either
        # -- also resolve, honestly noting why.
        if n_flagged_b <= n_flagged_a or n_candidates == 0:
            state.incident_active = False
            state.current_budget_level = 0.0
            if n_flagged_b <= n_flagged_a:
                state._log(STATUS_RESOLVED, "Incident resolved -- Database B back to nominal. Alert cleared.")
            else:
                state._log(
                    STATUS_RESOLVED,
                    f"All {n_flagged_b} remaining flagged documents have already been remediated or "
                    f"deferred; consistent with the detector's own baseline noise, not further-fixable "
                    f"right now. Alert cleared.",
                )
            return

        if state.status in (STATUS_STABLE, STATUS_RESOLVED):
            state._log(STATUS_ALERT, f"ALERT: Database B contamination detected ({n_flagged_b} documents flagged).")
            state.incident_initial_flagged_count = n_flagged_b

        state._log(
            STATUS_REMEDIATING,
            f"Tick {state.tick_count}: building prioritized schedule for {n_candidates} flagged documents "
            f"not yet remediated (budget level {budget_level:.2f})...",
        )
        incident_initial_flagged_count = max(state.incident_initial_flagged_count, 1)

    severity = compute_severity(scores_b[candidate_indices])
    traffic = ctx.benchmark.doc_traffic[candidate_indices]
    segments = build_efficient_frontier_segments(
        severity,
        traffic,
        ctx.cfg.remediation.cost_full_reembed,
        ctx.cfg.remediation.cost_patch,
        ctx.cfg.scheduler.recovery_full_reembed,
        ctx.cfg.scheduler.recovery_patch,
    )
    # Sized off the STABLE incident_initial_flagged_count, not the
    # shrinking n_candidates -- see LiveSentryState's field comment.
    budget_units = budget_level * incident_initial_flagged_count * ctx.cfg.remediation.cost_full_reembed
    plan = solve_greedy(segments, n_items=n_candidates, budget=budget_units)
    n_full = int(np.sum(plan.actions == "full_reembed"))
    n_patch = int(np.sum(plan.actions == "patch"))

    with state._lock:
        state._log(state.status, f"Plan: {n_full} full re-embeds, {n_patch} patches this tick.")

    report = execute_migration(
        db_b,
        ctx.benchmark.emb_a,
        ctx.patched_emb_b,
        candidate_indices,
        plan.actions,
        ctx.doc_ids,
        ctx.golden_queries,
        ctx.golden_query_embeddings,
        ctx.benchmark.qrels_eval,
        ctx.benchmark.query_traffic,
        regression_tolerance=ctx.cfg.migration.regression_tolerance,
    )

    with state._lock:
        if report.verification_passed:
            state.db_b_embeddings = report.live_embeddings
            acted_mask = plan.actions != "defer"
            state.db_b_remediated_once[candidate_indices[acted_mask]] = True
            state.consecutive_verification_failures = 0
            state._log(
                state.status,
                f"Verification PASSED (golden nDCG {report.golden_ndcg_before:.3f} -> "
                f"{report.golden_ndcg_after:.3f}). Cutover applied.",
            )
        else:
            state.consecutive_verification_failures += 1
            # Escalating budget only helps when the abort reason IS budget
            # scarcity (a bigger budget lets the scheduler afford a better
            # plan). If the SAME candidate set keeps producing the SAME
            # golden-query regression regardless of budget -- 2B's real
            # run showed this really happens, since a fix can shift
            # ranking for an unrelated query, not just under-fund one --
            # escalating forever at an already-maxed budget is a no-op
            # that would stall the incident permanently. Give up on this
            # candidate set instead: a real automated pipeline would flag
            # it for manual review rather than retry an identical failing
            # plan forever.
            already_at_max_budget = budget_level >= 1.0 - 1e-9
            if already_at_max_budget and state.consecutive_verification_failures >= MAX_CONSECUTIVE_FAILURES:
                state.db_b_deferred[candidate_indices] = True
                state.consecutive_verification_failures = 0
                state.current_budget_level = ctx.cfg.live_sentry.initial_budget_level
                state._log(
                    state.status,
                    f"Verification failed {MAX_CONSECUTIVE_FAILURES} times in a row at maximum budget for "
                    f"this exact candidate set ({len(candidate_indices)} documents) -- the fix itself, not "
                    f"the budget, is the blocker. Deferring for manual review; budget reset to "
                    f"{ctx.cfg.live_sentry.initial_budget_level:.2f} for any other documents.",
                )
            else:
                next_budget = min(1.0, budget_level + ctx.cfg.live_sentry.budget_escalation_step)
                state.current_budget_level = next_budget
                state._log(
                    state.status,
                    f"Verification FAILED (golden nDCG would drop {report.golden_ndcg_before:.3f} -> "
                    f"{report.golden_ndcg_after:.3f}). Migration ABORTED this tick; "
                    f"escalating to budget level {next_budget:.2f} next tick.",
                )


def monitor_loop(state: LiveSentryState, ctx: LiveSentryContext) -> None:
    """Background-thread target: calls `run_one_tick` forever, sleeping
    `state.get_poll_interval()` seconds between ticks, until
    `state.stop_requested`. Reads the interval fresh each iteration (not
    once at startup) so a live UI slider change takes effect without
    restarting the thread.
    """
    import time

    while True:
        with state._lock:
            if state.stop_requested:
                return
        run_one_tick(state, ctx)
        time.sleep(max(0.1, state.get_poll_interval()))
