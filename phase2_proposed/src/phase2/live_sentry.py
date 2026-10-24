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

Detection -- generation fingerprint, not a percentile of KNN-OOD scores:
a continuous monitor cannot assume a contamination ratio (it does not know
how big the next incident is, or whether there is one). The original
monitor calibrated a 99th-percentile KNN-OOD threshold on Database A, which
by construction raised ~1% false alarms on ANY database and, on the real
benchmark, caught only a small fraction of an injected incident (the
contaminated documents are each other's neighbours, so they look normal).
It now uses Phase 2's fingerprint detector (phase2/fingerprint.py):
P(generation B) > 0.5 against the probe batch, which needs no ratio and
raises almost no false alarms on the clean control. Database A is still
scored every tick as the control: "back to nominal" means Database B
flags no more documents than Database A does.

Remediation per tick is the same Phase 2 pipeline the sweep uses
(phase2/pipeline.py): expected-benefit MCKP, then a staged, verified
rollout (phase2.migrate.execute_staged_migration), so one bad slice of a
plan is rolled back without blocking the rest.
"""

from __future__ import annotations

import dataclasses
import datetime
import threading
from typing import Callable

import numpy as np

from common.benchmark import Benchmark, build_benchmark
from common.config import Config
from common.data import Query
from phase2.migrate import execute_staged_migration
from phase2.pipeline import Phase2Context, build_canary, build_phase2_context, plan_remediation, remediate

STATUS_STABLE = "stable"
STATUS_ALERT = "alert"
STATUS_REMEDIATING = "remediating"
STATUS_RESOLVED = "resolved"
STATUS_NEEDS_REVIEW = "needs_review"

# How many consecutive verification failures AT MAXIMUM BUDGET for the same
# candidate set before the monitor gives up and defers it for manual
# review, rather than retrying an identical failing plan forever -- see
# run_one_tick's verification-failure handling.
MAX_CONSECUTIVE_FAILURES = 3

# Clean-up retries per patched document before it is left on its patch.
MAX_UPGRADE_ATTEMPTS = 3

# Per-tick flagged counts kept for the dashboard's "over time" chart --
# bounded so a long-running demo doesn't grow memory without limit.
HISTORY_MAX_POINTS = 400


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
    canary: Callable[[np.ndarray], np.ndarray]  # per-query gate score for a candidate index (phase2.pipeline.build_canary)
    canary_description: str
    canary_size: int  # number of canary queries, for display
    phase2: Phase2Context  # fingerprint detector, adapter, replay-estimated action effects, traffic


def build_context(cfg: Config) -> LiveSentryContext:
    benchmark = build_benchmark(cfg)
    doc_ids = [d.doc_id for d in benchmark.documents]
    eval_rows = [i for i, q in enumerate(benchmark.queries) if q.query_id in benchmark.qrels_eval]
    eval_queries = [benchmark.queries[i] for i in eval_rows]
    query_embeddings = benchmark.query_emb_a[eval_rows]
    canary, description = build_canary(cfg, benchmark)

    return LiveSentryContext(
        cfg,
        benchmark,
        doc_ids,
        eval_queries,
        query_embeddings,
        canary,
        description,
        len(canary(benchmark.emb_a)),
        build_phase2_context(cfg, benchmark),
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
        self.incident_injected_count = 0
        self.incident_initial_remediated_count = 0
        self.last_incident_summary: dict | None = None
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
        # Documents whose live vector is a quick patch, an interim fix. Once an
        # incident is resolved the monitor upgrades them to exact re-embeds in
        # the background (see run_one_tick), busiest first.
        self.db_b_patched = np.zeros(len(self.db_b_embeddings), dtype=bool)
        # How many times the clean-up's exact re-embed of a patched document
        # has been rolled back; it is retried on later ticks, up to
        # MAX_UPGRADE_ATTEMPTS, because a rejection is usually transient
        # (other still-patched documents briefly shift the logged results).
        self.db_b_upgrade_attempts = np.zeros(len(self.db_b_embeddings), dtype=np.int16)
        self.consecutive_verification_failures = 0
        self.last_verification_before: float | None = None
        self.last_verification_after: float | None = None
        self.last_verification_passed: bool | None = None
        self.last_plan_rows: list[dict] = []
        self.last_plan_budget = 0.0
        self.last_plan_cost = 0.0
        self.last_plan_benefit = 0.0
        self.last_plan_candidate_count = 0
        self.demo_initial_budget_level: float | None = None
        self.demo_regression_tolerance: float | None = None
        # For the dashboard's charts: the live database's search health (the
        # canary score -- share of pre-incident results its logged queries
        # still get back, 1.0 = healthy), the latest rollout's per-batch
        # safety-check results, and the budget spent so far this incident.
        self.last_health: float | None = None
        # Which live documents the detector flagged on the latest check (for the web UI's index map).
        self.last_flagged_b = np.zeros(len(self.db_b_embeddings), dtype=bool)
        self.last_waves: list[dict] = []
        self.incident_budget_spent = 0.0
        # Every repair round's plan (rows, budget, waves) this incident, so the
        # dashboard can step through the decisions after the fact.
        self.incident_rounds: list[dict] = []
        # Mutable, lock-protected -- NOT read from Streamlit's session_state
        # by the background thread, which is a plain Python thread with no
        # Streamlit script-run context. The UI's slider calls
        # `set_poll_interval()` instead; `monitor_loop` reads it via
        # `get_poll_interval()`.
        self._poll_interval_seconds = poll_interval_seconds
        self.history: list[dict] = []
        self._log(STATUS_STABLE, "Live Sentry started. Monitoring Database A (stable) and Database B (live).")

    def _log(self, status: str, message: str) -> None:
        # Caller must already hold self._lock.
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.event_log.append((timestamp, message))
        self.status = status

    def _record_history(self) -> None:
        # Caller must already hold self._lock.
        self.history.append(
            {
                "time": datetime.datetime.now(),
                "tick": self.tick_count,
                "db_a_flagged": self.last_db_a_flagged,
                "db_b_flagged": self.last_db_b_flagged,
                "health": self.last_health,
                "n_remediated": int(self.db_b_remediated_once.sum()),
                "budget_spent": self.incident_budget_spent,
                "status": self.status,
            }
        )
        del self.history[:-HISTORY_MAX_POINTS]

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
                "n_remediated": int(self.db_b_remediated_once.sum()),
                "n_patched": int(self.db_b_patched.sum()),
                "last_verification_before": self.last_verification_before,
                "last_verification_after": self.last_verification_after,
                "last_verification_passed": self.last_verification_passed,
                "incident_initial_flagged_count": self.incident_initial_flagged_count,
                "incident_injected_count": self.incident_injected_count,
                "last_incident_summary": dict(self.last_incident_summary) if self.last_incident_summary else None,
                "last_plan_rows": list(self.last_plan_rows),
                "last_plan_budget": self.last_plan_budget,
                "last_plan_cost": self.last_plan_cost,
                "last_plan_benefit": self.last_plan_benefit,
                "last_plan_candidate_count": self.last_plan_candidate_count,
                "demo_initial_budget_level": self.demo_initial_budget_level,
                "demo_regression_tolerance": self.demo_regression_tolerance,
                "history": list(self.history),
                "last_health": self.last_health,
                "last_waves": list(self.last_waves),
                "incident_budget_spent": self.incident_budget_spent,
                "incident_rounds": list(self.incident_rounds),
            }

    def embedding_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Return consistent copies of both live databases for visualization.

        The dashboard uses this to show how the live embedding space moves
        relative to the untouched control without reading mutable arrays while
        the monitor thread is applying a cutover.
        """
        with self._lock:
            return self.db_a_embeddings.copy(), self.db_b_embeddings.copy()

    def get_poll_interval(self) -> float:
        with self._lock:
            return self._poll_interval_seconds

    def set_poll_interval(self, seconds: float) -> None:
        with self._lock:
            self._poll_interval_seconds = seconds

    def configure_demo(self, initial_budget_level: float, regression_tolerance: float | None) -> None:
        """Set run-level demo controls before the next incident starts."""
        with self._lock:
            self.demo_initial_budget_level = float(np.clip(initial_budget_level, 0.1, 1.0))
            self.demo_regression_tolerance = regression_tolerance

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
            self.current_budget_level = (
                ctx.cfg.live_sentry.initial_budget_level
                if self.demo_initial_budget_level is None
                else self.demo_initial_budget_level
            )
            self.incident_injected_count = batch_size
            self.incident_initial_remediated_count = int(self.db_b_remediated_once.sum())
            self.incident_budget_spent = 0.0
            self.last_waves = []
            self.incident_rounds = []
            self.last_incident_summary = None
            self.last_plan_rows = []
            self.last_plan_budget = 0.0
            self.last_plan_cost = 0.0
            self.last_plan_benefit = 0.0
            self.last_plan_candidate_count = 0
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
            self.incident_injected_count = 0
            self.incident_initial_remediated_count = 0
            self.last_incident_summary = None
            self.last_db_b_flagged = 0
            self.db_b_remediated_once = np.zeros(len(self.db_b_embeddings), dtype=bool)
            self.db_b_deferred = np.zeros(len(self.db_b_embeddings), dtype=bool)
            self.db_b_patched = np.zeros(len(self.db_b_embeddings), dtype=bool)
            self.db_b_upgrade_attempts = np.zeros(len(self.db_b_embeddings), dtype=np.int16)
            self.consecutive_verification_failures = 0
            self.last_verification_before = None
            self.last_verification_after = None
            self.last_verification_passed = None
            self.last_plan_rows = []
            self.last_plan_budget = 0.0
            self.last_plan_cost = 0.0
            self.last_plan_benefit = 0.0
            self.last_plan_candidate_count = 0
            self.history = []
            self.last_waves = []
            self.incident_budget_spent = 0.0
            self.incident_rounds = []
            self._log(STATUS_STABLE, "Reset: Database B restored to clean state.")


def run_one_tick(state: LiveSentryState, ctx: LiveSentryContext) -> None:
    """One iteration of the autonomous monitor loop: score both databases,
    and if Database B has an active incident, run detection -> schedule ->
    staged migrate/verify, applying whichever waves pass verification.
    Safe to call repeatedly from a background thread; every state mutation
    goes through `state._lock`.
    """
    with state._lock:
        db_a = state.db_a_embeddings.copy()
        db_b = state.db_b_embeddings.copy()
        incident_active = state.incident_active
        budget_level = state.current_budget_level
        initial_budget_level = (
            ctx.cfg.live_sentry.initial_budget_level
            if state.demo_initial_budget_level is None
            else state.demo_initial_budget_level
        )
        regression_tolerance = state.demo_regression_tolerance

    detector, threshold = ctx.phase2.detector, ctx.cfg.fingerprint.threshold
    # Database A is the control: its flag count is the detector's own
    # false-alarm floor, re-measured every tick.
    _, flagged_a = detector.flag(db_a, threshold)
    n_flagged_a = int(flagged_a.sum())

    # Score Database B even with no active incident -- its flagged count
    # should reflect reality (its own baseline noise, comparable to A's)
    # rather than being hardcoded, which would misrepresent a resolved
    # database as having literally zero anomalies.
    p_b, flagged_mask = detector.flag(db_b, threshold)
    flagged_indices = np.flatnonzero(flagged_mask)
    n_flagged_b = len(flagged_indices)
    health = float(np.mean(ctx.canary(db_b)))

    if not incident_active:
        with state._lock:
            state.last_health = health
            state.last_flagged_b = flagged_mask
            state.last_db_a_flagged = n_flagged_a
            state.last_db_b_flagged = n_flagged_b
            state.tick_count += 1
            state._record_history()
            if state.status == STATUS_NEEDS_REVIEW and n_flagged_b <= n_flagged_a and not state.db_b_deferred.any():
                if state.last_incident_summary:
                    state.last_incident_summary["final_flagged"] = n_flagged_b
                state._log(STATUS_RESOLVED, "Detector nominal -- the clean-up cleared the remaining flagged documents.")
            patched = np.flatnonzero(state.db_b_patched)
        if len(patched):
            _upgrade_patches(state, ctx, db_b, patched, initial_budget_level)
        return

    with state._lock:
        already_remediated = state.db_b_remediated_once.copy()
        deferred = state.db_b_deferred.copy()
        patched_now = state.db_b_patched.copy()

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
    # Exception: a document carrying a quick PATCH that the detector still
    # flags is not a no-op to fix again -- an exact re-embed would cure it --
    # so it stays a candidate, with the patch option withheld. (Excluding it
    # paused a real Tight-budget run for "human review" with 51 such
    # documents left.)
    retry_patched = already_remediated & patched_now
    candidate_indices = np.flatnonzero(flagged_mask & ~deferred & (~already_remediated | retry_patched))
    n_candidates = len(candidate_indices)

    with state._lock:
        state.last_health = health
        state.last_flagged_b = flagged_mask
        state.last_db_a_flagged = n_flagged_a
        state.last_db_b_flagged = n_flagged_b
        state.tick_count += 1
        state._record_history()

        # "Resolved" is relative to Database A's OWN baseline flagged count,
        # not literally zero: a percentile threshold calibrated from A's
        # distribution will, by construction, flag roughly (100-percentile)%
        # of ANY similarly-distributed sample -- including A's own re-scored
        # baseline every tick, and the still-clean majority of B. Requiring
        # exactly zero would mean these expected false positives keep the
        # incident "active" forever. Comparing against A's own
        # concurrently-measured noise floor is what a real anomaly monitor
        # does, and is what makes "back to nominal" well-defined here.
        # If flagged documents remain but none is eligible for a fresh
        # remediation attempt, the monitor stops safely and reports manual
        # review. That is deliberately NOT a resolved state.
        if n_flagged_b <= n_flagged_a:
            state.incident_active = False
            state.current_budget_level = 0.0
            state.last_incident_summary = {
                "injected": state.incident_injected_count,
                "initial_flagged": state.incident_initial_flagged_count,
                "final_flagged": n_flagged_b,
                "remediated": int(state.db_b_remediated_once.sum()) - state.incident_initial_remediated_count,
                "deferred": int(state.db_b_deferred.sum()),
            }
            state._log(
                STATUS_RESOLVED,
                "Detector nominal -- Database B is back within control noise. "
                "This does not prove every contaminated vector was detected.",
            )
            return

        if n_candidates == 0:
            state.incident_active = False
            state.current_budget_level = 0.0
            state.last_incident_summary = {
                "injected": state.incident_injected_count,
                "initial_flagged": state.incident_initial_flagged_count,
                "final_flagged": n_flagged_b,
                "remediated": int(state.db_b_remediated_once.sum()) - state.incident_initial_remediated_count,
                "deferred": int(state.db_b_deferred.sum()),
            }
            state._log(
                STATUS_NEEDS_REVIEW,
                f"Automation paused: Database B still has {n_flagged_b} flagged documents versus "
                f"{n_flagged_a} in the stable control, but all remaining candidates are deferred or "
                f"already attempted. Manual review required; this is not a clean resolution.",
            )
            return

        if state.status in (STATUS_STABLE, STATUS_RESOLVED, STATUS_NEEDS_REVIEW):
            state._log(STATUS_ALERT, f"ALERT: Database B contamination detected ({n_flagged_b} documents flagged).")
            state.incident_initial_flagged_count = n_flagged_b

        state._log(
            STATUS_REMEDIATING,
            f"Tick {state.tick_count}: building prioritized schedule for {n_candidates} flagged documents "
            f"not yet remediated (budget level {budget_level:.2f})...",
        )
        incident_initial_flagged_count = max(state.incident_initial_flagged_count, 1)

    # Sized off the STABLE incident_initial_flagged_count, not the
    # shrinking n_candidates -- see LiveSentryState's field comment.
    budget_units = budget_level * incident_initial_flagged_count * ctx.cfg.remediation.cost_full_reembed
    cfg = ctx.cfg
    if regression_tolerance is not None:  # the dashboard's per-run gate setting
        cfg = dataclasses.replace(cfg, migration=dataclasses.replace(cfg.migration, regression_tolerance=regression_tolerance))
    # Plan -> staged, verified rollout -> recycle rolled-back budget, exactly
    # as in the offline sweep (phase2.pipeline.remediate).
    outcome = remediate(
        cfg, ctx.phase2, db_b, ctx.benchmark.emb_a, candidate_indices, p_b[candidate_indices], budget_units,
        ctx.canary, seed=cfg.seed + state.tick_count, allow_patch=~patched_now[candidate_indices],
    )
    remediation = (
        outcome.rounds[0][0]
        if outcome.rounds
        else plan_remediation(cfg, ctx.phase2, candidate_indices, p_b[candidate_indices], budget_units)
    )
    plan = remediation.plan
    n_full = int(np.sum(plan.actions == "full_reembed"))
    n_patch = int(np.sum(plan.actions == "patch"))

    recovery_by_action = {
        "full_reembed": ctx.cfg.scheduler.recovery_full_reembed,
        "patch": ctx.phase2.recovery_patch,
        "defer": ctx.cfg.scheduler.recovery_defer,
    }
    segment_by_choice = {}
    for segment in remediation.segments:
        segment_by_choice[(segment.local_index, segment.to_action)] = segment
    # Raw impression traffic credits every one of a query's 10 results, so it
    # sums to ~4.5 across the corpus, not 1 -- normalize before calling it a
    # share. "x average" stays readable across the long tail, where shares
    # are thousandths of a percent.
    traffic_total = float(ctx.phase2.traffic.sum())
    traffic_mean = float(ctx.phase2.traffic.mean())
    rollout_rank = {int(local): rank for rank, local in enumerate(remediation.rollout_order, start=1)}
    plan_rows = []
    for local_index, action in enumerate(plan.actions):
        segment = segment_by_choice.get((local_index, action))
        expected = {
            "full_reembed": remediation.benefit_full[local_index],
            "patch": remediation.benefit_patch[local_index],
        }.get(action, 0.0)
        plan_rows.append(
            {
                "doc_id": ctx.doc_ids[candidate_indices[local_index]],
                "action": action,
                # P(contaminated) from the fingerprint detector (the dashboard's "Chance it is broken" column).
                "severity": float(remediation.p_contaminated[local_index]),
                "traffic_share": float(ctx.phase2.traffic[candidate_indices[local_index]]) / traffic_total,
                "traffic_vs_average": float(ctx.phase2.traffic[candidate_indices[local_index]]) / traffic_mean,
                "rollout_rank": rollout_rank.get(local_index),  # 1 = shipped in the first wave; None = deferred
                "doc_index": int(candidate_indices[local_index]),
                "cost": float(ctx.cfg.remediation.cost_full_reembed if action == "full_reembed" else ctx.cfg.remediation.cost_patch if action == "patch" else 0.0),
                "expected_recovery": float(recovery_by_action[action]),
                "expected_benefit": float(expected),
                "priority_ratio": float(segment.ratio if segment is not None else 0.0),
            }
        )
    plan_rows.sort(key=lambda row: (row["action"] == "defer", -row["priority_ratio"], -row["expected_benefit"]))

    with state._lock:
        state.last_plan_rows = plan_rows
        state.last_plan_budget = float(budget_units)
        state.last_plan_cost = float(plan.total_cost)
        state.last_plan_benefit = float(plan.total_benefit)
        state.last_plan_candidate_count = n_candidates
        state._log(state.status, f"Plan: {n_full} full re-embeds, {n_patch} patches this tick.")

    acted = bool(outcome.rounds)
    passed = (not acted) or bool(outcome.applied.any())
    canary_before = outcome.rounds[0][1].canary_before if acted else None
    canary_after = outcome.rounds[-1][1].canary_after if acted else None
    waves = [w for _, report in outcome.rounds for w in report.waves]

    with state._lock:
        state.last_verification_before = canary_before
        state.last_verification_after = canary_after
        state.last_verification_passed = passed
        state.incident_budget_spent += outcome.cost_spent
        state.last_waves = [
            {
                "round": round_number,
                "wave": w.wave,
                "depth": w.depth,
                "n_documents": w.n_documents,
                "passed": w.passed,
                "canary_before": w.canary_before,
                "canary_after": w.canary_after,
            }
            for round_number, (_, report) in enumerate(outcome.rounds, start=1)
            for w in report.waves
        ]
        if plan_rows:
            state.incident_rounds.append(
                {"rows": plan_rows, "budget": float(budget_units), "waves": list(state.last_waves)}
            )
        if passed:
            state.db_b_embeddings = outcome.live_embeddings
            state.db_b_remediated_once[candidate_indices[outcome.applied]] = True
            state.db_b_patched[candidate_indices[outcome.action_applied == "patch"]] = True
            state.db_b_patched[candidate_indices[outcome.action_applied == "full_reembed"]] = False
            state.consecutive_verification_failures = 0
            rolled_back = (
                f" {outcome.n_rolled_back} fix(es) rolled back by the safety check." if outcome.n_rolled_back else ""
            )
            if acted:
                state._log(
                    state.status,
                    f"Verification PASSED for {sum(w.passed for w in waves)} of {len(waves)} rollout wave(s) "
                    f"(canary {canary_before:.3f} -> {canary_after:.3f}). Cutover applied: "
                    f"{outcome.n_full_reembed} re-embedded, {outcome.n_patch} patched.{rolled_back}",
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
                state.current_budget_level = initial_budget_level
                state._log(
                    state.status,
                    f"Verification failed {MAX_CONSECUTIVE_FAILURES} times in a row at maximum budget for "
                    f"this exact candidate set ({len(candidate_indices)} documents) -- the fix itself, not "
                    f"the budget, is the blocker. Deferring for manual review; budget reset to "
                    f"{initial_budget_level:.2f} for any other documents.",
                )
            else:
                next_budget = min(1.0, budget_level + ctx.cfg.live_sentry.budget_escalation_step)
                state.current_budget_level = next_budget
                state._log(
                    state.status,
                    f"Verification FAILED (canary would drop {canary_before:.3f} -> "
                    f"{min(w.canary_after for w in waves):.3f}). Migration ABORTED this tick; "
                    f"escalating to budget level {next_budget:.2f} next tick.",
                )


def _upgrade_patches(
    state: LiveSentryState, ctx: LiveSentryContext, db_b: np.ndarray, patched: np.ndarray, budget_level: float
) -> None:
    """Post-incident clean-up: a quick patch is an interim mitigation (close
    to the right vector, not exact), so once the incident is resolved the
    monitor replaces patches with exact re-embeds, busiest documents first,
    one round's budget per tick, through the same verified staged rollout."""
    per_tick = max(1, int(round(budget_level * max(state.incident_initial_flagged_count, 1))))
    # Fewest failed attempts first, then busiest, so a document whose
    # re-embed was just rejected waits behind ones not yet tried.
    with state._lock:
        attempts = state.db_b_upgrade_attempts[patched].copy()
    order = patched[np.lexsort((-ctx.phase2.traffic[patched], attempts))][:per_tick]
    report = execute_staged_migration(
        db_b, ctx.benchmark.emb_a, ctx.phase2.adapter, order, np.full(len(order), "full_reembed", dtype=object),
        np.arange(len(order)), ctx.canary,
        n_waves=ctx.cfg.migration.n_waves,
        regression_tolerance=ctx.cfg.migration.regression_tolerance,
        gate_confidence=ctx.cfg.migration.gate_confidence,
        bootstrap_samples=ctx.cfg.migration.bootstrap_samples,
        bisect_depth=ctx.cfg.migration.bisect_depth,
        seed=ctx.cfg.seed + state.tick_count,
    )
    with state._lock:
        if state.incident_active:  # a new incident started meanwhile; leave this for later
            return
        state.db_b_embeddings = report.live_embeddings
        state.db_b_patched[order[report.applied]] = False
        rejected = order[~report.applied]
        state.db_b_upgrade_attempts[rejected] += 1
        state.db_b_patched[rejected[state.db_b_upgrade_attempts[rejected] >= MAX_UPGRADE_ATTEMPTS]] = False
        state.incident_budget_spent += report.n_full_reembed * ctx.cfg.remediation.cost_full_reembed
        if report.n_full_reembed:
            state._log(
                state.status,
                f"Clean-up: upgraded {report.n_full_reembed} patched documents to exact re-embeds "
                f"(canary {report.canary_before:.3f} -> {report.canary_after:.3f}); "
                f"{int(state.db_b_patched.sum())} patches left.",
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
