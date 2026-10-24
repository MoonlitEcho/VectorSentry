"""Phase 2's full decision pipeline in one place, shared by the sweep
scripts and the Live Sentry so they cannot drift apart:

    probe batch --> fit adapter (patch action) + fingerprint detector
                --> replay the query log to estimate what a patch is worth
    stored index --> P(contaminated) per document           (DETECT)
                --> expected-benefit MCKP under the budget   (PRIORITIZE)
                --> priority-ordered rollout order           (MIGRATE, see phase2.migrate)

Everything here is built from information a real operator has: the probe
batch they dual-embedded on purpose, their query/impression logs, and the
stored vectors. Ground-truth contamination labels are never read.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from common.benchmark import Benchmark
from common.config import Config
from common.data import Query, zipf_query_weights
from common.patch import Adapter, fit_adapter, sample_calibration_indices
from phase2.calibration import ActionEffects, estimate_action_effects
from phase2.fingerprint import FingerprintDetector
from phase2.migrate import (
    StagedMigrationReport,
    execute_staged_migration,
    labelled_canary_scorer,
    replay_canary_scorer,
)
from phase2.scheduler import (
    MCKPSegment,
    SchedulePlan,
    build_frontier_segments_from_benefits,
    expected_benefits,
    priority_order,
    solve_greedy,
)


@dataclasses.dataclass
class Phase2Context:
    probe_indices: np.ndarray
    adapter: Adapter
    detector: FingerprintDetector
    effects: ActionEffects | None  # None when the replay had nothing to measure
    recovery_patch: float
    harm_patch: float
    traffic: np.ndarray  # (N,) document traffic with the small prior for never-shown documents


@dataclasses.dataclass
class RemediationPlan:
    candidate_indices: np.ndarray  # global document indices the plan covers
    p_contaminated: np.ndarray  # aligned to candidate_indices
    benefit_full: np.ndarray
    benefit_patch: np.ndarray
    segments: list[MCKPSegment]
    plan: SchedulePlan
    rollout_order: np.ndarray  # local indices into candidate_indices, most valuable first


@dataclasses.dataclass
class CanarySet:
    """The migration gate's golden queries, their serving-model embeddings,
    and their relevance labels."""

    queries: list[Query]
    embeddings: np.ndarray
    qrels: dict[str, set[str]]


def build_canary_set(benchmark: Benchmark, n: int) -> CanarySet:
    """The `n` most popular queries that carry TRAIN-split relevance labels
    and are NOT evaluation queries -- an ops team's curated regression suite
    built from historically labelled traffic. Keeping it disjoint from the
    test-split evaluation queries means the gate never gets to peek at the
    labels the final numbers are scored on (the original 2B canary was the
    top-20 evaluation queries themselves). Falls back to the evaluation
    queries only when no train-split labels exist (small synthetic tests)."""
    labelled = benchmark.qrels_all or {}
    rows = [
        i for i, q in enumerate(benchmark.queries)
        if q.query_id in labelled and q.query_id not in benchmark.qrels_eval
    ]
    qrels = labelled
    if not rows:
        rows = [i for i, q in enumerate(benchmark.queries) if q.query_id in benchmark.qrels_eval]
        qrels = benchmark.qrels_eval
    rows = sorted(rows, key=lambda i: -benchmark.query_traffic.get(benchmark.queries[i].query_id, 0.0))[:n]
    queries = [benchmark.queries[i] for i in rows]
    if benchmark.query_emb_a is not None:
        embeddings = benchmark.query_emb_a[rows]
    else:
        raise ValueError("build_canary_set needs benchmark.query_emb_a (serving-model query embeddings).")
    return CanarySet(queries, embeddings, {q.query_id: qrels[q.query_id] for q in queries})


def build_canary(cfg: Config, benchmark: Benchmark) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    """The migration gate's per-query scorer, per `migration.canary` (see
    phase2.migrate's canary block comment). Returns (scorer, description).

    The replay canary replays the `migration.canary_queries` most popular
    logged queries that are NOT evaluation queries against each candidate
    index, and compares with what they returned on the pre-incident index
    (generation A). Falls back to the labelled canary when the benchmark has
    no query log (small synthetic tests)."""
    if cfg.migration.canary == "replay" and benchmark.query_emb_a is not None:
        rows = [i for i, q in enumerate(benchmark.queries) if q.query_id not in benchmark.qrels_eval]
        rows = sorted(rows, key=lambda i: -benchmark.query_traffic.get(benchmark.queries[i].query_id, 0.0))
        rows = rows[: cfg.migration.canary_queries]
        scorer = replay_canary_scorer(benchmark.query_emb_a[rows], benchmark.emb_a, cfg.traffic.impression_depth)
        return scorer, f"replay canary: {len(rows)} logged non-evaluation queries vs. pre-incident results"
    canary_set = build_canary_set(benchmark, cfg.migration.golden_query_count)
    doc_ids = [d.doc_id for d in benchmark.documents]
    scorer = labelled_canary_scorer(doc_ids, canary_set.queries, canary_set.embeddings, canary_set.qrels)
    return scorer, f"labelled canary: {len(canary_set.queries)} queries, nDCG@10"


def traffic_with_floor(doc_traffic: np.ndarray, floor_fraction: float) -> np.ndarray:
    """Adds a small prior to every document's traffic (see
    SchedulerConfig.traffic_floor_fraction)."""
    doc_traffic = np.asarray(doc_traffic, dtype=np.float64)
    positive = doc_traffic[doc_traffic > 0]
    floor = floor_fraction * (positive.min() if len(positive) else 1.0)
    return doc_traffic + floor


def build_phase2_context(cfg: Config, benchmark: Benchmark) -> Phase2Context:
    probe = sample_calibration_indices(len(benchmark.documents), cfg.remediation.calibration_sample_size, cfg.seed)
    adapter = fit_adapter(benchmark.emb_a, benchmark.emb_b, probe, cfg.remediation.adapter, cfg.remediation.ridge_alpha)
    detector = FingerprintDetector.fit(benchmark.emb_a, benchmark.emb_b, probe, k=cfg.fingerprint.k)

    effects = None
    if cfg.scheduler.recovery_source == "replay" and benchmark.query_emb_a is not None:
        query_weights = zipf_query_weights(len(benchmark.queries), cfg.traffic.zipf_s, cfg.seed)
        effects = estimate_action_effects(
            benchmark.emb_a,
            benchmark.emb_b,
            probe,
            benchmark.query_emb_a,
            query_weights,
            adapter_kind=cfg.remediation.adapter,
            ridge_alpha=cfg.remediation.ridge_alpha,
            depth=cfg.traffic.impression_depth,
            seed=cfg.seed,
        )
    if effects is not None:
        recovery_patch, harm_patch = effects.recovery_patch, effects.harm_patch
    else:
        # Configured constant, and the conservative assumption that a patch
        # on a clean document undoes as much as it would have fixed.
        recovery_patch, harm_patch = cfg.scheduler.recovery_patch, cfg.scheduler.recovery_patch

    traffic = traffic_with_floor(benchmark.doc_traffic, cfg.scheduler.traffic_floor_fraction)
    return Phase2Context(probe, adapter, detector, effects, recovery_patch, harm_patch, traffic)


def plan_remediation(
    cfg: Config,
    ctx: Phase2Context,
    candidate_indices: np.ndarray,
    p_contaminated: np.ndarray,
    budget_units: float,
    allow_patch: np.ndarray | None = None,
) -> RemediationPlan:
    """Expected-benefit MCKP over `candidate_indices` (with their detector
    probabilities), solved by the efficient-frontier greedy. `allow_patch`
    (aligned to candidate_indices) removes the patch option for documents
    whose patch already failed verification."""
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    traffic = ctx.traffic[candidate_indices]
    benefit_full, benefit_patch = expected_benefits(
        p_contaminated, traffic, cfg.scheduler.recovery_full_reembed, ctx.recovery_patch, ctx.harm_patch
    )
    if allow_patch is not None:
        benefit_patch = np.where(allow_patch, benefit_patch, -np.inf)
    segments = build_frontier_segments_from_benefits(
        benefit_full, benefit_patch, cfg.remediation.cost_full_reembed, cfg.remediation.cost_patch
    )
    plan = solve_greedy(segments, n_items=len(candidate_indices), budget=budget_units)
    return RemediationPlan(
        candidate_indices,
        np.asarray(p_contaminated, dtype=np.float64),
        benefit_full,
        benefit_patch,
        segments,
        plan,
        priority_order(segments, plan),
    )


@dataclasses.dataclass
class RemediationOutcome:
    live_embeddings: np.ndarray
    applied: np.ndarray  # (n_candidates,) bool: a fix for this candidate is now live
    action_applied: np.ndarray  # (n_candidates,) str: "full_reembed" | "patch" | "defer"
    rounds: list[tuple[RemediationPlan, StagedMigrationReport]]
    cost_spent: float
    n_rolled_back: int  # candidates that had a fix rolled back and never got a passing one

    @property
    def n_full_reembed(self) -> int:
        return int(np.sum(self.action_applied == "full_reembed"))

    @property
    def n_patch(self) -> int:
        return int(np.sum(self.action_applied == "patch"))


def remediate(
    cfg: Config,
    ctx: Phase2Context,
    stored: np.ndarray,
    emb_a: np.ndarray,
    candidate_indices: np.ndarray,
    p_contaminated: np.ndarray,
    budget_units: float,
    canary: Callable[[np.ndarray], np.ndarray],
    seed: int = 0,
    allow_patch: np.ndarray | None = None,
) -> RemediationOutcome:
    """Plan -> staged, verified rollout -> recycle, for up to
    `migration.recycle_rounds` rounds.

    Budget is only spent on fixes that actually ship. When the gate rolls a
    fix back, its budget returns to the pool and the scheduler re-plans: a
    document whose PATCH was rejected loses the patch option and competes
    for a full re-embed instead (an exact restoration, so it cannot push
    other documents out of their logged results); a document whose full
    re-embed was rejected is left alone. Repeats until nothing is rolled
    back, the budget is spent, or the round limit is hit. `allow_patch`
    (aligned to candidate_indices) withholds the patch option up front, e.g.
    for documents already carrying a patch the detector still flags.
    """
    candidate_indices = np.asarray(candidate_indices, dtype=int)
    p_contaminated = np.asarray(p_contaminated, dtype=np.float64)
    n = len(candidate_indices)
    live = stored
    applied = np.zeros(n, dtype=bool)
    action_applied = np.full(n, "defer", dtype=object)
    allow_patch = np.ones(n, dtype=bool) if allow_patch is None else np.asarray(allow_patch, dtype=bool).copy()
    excluded = np.zeros(n, dtype=bool)
    ever_rolled_back = np.zeros(n, dtype=bool)
    remaining_budget = float(budget_units)
    rounds: list[tuple[RemediationPlan, StagedMigrationReport]] = []
    costs = {"full_reembed": cfg.remediation.cost_full_reembed, "patch": cfg.remediation.cost_patch}
    min_cost = min(costs.values())

    for round_number in range(max(1, cfg.migration.recycle_rounds)):
        open_ = np.flatnonzero(~applied & ~excluded)
        if len(open_) == 0 or remaining_budget < min_cost - 1e-9:
            break
        plan = plan_remediation(
            cfg, ctx, candidate_indices[open_], p_contaminated[open_], remaining_budget, allow_patch[open_]
        )
        actions = plan.plan.actions
        if not np.any(actions != "defer"):
            break
        report = execute_staged_migration(
            live,
            emb_a,
            ctx.adapter,
            plan.candidate_indices,
            actions,
            plan.rollout_order,
            canary,
            n_waves=cfg.migration.n_waves,
            regression_tolerance=cfg.migration.regression_tolerance,
            gate_confidence=cfg.migration.gate_confidence,
            bootstrap_samples=cfg.migration.bootstrap_samples,
            bisect_depth=cfg.migration.bisect_depth,
            seed=seed + round_number,
        )
        rounds.append((plan, report))
        live = report.live_embeddings

        shipped = open_[report.applied]
        applied[shipped] = True
        action_applied[shipped] = actions[report.applied]
        remaining_budget -= sum(costs[a] for a in actions[report.applied])

        rolled_back = (actions != "defer") & ~report.applied
        ever_rolled_back[open_[rolled_back]] = True
        allow_patch[open_[rolled_back & (actions == "patch")]] = False
        excluded[open_[rolled_back & (actions == "full_reembed")]] = True
        if not rolled_back.any():
            break

    return RemediationOutcome(
        live_embeddings=live,
        applied=applied,
        action_applied=action_applied,
        rounds=rounds,
        cost_spent=float(budget_units) - remaining_budget,
        n_rolled_back=int(np.sum(ever_rolled_back & ~applied)),
    )
