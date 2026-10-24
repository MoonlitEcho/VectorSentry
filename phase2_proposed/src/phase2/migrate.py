"""Sub-phase 2B -- execute a scheduler plan (2A) using realistic migration
mechanics: build the proposed new state alongside the current one ("shadow"
index, never mutating the live one in place), verify it against a curated
"golden query" canary set, and only then atomically cut over -- aborting
and staying on the current state if verification finds a regression.

This models a real, safety-conscious deployment pattern deliberately, not
just "apply the plan and measure quality" (that's closer to what 2C's
head-to-head comparison does with the RESULT of this gate). The point of
2B specifically is the OPERATIONAL question: even if a scheduler's plan is
supposed to help, would a real system ever actually ship a change that
turns out, on verification, to make things worse? Here: no -- it aborts.

Two cutover strategies live here:
- `execute_migration`: the original single all-or-nothing cutover.
- `execute_staged_migration`: the redesigned priority-ordered, wave-by-wave
  rollout with a statistical gate and bisection -- what the Phase 2 sweep
  and Live Sentry now use. See the block comment above it.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

import numpy as np

from common.data import Query
from common.metrics import evaluate_retrieval, per_query_ndcg


def build_shadow_embeddings(
    current_embeddings: np.ndarray,
    emb_a: np.ndarray,
    patch: Callable[[np.ndarray], np.ndarray],
    flagged_indices: np.ndarray,
    plan_actions: np.ndarray,
) -> np.ndarray:
    """Builds the proposed post-migration embedding matrix WITHOUT
    mutating `current_embeddings` -- a real "shadow index build": every
    document keeps its current (possibly still-contaminated) embedding
    except the ones the plan acts on. `plan_actions` is aligned to
    `flagged_indices` (i.e. `plan_actions[i]` is the action for document
    `flagged_indices[i]`), matching `SchedulePlan.actions` from
    phase2.scheduler.

    - full_reembed: re-encode the document's text with the serving model,
      i.e. its true generation-A embedding (a no-op if it was clean).
    - patch: push the CURRENTLY STORED vector through the adapter
      (`common.patch.Adapter`). For a false positive that stored vector is
      already a correct generation-A one, so a patch damages it -- see
      Adapter's docstring for why this replaced the old lookup of the
      document's (unknowable) generation-B embedding.
    - defer: leave the stored vector as it is.
    """
    shadow = current_embeddings.copy()
    plan_actions = np.asarray(plan_actions)
    flagged_indices = np.asarray(flagged_indices, dtype=int)
    full = flagged_indices[plan_actions == "full_reembed"]
    patched = flagged_indices[plan_actions == "patch"]
    shadow[full] = emb_a[full]
    if len(patched):
        shadow[patched] = patch(current_embeddings[patched])
    return shadow


def select_golden_query_indices(eval_queries: list[Query], query_traffic: dict[str, float], n: int) -> np.ndarray:
    """Picks the top-`n` eval queries by traffic weight as the "golden
    query" canary set -- modeling the real practice of checking a small,
    curated, high-value query set before a deploy, not a full evaluation
    suite on every migration decision."""
    weights = np.array([query_traffic.get(q.query_id, 0.0) for q in eval_queries])
    n = min(n, len(eval_queries))
    return np.argsort(-weights)[:n]


@dataclasses.dataclass
class MigrationReport:
    verification_passed: bool
    golden_ndcg_before: float
    golden_ndcg_after: float
    n_full_reembed: int
    n_patch: int
    n_defer: int
    live_embeddings: np.ndarray  # shadow if verification passed, unchanged current if aborted


def execute_migration(
    current_embeddings: np.ndarray,
    emb_a: np.ndarray,
    patch: Callable[[np.ndarray], np.ndarray],
    flagged_indices: np.ndarray,
    plan_actions: np.ndarray,
    doc_ids: list[str],
    golden_queries: list[Query],
    golden_query_embeddings: np.ndarray,
    qrels_eval: dict[str, set[str]],
    query_traffic: dict[str, float],
    regression_tolerance: float = 0.0,
) -> MigrationReport:
    """Runs the full shadow-build -> verify -> swap-or-abort loop for one
    scheduler plan.

    Verification metric is unweighted nDCG@10 over ONLY the golden query
    set (every golden query is already high-traffic by construction, so a
    second traffic-weighting pass inside an already-curated canary set
    would be redundant). The gate: shadow must score at least
    `golden_ndcg_before - regression_tolerance`, or the migration aborts
    and `live_embeddings` stays exactly `current_embeddings` -- an
    all-or-nothing cutover, not a partial one, matching a real atomic
    alias-swap (it flips or it doesn't; there is no in-between state).
    """
    shadow = build_shadow_embeddings(current_embeddings, emb_a, patch, flagged_indices, plan_actions)

    before = evaluate_retrieval(
        current_embeddings, doc_ids, golden_queries, golden_query_embeddings, qrels_eval, query_traffic, k=10
    )
    after = evaluate_retrieval(
        shadow, doc_ids, golden_queries, golden_query_embeddings, qrels_eval, query_traffic, k=10
    )

    passed = after["ndcg@10"] >= before["ndcg@10"] - regression_tolerance
    live_embeddings = shadow if passed else current_embeddings

    n_full = int(np.sum(plan_actions == "full_reembed"))
    n_patch = int(np.sum(plan_actions == "patch"))
    n_defer = len(plan_actions) - n_full - n_patch

    return MigrationReport(
        verification_passed=passed,
        golden_ndcg_before=before["ndcg@10"],
        golden_ndcg_after=after["ndcg@10"],
        n_full_reembed=n_full,
        n_patch=n_patch,
        n_defer=n_defer,
        live_embeddings=live_embeddings,
    )


# --- Staged rollout (see DEVLOG.md's "Redesign" entry) ---------------------
#
# `execute_migration` above is a single all-or-nothing cutover behind a
# zero-tolerance gate. On the real sweep it aborted 10 of 30 plans -- every
# one a plan that would have raised full-corpus nDCG (by up to +0.117) --
# because a 20-query canary moves in whole-query steps, so ANY one query
# losing a rank sank thousands of good fixes. Production rollouts avoid this
# in two ways, both modelled here:
#   1. Waves: ship the plan in priority order, a slice at a time, verifying
#      each slice against the index as it stands after the previous ones. A
#      bad slice is rolled back alone; earlier good slices stay live.
#   2. A statistical NON-INFERIORITY gate: a wave ships only if a paired
#      bootstrap over the golden queries' per-query nDCG deltas shows, with
#      `confidence`, that it is not worse by more than a small margin
#      (`regression_tolerance`). One query wobbling inside the margin does
#      not block a wave; a wave whose harm cannot be ruled out does not ship.
#      A rejected wave is bisected and its halves retried, so a handful of
#      harmful fixes cannot block the rest.
#   3. Deltas are measured against the BEST canary state reached so far,
#      not the previous one, so the margin is a total allowance, not a
#      per-wave one.
#
# 2 and 3 were added after a negative control (run_migration.py: a plan
# that patches clean, heavily-searched documents and halves full-corpus
# nDCG) sailed through the first version of this gate. That version only
# rejected a wave when harm was statistically CONVINCING, and compared each
# wave with the one before it. Bisection then shrank harmful waves until no
# single slice was significant, and the slices passed one after another,
# adding up to the full damage.


def wave_is_unsafe(
    deltas: np.ndarray, tolerance: float, confidence: float, n_bootstrap: int, rng: np.random.Generator
) -> tuple[bool, float]:
    """Non-inferiority test on paired per-query nDCG deltas (shadow -
    reference). Returns (reject, p_worse), where p_worse is the bootstrap
    probability that the mean delta is below -tolerance. The wave is safe
    only if p_worse <= 1 - confidence."""
    deltas = np.asarray(deltas, dtype=np.float64)
    if len(deltas) == 0:
        return False, 0.0
    means = rng.choice(deltas, size=(n_bootstrap, len(deltas)), replace=True).mean(axis=1)
    p_worse = float(np.mean(means < -tolerance))
    return p_worse > 1.0 - confidence, p_worse


# --- Canary scorers ----------------------------------------------------------
#
# The gate needs a per-query quality score for any candidate index state.
# Two are provided:
#
# - labelled: nDCG@10 on a small curated set of queries with relevance
#   labels (the original 2B canary). Honest but low-coverage: 30 queries
#   only "see" the few dozen documents they retrieve, so harm to any other
#   document is invisible, while chance swings on a few queries block
#   good waves. The negative control showed both failure modes.
# - replay (default): shadow-traffic result diffing. Hundreds of popular
#   logged queries are replayed against the candidate index and compared with
#   the results they returned BEFORE the incident (the same impression logs
#   the traffic model uses). Per query: the position-discounted share of the
#   logged top-10 that the candidate still returns. A correct fix restores
#   logged results; a harmful change displaces them. No relevance labels
#   are needed, so the canary can be large; using non-evaluation queries
#   keeps it independent of the reported metric.


def labelled_canary_scorer(
    doc_ids: list[str], queries: list[Query], query_embeddings: np.ndarray, qrels: dict[str, set[str]], k: int = 10
) -> Callable[[np.ndarray], np.ndarray]:
    """Per-query nDCG@k over a labelled canary set."""
    return lambda embeddings: per_query_ndcg(embeddings, doc_ids, queries, query_embeddings, qrels, k)


def _ranked_top_k(query_embeddings: np.ndarray, embeddings: np.ndarray, k: int) -> np.ndarray:
    scores = query_embeddings @ embeddings.T
    top = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1)
    return np.take_along_axis(top, order, axis=1)


def replay_canary_scorer(
    query_embeddings: np.ndarray, pre_incident_embeddings: np.ndarray, k: int = 10
) -> Callable[[np.ndarray], np.ndarray]:
    """Per-query agreement with pre-incident logged results: the share of
    each query's logged top-k (weighted by the 1/log2(rank+2) position
    discount of where it was logged) that the candidate index still returns
    in its top-k."""
    k = min(k, len(pre_incident_embeddings))
    logged = _ranked_top_k(query_embeddings, pre_incident_embeddings, k)
    discount = 1.0 / np.log2(np.arange(k) + 2)

    def score(embeddings: np.ndarray) -> np.ndarray:
        current = _ranked_top_k(query_embeddings, embeddings, k)
        still_shown = np.array([np.isin(logged[row], current[row]) for row in range(len(logged))])
        return (still_shown * discount).sum(axis=1) / discount.sum()

    return score


@dataclasses.dataclass
class WaveResult:
    wave: int  # 1-based wave number in the rollout
    depth: int  # 0 = the wave as planned; >0 = a bisected part of a rejected wave
    n_documents: int
    passed: bool
    canary_before: float  # mean canary score of the live index before this attempt
    canary_after: float  # ... and of the shadow it tested
    p_worse: float


@dataclasses.dataclass
class StagedMigrationReport:
    live_embeddings: np.ndarray  # the index after every accepted wave
    applied: np.ndarray  # (n_items,) bool over the plan's local indices: True = its action is now live
    waves: list[WaveResult]
    canary_before: float  # mean canary score before the first wave
    canary_after: float  # ... and after the last accepted wave
    n_full_reembed: int  # actions actually applied, not merely planned
    n_patch: int
    n_rolled_back: int  # documents whose planned action was rolled back

    @property
    def verification_passed(self) -> bool:
        """False only if the plan acted on something and every wave of it
        was rolled back."""
        return bool(self.applied.any()) or self.n_rolled_back == 0


def execute_staged_migration(
    current_embeddings: np.ndarray,
    emb_a: np.ndarray,
    patch: Callable[[np.ndarray], np.ndarray],
    flagged_indices: np.ndarray,
    plan_actions: np.ndarray,
    rollout_order: np.ndarray,
    canary: Callable[[np.ndarray], np.ndarray],
    n_waves: int = 5,
    regression_tolerance: float = 0.0,
    gate_confidence: float = 0.95,
    bootstrap_samples: int = 2000,
    bisect_depth: int = 2,
    seed: int = 0,
) -> StagedMigrationReport:
    """Ships a scheduler plan in `n_waves` priority-ordered waves (see the
    block comment above), gating each on `canary` (a per-query scorer such
    as `replay_canary_scorer`). `rollout_order` lists the plan's local
    indices (positions in `flagged_indices`/`plan_actions`) in shipping
    order -- typically `phase2.scheduler.priority_order`. Deferred entries
    are skipped, and any acted-on entry missing from `rollout_order` ships
    last.
    """
    plan_actions = np.asarray(plan_actions)
    flagged_indices = np.asarray(flagged_indices, dtype=int)
    rng = np.random.default_rng(seed)
    acted = plan_actions != "defer"
    rollout = [int(i) for i in rollout_order if acted[int(i)]]
    in_rollout = set(rollout)
    rollout += [int(i) for i in np.flatnonzero(acted) if int(i) not in in_rollout]
    applied = np.zeros(len(plan_actions), dtype=bool)
    waves: list[WaveResult] = []

    live = current_embeddings
    live_scores = canary(live)
    canary_before = float(live_scores.mean()) if len(live_scores) else 0.0
    reference_scores = live_scores  # best canary state reached so far -- see block comment, point 3

    def attempt(chunk: np.ndarray, wave: int, depth: int) -> None:
        nonlocal live, live_scores, reference_scores
        shadow = build_shadow_embeddings(live, emb_a, patch, flagged_indices[chunk], plan_actions[chunk])
        shadow_scores = canary(shadow)
        reject, p_worse = wave_is_unsafe(
            shadow_scores - reference_scores, regression_tolerance, gate_confidence, bootstrap_samples, rng
        )
        waves.append(
            WaveResult(wave, depth, len(chunk), not reject, float(live_scores.mean()), float(shadow_scores.mean()), p_worse)
        )
        if not reject:
            live, live_scores = shadow, shadow_scores
            if shadow_scores.mean() >= reference_scores.mean():
                reference_scores = shadow_scores
            applied[chunk] = True
        elif depth < bisect_depth and len(chunk) > 1:
            half = len(chunk) // 2
            attempt(chunk[:half], wave, depth + 1)
            attempt(chunk[half:], wave, depth + 1)

    if rollout:
        for wave_number, chunk in enumerate(np.array_split(np.array(rollout), min(n_waves, len(rollout))), start=1):
            attempt(chunk, wave_number, 0)

    return StagedMigrationReport(
        live_embeddings=live,
        applied=applied,
        waves=waves,
        canary_before=canary_before,
        canary_after=float(live_scores.mean()) if len(live_scores) else 0.0,
        n_full_reembed=int(np.sum(applied & (plan_actions == "full_reembed"))),
        n_patch=int(np.sum(applied & (plan_actions == "patch"))),
        n_rolled_back=int(np.sum(acted & ~applied)),
    )
