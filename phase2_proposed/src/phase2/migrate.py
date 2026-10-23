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
"""

from __future__ import annotations

import dataclasses

import numpy as np

from common.data import Query
from common.metrics import evaluate_retrieval


def build_shadow_embeddings(
    current_embeddings: np.ndarray,
    emb_a: np.ndarray,
    patched_emb_b: np.ndarray,
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
    """
    shadow = current_embeddings.copy()
    for local_index, action in enumerate(plan_actions):
        global_index = flagged_indices[local_index]
        if action == "full_reembed":
            shadow[global_index] = emb_a[global_index]
        elif action == "patch":
            shadow[global_index] = patched_emb_b[global_index]
        # "defer": leave shadow[global_index] == current_embeddings[global_index]
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
    patched_emb_b: np.ndarray,
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
    shadow = build_shadow_embeddings(current_embeddings, emb_a, patched_emb_b, flagged_indices, plan_actions)

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
