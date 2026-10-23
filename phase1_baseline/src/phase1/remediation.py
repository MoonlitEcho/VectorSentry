"""Sub-phase 1D -- baseline (existing-methodology) remediation: NO
prioritization. Given a set of FLAGGED documents (from a 1C detector, not
oracle ground truth -- a real operator only has what detection gave them)
and a fixed budget, decide which documents get fixed in the order they were
detected/ingested, spending budget until it runs out. This is deliberately
the gap Phase 2 fills: nothing here looks at severity or traffic when
deciding WHICH flagged document to fix next -- only ingestion order.

Two strategies, both using this same FIFO order rule, differing only in
what "fixing" a document costs and how well it works:
- FIFO full re-embed: expensive, perfect recovery per document reached.
- Uniform patch: cheap (`common.patch`), imperfect recovery per document,
  but a fixed budget reaches far more documents.
"""

from __future__ import annotations

import numpy as np


def fifo_remediation_plan(
    flagged: np.ndarray, budget_units: float, cost_per_fix: float
) -> tuple[np.ndarray, float]:
    """Processes flagged documents in ingestion order (array index order --
    see common/data.py's RawBeirData.documents docstring for why BEIR's
    natural corpus order stands in for a real ingestion order), fixing as
    many as the budget affords at `cost_per_fix` each.

    Returns (fixed_mask, cost_spent): `fixed_mask[i]` is True iff document i
    was reached before the budget ran out. Documents that are contaminated
    but NOT flagged (a detector's false negatives) are never in `flagged`
    and so can never be fixed here -- an honest consequence of building on
    real (imperfect) detection rather than oracle ground truth.
    """
    flagged_indices = np.flatnonzero(flagged)  # already in ingestion order, since `flagged` is index-aligned
    if cost_per_fix <= 0:
        raise ValueError("cost_per_fix must be positive")

    cumulative_cost = np.arange(1, len(flagged_indices) + 1) * cost_per_fix
    n_affordable = int(np.searchsorted(cumulative_cost, budget_units, side="right"))

    fixed = np.zeros(len(flagged), dtype=bool)
    fixed[flagged_indices[:n_affordable]] = True
    cost_spent = n_affordable * cost_per_fix
    return fixed, cost_spent
