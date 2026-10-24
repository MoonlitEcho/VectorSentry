"""Sub-phase 2A -- VectorSentry's core novel contribution: a budget-aware
prioritization scheduler formulated as a Multiple-Choice Knapsack Problem
(MCKP).

Phase 1 (existing methodology) reuses the SAME detection infrastructure and
the SAME two migration actions (full re-embed, patch -- see common/patch.py)
Phase 2 does. The only thing that differs is the decision layer: Phase 1
applies one uniform strategy to every flagged document in ingestion order;
Phase 2 chooses, PER DOCUMENT, which of three actions to take --

    Full re-embed  (high cost,  full recovery)
    Patch          (low cost,   partial recovery)
    Defer          (zero cost,  zero recovery)

-- to maximize total benefit under ONE shared budget across all flagged
documents jointly. That per-document choice among K mutually exclusive
options, budget-constrained, is exactly a Multiple-Choice Knapsack Problem:
each flagged document is an "item group," each action is one "class" within
that group, and exactly one class must be chosen per group (Defer is always
free and always feasible, so the problem is never infeasible).

Benefit(i, action) = severity(i) * traffic(i) * recovery_factor(action)

- severity(i): the flagged document's continuous detector anomaly score
  (NOT oracle ground truth -- consistent with Phase 1's use of only what
  detection actually found), min-max normalized to [0, 1] across the
  flagged set.
- traffic(i): the document's simulated search-traffic share (common/data.py).
- recovery_factor(action): recovery_full_reembed=1.0 by construction (that
  action substitutes the document's TRUE embedding); recovery_defer=0.0 by
  construction; recovery_patch is NOT assumed from the literature -- it's
  empirically measured from Phase 1's real 1D results (see
  common/config.py's SchedulerConfig and DEVLOG.md's sub-phase 2A entry).

Three solvers:
- `solve_exact`: an exact 0/1 ILP via PuLP/CBC, for validating optimality
  on a TRACTABLE subsample (not the full corpus -- MCKP is NP-hard, and
  while single-knapsack-constraint MIPs are often fast in practice, running
  an exact solver on a subsample and reporting the gap is the standard,
  honest way to validate a heuristic without betting the whole pipeline's
  runtime on a solver that might not scale).
- `solve_greedy_naive`: a flat "sort every (document, action) candidate by
  benefit-to-cost ratio, take greedily" heuristic. Simple and fast, but
  has a real, measured weakness: once it grabs a cheap high-ratio action
  for a document, it can never reconsider that document even when
  abundant leftover budget would make a pricier, more valuable action
  strictly better. On the primary benchmark this doesn't matter (one
  action uniformly dominates the other in cost-efficiency, so the
  heuristic degenerates to an easy uniform-cost knapsack and IS optimal),
  but a stress-test config where the actions' efficiencies are close
  exposed an 88% optimality gap from exactly this flaw -- see DEVLOG.md's
  sub-phase 2A entry. Kept here as a labeled baseline, not the production
  scheduler.
- `solve_greedy`: the standard, textbook-correct MCKP greedy -- for each
  document, prunes any choice that is dominated on its LP-relaxation
  "efficient frontier" (see `build_efficient_frontier_segments`), then
  globally ranks the resulting INCREMENTAL upgrade steps (e.g.
  defer->patch, patch->full_reembed) by incremental benefit-to-cost ratio,
  applying them in order. This is provably optimal for the LP relaxation
  of the MCKP, with any remaining integer-rounding gap possible only at
  the single item where the budget is exhausted mid-upgrade -- which is
  why it closes most of `solve_greedy_naive`'s gap while staying
  O(n log n) and scaling to the full corpus. This is the production
  scheduler Phase 2's actual migration plan (2B) executes.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pulp


@dataclasses.dataclass(frozen=True)
class MCKPItem:
    """One (document, non-defer action) candidate. Defer is never
    represented explicitly -- it's the implicit zero-cost/zero-benefit
    fallback for any item no action gets assigned to."""

    local_index: int  # position within the flagged subset being scheduled, NOT the global document index
    action: str  # "full_reembed" or "patch"
    benefit: float
    cost: float

    @property
    def ratio(self) -> float:
        return self.benefit / self.cost if self.cost > 0 else float("inf")


def compute_severity(scores: np.ndarray) -> np.ndarray:
    """Min-max normalizes detector anomaly scores (restricted to the
    flagged subset) to [0, 1]. If every flagged document has the same
    score (degenerate case), returns uniform 1.0 severity for all of them
    rather than dividing by zero -- with no signal to differentiate them,
    treating them as equally severe is the only defensible default.
    """
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        return np.ones_like(scores, dtype=np.float64)
    return (scores - lo) / (hi - lo)


def build_mckp_items(
    severity: np.ndarray,
    traffic: np.ndarray,
    cost_full_reembed: float,
    cost_patch: float,
    recovery_full_reembed: float,
    recovery_patch: float,
) -> list[MCKPItem]:
    """Builds the two non-defer candidates for every flagged document.
    `severity` and `traffic` must be aligned arrays over the flagged
    subset (index i here == `local_index` in the resulting items)."""
    items: list[MCKPItem] = []
    for i in range(len(severity)):
        base_benefit = severity[i] * traffic[i]
        items.append(MCKPItem(i, "full_reembed", base_benefit * recovery_full_reembed, cost_full_reembed))
        items.append(MCKPItem(i, "patch", base_benefit * recovery_patch, cost_patch))
    return items


@dataclasses.dataclass
class SchedulePlan:
    actions: np.ndarray  # (n_items,) str array: "full_reembed" | "patch" | "defer"
    total_benefit: float
    total_cost: float


def solve_greedy_naive(items: list[MCKPItem], n_items: int, budget: float) -> SchedulePlan:
    """Flat benefit-to-cost ratio ranking greedy -- see the module
    docstring for its known weakness. Kept as a labeled baseline; use
    `solve_greedy` for actual scheduling.

    Process candidates in descending benefit/cost order; assign a
    document its candidate action if the document hasn't already been
    assigned one and the remaining budget covers it. A document whose
    higher-ratio candidate didn't fit the budget is still eligible for its
    OTHER (lower-ratio, typically cheaper) candidate later in the scan --
    it is only marked "assigned" on an actual successful assignment, never
    on a skip. What it CANNOT do is revisit a document that already got a
    cheap action, even if a pricier action would clearly have been better.
    """
    actions = np.full(n_items, "defer", dtype=object)
    remaining_budget = budget
    total_benefit = 0.0
    total_cost = 0.0
    # Costs/budgets are floats accumulated across many additions (e.g. three
    # 0.1 patches don't sum to exactly 0.3 in float64) -- without this
    # tolerance, a document that should legitimately be affordable can get
    # spuriously skipped by a last-bit rounding difference.
    epsilon = 1e-9

    for item in sorted(items, key=lambda it: (-it.ratio, -it.benefit)):
        if actions[item.local_index] != "defer":
            continue
        if item.cost <= remaining_budget + epsilon:
            actions[item.local_index] = item.action
            remaining_budget -= item.cost
            total_benefit += item.benefit
            total_cost += item.cost

    return SchedulePlan(actions, total_benefit, total_cost)


@dataclasses.dataclass(frozen=True)
class MCKPSegment:
    """One incremental upgrade step on a single document's efficient
    frontier: moving from `from_action` to `to_action` costs
    `incremental_cost` for `incremental_benefit` additional benefit.
    Every document contributes either one segment (defer -> full_reembed,
    if patch is dominated for it) or two (defer -> patch -> full_reembed,
    if patch is a genuinely useful intermediate stop) -- see
    `build_efficient_frontier_segments`.
    """

    local_index: int
    from_action: str  # "defer" or "patch"
    to_action: str  # "patch" or "full_reembed"
    incremental_cost: float
    incremental_benefit: float

    @property
    def ratio(self) -> float:
        return self.incremental_benefit / self.incremental_cost if self.incremental_cost > 0 else float("inf")


def build_efficient_frontier_segments(
    severity: np.ndarray,
    traffic: np.ndarray,
    cost_full_reembed: float,
    cost_patch: float,
    recovery_full_reembed: float,
    recovery_patch: float,
) -> list[MCKPSegment]:
    """For each flagged document, builds the incremental-upgrade segments
    along its cost-ascending efficient frontier: defer (0, 0) -> patch
    (cost_patch, benefit_patch) -> full_reembed (cost_full_reembed,
    benefit_full_reembed).

    Patch is only a genuinely useful intermediate stop if its OWN ratio
    from the origin (benefit_patch / cost_patch) exceeds full re-embed's
    OWN ratio from the origin (benefit_full / cost_full) -- geometrically,
    patch's point must sit ABOVE the straight line connecting defer to
    full_reembed on a cost-vs-benefit plot. If it doesn't, patch is
    "dominated" in the LP-relaxation sense for THAT document: no optimal
    solution would ever stop at patch rather than either deferring or
    going straight to full_reembed, so it's dropped from that document's
    frontier and only a single defer -> full_reembed segment is produced.

    This dominance check is per-document (it depends on that document's
    severity*traffic canceling out of the ratio comparison -- actually it
    does NOT depend on severity*traffic at all, since that factor is
    common to both of a document's benefit terms and cancels in the ratio
    comparison; whether patch is dominated is the SAME for every document
    given fixed costs/recovery factors). It is computed per-document here
    anyway (rather than once globally) so this function stays correct if
    a future version makes recovery factors document-dependent.
    """
    segments: list[MCKPSegment] = []
    for i in range(len(severity)):
        base = severity[i] * traffic[i]
        benefit_patch = base * recovery_patch
        benefit_full = base * recovery_full_reembed

        patch_ratio_from_defer = benefit_patch / cost_patch if cost_patch > 0 else float("inf")
        full_ratio_from_defer = benefit_full / cost_full_reembed if cost_full_reembed > 0 else float("inf")
        incremental_cost = cost_full_reembed - cost_patch

        if patch_ratio_from_defer > full_ratio_from_defer and incremental_cost > 0:
            segments.append(MCKPSegment(i, "defer", "patch", cost_patch, benefit_patch))
            segments.append(
                MCKPSegment(i, "patch", "full_reembed", incremental_cost, benefit_full - benefit_patch)
            )
        else:
            # Patch is dominated for this document (or costs more than
            # full re-embed, a degenerate config) -- go straight to full.
            segments.append(MCKPSegment(i, "defer", "full_reembed", cost_full_reembed, benefit_full))

    return segments


def solve_greedy(segments: list[MCKPSegment], n_items: int, budget: float) -> SchedulePlan:
    """Efficient-frontier-aware MCKP greedy (the production scheduler --
    see the module docstring for why this beats `solve_greedy_naive`).

    Sorts ALL incremental upgrade segments, across every document, by
    incremental benefit-to-cost ratio descending. A segment is only
    applied to its document if that document is currently AT the
    segment's `from_action` tier (upgrades must apply in frontier order --
    you can't take the patch->full_reembed segment before the
    defer->patch one) and the remaining budget covers the incremental
    cost.
    """
    current_action = np.full(n_items, "defer", dtype=object)
    remaining_budget = budget
    total_benefit = 0.0
    total_cost = 0.0
    epsilon = 1e-9

    for segment in sorted(segments, key=lambda s: (-s.ratio, -s.incremental_benefit)):
        if current_action[segment.local_index] != segment.from_action:
            continue
        if segment.incremental_cost <= remaining_budget + epsilon:
            current_action[segment.local_index] = segment.to_action
            remaining_budget -= segment.incremental_cost
            total_benefit += segment.incremental_benefit
            total_cost += segment.incremental_cost

    return SchedulePlan(current_action, total_benefit, total_cost)


def solve_exact(items: list[MCKPItem], n_items: int, budget: float) -> SchedulePlan:
    """Exact 0/1 ILP formulation of the same MCKP, solved via PuLP's
    bundled CBC solver. Intended for TRACTABLE instances (see
    SchedulerConfig.exact_solver_sample_size) -- used to measure the
    greedy heuristic's optimality gap, not as the production scheduler.
    """
    problem = pulp.LpProblem("mckp", pulp.LpMaximize)
    variables = {
        (item.local_index, item.action): pulp.LpVariable(f"x_{item.local_index}_{item.action}", cat="Binary")
        for item in items
    }

    problem += pulp.lpSum(item.benefit * variables[(item.local_index, item.action)] for item in items)
    problem += pulp.lpSum(item.cost * variables[(item.local_index, item.action)] for item in items) <= budget

    items_by_doc: dict[int, list[MCKPItem]] = {}
    for item in items:
        items_by_doc.setdefault(item.local_index, []).append(item)
    for local_index, doc_items in items_by_doc.items():
        # At most one non-defer action per document; Defer is implicit
        # (the case where neither variable is 1).
        problem += pulp.lpSum(variables[(local_index, it.action)] for it in doc_items) <= 1

    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    actions = np.full(n_items, "defer", dtype=object)
    total_benefit = 0.0
    total_cost = 0.0
    for item in items:
        if pulp.value(variables[(item.local_index, item.action)]) > 0.5:
            actions[item.local_index] = item.action
            total_benefit += item.benefit
            total_cost += item.cost

    return SchedulePlan(actions, total_benefit, total_cost)


# --- Redesigned benefit model (see DEVLOG.md's "Redesign" entry) -----------
#
# The functions above take a min-max-normalized detector distance as
# "severity" and one global recovery constant per action. The redesigned
# pipeline replaces both with quantities that have a real meaning:
#
#   Benefit(i, full_reembed) = p_i * traffic_i * recovery_full
#   Benefit(i, patch)        = (p_i * recovery_patch - (1 - p_i) * harm_patch) * traffic_i
#
# - p_i is the fingerprint detector's P(document i is contaminated), so each
#   benefit is an EXPECTED value: fixing a document only helps if it was
#   actually broken.
# - Re-embedding a clean document is a harmless no-op (it reproduces the
#   vector already there), so its downside is only the wasted cost.
#   Patching a clean document is NOT harmless: the adapter pushes a correct
#   vector through a B->A map, damaging it by harm_patch (estimated by
#   phase2.calibration). Patch is therefore only worth choosing when the
#   detector is confident.
# - Because p_i now enters the two actions differently, whether patch sits
#   on a document's efficient frontier depends on THAT document -- the
#   "one action uniformly dominates for everyone" degeneracy DEVLOG.md 2A
#   documents no longer holds by construction.


def expected_benefits(
    p_contaminated: np.ndarray,
    traffic: np.ndarray,
    recovery_full_reembed: float,
    recovery_patch: float,
    harm_patch: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-document expected benefit of (full_reembed, patch). See the block
    comment above for the formula and why each term is there."""
    p = np.asarray(p_contaminated, dtype=np.float64)
    t = np.asarray(traffic, dtype=np.float64)
    benefit_full = p * t * recovery_full_reembed
    benefit_patch = (p * recovery_patch - (1.0 - p) * harm_patch) * t
    return benefit_full, benefit_patch


def build_frontier_segments_from_benefits(
    benefit_full: np.ndarray,
    benefit_patch: np.ndarray,
    cost_full_reembed: float,
    cost_patch: float,
) -> list[MCKPSegment]:
    """General per-document efficient frontier for arbitrary (possibly
    negative) per-document benefits: the upper concave hull of
    {defer (0, 0), patch, full_reembed} on a cost-vs-benefit plot.

    A choice is dropped if it does not raise benefit over the cheaper choice
    before it (e.g. a patch with negative expected benefit), or if it sits
    under the straight line between its neighbours on the hull (it would
    never be the optimal stopping point in the LP relaxation). The surviving
    points become consecutive upgrade segments, exactly as in
    `build_efficient_frontier_segments`, and `solve_greedy` consumes them
    unchanged.
    """
    segments: list[MCKPSegment] = []
    for i in range(len(benefit_full)):
        points = sorted(
            [(cost_patch, float(benefit_patch[i]), "patch"), (cost_full_reembed, float(benefit_full[i]), "full_reembed")],
            key=lambda pt: pt[0],
        )
        hull = [(0.0, 0.0, "defer")]
        for point in points:
            if point[1] <= hull[-1][1] + 1e-15:
                continue  # costs more but gains nothing over the cheaper option
            while len(hull) >= 2:
                (x1, y1, _), (x2, y2, _) = hull[-2], hull[-1]
                # Pop the middle point if it lies on/under the chord to `point`.
                if (y2 - y1) * (point[0] - x1) <= (point[1] - y1) * (x2 - x1):
                    hull.pop()
                else:
                    break
            hull.append(point)
        for (x1, y1, a1), (x2, y2, a2) in zip(hull, hull[1:]):
            segments.append(MCKPSegment(i, a1, a2, x2 - x1, y2 - y1))
    return segments


def build_mckp_items_from_benefits(
    benefit_full: np.ndarray,
    benefit_patch: np.ndarray,
    cost_full_reembed: float,
    cost_patch: float,
) -> list[MCKPItem]:
    """Exact-solver counterpart of `build_frontier_segments_from_benefits`.
    Candidates with non-positive benefit are omitted: the ILP would never
    pick them over Defer anyway."""
    items: list[MCKPItem] = []
    for i in range(len(benefit_full)):
        if benefit_full[i] > 0:
            items.append(MCKPItem(i, "full_reembed", float(benefit_full[i]), cost_full_reembed))
        if benefit_patch[i] > 0:
            items.append(MCKPItem(i, "patch", float(benefit_patch[i]), cost_patch))
    return items


def priority_order(segments: list[MCKPSegment], plan: SchedulePlan) -> np.ndarray:
    """The order in which a plan's acted-on documents should be ROLLED OUT:
    by the incremental ratio of the first segment `solve_greedy` applied to
    each, i.e. the order the scheduler itself valued them. Used by the
    staged migration so the most valuable fixes ship in the first wave."""
    first_ratio: dict[int, float] = {}
    for segment in sorted(segments, key=lambda s: (-s.ratio, -s.incremental_benefit)):
        if plan.actions[segment.local_index] != "defer":
            first_ratio.setdefault(segment.local_index, segment.ratio)
    return np.array(sorted(first_ratio, key=lambda i: -first_ratio[i]), dtype=int)
