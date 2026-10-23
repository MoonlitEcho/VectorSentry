"""Fast, network-free unit tests for phase1.remediation's FIFO budget
allocation logic."""

import numpy as np
import pytest

from phase1.remediation import fifo_remediation_plan


def test_zero_budget_fixes_nothing():
    flagged = np.array([True, False, True, True])
    fixed, spent = fifo_remediation_plan(flagged, budget_units=0.0, cost_per_fix=1.0)
    assert not fixed.any()
    assert spent == 0.0


def test_ample_budget_fixes_every_flagged_document():
    flagged = np.array([True, False, True, True, False, True])
    fixed, spent = fifo_remediation_plan(flagged, budget_units=100.0, cost_per_fix=1.0)
    np.testing.assert_array_equal(fixed, flagged)
    assert spent == pytest.approx(4.0)  # 4 flagged docs x cost 1.0


def test_partial_budget_fixes_earliest_flagged_docs_first():
    # Flagged docs at indices 1, 3, 5. Budget affords exactly 2 fixes.
    flagged = np.array([False, True, False, True, False, True])
    fixed, spent = fifo_remediation_plan(flagged, budget_units=2.0, cost_per_fix=1.0)
    np.testing.assert_array_equal(fixed, [False, True, False, True, False, False])
    assert spent == pytest.approx(2.0)


def test_no_flagged_documents_fixes_nothing_regardless_of_budget():
    flagged = np.zeros(5, dtype=bool)
    fixed, spent = fifo_remediation_plan(flagged, budget_units=50.0, cost_per_fix=1.0)
    assert not fixed.any()
    assert spent == 0.0


def test_cheaper_cost_per_fix_reaches_more_documents_for_same_budget():
    flagged = np.array([True] * 10)
    fixed_expensive, _ = fifo_remediation_plan(flagged, budget_units=3.0, cost_per_fix=1.0)
    fixed_cheap, _ = fifo_remediation_plan(flagged, budget_units=3.0, cost_per_fix=0.1)
    assert fixed_cheap.sum() > fixed_expensive.sum()


def test_non_positive_cost_per_fix_raises():
    flagged = np.array([True, False])
    with pytest.raises(ValueError):
        fifo_remediation_plan(flagged, budget_units=1.0, cost_per_fix=0.0)
