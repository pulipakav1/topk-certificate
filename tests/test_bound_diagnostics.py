from fractions import Fraction

import numpy as np
from hypothesis import given, strategies as st

from topk_stability import full_certificate, validate_certificate, aggregate_full_stability


def test_large_other_bound_cannot_hide_local_failure():
    decision = full_certificate(np.array([2., 0.]), 1, np.array([.1, 10.]))
    after = np.array([2.2, 0.])
    diagnostics = validate_certificate(decision, after)
    assert not diagnostics["bound_holds"]
    assert not diagnostics["per_document_bound_holds"]
    assert diagnostics["max_bound_holds"]
    summary = aggregate_full_stability([decision], [after])
    assert summary["bound_holds_rate"] == 0
    assert summary["max_bound_holds_rate"] == 1


@given(seed=st.integers(0, 2**32 - 1))
def test_bound_diagnostics_match_exact_candidate_comparisons(seed):
    rng = np.random.default_rng(seed)
    old = rng.normal(size=5)
    bounds = rng.uniform(0., 2., size=5)
    after = old + rng.normal(size=5)
    decision = full_certificate(old, 2, bounds)
    expected = all(abs(Fraction(float(a)) - Fraction(float(b))) <= Fraction(float(cap))
                   for a, b, cap in zip(after, old, bounds))
    diagnostics = validate_certificate(decision, after)
    assert diagnostics["bound_holds"] == expected
    assert aggregate_full_stability([decision], [after])["bound_holds_rate"] == float(expected)
