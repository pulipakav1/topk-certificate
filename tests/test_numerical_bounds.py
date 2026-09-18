from fractions import Fraction as F
import io

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from numerical_bounds import bilinear_score, certified_bounds, score_roundoff_bound, spectral_norm_upper
from topk_stability import full_certificate, validate_certificate


def exact_score(w, q, d):
    dot = lambda a, b: sum((F(float(x)) * F(float(y)) for x, y in zip(a, b)), F(0))
    x, y = [dot(row, q) for row in w], [dot(row, d) for row in w]
    return sum((a * b for a, b in zip(x, y)), F(0))


@settings(max_examples=100, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), exponent=st.integers(-60, 60))
def test_float64_envelopes_against_exact_rational_arithmetic(seed, exponent):
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(3, 3)) * 2.**exponent
    q, d = rng.normal(size=(2, 3))
    error = abs(F(bilinear_score(w, q, d)) - exact_score(w, q, d))
    assert error <= F(score_roundoff_bound(w, q, d))
    # Every tested Rayleigh quotient is bounded by the validated operator norm.
    sn = F(spectral_norm_upper(w))
    image = [sum(F(float(a)) * F(float(b)) for a, b in zip(row, q)) for row in w]
    assert sum(x*x for x in image) <= sn*sn*sum(F(float(x))**2 for x in q)


@pytest.mark.parametrize("scale", [1., 1e60, 1e-60])
@pytest.mark.parametrize("step", [0., 2.**-52, .012])
def test_saturated_bounds_tiny_steps_and_large_norms(scale, step):
    old = np.array([np.eye(2) * scale, np.eye(2) * scale])
    proposed = old * (1 + step)
    q = np.array([1., 0.])
    docs = np.array([[1., 0.], [.5, 0.]])
    real, errors, bounds = certified_bounds(old, proposed, q, docs)
    before = np.array([bilinear_score(w, q, d) for w, d in zip(old, docs)])
    after = np.array([bilinear_score(w, q, d) for w, d in zip(proposed, docs)])
    for i in range(2):
        assert abs(F(float(after[i])) - F(float(before[i]))) <= F(float(bounds[i]))
        assert abs(exact_score(proposed[i], q, docs[i]) - exact_score(old[i], q, docs[i])) <= F(float(real[i]))
    decision = full_certificate(before, 1, bounds)
    assert not (decision.pairwise_certified and validate_certificate(decision, after)["topk_changed"])


def test_zero_margin_and_exact_equality_are_rejected():
    assert not full_certificate(np.array([1., 1.]), 1, np.zeros(2)).pairwise_certified
    assert not full_certificate(np.array([2., 1.]), 1, np.array([.5, .5])).pairwise_certified
    # Rounded sequential subtraction can incorrectly make an exact zero positive.
    scores = np.array([1., 0.])
    assert not full_certificate(scores, 1, np.array([np.nextafter(1., 0.), 2.**-53])).pairwise_certified


def test_nonfinite_enclosures_fail_closed():
    with pytest.raises(ValueError, match="no certificate"):
        spectral_norm_upper(np.array([[np.inf]]))


def test_tiny_positive_slack_is_reported_without_cancellation():
    decision = full_certificate(np.array([1., -2.**-54]), 1, np.array([.5, .5]))
    assert decision.pairwise_certified
    assert decision.min_pairwise_gap == 2.**-54
    assert float(format(decision.min_pairwise_gap, ".17e")) > 0


def test_reported_slack_survives_the_csv_round_trip():
    """A certified positive slack must not reach a reader as zero."""
    csv = pytest.importorskip("csv")
    pd = pytest.importorskip("pandas")
    decision = full_certificate(np.array([1., -2.**-54]), 1, np.array([.5, .5]))
    row = validate_certificate(decision, np.array([1., -2.**-54]))
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=sorted(row))
    writer.writeheader()
    writer.writerow(row)
    buffer.seek(0)
    # The same reader options corrected_results.load_corrected uses.
    read_back = pd.read_csv(buffer, float_precision="round_trip")
    gap = float(read_back.min_pairwise_gap.iloc[0])
    assert gap == decision.min_pairwise_gap > 0
