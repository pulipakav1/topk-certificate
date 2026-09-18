"""Randomized soundness properties; no datasets or encoders required."""

import numpy as np
import pytest

pytest.importorskip("hypothesis", reason="Hypothesis is required to execute property tests")
from hypothesis import given, settings, strategies as st

from topk_stability import full_certificate, per_document_bounds, spectral_norm, validate_certificate


@settings(max_examples=100, deadline=None)
@given(seed=st.integers(0, 2**32 - 1), k=st.integers(1, 5),
       scale=st.floats(min_value=0., max_value=.2, allow_nan=False))
def test_random_head_updates_obey_bounds_and_certificate_implications(seed, k, scale):
    rng = np.random.default_rng(seed)
    heads = rng.normal(size=(6, 3, 3))
    delta = rng.normal(size=heads.shape) * scale
    q, docs = rng.normal(size=3), rng.normal(size=(6, 3))
    scores = lambda ws: np.array([(w @ q) @ (w @ d) for w, d in zip(ws, docs)])
    bounds = per_document_bounds(np.linalg.norm(q), np.linalg.norm(docs, axis=1),
                                 np.array([spectral_norm(w) for w in heads]),
                                 np.array([spectral_norm(w) for w in delta]))
    old = scores(heads)
    decision = full_certificate(old, k, bounds)
    new = scores(heads + delta)
    assert np.all(np.abs(new - old) <= bounds + 1e-9)
    diagnostics = validate_certificate(decision, new)
    if decision.global_certified:
        assert decision.pairwise_certified
    if decision.pairwise_certified:
        assert not diagnostics["topk_changed"]
    if decision.ordered_certified:
        assert decision.pairwise_certified
        assert not diagnostics["ordering_changed"]


@settings(max_examples=100, deadline=None)
@given(seed=st.integers(0, 2**32 - 1))
def test_all_three_implications_for_deterministic_consensus(seed):
    rng = np.random.default_rng(seed)
    heads = np.eye(3)[None, :, :] + rng.normal(scale=.01, size=(6, 3, 3))
    # Fixed incoming stochastic weights, exactly one deterministic update.
    weights = rng.uniform(size=(6, 6))
    weights /= weights.sum(axis=1, keepdims=True)
    proposed = .99 * heads + .01 * np.einsum("ij,jkl->ikl", weights, heads)
    q = np.array([1., 0., 0.])
    docs = np.array([[x, 0., 0.] for x in [3., 2., 1., 0., -1., -2.]])
    score = lambda ws: np.array([(w @ q) @ (w @ d) for w, d in zip(ws, docs)])
    bounds = per_document_bounds(1., np.linalg.norm(docs, axis=1),
                                 np.array([spectral_norm(w) for w in heads]),
                                 np.array([spectral_norm(w) for w in proposed - heads]))
    decision = full_certificate(score(heads), 3, bounds)
    assert decision.global_certified and decision.pairwise_certified and decision.ordered_certified
    diagnostics = validate_certificate(decision, score(proposed))
    assert diagnostics["per_document_bound_holds"]
    assert not diagnostics["topk_changed"] and not diagnostics["ordering_changed"]
