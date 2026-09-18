"""Outward-rounded float64 bounds for the unchanged bilinear-score theorem.

Reference CPU implementation: IEEE-754 binary64, gradual underflow, round-to-nearest.
No unproved accuracy assumption is made about LAPACK's singular values: an interval
reconstruction residual and interval Gram matrices validate the returned factors.

Derivation of the returned enclosures:

`spectral_norm_upper(A)` takes the SVD `A ~= U S V`, which is only a guess. It then
bounds `||A||_2 <= ||U||_2 S_max ||V||_2 + ||A - U S V||_2`, where the two factor
norms come from interval Gram matrices (`||U||_2^2 <= max_i sum_j |(U^T U)_ij|`, a
rounded-up Schur bound) and the residual is bounded by `sqrt(||R||_1 ||R||_inf)`.
Every intermediate product and sum is rounded outward, so the result is an upper
bound on the operator 2-norm whatever the SVD returned.

`update_norm_upper(W, W')` bounds `||W' - W||_2`. The stored difference is itself a
rounded quantity, so the enclosure adds the subtraction's own interval radius to the
spectral bound of the computed difference.

`score_roundoff_bound(W, q, d)` bounds the absolute error of `bilinear_score` using
magnitudes only, never the exact value. With `x_i = sum_j W_ij q_j` accumulated in
`2n` operations, the standard `gamma_L = Lu/(1 - Lu)` factor (plus an `L*eta/(1-Lu)`
absolute underflow term) bounds each `|x_i|` and `|y_i|` error; those propagate
through `sum_i x_i y_i` as `|x||ey| + |y||ex| + |ex||ey|`, and the final accumulation
contributes its own `gamma_{2m}` term.

`certified_bounds` returns, per candidate, the real-arithmetic bound `B`, the sum of
the old and proposed scoring-error envelopes, and their outward-rounded total. Only
that total is handed to `full_certificate`, so a certificate covers both the real
score change and the rounding of the two computed scores it is compared against.

Cost is `O(n^3)` per interval matrix product and `O(n^2)` per score, with an
interpreted inner loop; this is a reference implementation, not a fast path.
"""

import numpy as np


def up(x):
    return np.nextafter(np.asarray(x, dtype=np.float64), np.inf)


def down(x):
    return np.nextafter(np.asarray(x, dtype=np.float64), -np.inf)


def add_upper(a, b):
    return up(np.asarray(a) + b)


def mul_upper(a, b):
    return up(np.asarray(a) * b)


def sum_upper(x, axis=0):
    x = np.moveaxis(np.asarray(x, dtype=np.float64), axis, 0)
    total = np.zeros(x.shape[1:], dtype=np.float64)
    for row in x:
        total = add_upper(total, row)
    return total


def _finite(value):
    if not np.all(np.isfinite(value)):
        raise ValueError("Numerical enclosure overflow/nonfinite input: no certificate can be reported")
    return value


def interval_product(al, ah, bl, bh):
    terms = np.stack(np.broadcast_arrays(al * bl, al * bh, ah * bl, ah * bh))
    return down(terms.min(axis=0)), up(terms.max(axis=0))


def interval_matmul(al, ah, bl, bh):
    lo = np.zeros((al.shape[0], bl.shape[1]), dtype=np.float64)
    hi = lo.copy()
    for k in range(al.shape[1]):
        pl, ph = interval_product(al[:, k, None], ah[:, k, None], bl[None, k, :], bh[None, k, :])
        lo, hi = down(lo + pl), up(hi + ph)
    return _finite(lo), _finite(hi)


def _matrix_norm_upper_from_abs(a):
    """Residual norm bound sqrt(||a||_1 ||a||_inf), rounded upward."""
    return up(np.sqrt(mul_upper(sum_upper(a, 0).max(), sum_upper(a, 1).max())))


def spectral_norm_upper(matrix):
    """Validated upper enclosure of the operator 2-norm, not a replacement norm."""
    a = _finite(np.asarray(matrix, dtype=np.float64))
    if not np.any(a):
        return 0.0
    u, singular, vh = np.linalg.svd(a, full_matrices=False)
    ul, uh = interval_matmul(u.T, u.T, u, u)
    vl, vhigh = interval_matmul(vh, vh, vh.T, vh.T)
    un = up(np.sqrt(sum_upper(np.maximum(abs(ul), abs(uh)), 1).max()))
    vn = up(np.sqrt(sum_upper(np.maximum(abs(vl), abs(vhigh)), 1).max()))
    scaled_lo, scaled_hi = down(u * singular), up(u * singular)
    rl, rh = interval_matmul(scaled_lo, scaled_hi, vh, vh)
    residual_abs = np.maximum(abs(down(a - rh)), abs(up(a - rl)))
    estimate = mul_upper(mul_upper(un, singular.max()), vn)
    return float(_finite(add_upper(estimate, _matrix_norm_upper_from_abs(residual_abs))))


def update_norm_upper(old, proposed):
    """Include subtraction error in the exact stored-parameter difference."""
    delta = proposed - old
    low, high = down(delta), up(delta)
    radius = np.maximum(abs(down(delta - high)), abs(up(delta - low)))
    if np.array_equal(old, proposed):
        return 0.0
    return float(add_upper(spectral_norm_upper(delta), _matrix_norm_upper_from_abs(radius)))


def vector_norm_upper(x):
    x = np.asarray(x, dtype=np.float64)
    if not np.any(x):
        return 0.0
    return float(_finite(up(np.sqrt(sum_upper(mul_upper(abs(x), abs(x)))))))


def bilinear_score(w, q, d):
    """Specified float64 multiply/add order; no BLAS, GPU, or TF32 reductions."""
    w, q, d = (np.asarray(x, dtype=np.float64) for x in (w, q, d))
    x, y = np.zeros(w.shape[0]), np.zeros(w.shape[0])
    for j in range(w.shape[1]):
        x = x + w[:, j] * q[j]
        y = y + w[:, j] * d[j]
    value = np.float64(0)
    for j in range(w.shape[0]):
        value = value + x[j] * y[j]
    return float(_finite(value))


def _roundoff_constants(operations):
    # For L operations: gamma_L=Lu/(1-Lu), absolute underflow term L*eta/(1-Lu).
    unit = np.finfo(np.float64).eps / 2
    eta = np.nextafter(np.float64(0), np.float64(1))
    lu = mul_upper(operations, unit)
    denominator = down(1 - lu)
    if denominator <= 0:
        raise ValueError("Too many operations for a finite rounding-error bound")
    return up(lu / denominator), up(mul_upper(operations, eta) / denominator)


def score_roundoff_bound(w, q, d):
    """Absolute error envelope from magnitudes only; never scores the proposal."""
    w, q, d = (np.abs(np.asarray(x, dtype=np.float64)) for x in (w, q, d))
    ax = sum_upper(mul_upper(w, q), 1)
    ay = sum_upper(mul_upper(w, d), 1)
    gamma, underflow = _roundoff_constants(2 * w.shape[1])
    ex, ey = add_upper(mul_upper(gamma, ax), underflow), add_upper(mul_upper(gamma, ay), underflow)
    propagation = sum_upper(add_upper(add_upper(mul_upper(ax, ey), mul_upper(ay, ex)), mul_upper(ex, ey)))
    gamma, underflow = _roundoff_constants(2 * w.shape[0])
    magnitude = sum_upper(mul_upper(add_upper(ax, ex), add_upper(ay, ey)))
    return float(_finite(add_upper(propagation, add_upper(mul_upper(gamma, magnitude), underflow))))


def certified_bounds(old, proposed, q, docs, norm_data=None):
    """Return (upper real B, old+new scoring error, total) for each candidate."""
    old, proposed, q, docs = (np.asarray(x, dtype=np.float64) for x in (old, proposed, q, docs))
    if norm_data is None:
        norm_data = [(spectral_norm_upper(w), update_norm_upper(w, p)) for w, p in zip(old, proposed, strict=True)]
    real, errors = [], []
    qn = vector_norm_upper(q)
    for w, p, d, (wn, dn) in zip(old, proposed, docs, norm_data, strict=True):
        polynomial = add_upper(mul_upper(mul_upper(2., wn), dn), mul_upper(dn, dn))
        bound = mul_upper(mul_upper(qn, vector_norm_upper(d)), polynomial)
        if dn == 0:
            bound = np.float64(0)
        real.append(float(bound))
        errors.append(float(add_upper(score_roundoff_bound(w, q, d), score_roundoff_bound(p, q, d))))
    real, errors = np.array(real), np.array(errors)
    total = _finite(add_upper(real, errors))
    return real, errors, total
