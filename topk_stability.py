"""topk toolkit"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Dict, List, Tuple

import numpy as np


# --------------------------------------------------------------------------
# top-k margin
# --------------------------------------------------------------------------

def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Rank by descending score, then ascending candidate-slot ID (array index)."""
    k = min(k, len(scores))
    return np.argsort(-scores, kind="stable")[:k]


def topk_boundary_margin(scores: np.ndarray, k: int) -> float:
    """rank gap"""
    n = len(scores)
    if k <= 0 or k >= n:
        raise ValueError(f"k must satisfy 0 < k < n (n={n}); got k={k}")
    sorted_scores = np.sort(scores)[::-1]
    return float(sorted_scores[k - 1] - sorted_scores[k])


# --------------------------------------------------------------------------
# parameter / score change
# --------------------------------------------------------------------------

def param_change_norm(theta_old: np.ndarray, theta_new: np.ndarray) -> float:
    """theta jump"""
    return float(np.linalg.norm(theta_new - theta_old, axis=1).mean())


def max_abs_score_delta(scores_old: np.ndarray, scores_new: np.ndarray) -> float:
    """score jump"""
    return float(np.max(np.abs(scores_new - scores_old)))


# --------------------------------------------------------------------------
# empirical certificate (diagnostic only, see predictive certificate below)
# --------------------------------------------------------------------------
#
# wide margin
# small jump
# no swap

def topk_stability_certificate(scores_old: np.ndarray, scores_new: np.ndarray, k: int) -> Dict:
    """stability check"""
    margin_k = topk_boundary_margin(scores_old, k)
    delta = max_abs_score_delta(scores_old, scores_new)
    certified = bool(margin_k > 2.0 * delta)
    topk_changed = bool(
        set(topk_indices(scores_old, k).tolist()) != set(topk_indices(scores_new, k).tolist())
    )
    return {
        "margin_k": margin_k,
        "delta": delta,
        "certified": certified,
        "topk_changed": topk_changed,
    }


def aggregate_topk_stability(
    per_query_scores_old: List[np.ndarray], per_query_scores_new: List[np.ndarray], k: int
) -> Dict:
    """batch check"""
    margins, deltas, certified, changed = [], [], [], []
    violations = 0
    for s_old, s_new in zip(per_query_scores_old, per_query_scores_new):
        d = topk_stability_certificate(s_old, s_new, k)
        margins.append(d["margin_k"])
        deltas.append(d["delta"])
        certified.append(d["certified"])
        changed.append(d["topk_changed"])
        if d["certified"] and d["topk_changed"]:
            violations += 1  # proof break

    n = len(margins)
    return {
        "mean_topk_margin": float(np.mean(margins)) if n else float("nan"),
        "mean_score_delta": float(np.mean(deltas)) if n else float("nan"),
        "certificate_satisfied_rate": float(np.mean(certified)) if n else float("nan"),
        "topk_changed_rate": float(np.mean(changed)) if n else float("nan"),
        "certificate_violations": violations,
        "n_queries_used": n,
    }


# --------------------------------------------------------------------------
# score-change bound
# --------------------------------------------------------------------------
#
# s(q,d) = (Wq).(Wd), bias-free head
# bound uses spectral norm, submultiplicative in the L2 sense

def spectral_norm(W: np.ndarray) -> float:
    """matrix norm"""
    return float(np.linalg.norm(W, ord=2))


def score_change_bound(w_norm: float, delta_w_norm: float, q_norm: float, d_norm: float) -> float:
    """score cap"""
    return float(q_norm * d_norm * (2.0 * w_norm * delta_w_norm + delta_w_norm ** 2))


def per_document_bounds(
    q_norm: float, doc_norms: np.ndarray, w_norms: np.ndarray, delta_w_norms: np.ndarray
) -> np.ndarray:
    """doc caps"""
    return q_norm * doc_norms * (2.0 * w_norms * delta_w_norms + delta_w_norms ** 2)


def query_score_bound(
    q_norm: float, doc_norms: np.ndarray, w_norms: np.ndarray, delta_w_norms: np.ndarray
) -> float:
    """query cap"""
    return float(np.max(per_document_bounds(q_norm, doc_norms, w_norms, delta_w_norms)))


# --------------------------------------------------------------------------
# agent disagreement
# --------------------------------------------------------------------------

def agent_disagreement(theta: np.ndarray) -> float:
    """agent spread"""
    center = theta.mean(axis=0)
    return float(np.mean(np.linalg.norm(theta - center, axis=1) ** 2))


# --------------------------------------------------------------------------
# predictive certificate
# --------------------------------------------------------------------------
#
# margin from before scores
# cap from before update
# no rescoring needed

def predictive_certificate(
    scores_old: np.ndarray, k: int, b_max: float, *, noise_std: float = 0.0
) -> "CertificateDecision":
    """Pre-update global certificate, with a uniform per-document bound."""
    return full_certificate(scores_old, k, np.full(len(scores_old), b_max), noise_std=noise_std)


def aggregate_predictive_stability(
    decisions: List[CertificateDecision],
    per_query_scores_new: List[np.ndarray],
) -> Dict:
    """batch predictive"""
    margins, b_maxes, deltas = [], [], []
    certified, changed, bound_holds = [], [], []
    violations = 0
    for decision, s_new in zip(decisions, per_query_scores_new, strict=True):
        d = validate_certificate(decision, s_new)
        margins.append(d["margin_k"])
        b_maxes.append(d["b_max"])
        deltas.append(d["empirical_score_delta"])
        certified.append(d["certified"])
        changed.append(d["topk_changed"])
        bound_holds.append(d["bound_holds"])
        if d["certified"] and d["topk_changed"]:
            violations += 1  # proof break

    n = len(margins)
    return {
        "mean_topk_margin": float(np.mean(margins)) if n else float("nan"),
        "mean_score_bound": float(np.mean(b_maxes)) if n else float("nan"),
        "mean_empirical_score_delta": float(np.mean(deltas)) if n else float("nan"),
        "certificate_satisfied_rate": float(np.mean(certified)) if n else float("nan"),
        "topk_changed_rate": float(np.mean(changed)) if n else float("nan"),
        "bound_holds_rate": float(np.mean(bound_holds)) if n else float("nan"),
        "certificate_violations": violations,
        "n_queries_used": n,
    }


# --------------------------------------------------------------------------
# pairwise certificate
# --------------------------------------------------------------------------
#
# own cap per doc
# check every pair
# tighter than global

def pairwise_min_gap(scores_old: np.ndarray, b: np.ndarray, k: int) -> float:
    """pairwise gap"""
    n = len(scores_old)
    if k <= 0 or k >= n:
        raise ValueError(f"k must satisfy 0 < k < n (n={n}); got k={k}")
    order = np.argsort(-scores_old, kind="stable")
    top_idx, rest_idx = order[:k], order[k:]
    gaps = (
        scores_old[top_idx][:, None] - scores_old[rest_idx][None, :]
        - b[top_idx][:, None] - b[rest_idx][None, :]
    )
    return float(gaps.min())


def dual_certificate(
    scores_old: np.ndarray, k: int, b: np.ndarray, *, noise_std: float = 0.0
) -> "CertificateDecision":
    """Pre-update global and pairwise decisions."""
    return full_certificate(scores_old, k, b, noise_std=noise_std)


def aggregate_dual_stability(
    decisions: List[CertificateDecision],
    per_query_scores_new: List[np.ndarray],
) -> Dict:
    """batch dual"""
    margins, b_maxes, gaps, deltas = [], [], [], []
    global_certified, pairwise_certified, changed, bound_holds = [], [], [], []
    global_violations = 0
    pairwise_violations = 0
    for decision, s_new in zip(decisions, per_query_scores_new, strict=True):
        d = validate_certificate(decision, s_new)
        margins.append(d["margin_k"])
        b_maxes.append(d["b_max"])
        gaps.append(d["min_pairwise_gap"])
        deltas.append(d["empirical_score_delta"])
        global_certified.append(d["global_certified"])
        pairwise_certified.append(d["pairwise_certified"])
        changed.append(d["topk_changed"])
        bound_holds.append(d["bound_holds"])
        if d["global_certified"] and d["topk_changed"]:
            global_violations += 1  # proof break
        if d["pairwise_certified"] and d["topk_changed"]:
            pairwise_violations += 1  # proof break

    n = len(margins)
    return {
        "mean_topk_margin": float(np.mean(margins)) if n else float("nan"),
        "mean_score_bound": float(np.mean(b_maxes)) if n else float("nan"),
        "mean_pairwise_gap": float(np.mean(gaps)) if n else float("nan"),
        "mean_empirical_score_delta": float(np.mean(deltas)) if n else float("nan"),
        "bound_holds_rate": float(np.mean(bound_holds)) if n else float("nan"),
        "global_certificate_rate": float(np.mean(global_certified)) if n else float("nan"),
        "pairwise_certificate_rate": float(np.mean(pairwise_certified)) if n else float("nan"),
        "topk_changed_rate": float(np.mean(changed)) if n else float("nan"),
        "global_certificate_violations": global_violations,
        "pairwise_certificate_violations": pairwise_violations,
        "n_queries_used": n,
    }


# --------------------------------------------------------------------------
# ordered certificate
# --------------------------------------------------------------------------
#
# ranks in order
# adjacent gaps
# order can't flip

def ordered_topk_indices(scores: np.ndarray, k: int) -> Tuple[int, ...]:
    """ranked picks"""
    k = min(k, len(scores))
    return tuple(int(i) for i in np.argsort(-scores, kind="stable")[:k])


def adjacent_rank_min_gap(scores_old: np.ndarray, b: np.ndarray, k: int) -> float:
    """rank gaps"""
    n = len(scores_old)
    if k <= 0 or k > n:
        raise ValueError(f"k must satisfy 0 < k <= n (n={n}); got k={k}")
    if k < 2:
        return float("inf")  # no adjacent rank to compare
    order = np.argsort(-scores_old, kind="stable")[:k]
    s_sorted = scores_old[order]
    b_sorted = b[order]
    gaps = (s_sorted[:-1] - s_sorted[1:]) - (b_sorted[:-1] + b_sorted[1:])
    return float(gaps.min())


@dataclass(frozen=True)
class CertificateDecision:
    """Immutable snapshot. Candidate IDs are fixed zero-based pool slots.

    Tuples copy all decision inputs so later array/head mutations cannot change
    a stored decision. The bounds cover only the deterministic proposal.
    """
    scores_old: Tuple[float, ...]
    bounds: Tuple[float, ...]
    candidate_ids: Tuple[int, ...]
    k: int
    noise_std: float
    topk: Tuple[int, ...]
    margin_k: float
    b_max: float
    min_pairwise_gap: float
    min_rank_gap: float
    global_certified: bool
    pairwise_certified: bool
    order_certified: bool
    ordered_certified: bool
    certificate_kind: str = "real_arithmetic_assumed_bounds"
    real_bounds: Tuple[float, ...] = ()
    score_error_bounds: Tuple[float, ...] = ()

    def __getitem__(self, key):
        return getattr(self, key)


def require_deterministic_noise(noise_std: float) -> None:
    """Reject noisy updates, including invalid/nonfinite noise values."""
    if noise_std != 0.0:
        raise ValueError("Certificate runs require noise_std=0; post-consensus noise is outside the bound.")


def full_certificate(
    scores_old: np.ndarray, k: int, b: np.ndarray, *, noise_std: float = 0.0,
    real_bounds=None, score_error_bounds=None,
) -> CertificateDecision:
    """Decide using pre-update scores and proposed-update bounds only."""
    require_deterministic_noise(noise_std)
    scores_old = np.asarray(scores_old, dtype=float)
    b = np.asarray(b, dtype=float)
    if scores_old.ndim != 1 or b.shape != scores_old.shape:
        raise ValueError("scores_old and bounds must be matching one-dimensional arrays")
    if not np.all(np.isfinite(scores_old)) or not np.all(np.isfinite(b)) or np.any(b < 0):
        raise ValueError("scores and bounds must be finite, with nonnegative bounds")
    if not isinstance(k, (int, np.integer)):
        raise ValueError("k must be an integer")
    margin = topk_boundary_margin(scores_old, k)
    b_max = float(np.max(b))
    # Exact arithmetic on the binary input values makes strict comparisons safe
    # even when subtraction rounds a zero/negative slack upward.
    order_idx = topk_indices(scores_old, len(scores_old))
    s = [Fraction(float(x)) for x in scores_old]
    bounds = [Fraction(float(x)) for x in b]
    gap = min(s[i] - s[j] - bounds[i] - bounds[j]
              for i in order_idx[:k] for j in order_idx[k:])
    rank = min((s[i] - s[j] - bounds[i] - bounds[j]
                for i, j in zip(order_idx[:k-1], order_idx[1:k])), default=None)
    pairwise = gap > 0
    order = rank is None or rank > 0
    # Round only after exact subtraction, so tiny positive slack keeps its sign.
    try:
        pair_gap = float(gap)
    except OverflowError:
        pair_gap = float("inf") if gap > 0 else -float("inf")
    try:
        rank_gap = float(rank) if rank is not None else float("inf")
    except OverflowError:
        rank_gap = float("inf") if rank > 0 else -float("inf")
    global_ok = s[order_idx[k-1]] - s[order_idx[k]] > 2 * max(bounds)
    kind = "real_arithmetic_assumed_bounds"
    if real_bounds is not None or score_error_bounds is not None:
        real_bounds = np.asarray(real_bounds, dtype=float)
        score_error_bounds = np.asarray(score_error_bounds, dtype=float)
        if (real_bounds.shape != b.shape or score_error_bounds.shape != b.shape or
                not np.all(np.isfinite(real_bounds)) or not np.all(np.isfinite(score_error_bounds)) or
                np.any(real_bounds < 0) or np.any(score_error_bounds < 0)):
            raise ValueError("Numerical certificate requires matching finite nonnegative error envelopes")
        if any(bound < Fraction(float(r)) + Fraction(float(e))
               for bound, r, e in zip(bounds, real_bounds, score_error_bounds)):
            raise ValueError("Total bounds must enclose real bounds plus score errors")
        kind = "float64_guarded"
    return CertificateDecision(
        tuple(float(s) for s in scores_old), tuple(float(x) for x in b),
        tuple(range(len(scores_old))), int(k), float(noise_std),
        ordered_topk_indices(scores_old, k), margin, b_max, pair_gap, rank_gap,
        bool(global_ok), bool(pairwise), bool(order), bool(pairwise and order),
        kind, tuple(real_bounds) if real_bounds is not None else (),
        tuple(score_error_bounds) if score_error_bounds is not None else (),
    )


def validate_certificate(decision: CertificateDecision, scores_new: np.ndarray) -> Dict:
    """Measure post-update behavior without computing or changing decisions."""
    scores_new = np.asarray(scores_new, dtype=float)
    old = np.asarray(decision.scores_old)
    if scores_new.shape != old.shape or not np.all(np.isfinite(scores_new)):
        raise ValueError("Post-update scores must be finite and match the certified candidate slots")
    delta = max_abs_score_delta(old, scores_new)
    new_topk = ordered_topk_indices(scores_new, decision.k)
    changes = [abs(Fraction(float(a)) - Fraction(float(b))) for a, b in zip(scores_new, old)]
    per_document_holds = all(change <= Fraction(float(cap))
                             for change, cap in zip(changes, decision.bounds))
    return {
        **asdict(decision),
        "certified": decision.global_certified,
        "empirical_score_delta": delta,
        "topk_changed": set(decision.topk) != set(new_topk),
        "ordering_changed": decision.topk != new_topk,
        "bound_holds": per_document_holds,
        "per_document_bound_holds": per_document_holds,
        "max_bound_holds": all(change <= Fraction(decision.b_max) for change in changes),
        "empirical_to_bound_ratio": delta / decision.b_max if decision.b_max > 1e-12 else float("nan"),
    }


def aggregate_full_stability(
    decisions: List[CertificateDecision],
    per_query_scores_new: List[np.ndarray],
) -> Dict:
    """batch full"""
    margins, b_maxes, gaps, rank_gaps, deltas, ratios = [], [], [], [], [], []
    global_certified, pairwise_certified, ordered_certified = [], [], []
    changed, ordering_changed, bound_holds = [], [], []
    max_bound_holds = []
    global_violations = 0
    pairwise_violations = 0
    ordered_violations = 0
    for decision, s_new in zip(decisions, per_query_scores_new, strict=True):
        d = validate_certificate(decision, s_new)
        margins.append(d["margin_k"])
        b_maxes.append(d["b_max"])
        gaps.append(d["min_pairwise_gap"])
        rank_gaps.append(d["min_rank_gap"])
        deltas.append(d["empirical_score_delta"])
        if not np.isnan(d["empirical_to_bound_ratio"]):
            ratios.append(d["empirical_to_bound_ratio"])
        global_certified.append(d["global_certified"])
        pairwise_certified.append(d["pairwise_certified"])
        ordered_certified.append(d["ordered_certified"])
        changed.append(d["topk_changed"])
        ordering_changed.append(d["ordering_changed"])
        bound_holds.append(d["bound_holds"])
        max_bound_holds.append(d["max_bound_holds"])
        if d["global_certified"] and d["topk_changed"]:
            global_violations += 1  # proof break
        if d["pairwise_certified"] and d["topk_changed"]:
            pairwise_violations += 1  # proof break
        if d["ordered_certified"] and d["ordering_changed"]:
            ordered_violations += 1  # proof break

    n = len(margins)
    return {
        "mean_topk_margin": float(np.mean(margins)) if n else float("nan"),
        "mean_score_bound": float(np.mean(b_maxes)) if n else float("nan"),
        "mean_pairwise_gap": float(np.mean(gaps)) if n else float("nan"),
        "mean_rank_gap": float(np.mean(rank_gaps)) if n else float("nan"),
        "mean_empirical_score_delta": float(np.mean(deltas)) if n else float("nan"),
        "mean_empirical_to_bound_ratio": float(np.mean(ratios)) if ratios else float("nan"),
        "bound_holds_rate": float(np.mean(bound_holds)) if n else float("nan"),
        "max_bound_holds_rate": float(np.mean(max_bound_holds)) if n else float("nan"),
        "global_certificate_rate": float(np.mean(global_certified)) if n else float("nan"),
        "pairwise_certificate_rate": float(np.mean(pairwise_certified)) if n else float("nan"),
        "ordered_certificate_rate": float(np.mean(ordered_certified)) if n else float("nan"),
        "topk_changed_rate": float(np.mean(changed)) if n else float("nan"),
        "ordering_changed_rate": float(np.mean(ordering_changed)) if n else float("nan"),
        "global_certificate_violations": global_violations,
        "pairwise_certificate_violations": pairwise_violations,
        "ordered_certificate_violations": ordered_violations,
        "n_queries_used": n,
    }
