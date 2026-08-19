"""
theorem1_diagnostics.py
=========================

Numerically operationalizes the quantities in the paper's Theorem 1
(spectral bound on non-consensus representation variance under H-hop
graph diffusion with Fisher-Rao-bounded local perturbations), and a
gold-vs-distractor routing-mass diagnostic, against the *real* HotpotQA
run in `evaluation_harness.py::run_hotpotqa_experiment`.

Two honesty notes up front, because both matter for whether a "Theorem 1
verified on real data" claim is actually true:

1. Theorem 1's x^(h) in R^n is a *scalar* graph signal per agent
   (standard graph-signal-processing setup). This pipeline diffuses a
   (n_agents, proj_dim) parameter matrix Theta, not a scalar per agent --
   but the update rule is applied identically, independently, per
   parameter coordinate (Theta_new = (1-alpha)*Theta + alpha * P^T @
   Theta is linear and separable across columns), so treating each of
   the `proj_dim` columns as an independent realization of the scalar
   signal x and using the Frobenius norm (sum of squared column-wise L2
   norms) for ||Pi_perp x||^2 is the natural, direct multi-channel
   generalization -- not a different quantity in disguise.

2. The actual diffusion update in `run_hotpotqa_experiment` is
   Theta_new = (1-alpha)*Theta + alpha*(P^T @ Theta) + noise, i.e. a
   *damped* consensus step (self-retention weight 1-alpha) plus additive
   noise, not the raw T = I - L propagation Theorem 1's proof assumes.
   So this checks whether the empirical (damped, noisy) system still
   respects the bound derived for the idealized (undamped, noiseless)
   one -- an informative empirical stress-test, not a formal proof
   verification. Report it as exactly that.

Also: Theorem 1 explicitly assumes a *connected* graph (states this in
its own hypothesis). When a hop's routing graph is edgeless or
disconnected, the degree-weighted consensus direction u_1 = D^1/2 1 /
||D^1/2 1|| is undefined (0/0) or only removes one of several
independent per-component consensus directions -- `non_consensus_variance`
returns None for the fully-disconnected (all-zero-degree) case rather
than silently reporting 0.0 or NaN-propagating a division.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import networkx as nx
import numpy as np
import torch

from fisher_information_geometry import geodesic_distance


# --------------------------------------------------------------------------
# Theorem 1 quantities
# --------------------------------------------------------------------------

def max_edge_geodesic_distance_sq(
    G: nx.DiGraph,
    thetas: Dict[str, torch.Tensor],
    fishers: Dict[str, torch.Tensor],
    diagonal: bool = True,
) -> float:
    """
    Delta_IG^(h) := max_{(i,j) in E} d_IG^2(theta_i, theta_j), the squared
    Fisher-Rao geodesic distance between the endpoints of the routing
    graph's *actual* edges at this hop (not all pairs) -- the max local
    perturbation magnitude the theorem allows a single edge to inject.

    Uses the directed edge's source-node Fisher F_i (matching this
    codebase's other directed-edge convention in
    `fisher_information_geometry.fisher_rao_edge_weights` with
    symmetrize=None). An edgeless graph has no edges to bound a
    perturbation over, so returns 0.0.
    """
    edges = list(G.edges())
    if not edges:
        return 0.0
    dists_sq = []
    for u, v in edges:
        d = geodesic_distance(thetas[u], thetas[v], fishers[u], diagonal=diagonal)
        dists_sq.append(float(d) ** 2)
    return max(dists_sq)


def non_consensus_variance(Theta: np.ndarray, A_sym: np.ndarray) -> Optional[float]:
    """
    ||Pi_perp x^(h)||_2^2 generalized to a (n_agents, proj_dim) state
    matrix via the Frobenius norm (see module docstring, point 1):

        u_1 = D^{1/2} 1 / || D^{1/2} 1 ||_2        (consensus direction)
        Pi_perp = I - u_1 u_1^T
        ||Pi_perp Theta||_F^2 = sum_j || Pi_perp Theta[:, j] ||_2^2

    Returns None (not 0.0) when every node has degree 0 -- the
    degree-weighted consensus direction is undefined there (0/0), and
    Theorem 1's own hypothesis requires a connected graph.
    """
    d = A_sym.sum(axis=1)
    total = float(d.sum())
    if total <= 1e-12:
        return None
    u1 = np.sqrt(d)
    u1 = u1 / np.linalg.norm(u1)
    proj_coeff = u1 @ Theta              # (proj_dim,)
    consensus = np.outer(u1, proj_coeff)  # (n_agents, proj_dim)
    perp = Theta - consensus
    return float(np.sum(perp ** 2))


def time_inhomogeneous_bound(lambda2_seq: Sequence[float], delta_ig_sq_seq: Sequence[float]) -> List[float]:
    """
    B(h) for h = 1..H, where

        B(H) = sum_{h=1}^{H} [ prod_{k=h+1}^{H} (1 - lambda2^(k))^2 ] * Delta_IG^(h)

    computed via the equivalent linear recursion B(h) = (1-lambda2_h)^2 *
    B(h-1) + Delta_IG_h, B(0) = 0 (verify by expanding the telescoping
    product -- both give identical closed forms). Returns the full
    prefix sequence [B(1), ..., B(H)], not just B(H), so partial-hop
    bounds can be checked too. `1 - lambda2` is clamped at 0 from below
    (lambda2 can theoretically reach up to 2 for the normalized
    Laplacian, which would make 1-lambda2 negative; squaring makes the
    sign irrelevant, but the clamp keeps every intermediate value a
    valid non-negative contraction factor).
    """
    if len(lambda2_seq) != len(delta_ig_sq_seq):
        raise ValueError("lambda2_seq and delta_ig_sq_seq must be the same length (one entry per hop).")
    B = 0.0
    out = []
    for lam2, delta in zip(lambda2_seq, delta_ig_sq_seq):
        contraction = max(0.0, 1.0 - lam2) ** 2
        B = contraction * B + delta
        out.append(B)
    return out


# --------------------------------------------------------------------------
# Gold-vs-distractor routing mass diagnostic
# --------------------------------------------------------------------------

def routing_mass_diagnostic_batch(A: np.ndarray, records: List[dict], n_power: int = 5) -> dict:
    """
    Row-stochastic (out-degree normalized) k-hop reachability diagnostic
    for one hop's shared agent-routing matrix A, averaged over every
    query's own (per-query) gold/distractor paragraph-slot split.

    NOTE this is a *decoupled*, idealized diagnostic: it asks "if this
    hop's graph were used for `n_power` steps of a plain row-stochastic
    random walk starting from a point mass at each node, where does the
    mass end up on average, gold slots vs. distractor slots?" -- not a
    replay of the pipeline's actual (damped, noisy) diffusion update. It
    isolates whether SIR's routing *topology* itself is gold-seeking or
    gold-agnostic, independent of the diffusion_alpha/noise_std knobs.

    A is shared across all queries (it's the agent-level routing graph,
    not query-specific); only each record's gold/distractor split
    varies, so P^n_power is computed once and reused across records.
    """
    n = A.shape[0]
    out_deg = A.sum(axis=1)
    nz = out_deg > 1e-12
    P = np.zeros_like(A)
    P[nz] = A[nz] / out_deg[nz, None]
    P_k = np.linalg.matrix_power(P, n_power) if n_power > 0 else np.eye(n)

    gold_inflows, distractor_inflows = [], []
    for rec in records:
        gold_idx = np.where(rec["relevance"] > 0)[0]
        distractor_idx = np.where(rec["relevance"] <= 0)[0]
        if len(gold_idx) == 0 or len(distractor_idx) == 0:
            continue  # degenerate query (all-gold or all-distractor slot split); skip, don't distort the mean
        gold_inflows.append(float(np.mean(P_k[:, gold_idx])))
        distractor_inflows.append(float(np.mean(P_k[:, distractor_idx])))

    gold_mean = float(np.mean(gold_inflows)) if gold_inflows else float("nan")
    distractor_mean = float(np.mean(distractor_inflows)) if distractor_inflows else float("nan")
    ratio = gold_mean / (distractor_mean + 1e-12) if gold_inflows else float("nan")

    row_entropies = []
    for i in range(n):
        if nz[i]:
            p_pos = P[i][P[i] > 0]
            row_entropies.append(float(-np.sum(p_pos * np.log(p_pos))))
    mean_entropy = float(np.mean(row_entropies)) if row_entropies else 0.0

    return {
        "gold_inflow_mass": gold_mean,
        "distractor_inflow_mass": distractor_mean,
        "gold_to_distractor_ratio": ratio,
        "mean_transition_entropy": mean_entropy,
        "n_isolated_senders": int(n - nz.sum()),
        "n_queries_used": len(gold_inflows),
    }
