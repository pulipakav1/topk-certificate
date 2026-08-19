"""theorem diagnostics"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import networkx as nx
import numpy as np
import torch

from fisher_information_geometry import geodesic_distance


# --------------------------------------------------------------------------
# theorem quantities
# --------------------------------------------------------------------------

def max_edge_geodesic_distance_sq(
    G: nx.DiGraph,
    thetas: Dict[str, torch.Tensor],
    fishers: Dict[str, torch.Tensor],
    diagonal: bool = True,
) -> float:
    """max distance"""
    edges = list(G.edges())
    if not edges:
        return 0.0
    dists_sq = []
    for u, v in edges:
        d = geodesic_distance(thetas[u], thetas[v], fishers[u], diagonal=diagonal)
        dists_sq.append(float(d) ** 2)
    return max(dists_sq)


def non_consensus_variance(Theta: np.ndarray, A_sym: np.ndarray) -> Optional[float]:
    """non consensus"""
    d = A_sym.sum(axis=1)
    total = float(d.sum())
    if total <= 1e-12:
        return None
    u1 = np.sqrt(d)
    u1 = u1 / np.linalg.norm(u1)
    proj_coeff = u1 @ Theta              # shape hint
    consensus = np.outer(u1, proj_coeff)  # shape hint
    perp = Theta - consensus
    return float(np.sum(perp ** 2))


def time_inhomogeneous_bound(lambda2_seq: Sequence[float], delta_ig_sq_seq: Sequence[float]) -> List[float]:
    """theorem bound"""
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
# routing mass
# --------------------------------------------------------------------------

def routing_mass_diagnostic_batch(A: np.ndarray, records: List[dict], n_power: int = 5) -> dict:
    """routing mass"""
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
            continue  # skip degenerate
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
