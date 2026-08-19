"""graph toolkit"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from typing import Iterable, Optional

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# build graph
# --------------------------------------------------------------------------

def build_retrieval_graph(
    n_agents: int,
    edge_prob: float = 0.15,
    weight_range: tuple[float, float] = (0.1, 1.0),
    seed: Optional[int] = None,
    ensure_weakly_connected: bool = True,
) -> nx.DiGraph:
    """build graph"""
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    G.add_nodes_from(f"agent_{i}" for i in range(n_agents))
    nodes = list(G.nodes())

    if ensure_weakly_connected and n_agents > 1:
        # ring backbone
        # base backbone
        order = [str(x) for x in rng.permutation(nodes)]
        rotated = order[1:] + order[:1]  # avoid reboxing
        for u, v in zip(order, rotated):
            w = float(rng.uniform(*weight_range))
            G.add_edge(u, v, weight=w)

    for i, u in enumerate(nodes):
        for v in nodes:
            if u == v:
                continue
            if G.has_edge(u, v):
                continue
            if rng.random() < edge_prob:
                w = float(rng.uniform(*weight_range))
                G.add_edge(u, v, weight=w)

    logger.info(
        "Built retrieval graph: |V|=%d, |E|=%d", G.number_of_nodes(), G.number_of_edges()
    )
    return G


# --------------------------------------------------------------------------
# get adjacency
# --------------------------------------------------------------------------

def weighted_adjacency(G: nx.DiGraph, nodelist: Optional[list] = None) -> sp.csr_matrix:
    """get adjacency"""
    nodelist = nodelist or list(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight="weight", format="csr")
    return A.astype(np.float64)


def symmetrize(A: sp.spmatrix) -> sp.csr_matrix:
    """average symmetric"""
    A_sym = (A + A.T) * 0.5
    return A_sym.tocsr()


# --------------------------------------------------------------------------
# build laplacian
# --------------------------------------------------------------------------

def normalized_laplacian(A: sp.spmatrix, eps: float = 1e-12) -> sp.csr_matrix:
    """build laplacian"""
    A = A.tocsr()
    n = A.shape[0]
    degree = np.asarray(A.sum(axis=1)).flatten()

    d_inv_sqrt = np.zeros(n, dtype=np.float64)
    nonzero = degree > eps
    d_inv_sqrt[nonzero] = 1.0 / np.sqrt(degree[nonzero])

    D_inv_sqrt = sp.diags(d_inv_sqrt)
    I = sp.identity(n, format="csr", dtype=np.float64)

    L = I - D_inv_sqrt @ A @ D_inv_sqrt
    return L.tocsr()


# --------------------------------------------------------------------------
# find lambda2
# --------------------------------------------------------------------------

def algebraic_connectivity(
    L: sp.spmatrix,
    tol: float = 1e-8,
    max_iter: int = 5000,
) -> float:
    """find lambda2"""
    n = L.shape[0]
    if n < 2:
        return 0.0

    if n <= 3:
        # small fallback
        # dense fallback
        eigvals = np.linalg.eigvalsh(L.toarray())
        eigvals.sort()
        return float(eigvals[1])

    try:
        # smallest eigenvalues
        # two smallest
        eigvals = spla.eigsh(
            L, k=2, which="SA", tol=tol, maxiter=max_iter, return_eigenvectors=False
        )
    except spla.ArpackNoConvergence as exc:
        logger.warning("eigsh did not fully converge (%s); using partial result.", exc)
        eigvals = exc.eigenvalues
        if eigvals is None or len(eigvals) < 2:
            # dense fallback
            eigvals = np.linalg.eigvalsh(L.toarray())

    eigvals = np.sort(np.real(eigvals))
    return float(eigvals[1]) if len(eigvals) > 1 else 0.0


def compute_pipeline_lambda2(G: nx.DiGraph, nodelist: Optional[list] = None) -> float:
    """wrapper lambda2"""
    A = weighted_adjacency(G, nodelist=nodelist)
    A_sym = symmetrize(A)

    if A_sym.nnz == 0 or A_sym.max() <= 0:
        return 0.0  # fully disconnected

    n_components, _ = connected_components(A_sym, directed=False)
    if n_components > 1:
        return 0.0

    L = normalized_laplacian(A_sym)
    return algebraic_connectivity(L)


# --------------------------------------------------------------------------
# prune edges
# --------------------------------------------------------------------------

@dataclass
class PruneStep:
    step: int
    removed_edge: tuple
    removed_weight: float
    n_edges_remaining: int
    lambda2: float
    is_disconnecting: bool = field(default=False)


def prune_edges(
    G: nx.DiGraph,
    strategy: str = "min_weight",
    max_steps: Optional[int] = None,
    stop_at_zero: bool = True,
    zero_tol: float = 1e-9,
) -> list[PruneStep]:
    """prune edges"""
    H = G.copy()
    nodelist = list(H.nodes())  # fixed ordering
    edges = list(H.edges(data="weight"))

    rng = np.random.default_rng(0)
    if strategy == "min_weight":
        edges.sort(key=lambda e: e[2])
    elif strategy == "max_betweenness":
        bc = nx.edge_betweenness_centrality(H, weight="weight")
        edges.sort(key=lambda e: bc.get((e[0], e[1]), 0.0), reverse=True)
    elif strategy == "random":
        rng.shuffle(edges)
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    if max_steps is not None:
        edges = edges[:max_steps]

    history: list[PruneStep] = []
    for i, (u, v, w) in enumerate(edges, start=1):
        if not H.has_edge(u, v):
            continue  # already removed
        H.remove_edge(u, v)

        lam2 = compute_pipeline_lambda2(H, nodelist=nodelist)
        is_disconnecting = lam2 < zero_tol

        history.append(
            PruneStep(
                step=i,
                removed_edge=(u, v),
                removed_weight=float(w),
                n_edges_remaining=H.number_of_edges(),
                lambda2=lam2,
                is_disconnecting=is_disconnecting,
            )
        )
        logger.info(
            "[%03d] removed (%s -> %s, w=%.3f) | edges left=%d | lambda_2=%.6f%s",
            i, u, v, w, H.number_of_edges(), lam2,
            "  <-- BOTTLENECK" if is_disconnecting else "",
        )

        if stop_at_zero and is_disconnecting:
            logger.info("Graph fragmented (lambda_2 ~ 0) after %d removals; stopping.", i)
            break

    return history


# --------------------------------------------------------------------------
# find bottlenecks
# --------------------------------------------------------------------------

def find_bottlenecks(history: Iterable[PruneStep], zero_tol: float = 1e-9) -> list[PruneStep]:
    """find bottlenecks"""
    return [step for step in history if step.lambda2 < zero_tol]


def summarize_bottlenecks(history: list[PruneStep], zero_tol: float = 1e-9) -> str:
    """summarize bottlenecks"""
    if not history:
        return "No edges were pruned."

    bottlenecks = find_bottlenecks(history, zero_tol=zero_tol)
    lines = [
        f"Pruned {len(history)} edge(s); lambda_2 trajectory: "
        f"{history[0].lambda2:.4f} -> {history[-1].lambda2:.4f}",
    ]
    if bottlenecks:
        lines.append(f"Found {len(bottlenecks)} bottleneck edge(s):")
        for b in bottlenecks:
            lines.append(
                f"  step {b.step}: edge {b.removed_edge} (weight={b.removed_weight:.3f}) "
                f"collapsed lambda_2 to {b.lambda2:.2e} after {b.n_edges_remaining} edges remained"
            )
    else:
        lines.append("No pruning step fully disconnected the graph within the simulated range.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# run demo
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze structural bottlenecks in a multi-agent retrieval graph."
    )
    parser.add_argument("--n-agents", type=int, default=12)
    parser.add_argument("--edge-prob", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strategy",
        choices=["min_weight", "max_betweenness", "random"],
        default="max_betweenness",
        help="Edge removal order for the pruning simulation.",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    G = build_retrieval_graph(args.n_agents, edge_prob=args.edge_prob, seed=args.seed)

    lam2_initial = compute_pipeline_lambda2(G)
    logger.info("Initial algebraic connectivity lambda_2 = %.6f", lam2_initial)

    history = prune_edges(G, strategy=args.strategy, max_steps=args.max_steps)
    print("\n" + summarize_bottlenecks(history))


if __name__ == "__main__":
    main()
