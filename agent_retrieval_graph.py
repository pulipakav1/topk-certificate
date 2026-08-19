"""
agent_retrieval_graph.py
=========================

Modular toolkit for analyzing a directed, weighted multi-agent retrieval
graph G = (V, E, W) via spectral graph theory.

Pipeline
--------
1. build_retrieval_graph   -- construct/ingest a directed weighted graph
2. weighted_adjacency      -- G -> scipy.sparse adjacency matrix
3. normalized_laplacian    -- A -> L = I - D^-1/2 A D^-1/2  (symmetrized)
4. algebraic_connectivity  -- L -> lambda_2 (Fiedler value) via eigsh
5. prune_edges             -- iteratively remove edges, tracking lambda_2
6. find_bottlenecks        -- flag prunings that collapse lambda_2 -> 0

Notes on directedness
----------------------
The normalized graph Laplacian I - D^-1/2 A D^-1/2 is only guaranteed
real-symmetric (hence diagonalizable with orthogonal eigenvectors, and
usable with scipy.sparse.linalg.eigsh) when A is symmetric. Since G is a
directed multi-agent communication/retrieval graph, we analyze
connectivity on its symmetrized "communication skeleton"
A_sym = (A + A^T) / 2, which preserves W (edge weights survive averaging)
while treating a bidirectional retrieval channel as a single undirected
link for the purposes of measuring structural connectivity /
bottlenecks. This is the standard trick for spectral analysis of
directed graphs when a symmetric Laplacian is required.

Author: generated for structural bottleneck analysis of agent graphs.
"""

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
# 1. Graph construction
# --------------------------------------------------------------------------

def build_retrieval_graph(
    n_agents: int,
    edge_prob: float = 0.15,
    weight_range: tuple[float, float] = (0.1, 1.0),
    seed: Optional[int] = None,
    ensure_weakly_connected: bool = True,
) -> nx.DiGraph:
    """
    Build a directed multi-agent retrieval graph G = (V, E, W).

    Each node represents an agent (or retrieval unit); each directed edge
    (u, v, weight) represents a retrieval/communication channel from
    agent u to agent v with strength `weight` (e.g. retrieval relevance,
    trust score, or bandwidth).

    Parameters
    ----------
    n_agents : number of agents/nodes |V|.
    edge_prob : probability of a directed edge existing between any
        ordered pair of distinct nodes (Erdos-Renyi style).
    weight_range : (low, high) uniform range for edge weights W.
    seed : RNG seed for reproducibility.
    ensure_weakly_connected : if True, stitch a random spanning ring
        of edges through all nodes first, so the graph never starts
        out trivially disconnected (lambda_2 = 0 by construction).

    Returns
    -------
    nx.DiGraph with a 'weight' attribute on every edge.
    """
    rng = np.random.default_rng(seed)
    G = nx.DiGraph()
    G.add_nodes_from(f"agent_{i}" for i in range(n_agents))
    nodes = list(G.nodes())

    if ensure_weakly_connected and n_agents > 1:
        # Random permutation ring guarantees weak connectivity as a
        # baseline "backbone" before random edges are layered on top.
        order = [str(x) for x in rng.permutation(nodes)]
        rotated = order[1:] + order[:1]  # avoid np.roll: it re-boxes str as np.str_
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
# 2. Adjacency extraction
# --------------------------------------------------------------------------

def weighted_adjacency(G: nx.DiGraph, nodelist: Optional[list] = None) -> sp.csr_matrix:
    """Return the weighted (directed) adjacency matrix A as scipy.sparse.csr_matrix."""
    nodelist = nodelist or list(G.nodes())
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight="weight", format="csr")
    return A.astype(np.float64)


def symmetrize(A: sp.spmatrix) -> sp.csr_matrix:
    """Symmetrize a (possibly directed) adjacency matrix: A_sym = (A + A^T) / 2."""
    A_sym = (A + A.T) * 0.5
    return A_sym.tocsr()


# --------------------------------------------------------------------------
# 3. Normalized Laplacian
# --------------------------------------------------------------------------

def normalized_laplacian(A: sp.spmatrix, eps: float = 1e-12) -> sp.csr_matrix:
    """
    Compute the symmetric normalized graph Laplacian

        L = I - D^{-1/2} A D^{-1/2}

    where D is the diagonal degree matrix of A (row sums) and A is assumed
    symmetric (use `symmetrize` first for directed graphs).

    Isolated nodes (degree 0) are handled by zeroing their D^{-1/2} entry
    (they contribute a trivial 0 row/col rather than a division by zero).
    """
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
# 4. Algebraic connectivity (Fiedler value)
# --------------------------------------------------------------------------

def algebraic_connectivity(
    L: sp.spmatrix,
    tol: float = 1e-8,
    max_iter: int = 5000,
) -> float:
    """
    Compute lambda_2, the second-smallest eigenvalue of the normalized
    Laplacian L, using scipy.sparse.linalg.eigsh.

    lambda_2 (algebraic connectivity / Fiedler value) is 0 iff the graph
    is disconnected; larger values indicate stronger, more robust
    connectivity (harder to fragment the information-flow structure).

    For very small graphs (n <= 3) eigsh's Lanczos iteration cannot
    produce enough Krylov vectors, so we fall back to a dense solve.
    """
    n = L.shape[0]
    if n < 2:
        return 0.0

    if n <= 3:
        # eigsh requires k < n and enough dimensions for Lanczos; use
        # a dense fallback for tiny graphs.
        eigvals = np.linalg.eigvalsh(L.toarray())
        eigvals.sort()
        return float(eigvals[1])

    try:
        # 'SA' = smallest algebraic eigenvalues; L is symmetric PSD so
        # this reliably returns the two smallest (lambda_1 ~ 0, lambda_2).
        eigvals = spla.eigsh(
            L, k=2, which="SA", tol=tol, maxiter=max_iter, return_eigenvectors=False
        )
    except spla.ArpackNoConvergence as exc:
        logger.warning("eigsh did not fully converge (%s); using partial result.", exc)
        eigvals = exc.eigenvalues
        if eigvals is None or len(eigvals) < 2:
            # Dense fallback if ARPACK gives us nothing usable.
            eigvals = np.linalg.eigvalsh(L.toarray())

    eigvals = np.sort(np.real(eigvals))
    return float(eigvals[1]) if len(eigvals) > 1 else 0.0


def compute_pipeline_lambda2(G: nx.DiGraph, nodelist: Optional[list] = None) -> float:
    """
    Convenience wrapper: DiGraph -> symmetrized adjacency -> L -> lambda_2.

    lambda_2 is only a meaningful "how well can information actually flow
    through this graph" signal when there's at least one edge and the graph
    is a single connected component -- by definition, lambda_2 = 0 iff the
    graph is disconnected (which includes "zero edges", the most
    disconnected case possible: n separate isolated-vertex components).

    The symmetric normalized Laplacian I - D^-1/2 A D^-1/2 has a
    well-known quirk here (see e.g. von Luxburg, "A Tutorial on Spectral
    Clustering", Sec. 3): an isolated (degree-0) vertex contributes
    eigenvalue *1*, not 0, to the spectrum, because D^-1/2 is defined as 0
    there to avoid a division by zero, collapsing that vertex's row/column
    of L to the identity. A totally edgeless graph is therefore L = I_n,
    whose eigenvalues are *all* exactly 1.0 -- naively reading that off as
    lambda_2 = 1.0 reports the *maximum possible* algebraic connectivity
    for a graph with zero information flow, exactly backwards from what
    the metric is supposed to mean. We detect the "no edges" / "more than
    one connected component" cases explicitly up front and short-circuit
    to the mathematically correct lambda_2 = 0.0 rather than handing an
    all-zero or block-isolated adjacency matrix to the eigensolver.
    """
    A = weighted_adjacency(G, nodelist=nodelist)
    A_sym = symmetrize(A)

    if A_sym.nnz == 0 or A_sym.max() <= 0:
        return 0.0  # no edges at all -> n isolated components -> maximally disconnected

    n_components, _ = connected_components(A_sym, directed=False)
    if n_components > 1:
        return 0.0

    L = normalized_laplacian(A_sym)
    return algebraic_connectivity(L)


# --------------------------------------------------------------------------
# 5. Edge-pruning simulation
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
    """
    Simulate progressive edge removal and track algebraic connectivity
    after each removal, to surface structural information bottlenecks
    (edges whose loss drives lambda_2 -> 0, i.e. fragments or nearly
    fragments the agent communication graph).

    Parameters
    ----------
    G : the retrieval graph (not mutated -- operates on a copy).
    strategy : edge removal order.
        - 'min_weight'   : remove weakest (lowest-trust/relevance) edges first.
        - 'max_betweenness': remove highest edge-betweenness ("critical
                              bridge") edges first -- the classic recipe
                              for fast fragmentation.
        - 'random'       : remove in random order (baseline/control).
    max_steps : cap on number of removals (default: all edges).
    stop_at_zero : halt as soon as lambda_2 first drops to ~0
        (i.e. the graph has just become disconnected).
    zero_tol : threshold below which lambda_2 is considered "zero".

    Returns
    -------
    List of PruneStep records, one per edge removed, in removal order.
    """
    H = G.copy()
    nodelist = list(H.nodes())  # fixed node ordering across all steps
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
            continue  # already removed as part of an earlier tie
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
# 6. Bottleneck identification
# --------------------------------------------------------------------------

def find_bottlenecks(history: Iterable[PruneStep], zero_tol: float = 1e-9) -> list[PruneStep]:
    """
    From a pruning history, return the steps that drove lambda_2 to ~0,
    i.e. the specific edges whose removal created a structural
    information bottleneck (fragmenting or near-fragmenting the graph).
    """
    return [step for step in history if step.lambda2 < zero_tol]


def summarize_bottlenecks(history: list[PruneStep], zero_tol: float = 1e-9) -> str:
    """Human-readable summary of the pruning run and any bottlenecks found."""
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
# CLI entry point
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
