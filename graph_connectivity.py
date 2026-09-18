"""Algebraic connectivity (lambda_2) of the consensus graph."""

from __future__ import annotations

import logging
from typing import Optional

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components

logger = logging.getLogger(__name__)


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
