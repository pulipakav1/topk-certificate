"""verify invariants"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
import torch.nn as nn

from agent_retrieval_graph import (
    build_retrieval_graph,
    compute_pipeline_lambda2,
    normalized_laplacian,
    symmetrize,
    weighted_adjacency,
)
from fisher_information_geometry import (
    compute_empirical_fisher,
    edge_weight_kernel,
    empirical_fisher,
    fisher_rao_edge_weights,
    iter_single_examples,
    linear_gaussian_log_likelihood,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# check laplacian
# --------------------------------------------------------------------------

def check_laplacian_spectrum(
    L: sp.spmatrix, zero_tol: float = 1e-10, upper_tol: float = 1e-8, dense_cutoff: int = 200
) -> dict:
    """check laplacian"""
    n = L.shape[0]
    if n < 2:
        raise ValueError("Laplacian must have at least 2 nodes to define lambda_2.")

    if n <= dense_cutoff:
        eigvals = np.sort(np.linalg.eigvalsh(L.toarray()))
        lambda1, lambda2, lambda_n = eigvals[0], eigvals[1], eigvals[-1]
        diffs = np.diff(eigvals)
        assert diffs.min() >= -zero_tol, (
            f"Eigenvalue spectrum is not non-decreasing (min gap {diffs.min():.3e}) -- "
            "sorted eigenvalues should never decrease."
        )
    else:
        lo = np.sort(spla.eigsh(L, k=2, which="SA", return_eigenvectors=False))
        hi = spla.eigsh(L, k=1, which="LA", return_eigenvectors=False)
        lambda1, lambda2, lambda_n = lo[0], lo[1], hi[0]

    assert lambda1 > -zero_tol, (
        f"lambda_1={lambda1:.3e} is significantly negative -- L is not PSD "
        f"(violates L = I - D^-1/2 A D^-1/2 >= 0)."
    )
    assert abs(lambda1) < zero_tol, (
        f"lambda_1={lambda1:.3e} is not approximately 0 (tol={zero_tol:.1e})."
    )
    assert lambda2 >= -zero_tol, f"lambda_2={lambda2:.3e} is negative (algebraic connectivity must be >= 0)."
    assert lambda_n <= 2 + upper_tol, f"lambda_n={lambda_n:.6f} exceeds the theoretical upper bound of 2."

    return {"n": n, "lambda1": float(lambda1), "lambda2": float(lambda2), "lambda_n": float(lambda_n)}


def test_laplacian_spectrum_connected_graph() -> None:
    G = build_retrieval_graph(n_agents=12, edge_prob=0.3, seed=1)
    L = normalized_laplacian(symmetrize(weighted_adjacency(G)))
    check_laplacian_spectrum(L)


def test_laplacian_spectrum_disconnected_graph() -> None:
    """disconnected test"""
    G1 = build_retrieval_graph(n_agents=5, edge_prob=0.5, seed=2)
    G2 = build_retrieval_graph(n_agents=5, edge_prob=0.5, seed=3)
    G = nx.DiGraph()
    G.add_nodes_from((f"a_{n}" for n in G1.nodes()))
    G.add_nodes_from((f"b_{n}" for n in G2.nodes()))
    for u, v, w in G1.edges(data="weight"):
        G.add_edge(f"a_{u}", f"a_{v}", weight=w)
    for u, v, w in G2.edges(data="weight"):
        G.add_edge(f"b_{u}", f"b_{v}", weight=w)

    L = normalized_laplacian(symmetrize(weighted_adjacency(G)))
    result = check_laplacian_spectrum(L)
    assert result["lambda2"] < 1e-8, (
        f"Two disjoint components should give lambda_2 ~ 0, got {result['lambda2']:.3e} "
        "-- multiplicity of the zero eigenvalue should equal the number of connected components."
    )


def test_laplacian_spectrum_larger_graph_sparse_path() -> None:
    """sparse test"""
    G = build_retrieval_graph(n_agents=40, edge_prob=0.2, seed=4)
    L = normalized_laplacian(symmetrize(weighted_adjacency(G)))
    check_laplacian_spectrum(L, dense_cutoff=10)


def test_pipeline_lambda2_zero_edges_is_zero_not_one() -> None:
    """edgeless test"""
    G = nx.DiGraph()
    G.add_nodes_from(f"agent_{i}" for i in range(10))  # no edges
    lam2 = compute_pipeline_lambda2(G)
    assert lam2 == 0.0, f"Edgeless graph must report lambda_2 = 0.0 (maximally disconnected), got {lam2!r}."


def test_pipeline_lambda2_isolated_vertex_within_graph() -> None:
    """isolated test"""
    G = build_retrieval_graph(n_agents=9, edge_prob=0.5, seed=5)
    G.add_node("isolated_agent")  # zero degree
    lam2 = compute_pipeline_lambda2(G)
    assert lam2 == 0.0, (
        f"Graph with an isolated vertex must report lambda_2 = 0.0 (it's a second, trivial "
        f"connected component), got {lam2!r}."
    )


# --------------------------------------------------------------------------
# check fisher
# --------------------------------------------------------------------------

def check_fisher_psd(fisher: torch.Tensor, diagonal: bool = True, tol: float = -1e-8) -> dict:
    """check fisher"""
    F = fisher.detach().cpu().double()

    if diagonal:
        assert F.dim() == 1, f"diagonal=True expects a 1D tensor; got shape {tuple(F.shape)}"
        min_val = float(F.min())
        assert min_val >= tol, f"Diagonal Fisher has entry {min_val:.3e} < 0 (tol={tol:.1e})."
        return {
            "diagonal": True,
            "min_entry": min_val,
            "max_entry": float(F.max()),
            "n_negative": int((F < 0).sum()),
        }

    assert F.dim() == 2 and F.shape[0] == F.shape[1], f"full Fisher must be square; got {tuple(F.shape)}"
    asym = float((F - F.T).abs().max())
    assert asym < 1e-6, f"Full Fisher matrix is not symmetric (max |F - F^T| = {asym:.3e})."
    eigvals = torch.linalg.eigvalsh(F)
    min_eig = float(eigvals.min())
    assert min_eig >= tol, f"Full Fisher matrix has eigenvalue {min_eig:.3e} < 0 -- not PSD (tol={tol:.1e})."
    return {
        "diagonal": False,
        "min_eigenvalue": min_eig,
        "max_eigenvalue": float(eigvals.max()),
        "symmetry_error": asym,
    }


def _tiny_linear_agent(seed: int, in_dim: int = 4, out_dim: int = 2) -> nn.Linear:
    g = torch.Generator().manual_seed(seed)
    model = nn.Linear(in_dim, out_dim)
    with torch.no_grad():
        model.weight.copy_(torch.randn(model.weight.shape, generator=g))
        model.bias.copy_(torch.randn(model.bias.shape, generator=g))
    return model


def test_fisher_psd_diagonal_from_real_gradients() -> None:
    model = _tiny_linear_agent(seed=10)
    params = [p for p in model.parameters() if p.requires_grad]
    x = torch.randn(64, 4)
    y = model(x).detach() + 0.1 * torch.randn(64, 2)
    examples = iter_single_examples({"x": x, "y": y})
    _, fisher, _ = compute_empirical_fisher(
        model, examples, linear_gaussian_log_likelihood, params=params, diagonal=True
    )
    check_fisher_psd(fisher, diagonal=True)


def test_fisher_psd_full_from_real_gradients() -> None:
    model = _tiny_linear_agent(seed=11)
    params = [p for p in model.parameters() if p.requires_grad]
    x = torch.randn(64, 4)
    y = model(x).detach() + 0.1 * torch.randn(64, 2)
    examples = iter_single_examples({"x": x, "y": y})
    _, fisher, _ = compute_empirical_fisher(
        model, examples, linear_gaussian_log_likelihood, params=params, diagonal=False
    )
    check_fisher_psd(fisher, diagonal=False)


def test_fisher_psd_degenerate_all_zero_gradients() -> None:
    """zero test"""
    grad_logs = torch.zeros(20, 15, dtype=torch.float64)
    check_fisher_psd(empirical_fisher(grad_logs, diagonal=True), diagonal=True)
    check_fisher_psd(empirical_fisher(grad_logs, diagonal=False), diagonal=False)


def test_fisher_psd_rejects_synthetic_negative_entry() -> None:
    """reject test"""
    bad_diag = torch.tensor([0.1, -0.5, 0.2])
    try:
        check_fisher_psd(bad_diag, diagonal=True)
    except AssertionError:
        return
    raise AssertionError("check_fisher_psd failed to reject a diagonal Fisher with a negative entry.")


# --------------------------------------------------------------------------
# check kernel
# --------------------------------------------------------------------------

def check_edge_weight_range(
    W: torch.Tensor, zero_diagonal: bool = True, upper: float = 1.0, tol: float = 1e-9
) -> dict:
    """check kernel"""
    Wn = W.detach().cpu().double().numpy()
    assert np.isfinite(Wn).all(), "W contains NaN or Inf entries -- kernel transform produced an undefined value."

    n = Wn.shape[0]
    off_mask = ~np.eye(n, dtype=bool)
    off_vals = Wn[off_mask]

    assert off_vals.max() <= upper + tol, f"W_ij exceeds upper bound {upper}: max={off_vals.max():.6f}."
    assert off_vals.min() >= 0.0 - tol, f"W_ij is negative: min={off_vals.min():.6f}."

    diag_vals = np.diag(Wn)
    expected_diag = 0.0 if zero_diagonal else 1.0
    assert np.allclose(diag_vals, expected_diag, atol=1e-8), (
        f"Diagonal of W does not match zero_diagonal={zero_diagonal} convention "
        f"(expected {expected_diag}, got {diag_vals})."
    )

    n_zero = int((off_vals <= tol).sum())
    warnings: List[str] = []
    if n_zero > 0:
        warnings.append(
            f"{n_zero}/{off_vals.size} off-diagonal weights underflowed to ~0 "
            "(gamma * d_IG too large for float64 exp) -- consider a smaller gamma "
            "if a strictly positive floor is required."
        )
    return {"min": float(off_vals.min()), "max": float(off_vals.max()), "n_zero_underflow": n_zero, "warnings": warnings}


def _make_agent_fishers(n_agents: int = 5, seed: int = 20):
    thetas, fishers = {}, {}
    for i in range(n_agents):
        model = _tiny_linear_agent(seed=seed + i)
        params = [p for p in model.parameters() if p.requires_grad]
        x = torch.randn(32, 4)
        y = model(x).detach() + 0.1 * torch.randn(32, 2)
        examples = iter_single_examples({"x": x, "y": y})
        theta, fisher, _ = compute_empirical_fisher(
            model, examples, linear_gaussian_log_likelihood, params=params, diagonal=True
        )
        name = f"agent_{i}"
        thetas[name], fishers[name] = theta, fisher
    return thetas, fishers


def test_edge_weight_range_default_gamma() -> None:
    thetas, fishers = _make_agent_fishers()
    W, _ = fisher_rao_edge_weights(thetas, fishers, diagonal=True, gamma=1.0)
    result = check_edge_weight_range(W)
    assert result["n_zero_underflow"] == 0, "Unexpected underflow at gamma=1.0 with well-separated tiny agents."


def test_edge_weight_range_extreme_gamma_underflow_is_caught() -> None:
    """underflow test"""
    thetas, fishers = _make_agent_fishers(seed=30)
    W, _ = fisher_rao_edge_weights(thetas, fishers, diagonal=True, gamma=1e4)
    result = check_edge_weight_range(W)  # no raise
    assert result["n_zero_underflow"] > 0, "Expected extreme gamma to underflow at least one edge weight."


def test_edge_weight_kernel_rejects_invalid_gamma() -> None:
    d = torch.tensor([0.5, 1.0])
    for bad_gamma in (0.0, -1.0, float("nan"), float("inf")):
        try:
            edge_weight_kernel(d, gamma=bad_gamma)
        except ValueError:
            continue
        raise AssertionError(f"edge_weight_kernel accepted invalid gamma={bad_gamma}")


def test_edge_weight_kernel_no_division_bounded_output() -> None:
    d = torch.tensor([0.0, 1.0, 10.0, 100.0])
    w = edge_weight_kernel(d, gamma=1.0)
    assert torch.isfinite(w).all()
    assert float(w[0]) == 1.0  # self affinity
    assert bool((w >= 0).all() and (w <= 1).all())
    assert bool((w[:-1] > w[1:]).all())  # monotonic decrease


# --------------------------------------------------------------------------
# print report
# --------------------------------------------------------------------------

@dataclass
class CheckOutcome:
    name: str
    passed: bool
    detail: str = ""


ALL_CHECKS: List[Callable[[], None]] = [
    test_laplacian_spectrum_connected_graph,
    test_laplacian_spectrum_disconnected_graph,
    test_laplacian_spectrum_larger_graph_sparse_path,
    test_pipeline_lambda2_zero_edges_is_zero_not_one,
    test_pipeline_lambda2_isolated_vertex_within_graph,
    test_fisher_psd_diagonal_from_real_gradients,
    test_fisher_psd_full_from_real_gradients,
    test_fisher_psd_degenerate_all_zero_gradients,
    test_fisher_psd_rejects_synthetic_negative_entry,
    test_edge_weight_range_default_gamma,
    test_edge_weight_range_extreme_gamma_underflow_is_caught,
    test_edge_weight_kernel_rejects_invalid_gamma,
    test_edge_weight_kernel_no_division_bounded_output,
]


def run_all_checks() -> List[CheckOutcome]:
    outcomes = []
    for fn in ALL_CHECKS:
        try:
            fn()
            outcomes.append(CheckOutcome(fn.__name__, True))
        except AssertionError as exc:
            outcomes.append(CheckOutcome(fn.__name__, False, str(exc)))
        except Exception as exc:  # keep running
            outcomes.append(CheckOutcome(fn.__name__, False, f"{type(exc).__name__}: {exc}"))
    return outcomes


def print_report(outcomes: List[CheckOutcome]) -> bool:
    print("\n=== Verification Checklist ===")
    width = max(len(o.name) for o in outcomes) + 2
    n_pass = 0
    for o in outcomes:
        status = "[PASS]" if o.passed else "[FAIL]"
        n_pass += int(o.passed)
        print(f"{status} {o.name:<{width}}{('- ' + o.detail) if o.detail else ''}")
    print(f"\n{n_pass}/{len(outcomes)} checks passed.")
    return n_pass == len(outcomes)


def main() -> None:
    outcomes = run_all_checks()
    all_passed = print_report(outcomes)
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
