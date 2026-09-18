"""verify invariants"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List

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
from topk_stability import (
    adjacent_rank_min_gap,
    agent_disagreement,
    aggregate_dual_stability,
    aggregate_full_stability,
    aggregate_predictive_stability,
    aggregate_topk_stability,
    dual_certificate,
    full_certificate,
    validate_certificate,
    max_abs_score_delta,
    ordered_topk_indices,
    pairwise_min_gap,
    param_change_norm,
    per_document_bounds,
    predictive_certificate,
    query_score_bound,
    score_change_bound,
    spectral_norm,
    topk_boundary_margin,
    topk_indices,
    topk_stability_certificate,
)
from dataset_loaders import RawExample, _assemble_candidates, known_datasets
from model_adapters import (
    format_passage,
    format_query,
    known_retrievers,
    pool_hidden_states,
    pooling_strategy,
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
# check topk
# --------------------------------------------------------------------------

def test_topk_boundary_margin_matches_manual_sort() -> None:
    """margin test"""
    scores = np.array([0.1, 0.9, 0.5, 0.3, 0.7])  # sorted desc
    margin = topk_boundary_margin(scores, k=2)
    assert abs(margin - (0.7 - 0.5)) < 1e-9, f"Expected margin=0.2, got {margin!r}"


def test_topk_boundary_margin_rejects_invalid_k() -> None:
    """bounds test"""
    scores = np.array([0.1, 0.2, 0.3])
    for bad_k in (0, -1, 3, 10):
        try:
            topk_boundary_margin(scores, k=bad_k)
        except ValueError:
            continue
        raise AssertionError(f"topk_boundary_margin accepted invalid k={bad_k}")


def test_topk_indices_matches_argsort() -> None:
    """indices test"""
    scores = np.array([3.0, 1.0, 4.0, 1.5, 5.0])
    idx = set(topk_indices(scores, k=2).tolist())
    assert idx == {4, 2}, f"Expected indices of the two largest scores {{4, 2}}, got {idx!r}"


def test_param_change_norm_zero_when_unchanged() -> None:
    """drift test"""
    theta = np.random.default_rng(0).normal(size=(5, 8))
    assert param_change_norm(theta, theta) == 0.0


def test_param_change_norm_matches_manual() -> None:
    """drift value"""
    rng = np.random.default_rng(1)
    theta_old = rng.normal(size=(4, 6))
    theta_new = theta_old + 1.0  # uniform shift
    expected = float(np.linalg.norm(np.ones(6)))
    got = param_change_norm(theta_old, theta_new)
    assert abs(got - expected) < 1e-9, f"Expected {expected:.6f}, got {got:.6f}"


def test_max_abs_score_delta_basic() -> None:
    """delta test"""
    s_old = np.array([1.0, 2.0, 3.0])
    s_new = np.array([1.0, 2.5, 2.0])
    assert abs(max_abs_score_delta(s_old, s_new) - 1.0) < 1e-9  # largest move


def test_topk_stability_certificate_certifies_small_perturbation() -> None:
    """certified test"""
    s_old = np.array([5.0, 4.0, 1.0, 0.5])   # wide margin
    s_new = s_old + np.array([0.1, -0.1, 0.05, -0.05])  # small jump
    result = topk_stability_certificate(s_old, s_new, k=2)
    assert result["certified"] is True
    assert result["topk_changed"] is False


def test_topk_stability_certificate_detects_uncertified_and_changed() -> None:
    """uncertified test"""
    s_old = np.array([5.0, 4.0, 3.9, 0.5])   # tight margin
    s_new = np.array([5.0, 3.0, 4.5, 0.5])   # docs swap
    result = topk_stability_certificate(s_old, s_new, k=2)
    assert result["certified"] is False
    assert result["topk_changed"] is True


def test_topk_stability_certificate_soundness_random_trials() -> None:
    """soundness test"""
    rng = np.random.default_rng(123)
    violations = 0
    n_trials = 500
    for _ in range(n_trials):
        n_docs = int(rng.integers(4, 12))
        k = int(rng.integers(1, n_docs))  # valid k
        s_old = rng.normal(size=n_docs)
        noise_scale = rng.uniform(0.0, 3.0)
        s_new = s_old + rng.normal(scale=noise_scale, size=n_docs)
        result = topk_stability_certificate(s_old, s_new, k)
        if result["certified"] and result["topk_changed"]:
            violations += 1
    assert violations == 0, f"Certificate soundness violated in {violations}/{n_trials} random trials."


def test_aggregate_topk_stability_batch_matches_manual_rates() -> None:
    """batch test"""
    s_old_1 = np.array([5.0, 4.0, 1.0, 0.5])
    s_new_1 = s_old_1 + np.array([0.1, -0.1, 0.05, -0.05])  # stable case
    s_old_2 = np.array([5.0, 4.0, 3.9, 0.5])
    s_new_2 = np.array([5.0, 3.0, 4.5, 0.5])                # unstable case

    agg = aggregate_topk_stability([s_old_1, s_old_2], [s_new_1, s_new_2], k=2)
    assert agg["n_queries_used"] == 2
    assert abs(agg["certificate_satisfied_rate"] - 0.5) < 1e-9
    assert abs(agg["topk_changed_rate"] - 0.5) < 1e-9
    assert agg["certificate_violations"] == 0


# --------------------------------------------------------------------------
# check bound
# --------------------------------------------------------------------------

def test_spectral_norm_orthogonal_is_one() -> None:
    """spectral test"""
    theta = np.pi / 5
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    assert abs(spectral_norm(rot) - 1.0) < 1e-9


def test_spectral_norm_matches_largest_singular_value() -> None:
    """singular test"""
    U, _ = np.linalg.qr(np.random.default_rng(2).normal(size=(4, 4)))
    V, _ = np.linalg.qr(np.random.default_rng(3).normal(size=(4, 4)))
    singulars = np.array([5.0, 3.0, 1.0, 0.2])
    W = U @ np.diag(singulars) @ V.T
    assert abs(spectral_norm(W) - 5.0) < 1e-7


def test_score_change_bound_matches_manual_formula() -> None:
    """cap formula test"""
    bound = score_change_bound(w_norm=2.0, delta_w_norm=0.5, q_norm=3.0, d_norm=4.0)
    assert abs(bound - 27.0) < 1e-9  # 3*4*(2*2*0.5 + 0.25)


def test_score_change_bound_dominates_actual_change_random_trials() -> None:
    """cap soundness test"""
    rng = np.random.default_rng(7)
    n_trials = 300
    violations = 0
    for _ in range(n_trials):
        dim = int(rng.integers(2, 6))
        W = rng.normal(size=(dim, dim)) * 0.5
        delta_w = rng.normal(size=(dim, dim)) * rng.uniform(0.0, 0.3)
        q = rng.normal(size=dim)
        d = rng.normal(size=dim)

        s_old = float((W @ q) @ (W @ d))
        s_new = float(((W + delta_w) @ q) @ ((W + delta_w) @ d))
        actual = abs(s_new - s_old)

        bound = score_change_bound(
            spectral_norm(W), spectral_norm(delta_w), float(np.linalg.norm(q)), float(np.linalg.norm(d))
        )
        if actual > bound + 1e-8:
            violations += 1
    assert violations == 0, f"score_change_bound violated in {violations}/{n_trials} random trials."


def test_query_score_bound_picks_max_over_docs() -> None:
    """query cap test"""
    q_norm = 2.0
    doc_norms = np.array([1.0, 3.0, 2.0])
    w_norms = np.array([1.0, 1.0, 1.0])
    delta_w_norms = np.array([0.1, 0.1, 0.1])
    per_doc = [score_change_bound(w_norms[j], delta_w_norms[j], q_norm, doc_norms[j]) for j in range(3)]
    got = query_score_bound(q_norm, doc_norms, w_norms, delta_w_norms)
    assert abs(got - max(per_doc)) < 1e-9


def test_agent_disagreement_zero_when_identical() -> None:
    """spread zero test"""
    theta = np.tile(np.array([1.0, 2.0, 3.0]), (4, 1))
    assert agent_disagreement(theta) == 0.0


def test_agent_disagreement_matches_manual() -> None:
    """spread value test"""
    theta = np.array([[0.0, 0.0], [2.0, 0.0]])  # center (1,0), each row 1 away
    assert abs(agent_disagreement(theta) - 1.0) < 1e-9


# --------------------------------------------------------------------------
# check predictive
# --------------------------------------------------------------------------

def test_predictive_certificate_certifies_within_bound() -> None:
    """predictive certified test"""
    s_old = np.array([5.0, 4.0, 1.0, 0.5])   # wide margin
    s_new = s_old + np.array([0.1, -0.1, 0.05, -0.05])  # small actual move
    result = validate_certificate(predictive_certificate(s_old, k=2, b_max=0.2), s_new)
    assert result["certified"] is True
    assert result["topk_changed"] is False
    assert result["bound_holds"] is True


def test_predictive_certificate_flags_bound_break() -> None:
    """bound break test"""
    s_old = np.array([5.0, 4.0, 1.0, 0.5])
    s_new = s_old + np.array([0.2, -0.2, 0.0, 0.0])  # actual exceeds the cap
    result = validate_certificate(predictive_certificate(s_old, k=2, b_max=0.05), s_new)
    assert result["bound_holds"] is False


def test_predictive_certificate_soundness_random_trials() -> None:
    """predictive soundness test"""
    rng = np.random.default_rng(99)
    n_trials = 300
    violations = 0
    for _ in range(n_trials):
        n_docs = int(rng.integers(4, 9))
        w_norms = rng.uniform(0.1, 2.0, size=n_docs)
        delta_w_norms = rng.uniform(0.0, 0.5, size=n_docs)
        q_norm = rng.uniform(0.1, 2.0)
        doc_norms = rng.uniform(0.1, 2.0, size=n_docs)

        s_old = rng.normal(size=n_docs)
        per_doc_cap = q_norm * doc_norms * (2.0 * w_norms * delta_w_norms + delta_w_norms ** 2)
        s_new = s_old + rng.uniform(-1.0, 1.0, size=n_docs) * per_doc_cap  # respects its own cap

        k = int(rng.integers(1, n_docs))
        b_max = query_score_bound(q_norm, doc_norms, w_norms, delta_w_norms)
        result = validate_certificate(predictive_certificate(s_old, k, b_max), s_new)
        if result["certified"] and result["topk_changed"]:
            violations += 1
    assert violations == 0, f"Predictive certificate soundness violated in {violations}/{n_trials} random trials."


def test_aggregate_predictive_stability_batch_matches_manual() -> None:
    """predictive batch test"""
    s_old_1 = np.array([5.0, 4.0, 1.0, 0.5])
    s_new_1 = s_old_1 + np.array([0.1, -0.1, 0.05, -0.05])  # stable case, wide margin
    s_old_2 = np.array([5.0, 4.0, 3.9, 0.5])
    s_new_2 = np.array([5.0, 3.0, 4.5, 0.5])                # unstable case, tight margin

    agg = aggregate_predictive_stability([predictive_certificate(s_old_1, 2, 0.05), predictive_certificate(s_old_2, 2, 0.2)], [s_new_1, s_new_2])
    assert agg["n_queries_used"] == 2
    assert abs(agg["certificate_satisfied_rate"] - 0.5) < 1e-9
    assert abs(agg["topk_changed_rate"] - 0.5) < 1e-9
    assert agg["bound_holds_rate"] == 0.0
    assert agg["certificate_violations"] == 0


# --------------------------------------------------------------------------
# check pairwise
# --------------------------------------------------------------------------

def test_per_document_bounds_matches_scalar_formula() -> None:
    """doc caps test"""
    q_norm = 2.0
    doc_norms = np.array([1.0, 3.0])
    w_norms = np.array([1.0, 2.0])
    delta_w_norms = np.array([0.1, 0.2])
    got = per_document_bounds(q_norm, doc_norms, w_norms, delta_w_norms)
    expected = np.array([
        score_change_bound(w_norms[0], delta_w_norms[0], q_norm, doc_norms[0]),
        score_change_bound(w_norms[1], delta_w_norms[1], q_norm, doc_norms[1]),
    ])
    assert np.allclose(got, expected)


def test_pairwise_min_gap_matches_manual() -> None:
    """pairwise gap test"""
    scores = np.array([5.0, 4.0, 1.0, 0.5])
    b = np.full(4, 0.1)
    got = pairwise_min_gap(scores, b, k=2)
    assert abs(got - 2.8) < 1e-9  # tightest pair is (score=4.0) vs (score=1.0)


def test_pairwise_min_gap_rejects_invalid_k() -> None:
    """pairwise bounds test"""
    scores = np.array([0.1, 0.2, 0.3])
    b = np.full(3, 0.01)
    for bad_k in (0, -1, 3, 10):
        try:
            pairwise_min_gap(scores, b, k=bad_k)
        except ValueError:
            continue
        raise AssertionError(f"pairwise_min_gap accepted invalid k={bad_k}")


def test_pairwise_certifies_when_global_blocked_by_one_outlier_bound() -> None:
    """pairwise wins test"""
    s_old = np.array([5.0, 4.0, 1.0, 0.9, -100.0])
    b = np.array([0.1, 0.1, 0.1, 0.1, 5.0])  # one huge, irrelevant bound
    result = validate_certificate(dual_certificate(s_old, k=2, b=b), s_old)
    assert result["global_certified"] is False
    assert result["pairwise_certified"] is True


def test_global_certified_implies_pairwise_certified_random_trials() -> None:
    """implication test"""
    rng = np.random.default_rng(11)
    n_trials = 300
    for _ in range(n_trials):
        n_docs = int(rng.integers(4, 10))
        k = int(rng.integers(1, n_docs))
        scores = rng.normal(size=n_docs)
        b = rng.uniform(0.0, 1.0, size=n_docs)

        margin_k = topk_boundary_margin(scores, k)
        global_certified = margin_k > 2.0 * float(np.max(b))
        pairwise_certified = pairwise_min_gap(scores, b, k) > 0.0

        if global_certified:
            assert pairwise_certified, "global certified but pairwise not -- pairwise should be weaker"


def test_dual_certificate_pairwise_soundness_random_trials() -> None:
    """pairwise soundness test"""
    rng = np.random.default_rng(21)
    n_trials = 300
    violations = 0
    for _ in range(n_trials):
        n_docs = int(rng.integers(4, 10))
        k = int(rng.integers(1, n_docs))
        s_old = rng.normal(size=n_docs)
        b = rng.uniform(0.0, 1.0, size=n_docs)
        s_new = s_old + rng.uniform(-1.0, 1.0, size=n_docs) * b  # each move respects its own cap

        result = validate_certificate(dual_certificate(s_old, k, b), s_new)
        if result["pairwise_certified"] and result["topk_changed"]:
            violations += 1
    assert violations == 0, f"Pairwise certificate soundness violated in {violations}/{n_trials} random trials."


def test_aggregate_dual_stability_batch_matches_manual() -> None:
    """dual batch test"""
    s_old_1 = np.array([5.0, 4.0, 1.0, 0.5])
    s_new_1 = s_old_1 + np.array([0.1, -0.1, 0.05, -0.05])  # stable case
    b_1 = np.full(4, 0.05)
    s_old_2 = np.array([5.0, 4.0, 3.9, 0.5])
    s_new_2 = np.array([5.0, 3.0, 4.5, 0.5])                # unstable case
    b_2 = np.full(4, 0.2)

    agg = aggregate_dual_stability([dual_certificate(s_old_1, 2, b_1), dual_certificate(s_old_2, 2, b_2)], [s_new_1, s_new_2])
    assert agg["n_queries_used"] == 2
    assert abs(agg["global_certificate_rate"] - 0.5) < 1e-9
    assert abs(agg["pairwise_certificate_rate"] - 0.5) < 1e-9
    assert abs(agg["topk_changed_rate"] - 0.5) < 1e-9
    assert agg["global_certificate_violations"] == 0
    assert agg["pairwise_certificate_violations"] == 0


# --------------------------------------------------------------------------
# check ordered
# --------------------------------------------------------------------------

def test_ordered_topk_indices_matches_argsort() -> None:
    """ordered picks test"""
    scores = np.array([3.0, 1.0, 4.0, 1.5, 5.0])
    assert ordered_topk_indices(scores, k=2) == (4, 2)


def test_adjacent_rank_min_gap_matches_manual() -> None:
    """rank gaps test"""
    scores = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    b = np.full(5, 0.1)
    got = adjacent_rank_min_gap(scores, b, k=3)
    assert abs(got - 0.8) < 1e-9  # (5-4-.2) and (4-3-.2), both 0.8


def test_full_certificate_ordered_certifies_small_perturbation() -> None:
    """ordered certified test"""
    s_old = np.array([5.0, 4.0, 3.0, 0.5])
    s_new = s_old + np.array([0.01, -0.01, 0.005, -0.005])
    b = np.full(4, 0.05)
    result = validate_certificate(full_certificate(s_old, k=2, b=b), s_new)
    assert result["order_certified"] is True
    assert result["pairwise_certified"] is True
    assert result["ordered_certified"] is True
    assert result["ordering_changed"] is False


def test_full_certificate_ordered_soundness_random_trials() -> None:
    """ordered soundness test"""
    rng = np.random.default_rng(55)
    n_trials = 300
    violations = 0
    for _ in range(n_trials):
        n_docs = int(rng.integers(4, 10))
        k = int(rng.integers(2, n_docs))  # k>=2 for a nontrivial rank check
        s_old = rng.normal(size=n_docs)
        b = rng.uniform(0.0, 1.0, size=n_docs)
        s_new = s_old + rng.uniform(-1.0, 1.0, size=n_docs) * b  # respects its own cap

        result = validate_certificate(full_certificate(s_old, k, b), s_new)
        if result["ordered_certified"] and result["ordering_changed"]:
            violations += 1
    assert violations == 0, f"Ordered certificate soundness violated in {violations}/{n_trials} random trials."


def test_certificate_slack_sign_matches_pairwise_certified_random_trials() -> None:
    """slack sign test"""
    rng = np.random.default_rng(66)
    for _ in range(200):
        n_docs = int(rng.integers(4, 10))
        k = int(rng.integers(1, n_docs))
        s_old = rng.normal(size=n_docs)
        s_new = s_old + rng.normal(scale=0.1, size=n_docs)
        b = rng.uniform(0.0, 1.0, size=n_docs)
        result = validate_certificate(full_certificate(s_old, k, b), s_new)
        assert (result["min_pairwise_gap"] > 0.0) == result["pairwise_certified"]


def test_aggregate_full_stability_batch_matches_manual() -> None:
    """full batch test"""
    s_old_1 = np.array([5.0, 4.0, 1.0, 0.5])
    s_new_1 = s_old_1 + np.array([0.1, -0.1, 0.05, -0.05])  # stable, well-ordered
    b_1 = np.full(4, 0.05)
    s_old_2 = np.array([5.0, 4.0, 3.9, 0.5])
    s_new_2 = np.array([5.0, 3.0, 4.5, 0.5])                # unstable, reorders
    b_2 = np.full(4, 0.2)

    agg = aggregate_full_stability([full_certificate(s_old_1, 2, b_1), full_certificate(s_old_2, 2, b_2)], [s_new_1, s_new_2])
    assert agg["n_queries_used"] == 2
    assert abs(agg["global_certificate_rate"] - 0.5) < 1e-9
    assert abs(agg["pairwise_certificate_rate"] - 0.5) < 1e-9
    assert abs(agg["ordered_certificate_rate"] - 0.5) < 1e-9
    assert abs(agg["topk_changed_rate"] - 0.5) < 1e-9
    assert abs(agg["ordering_changed_rate"] - 0.5) < 1e-9
    assert agg["global_certificate_violations"] == 0
    assert agg["pairwise_certificate_violations"] == 0
    assert agg["ordered_certificate_violations"] == 0


# --------------------------------------------------------------------------
# check adapters
# --------------------------------------------------------------------------

def test_format_query_passage_preserve_text_all_models() -> None:
    """formatter test"""
    text = "what is the capital of France"
    for model in known_retrievers():
        q = format_query(model, text)
        p = format_passage(model, text)
        assert text in q, f"{model} query formatting dropped the original text"
        assert text in p, f"{model} passage formatting dropped the original text"
        assert isinstance(q, str) and isinstance(p, str)


def test_known_registries_have_expected_entries() -> None:
    """registry test"""
    assert set(known_retrievers()) == {
        "sentence-transformers/all-MiniLM-L6-v2", "intfloat/e5-small-v2", "BAAI/bge-small-en-v1.5",
    }
    assert set(known_datasets()) == {"hotpotqa", "musique", "2wikimultihopqa"}


def test_e5_formatting_uses_exact_prefixes() -> None:
    """e5 prefix test"""
    assert format_query("intfloat/e5-small-v2", "cats") == "query: cats"
    assert format_passage("intfloat/e5-small-v2", "cats") == "passage: cats"


def test_bge_formatting_uses_exact_prefixes() -> None:
    """bge prefix test"""
    assert format_query("BAAI/bge-small-en-v1.5", "cats") == (
        "Represent this sentence for searching relevant passages: cats"
    )
    assert format_passage("BAAI/bge-small-en-v1.5", "cats") == "cats"  # no passage instruction


def test_minilm_formatting_is_identity() -> None:
    """minilm prefix test"""
    assert format_query("sentence-transformers/all-MiniLM-L6-v2", "cats") == "cats"
    assert format_passage("sentence-transformers/all-MiniLM-L6-v2", "cats") == "cats"


def test_pooling_strategy_per_model() -> None:
    """pooling choice test"""
    assert pooling_strategy("BAAI/bge-small-en-v1.5") == "cls"
    assert pooling_strategy("intfloat/e5-small-v2") == "mean"
    assert pooling_strategy("sentence-transformers/all-MiniLM-L6-v2") == "mean"


def test_pool_hidden_states_cls_picks_first_token() -> None:
    """cls pool test"""
    hidden = torch.tensor([
        [[1.0, 1.0], [9.0, 9.0], [9.0, 9.0]],
        [[2.0, 2.0], [9.0, 9.0], [9.0, 9.0]],
    ])
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    pooled = pool_hidden_states(hidden, mask, "BAAI/bge-small-en-v1.5")
    assert torch.allclose(pooled, torch.tensor([[1.0, 1.0], [2.0, 2.0]]))


def test_pool_hidden_states_mean_respects_attention_mask() -> None:
    """mean pool test"""
    hidden = torch.tensor([
        [[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]],  # last token is padding
    ])
    mask = torch.tensor([[1, 1, 0]])
    pooled = pool_hidden_states(hidden, mask, "intfloat/e5-small-v2")
    assert torch.allclose(pooled, torch.tensor([[2.0, 2.0]]))  # mean of (1,1) and (3,3), padding excluded


def test_pool_then_normalize_gives_unit_norm() -> None:
    """unit norm test"""
    rng = torch.Generator().manual_seed(0)
    hidden = torch.randn(5, 4, 8, generator=rng)
    mask = torch.ones(5, 4, dtype=torch.long)
    for model in known_retrievers():
        pooled = pool_hidden_states(hidden, mask, model)
        normed = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        norms = normed.norm(dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), (
            f"{model}: normalized embedding norm != 1 (got {norms.tolist()})"
        )


# --------------------------------------------------------------------------
# check candidates
# --------------------------------------------------------------------------

def test_assemble_candidates_keeps_all_gold_and_hits_target_count() -> None:
    """assemble test"""
    ex = RawExample(
        question="q",
        gold_titles={"A", "B"},
        paragraphs=[("A", "gold a"), ("B", "gold b"), ("C", "d1"), ("D", "d2"), ("E", "d3")],
    )
    rng = np.random.default_rng(0)
    paragraphs, relevance = _assemble_candidates(ex, n_candidates=4, pool=[], rng=rng)
    assert len(paragraphs) == 4
    assert relevance.sum() == 2  # both gold titles survived


def test_assemble_candidates_pads_from_cross_query_pool_when_short() -> None:
    """pad test"""
    ex = RawExample(question="q", gold_titles={"A"}, paragraphs=[("A", "gold a"), ("C", "d1")])
    pool = [("X", "other q distractor 1"), ("Y", "other q distractor 2"), ("Z", "other q distractor 3")]
    rng = np.random.default_rng(1)
    paragraphs, relevance = _assemble_candidates(ex, n_candidates=5, pool=pool, rng=rng)
    assert len(paragraphs) == 5
    assert relevance.sum() == 1


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
    test_topk_boundary_margin_matches_manual_sort,
    test_topk_boundary_margin_rejects_invalid_k,
    test_topk_indices_matches_argsort,
    test_param_change_norm_zero_when_unchanged,
    test_param_change_norm_matches_manual,
    test_max_abs_score_delta_basic,
    test_topk_stability_certificate_certifies_small_perturbation,
    test_topk_stability_certificate_detects_uncertified_and_changed,
    test_topk_stability_certificate_soundness_random_trials,
    test_aggregate_topk_stability_batch_matches_manual_rates,
    test_spectral_norm_orthogonal_is_one,
    test_spectral_norm_matches_largest_singular_value,
    test_score_change_bound_matches_manual_formula,
    test_score_change_bound_dominates_actual_change_random_trials,
    test_query_score_bound_picks_max_over_docs,
    test_agent_disagreement_zero_when_identical,
    test_agent_disagreement_matches_manual,
    test_predictive_certificate_certifies_within_bound,
    test_predictive_certificate_flags_bound_break,
    test_predictive_certificate_soundness_random_trials,
    test_aggregate_predictive_stability_batch_matches_manual,
    test_per_document_bounds_matches_scalar_formula,
    test_pairwise_min_gap_matches_manual,
    test_pairwise_min_gap_rejects_invalid_k,
    test_pairwise_certifies_when_global_blocked_by_one_outlier_bound,
    test_global_certified_implies_pairwise_certified_random_trials,
    test_dual_certificate_pairwise_soundness_random_trials,
    test_aggregate_dual_stability_batch_matches_manual,
    test_ordered_topk_indices_matches_argsort,
    test_adjacent_rank_min_gap_matches_manual,
    test_full_certificate_ordered_certifies_small_perturbation,
    test_full_certificate_ordered_soundness_random_trials,
    test_certificate_slack_sign_matches_pairwise_certified_random_trials,
    test_aggregate_full_stability_batch_matches_manual,
    test_format_query_passage_preserve_text_all_models,
    test_known_registries_have_expected_entries,
    test_e5_formatting_uses_exact_prefixes,
    test_bge_formatting_uses_exact_prefixes,
    test_minilm_formatting_is_identity,
    test_pooling_strategy_per_model,
    test_pool_hidden_states_cls_picks_first_token,
    test_pool_hidden_states_mean_respects_attention_mask,
    test_pool_then_normalize_gives_unit_norm,
    test_assemble_candidates_keeps_all_gold_and_hits_target_count,
    test_assemble_candidates_pads_from_cross_query_pool_when_short,
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
