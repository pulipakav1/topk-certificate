"""
fisher_information_geometry.py
================================

PyTorch toolkit for turning per-agent parameter states of a (Hugging Face)
causal LM into information-geometric edge weights for a multi-agent graph.

Pipeline (mirrors the three requested stages 1:1)
---------------------------------------------------
1. collect_score_gradients   -- gradient logs  g_n = grad_theta log p_theta(y_n | x_n)
                                 stacked over a batch of examples.
2. empirical_fisher           -- empirical FIM (or its diagonal) built from
                                 those gradient logs:
                                     F = (1/N) sum_n g_n g_n^T          (full)
                                     F_diag = (1/N) sum_n g_n ⊙ g_n     (diagonal)
3. geodesic_distance           -- local quadratic (Mahalanobis) approximation
                                 to the Fisher-Rao geodesic distance between
                                 two agents' parameter states:
                                     d_IG(theta_i, theta_j)
                                       = sqrt((theta_i - theta_j)^T F(theta_i) (theta_i - theta_j))
4. fisher_rao_edge_weights     -- W_ij = exp(-d_IG(theta_i, theta_j)), the
                                 updated edge-weight matrix for the agent graph.

Design notes
------------
* The **full** FIM is O(P^2) in parameter count P and is only tractable for
  tiny models (see `demo_synthetic_linear_agents`). For any real Hugging
  Face causal LM checkpoint, use `diagonal=True` (the default) everywhere --
  this is the standard "empirical Fisher diagonal" approximation used in
  EWC-style continual learning and elastic weight consolidation.
* The empirical Fisher requires **per-example** score vectors
  g_n = grad_theta log p_theta(y_n | x_n) -- gradients must NOT be averaged
  across a mini-batch before squaring/outer-producing them, or the estimator
  is biased. `collect_score_gradients` therefore always evaluates the
  log-likelihood one example at a time.
* d_IG as specified uses F evaluated at theta_i only, so it is a *directed*
  quasi-distance in general (d_IG(i, j) != d_IG(j, i)) -- this is
  intentional and matches the directed multi-agent retrieval graph
  G = (V, E, W) from `agent_retrieval_graph.py`. Pass `symmetrize=` to
  `fisher_rao_edge_weights` if an undirected graph is desired instead.

See `demo_synthetic_linear_agents()` and `demo_hf_causal_lm_agents()` at the
bottom for runnable, end-to-end examples.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

Batch = Dict[str, torch.Tensor]
LogLikelihoodFn = Callable[[nn.Module, Batch], torch.Tensor]


# --------------------------------------------------------------------------
# 1. Gradient logs:  g_n = grad_theta log p_theta(y_n | x_n)
# --------------------------------------------------------------------------

def collect_score_gradients(
    model: nn.Module,
    examples: Iterable[Batch],
    log_likelihood_fn: LogLikelihoodFn,
    params: Optional[List[nn.Parameter]] = None,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, List[nn.Parameter]]:
    """
    Compute the per-example score vectors (gradient logs)
    g_n = grad_theta log p_theta(y_n | x_n) for every example in `examples`,
    and stack them into a single (N, P) tensor, N = #examples, P = #params.

    Parameters
    ----------
    model : the model defining p_theta(y | x). Its current `.parameters()`
        values ARE theta -- call this once per agent state you want a
        Fisher estimate for.
    examples : an iterable of single-example batches (batch dimension 1),
        e.g. `{"input_ids": (1, T), "attention_mask": (1, T), "labels": (1, T)}`
        for a causal LM, or `{"x": (1, d_in), "y": (1, d_out)}` for a
        synthetic regression model. Use `iter_single_examples` to turn a
        normal multi-example DataLoader batch into this form.
    log_likelihood_fn : (model, single_example_batch) -> scalar tensor
        log p_theta(y | x) for that one example (sum over tokens/outputs,
        NOT averaged -- averaging changes the scale of the score).
    params : which parameters to differentiate w.r.t. Defaults to all
        `model.parameters()` with `requires_grad=True`.
    max_samples : optional cap on how many examples to process.

    Returns
    -------
    (grad_logs, params) where grad_logs has shape (N, P) and params is the
    (ordered) parameter list used to build the flattened P-dimensional
    vectors -- reuse the *same* `params` list when flattening theta itself
    (see `flatten_params`) so indices line up.
    """
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found on `model`.")

    rows: List[torch.Tensor] = []
    for n, example in enumerate(examples):
        if max_samples is not None and n >= max_samples:
            break

        model.zero_grad(set_to_none=True)
        log_p = log_likelihood_fn(model, example)
        if log_p.dim() != 0:
            raise ValueError(
                "log_likelihood_fn must return a scalar (single-example) "
                f"log-likelihood; got shape {tuple(log_p.shape)}."
            )

        grads = torch.autograd.grad(log_p, params, retain_graph=False, allow_unused=True)
        flat = torch.cat(
            [
                (g if g is not None else torch.zeros_like(p)).reshape(-1).double()
                for g, p in zip(grads, params)
            ]
        )
        rows.append(flat)

    if not rows:
        raise ValueError("`examples` produced zero samples -- nothing to differentiate.")

    grad_logs = torch.stack(rows, dim=0)  # (N, P)
    logger.info("Collected gradient logs: N=%d examples, P=%d parameters", *grad_logs.shape)
    return grad_logs, params


def iter_single_examples(batch: Batch) -> Iterable[Batch]:
    """Split a standard (multi-example) batch dict of tensors into single-example dicts."""
    keys = list(batch.keys())
    bsz = batch[keys[0]].shape[0]
    for i in range(bsz):
        yield {k: v[i : i + 1] for k, v in batch.items()}


def flatten_params(params: List[nn.Parameter]) -> torch.Tensor:
    """Flatten a list of parameters into a single theta vector (double precision)."""
    return torch.cat([p.detach().reshape(-1).double() for p in params])


# --------------------------------------------------------------------------
# 2. Empirical Fisher Information Matrix (full or diagonal)
# --------------------------------------------------------------------------

_FULL_FISHER_PARAM_WARN_THRESHOLD = 20_000  # ~3.2GB for a float64 P x P matrix


def empirical_fisher(grad_logs: torch.Tensor, diagonal: bool = True) -> torch.Tensor:
    """
    Build the empirical Fisher Information Matrix from stacked gradient logs.

        F        = (1/N) sum_n g_n g_n^T        (diagonal=False, shape (P, P))
        F_diag   = (1/N) sum_n g_n ⊙ g_n         (diagonal=True,  shape (P,))

    This is the classic "empirical Fisher" estimator: the true FIM is
    E_{y ~ p_theta(.|x)}[grad log p . grad log p^T], and we approximate the
    expectation with an empirical average over observed (x, y) pairs.

    Parameters
    ----------
    grad_logs : (N, P) tensor of per-example score vectors, as returned by
        `collect_score_gradients`.
    diagonal : if True (default, and the only tractable option for real
        LM-scale P), return only the diagonal as a (P,) vector. If False,
        return the full (P, P) matrix -- only use for small P (a warning
        is logged above `_FULL_FISHER_PARAM_WARN_THRESHOLD`).
    """
    if grad_logs.dim() != 2:
        raise ValueError(f"grad_logs must be 2D (N, P); got shape {tuple(grad_logs.shape)}")
    n, p = grad_logs.shape
    grad_logs = grad_logs.double()

    if diagonal:
        return (grad_logs ** 2).mean(dim=0)  # (P,)

    if p > _FULL_FISHER_PARAM_WARN_THRESHOLD:
        logger.warning(
            "Computing a full %d x %d Fisher matrix (%.2f GB as float64) -- "
            "consider diagonal=True for models at this scale.",
            p, p, (p * p * 8) / 1e9,
        )
    return (grad_logs.T @ grad_logs) / n  # (P, P)


def compute_empirical_fisher(
    model: nn.Module,
    examples: Iterable[Batch],
    log_likelihood_fn: LogLikelihoodFn,
    params: Optional[List[nn.Parameter]] = None,
    diagonal: bool = True,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[nn.Parameter]]:
    """
    Convenience wrapper: model + examples -> (theta, F(theta), params).

    Combines `collect_score_gradients` + `flatten_params` + `empirical_fisher`
    into a single call for one agent's current parameter state.
    """
    grad_logs, params = collect_score_gradients(
        model, examples, log_likelihood_fn, params=params, max_samples=max_samples
    )
    theta = flatten_params(params)
    fisher = empirical_fisher(grad_logs, diagonal=diagonal)
    return theta, fisher, params


# --------------------------------------------------------------------------
# 3. Fisher-Rao geodesic distance approximation
# --------------------------------------------------------------------------

def geodesic_distance(
    theta_i: torch.Tensor,
    theta_j: torch.Tensor,
    fisher_i: torch.Tensor,
    diagonal: bool = True,
) -> torch.Tensor:
    """
    Local quadratic (second-order / Mahalanobis) approximation to the
    Fisher-Rao geodesic distance between two parameter states, using the
    Fisher metric evaluated at theta_i:

        d_IG(theta_i, theta_j) = sqrt( (theta_i - theta_j)^T F(theta_i) (theta_i - theta_j) )

    Note this is generally asymmetric in (i, j) since the metric tensor is
    evaluated only at theta_i (the curvature at theta_j may differ); see
    `fisher_rao_edge_weights(symmetrize=...)` to symmetrize if needed.

    Parameters
    ----------
    theta_i, theta_j : flat parameter vectors, shape (P,).
    fisher_i : F(theta_i), either the diagonal (P,) or full (P, P) matrix.
    diagonal : must match the shape of `fisher_i`.

    Returns
    -------
    Scalar tensor >= 0.
    """
    if theta_i.shape != theta_j.shape:
        raise ValueError(f"theta_i and theta_j must match shape; got {theta_i.shape} vs {theta_j.shape}")
    delta = (theta_i - theta_j).double()

    if diagonal:
        if fisher_i.dim() != 1 or fisher_i.shape[0] != delta.shape[0]:
            raise ValueError(f"diagonal fisher_i must have shape ({delta.shape[0]},); got {tuple(fisher_i.shape)}")
        quad = torch.sum(fisher_i.clamp_min(0.0) * delta * delta)
    else:
        if fisher_i.dim() != 2 or fisher_i.shape != (delta.shape[0], delta.shape[0]):
            raise ValueError(f"full fisher_i must have shape {(delta.shape[0], delta.shape[0])}; got {tuple(fisher_i.shape)}")
        quad = delta @ fisher_i @ delta

    # F is PSD in exact arithmetic; clamp away tiny negative numerical noise.
    quad = torch.clamp(quad, min=0.0)
    return torch.sqrt(quad)


# --------------------------------------------------------------------------
# 4. Edge-weight matrix W_ij = exp(-gamma * d_IG(theta_i, theta_j))
# --------------------------------------------------------------------------

def edge_weight_kernel(distance: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    W = exp(-gamma * distance), the RBF-style kernel turning a (non-negative)
    geodesic distance into a bounded edge weight.

    No division occurs in this transform, so it cannot divide by zero; the
    only floating-point failure mode is *underflow* of W to exact 0.0 when
    gamma * distance is large enough (> ~745 in float64) that exp() flushes
    to zero -- this is a representable, expected value (a maximally "far"
    pair of agents gets ~0 edge weight, i.e. effectively no edge), not a
    numerical error, but it does mean W_ij can technically leave the open
    interval (0, 1] and hit the closed boundary 0. Callers that need a
    strictly positive floor (e.g. to keep a routing graph fully connected)
    should pass a smaller `gamma` or add an epsilon floor themselves.
    """
    if not (math.isfinite(gamma) and gamma > 0):
        raise ValueError(f"gamma must be a finite positive float; got {gamma!r}")
    distance = distance.clamp_min(0.0)  # d_IG is defined as non-negative
    return torch.exp(-gamma * distance)


def auto_kernel_gamma(
    thetas: Dict[Any, torch.Tensor],
    fishers: Dict[Any, torch.Tensor],
    diagonal: bool = True,
    target_median_distance: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """
    Median-heuristic bandwidth selection for `edge_weight_kernel`: picks
    gamma so that exp(-gamma * median(d_IG)) ~= exp(-target_median_distance),
    keeping W's dynamic range well-scaled regardless of the raw magnitude of
    theta / F(theta). Real model checkpoints can have parameter/Fisher
    scales many orders of magnitude away from a small synthetic test model;
    without this, gamma=1.0 risks *every* W_ij underflowing to exactly 0
    (see `edge_weight_kernel`'s docstring) -- which silently produces a
    fully edgeless routing graph instead of a properly ranked one.
    """
    nodes = list(thetas.keys())
    dists = [
        float(geodesic_distance(thetas[i], thetas[j], fishers[i], diagonal=diagonal))
        for a, i in enumerate(nodes)
        for b, j in enumerate(nodes)
        if a != b
    ]
    median_d = float(np.median(dists)) if dists else 0.0
    if median_d < eps:
        return 1.0
    return target_median_distance / median_d


def fisher_rao_edge_weights(
    thetas: Dict[Any, torch.Tensor],
    fishers: Dict[Any, torch.Tensor],
    diagonal: bool = True,
    symmetrize: Optional[str] = None,
    zero_diagonal: bool = True,
    gamma: float = 1.0,
) -> Tuple[torch.Tensor, List[Any]]:
    """
    Compute the full pairwise edge-weight matrix W for a set of agents,
    given each agent's current parameter state theta_i and Fisher F(theta_i).

        d_IG(theta_i, theta_j) = sqrt((theta_i - theta_j)^T F(theta_i) (theta_i - theta_j))
        W_ij = exp(-gamma * d_IG(theta_i, theta_j))

    Parameters
    ----------
    thetas : {agent_id: theta}, flat parameter vectors (all same shape).
    fishers : {agent_id: F(theta)}, same keys as `thetas`.
    diagonal : whether fishers hold diagonal (P,) or full (P, P) FIMs.
    symmetrize : None (default) keeps the raw directed d_IG(i, j) as given
        by the formula (uses F evaluated at the source node i, matching a
        directed multi-agent graph G=(V,E,W)); 'average' uses
        0.5*(d_ij + d_ji); 'max' uses max(d_ij, d_ji) -- either yields a
        symmetric metric suitable for an undirected graph.
    zero_diagonal : if True, W_ii = 0 (no self-loop weight); otherwise
        W_ii = exp(0) = 1.
    gamma : bandwidth of the exponential kernel (see `edge_weight_kernel`).
        Larger gamma sharpens the falloff and increases underflow risk for
        widely-separated agents; smaller gamma flattens W towards 1
        everywhere. Must be finite and > 0.

    Returns
    -------
    (W, nodes) where W is a (n_agents, n_agents) torch.DoubleTensor with
    W[a, b] = W_ij for nodes[a] -> nodes[b], and `nodes` is the row/column
    order used.
    """
    nodes = list(thetas.keys())
    if set(nodes) != set(fishers.keys()):
        raise ValueError("`thetas` and `fishers` must have exactly the same agent keys.")
    n = len(nodes)

    D = torch.zeros(n, n, dtype=torch.float64)
    for a, i in enumerate(nodes):
        for b, j in enumerate(nodes):
            if a == b:
                continue
            D[a, b] = geodesic_distance(thetas[i], thetas[j], fishers[i], diagonal=diagonal)

    if symmetrize == "average":
        D = 0.5 * (D + D.T)
    elif symmetrize == "max":
        D = torch.maximum(D, D.T)
    elif symmetrize is not None:
        raise ValueError(f"Unknown symmetrize option: {symmetrize!r}")

    W = edge_weight_kernel(D, gamma=gamma)
    if zero_diagonal:
        W.fill_diagonal_(0.0)
    else:
        W.fill_diagonal_(1.0)

    logger.info(
        "Built %d x %d Fisher-Rao edge-weight matrix (symmetrize=%s, diagonal=%s, gamma=%.4g)",
        n, n, symmetrize, diagonal, gamma,
    )
    return W, nodes


# --------------------------------------------------------------------------
# Optional bridge into agent_retrieval_graph.py's structural-bottleneck tools
# --------------------------------------------------------------------------

def weights_to_digraph(W: torch.Tensor, nodes: List[Any], weight_floor: float = 1e-9):
    """
    Convert an edge-weight matrix W (as returned by `fisher_rao_edge_weights`)
    into an `nx.DiGraph`, so it can be fed directly into the algebraic-
    connectivity / edge-pruning pipeline in `agent_retrieval_graph.py`
    (normalized_laplacian, algebraic_connectivity, prune_edges, ...).
    """
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    W_np = W.detach().cpu().numpy()
    for a, u in enumerate(nodes):
        for b, v in enumerate(nodes):
            if a == b:
                continue
            w = float(W_np[a, b])
            if w > weight_floor:
                G.add_edge(u, v, weight=w)
    return G


def sparsify_top_k(W: torch.Tensor, k: int) -> torch.Tensor:
    """
    Row-wise top-k sparsification of an edge-weight matrix: for each source
    node i, keep only its k strongest *outgoing* weights (W[i, :]) and zero
    the rest, preserving directed out-degree <= k per node (self-loops,
    i.e. the diagonal, are never selected). Ties are broken arbitrarily by
    argsort order.

    This is a per-node cap, which is a different sparsification policy from
    `sir_prune_to_budget` in evaluation_harness.py (which removes globally
    weakest edges down to a *total* edge-count budget, so out-degree can
    vary a lot node to node). Row-wise top-k instead guarantees every agent
    keeps its k best routing partners regardless of how "central" or
    "peripheral" that agent's Fisher-Rao neighborhood is.

    Only entries with strictly positive weight are kept, so a node with
    fewer than k positive outgoing weights ends up with fewer than k edges
    rather than picking up zero/negative padding.
    """
    if k < 0:
        raise ValueError(f"k must be >= 0; got {k!r}")
    n = W.shape[0]
    W_np = W.detach().cpu().numpy()
    out = np.zeros_like(W_np)
    for i in range(n):
        row = W_np[i].copy()
        row[i] = -np.inf  # never select the self-loop
        if k > 0:
            top_idx = np.argsort(row)[-k:]
            for j in top_idx:
                if row[j] > 0:
                    out[i, j] = W_np[i, j]
    return torch.as_tensor(out, dtype=W.dtype)


# --------------------------------------------------------------------------
# Log-likelihood functions
# --------------------------------------------------------------------------

def linear_gaussian_log_likelihood(model: nn.Module, example: Batch) -> torch.Tensor:
    """
    log p_theta(y | x) for a linear-Gaussian model y ~ N(model(x), I),
    single example. Constant terms independent of theta are dropped (they
    don't affect gradients, and only the gradient/curvature of log p is
    ever used downstream).
    """
    x, y = example["x"], example["y"]
    pred = model(x)
    return -0.5 * ((y - pred) ** 2).sum()


def hf_causal_lm_log_likelihood(model: nn.Module, example: Batch) -> torch.Tensor:
    """
    log p_theta(y | x) for a Hugging Face causal LM, single example.
    Relies on the model's internal shifted cross-entropy loss (mean over
    non-ignored target tokens) and rescales by the active-token count to
    recover the *summed* log-likelihood for that example.
    """
    outputs = model(**example)
    loss = outputs.loss  # mean NLL over active (non -100) label positions
    labels = example["labels"]
    num_active = (labels != -100).sum().clamp_min(1)
    return -loss * num_active


# --------------------------------------------------------------------------
# Demo 1: synthetic linear-Gaussian "agents" (always runs, no downloads)
# --------------------------------------------------------------------------

def _make_agent_model(in_dim: int, out_dim: int, seed: int) -> nn.Linear:
    g = torch.Generator().manual_seed(seed)
    model = nn.Linear(in_dim, out_dim)
    with torch.no_grad():
        model.weight.copy_(torch.randn(model.weight.shape, generator=g))
        model.bias.copy_(torch.randn(model.bias.shape, generator=g))
    return model


def demo_synthetic_linear_agents() -> None:
    """
    3 agents, each a small linear-Gaussian model whose weights have
    diverged (simulating independently-updated local agent states). For
    each agent we estimate its full empirical FIM from a handful of local
    synthetic (x, y) samples, then build the directed Fisher-Rao
    edge-weight matrix W_ij = exp(-d_IG(theta_i, theta_j)).
    """
    torch.manual_seed(0)
    in_dim, out_dim = 4, 2
    n_local_samples = 64

    agent_ids = ["agent_0", "agent_1", "agent_2"]
    thetas: Dict[str, torch.Tensor] = {}
    fishers: Dict[str, torch.Tensor] = {}

    for seed, agent_id in enumerate(agent_ids):
        model = _make_agent_model(in_dim, out_dim, seed=seed)
        params = [p for p in model.parameters() if p.requires_grad]

        x = torch.randn(n_local_samples, in_dim)
        y = model(x).detach() + 0.1 * torch.randn(n_local_samples, out_dim)
        examples = iter_single_examples({"x": x, "y": y})

        theta, fisher, _ = compute_empirical_fisher(
            model, examples, linear_gaussian_log_likelihood, params=params, diagonal=False
        )
        thetas[agent_id] = theta
        fishers[agent_id] = fisher

    W, nodes = fisher_rao_edge_weights(thetas, fishers, diagonal=False, symmetrize=None)

    print("\n=== Synthetic linear-agent demo ===")
    print("Agents:", nodes)
    print("Edge-weight matrix W_ij = exp(-d_IG(theta_i, theta_j)):")
    print(W.numpy().round(4))

    try:
        G = weights_to_digraph(W, nodes)
        from agent_retrieval_graph import compute_pipeline_lambda2  # sibling script

        lam2 = compute_pipeline_lambda2(G)
        print(f"Resulting graph: |V|={G.number_of_nodes()}, |E|={G.number_of_edges()}, "
              f"algebraic connectivity lambda_2={lam2:.6f}")
    except ImportError:
        logger.info("agent_retrieval_graph.py not importable from here; skipping graph-connectivity bridge.")


# --------------------------------------------------------------------------
# Demo 2: real Hugging Face causal LM agents (diagonal Fisher; optional)
# --------------------------------------------------------------------------

def demo_hf_causal_lm_agents(checkpoint: str = "sshleifer/tiny-gpt2") -> None:
    """
    Same pipeline as demo 1, but with two "agents" that are copies of a
    real (tiny) Hugging Face causal LM checkpoint, one of them fine-tuned
    for a few synthetic steps so their parameters diverge. Uses the
    diagonal empirical Fisher, as required for any real LM-scale model.

    Requires `transformers` and network access to fetch the checkpoint;
    fails gracefully (logs and returns) if either is unavailable.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        logger.warning("transformers is not installed; skipping HF causal LM demo.")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(checkpoint)
    except Exception as exc:  # offline / hub error / bad checkpoint name
        logger.warning("Could not load checkpoint %r (%s); skipping HF causal LM demo.", checkpoint, exc)
        return

    texts = [
        "The retrieval agent queried the knowledge base.",
        "Graph connectivity determines information flow between agents.",
        "Fisher information measures curvature of the log-likelihood.",
        "Empirical Fisher approximates the true Fisher via observed samples.",
    ]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    labels = enc["input_ids"].clone()
    labels[enc["attention_mask"] == 0] = -100
    batch = {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}

    def agent_fisher(model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor, List[nn.Parameter]]:
        params = [p for p in model.parameters() if p.requires_grad]
        examples = list(iter_single_examples(batch))
        return compute_empirical_fisher(
            model, examples, hf_causal_lm_log_likelihood, params=params, diagonal=True
        )

    model_a = base_model
    theta_a, fisher_a, params_a = agent_fisher(model_a)

    # Simulate a second agent that has drifted via a few local SGD steps.
    model_b = AutoModelForCausalLM.from_pretrained(checkpoint)
    opt = torch.optim.SGD(model_b.parameters(), lr=0.05)
    for _ in range(3):
        opt.zero_grad()
        out = model_b(**batch)
        out.loss.backward()
        opt.step()
    theta_b, fisher_b, params_b = agent_fisher(model_b)

    thetas = {"agent_a": theta_a, "agent_b": theta_b}
    fishers = {"agent_a": fisher_a, "agent_b": fisher_b}
    W, nodes = fisher_rao_edge_weights(thetas, fishers, diagonal=True, symmetrize="average")

    print(f"\n=== HF causal LM demo ({checkpoint}) ===")
    print("Agents:", nodes)
    print("Edge-weight matrix W_ij = exp(-d_IG(theta_i, theta_j)):")
    print(W.numpy().round(6))


if __name__ == "__main__":
    demo_synthetic_linear_agents()
    demo_hf_causal_lm_agents()
