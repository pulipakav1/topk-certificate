"""fisher toolkit"""

from __future__ import annotations

import logging
import math
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

Batch = Dict[str, torch.Tensor]
LogLikelihoodFn = Callable[[nn.Module, Batch], torch.Tensor]


# --------------------------------------------------------------------------
# score gradients
# --------------------------------------------------------------------------

def collect_score_gradients(
    model: nn.Module,
    examples: Iterable[Batch],
    log_likelihood_fn: LogLikelihoodFn,
    params: Optional[List[nn.Parameter]] = None,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, List[nn.Parameter]]:
    """score gradients"""
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

    grad_logs = torch.stack(rows, dim=0)  # shape hint
    logger.info("Collected gradient logs: N=%d examples, P=%d parameters", *grad_logs.shape)
    return grad_logs, params


def iter_single_examples(batch: Batch) -> Iterable[Batch]:
    """split batch"""
    keys = list(batch.keys())
    bsz = batch[keys[0]].shape[0]
    for i in range(bsz):
        yield {k: v[i : i + 1] for k, v in batch.items()}


def flatten_params(params: List[nn.Parameter]) -> torch.Tensor:
    """flatten params"""
    return torch.cat([p.detach().reshape(-1).double() for p in params])


# --------------------------------------------------------------------------
# build fisher
# --------------------------------------------------------------------------

_FULL_FISHER_PARAM_WARN_THRESHOLD = 20_000  # memory warning


def empirical_fisher(grad_logs: torch.Tensor, diagonal: bool = True) -> torch.Tensor:
    """build fisher"""
    if grad_logs.dim() != 2:
        raise ValueError(f"grad_logs must be 2D (N, P); got shape {tuple(grad_logs.shape)}")
    n, p = grad_logs.shape
    grad_logs = grad_logs.double()

    if diagonal:
        return (grad_logs ** 2).mean(dim=0)  # shape hint

    if p > _FULL_FISHER_PARAM_WARN_THRESHOLD:
        logger.warning(
            "Computing a full %d x %d Fisher matrix (%.2f GB as float64) -- "
            "consider diagonal=True for models at this scale.",
            p, p, (p * p * 8) / 1e9,
        )
    return (grad_logs.T @ grad_logs) / n  # shape hint


def compute_empirical_fisher(
    model: nn.Module,
    examples: Iterable[Batch],
    log_likelihood_fn: LogLikelihoodFn,
    params: Optional[List[nn.Parameter]] = None,
    diagonal: bool = True,
    max_samples: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[nn.Parameter]]:
    """wrapper fisher"""
    grad_logs, params = collect_score_gradients(
        model, examples, log_likelihood_fn, params=params, max_samples=max_samples
    )
    theta = flatten_params(params)
    fisher = empirical_fisher(grad_logs, diagonal=diagonal)
    return theta, fisher, params


# --------------------------------------------------------------------------
# geodesic distance
# --------------------------------------------------------------------------

def geodesic_distance(
    theta_i: torch.Tensor,
    theta_j: torch.Tensor,
    fisher_i: torch.Tensor,
    diagonal: bool = True,
) -> torch.Tensor:
    """geodesic distance"""
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

    # clamp noise
    quad = torch.clamp(quad, min=0.0)
    return torch.sqrt(quad)


# --------------------------------------------------------------------------
# edge weights
# --------------------------------------------------------------------------

def edge_weight_kernel(distance: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """edge weights"""
    if not (math.isfinite(gamma) and gamma > 0):
        raise ValueError(f"gamma must be a finite positive float; got {gamma!r}")
    distance = distance.clamp_min(0.0)  # always nonnegative
    return torch.exp(-gamma * distance)


def auto_kernel_gamma(
    thetas: Dict[Any, torch.Tensor],
    fishers: Dict[Any, torch.Tensor],
    diagonal: bool = True,
    target_median_distance: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """pick gamma"""
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
    """pairwise weights"""
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
# graph bridge
# --------------------------------------------------------------------------

def weights_to_digraph(W: torch.Tensor, nodes: List[Any], weight_floor: float = 1e-9):
    """make graph"""
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
    """topk sparsify"""
    if k < 0:
        raise ValueError(f"k must be >= 0; got {k!r}")
    n = W.shape[0]
    W_np = W.detach().cpu().numpy()
    out = np.zeros_like(W_np)
    for i in range(n):
        row = W_np[i].copy()
        row[i] = -np.inf  # skip self-loop
        if k > 0:
            top_idx = np.argsort(row)[-k:]
            for j in top_idx:
                if row[j] > 0:
                    out[i, j] = W_np[i, j]
    return torch.as_tensor(out, dtype=W.dtype)


# --------------------------------------------------------------------------
# log likelihood
# --------------------------------------------------------------------------

def linear_gaussian_log_likelihood(model: nn.Module, example: Batch) -> torch.Tensor:
    """gaussian likelihood"""
    x, y = example["x"], example["y"]
    pred = model(x)
    return -0.5 * ((y - pred) ** 2).sum()


def hf_causal_lm_log_likelihood(model: nn.Module, example: Batch) -> torch.Tensor:
    """lm likelihood"""
    outputs = model(**example)
    loss = outputs.loss  # mask padding
    labels = example["labels"]
    num_active = (labels != -100).sum().clamp_min(1)
    return -loss * num_active


# --------------------------------------------------------------------------
# synthetic demo
# --------------------------------------------------------------------------

def _make_agent_model(in_dim: int, out_dim: int, seed: int) -> nn.Linear:
    g = torch.Generator().manual_seed(seed)
    model = nn.Linear(in_dim, out_dim)
    with torch.no_grad():
        model.weight.copy_(torch.randn(model.weight.shape, generator=g))
        model.bias.copy_(torch.randn(model.bias.shape, generator=g))
    return model


def demo_synthetic_linear_agents() -> None:
    """synthetic demo"""
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
# real demo
# --------------------------------------------------------------------------

def demo_hf_causal_lm_agents(checkpoint: str = "sshleifer/tiny-gpt2") -> None:
    """real demo"""
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
    except Exception as exc:  # network fallback
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

    # drift agent
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
