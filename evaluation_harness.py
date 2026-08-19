"""run evaluation"""

from __future__ import annotations

import csv
import logging
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf

from agent_retrieval_graph import compute_pipeline_lambda2, weighted_adjacency
from fisher_information_geometry import (
    auto_kernel_gamma,
    compute_empirical_fisher,
    fisher_rao_edge_weights,
    flatten_params,
    geodesic_distance,
    iter_single_examples,
    linear_gaussian_log_likelihood,
    sparsify_top_k,
    weights_to_digraph,
)
from theorem1_diagnostics import (
    max_edge_geodesic_distance_sq,
    non_consensus_variance,
    routing_mass_diagnostic_batch,
)
from margin_diagnostics import aggregate_margin_diagnostics

try:
    from datasets import load_dataset
    _DATASETS_AVAILABLE = True
except ImportError:
    _DATASETS_AVAILABLE = False

try:
    from transformers import AutoModel, AutoTokenizer
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# config metrics
# --------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    n_agents: int = 10
    d_in: int = 6
    d_out: int = 5
    n_docs: int = 24
    n_hops: int = 5
    k: int = 5                     # eval depth
    diffusion_alpha: float = 0.6   # blend rate
    baseline_top_k: int = 3        # neighbor count
    local_fisher_samples: int = 30
    noise_std: float = 0.01
    seed: int = 42


@dataclass
class HopMetrics:
    hop: int
    lambda2: float
    ndcg_at_k: float
    f1_at_k: float
    mean_param_drift: float


# --------------------------------------------------------------------------
# retrieval metrics
# --------------------------------------------------------------------------

def ndcg_at_k(scores: np.ndarray, relevance: np.ndarray, k: int) -> float:
    """compute ndcg"""
    k = min(k, len(scores))
    order = np.argsort(-scores)[:k]
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float(np.sum(relevance[order] * discounts))
    ideal_order = np.argsort(-relevance)[:k]
    idcg = float(np.sum(relevance[ideal_order] * discounts))
    return dcg / idcg if idcg > 0 else 0.0


def f1_at_k(scores: np.ndarray, relevance_binary: np.ndarray, k: int) -> float:
    """compute f1"""
    k = min(k, len(scores))
    retrieved = set(np.argsort(-scores)[:k].tolist())
    relevant = set(np.where(relevance_binary > 0)[0].tolist())
    if not relevant:
        return 0.0
    tp = len(retrieved & relevant)
    precision = tp / k
    recall = tp / len(relevant)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# --------------------------------------------------------------------------
# build routing
# --------------------------------------------------------------------------

def build_baseline_graph(theta_matrix: np.ndarray, nodes: List[str], top_k: int) -> nx.DiGraph:
    """baseline graph"""
    X = theta_matrix / (np.linalg.norm(theta_matrix, axis=1, keepdims=True) + 1e-12)
    sim = X @ X.T
    np.fill_diagonal(sim, -np.inf)

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    n = len(nodes)
    for v in range(n):
        top_u = np.argsort(-sim[v])[:top_k]
        for u in top_u:
            w = float(max(sim[v, u], 0.0))
            if w > 0:
                G.add_edge(nodes[u], nodes[v], weight=w)
    return G


def sir_prune_to_budget(G: nx.DiGraph, target_edges: int, zero_tol: float = 1e-9) -> nx.DiGraph:
    """prune budget"""
    H = G.copy()
    target_edges = max(target_edges, H.number_of_nodes() - 1)
    weakest_first = sorted(H.edges(data="weight"), key=lambda e: e[2])

    for u, v, w in weakest_first:
        if H.number_of_edges() <= target_edges:
            break
        if not H.has_edge(u, v):
            continue
        H.remove_edge(u, v)
        if compute_pipeline_lambda2(H) < zero_tol:
            H.add_edge(u, v, weight=w)  # keep bottleneck
    return H


def build_sir_graph(
    models: List[nn.Module],
    params_list: List[List[nn.Parameter]],
    nodes: List[str],
    cfg: ExperimentConfig,
    target_edges: int,
) -> nx.DiGraph:
    """sir graph"""
    thetas: Dict[str, torch.Tensor] = {}
    fishers: Dict[str, torch.Tensor] = {}

    for name, model, params in zip(nodes, models, params_list):
        x = torch.randn(cfg.local_fisher_samples, cfg.d_in)
        with torch.no_grad():
            y = model(x) + 0.05 * torch.randn(cfg.local_fisher_samples, cfg.d_out)
        examples = list(iter_single_examples({"x": x, "y": y}))
        theta, fisher, _ = compute_empirical_fisher(
            model, examples, linear_gaussian_log_likelihood, params=params, diagonal=True
        )
        thetas[name] = theta
        fishers[name] = fisher

    W, node_order = fisher_rao_edge_weights(
        thetas, fishers, diagonal=True, symmetrize="average", zero_diagonal=True
    )
    G_full = weights_to_digraph(W, node_order, weight_floor=1e-9)
    return sir_prune_to_budget(G_full, target_edges=target_edges)


# --------------------------------------------------------------------------
# shared setup
# --------------------------------------------------------------------------

def unflatten_into(params: List[nn.Parameter], flat: torch.Tensor) -> None:
    idx = 0
    with torch.no_grad():
        for p in params:
            n = p.numel()
            p.copy_(flat[idx : idx + n].reshape(p.shape).to(p.dtype))
            idx += n


def build_shared_setup(cfg: ExperimentConfig) -> SimpleNamespace:
    g = torch.Generator().manual_seed(cfg.seed)

    true_model = nn.Linear(cfg.d_in, cfg.d_out)
    with torch.no_grad():
        true_model.weight.copy_(torch.randn(true_model.weight.shape, generator=g))
        true_model.bias.copy_(torch.randn(true_model.bias.shape, generator=g))

    probe_x = torch.randn(1, cfg.d_in, generator=g)
    docs = torch.randn(cfg.n_docs, cfg.d_out, generator=g)
    docs = docs / docs.norm(dim=1, keepdim=True)

    with torch.no_grad():
        q_true = true_model(probe_x).squeeze(0)
    q_true = q_true / q_true.norm()
    sim_to_truth = (docs @ q_true).numpy()
    order = np.argsort(-sim_to_truth)

    relevance = np.zeros(cfg.n_docs, dtype=np.float64)
    n_top = max(1, cfg.n_docs // 5)
    n_mid = max(1, cfg.n_docs // 3)
    relevance[order[:n_top]] = 2.0
    relevance[order[n_top : n_top + n_mid]] = 1.0
    relevance_binary = (relevance >= 1.0).astype(np.float64)

    agent_states = []
    expert_idx = 0
    for i in range(cfg.n_agents):
        m = nn.Linear(cfg.d_in, cfg.d_out)
        if i == expert_idx:
            m.load_state_dict(true_model.state_dict())
        else:
            gi = torch.Generator().manual_seed(cfg.seed * 1000 + i)
            with torch.no_grad():
                m.weight.copy_(torch.randn(m.weight.shape, generator=gi))
                m.bias.copy_(torch.randn(m.bias.shape, generator=gi))
        agent_states.append({k: v.clone() for k, v in m.state_dict().items()})

    nodes = [f"agent_{i}" for i in range(cfg.n_agents)]
    return SimpleNamespace(
        probe_x=probe_x, docs=docs, relevance=relevance, relevance_binary=relevance_binary,
        agent_states=agent_states, nodes=nodes, expert_idx=expert_idx,
    )


# --------------------------------------------------------------------------
# run strategy
# --------------------------------------------------------------------------

def run_experiment(strategy: str, cfg: ExperimentConfig, setup: SimpleNamespace) -> List[HopMetrics]:
    assert strategy in ("baseline", "sir")
    nodes = setup.nodes
    n = cfg.n_agents

    models: List[nn.Module] = []
    params_list: List[List[nn.Parameter]] = []
    for i in range(n):
        m = nn.Linear(cfg.d_in, cfg.d_out)
        m.load_state_dict({k: v.clone() for k, v in setup.agent_states[i].items()})
        models.append(m)
        params_list.append([p for p in m.parameters() if p.requires_grad])

    theta0 = np.stack([flatten_params(params_list[i]).numpy() for i in range(n)], axis=0)
    Theta = theta0.copy()

    history: List[HopMetrics] = []

    for hop in range(1, cfg.n_hops + 1):
        torch.manual_seed(cfg.seed * 7919 + hop)
        np.random.seed(cfg.seed * 7919 + hop)

        if strategy == "baseline":
            G = build_baseline_graph(Theta, nodes, top_k=cfg.baseline_top_k)
        else:
            G = build_sir_graph(models, params_list, nodes, cfg, target_edges=n * cfg.baseline_top_k)

        lam2 = compute_pipeline_lambda2(G)

        A = weighted_adjacency(G, nodelist=nodes).toarray()  # edge weight
        in_deg = A.sum(axis=0)
        P = np.zeros_like(A)
        receiving = in_deg > 1e-12
        P[:, receiving] = A[:, receiving] / in_deg[receiving]
        incoming_avg = P.T @ Theta  # weighted average

        Theta_new = Theta.copy()
        Theta_new[receiving] = (
            (1 - cfg.diffusion_alpha) * Theta[receiving] + cfg.diffusion_alpha * incoming_avg[receiving]
        )
        Theta_new += cfg.noise_std * np.random.randn(*Theta_new.shape)
        Theta = Theta_new

        for i in range(n):
            unflatten_into(params_list[i], torch.from_numpy(Theta[i]))

        ndcgs, f1s = [], []
        with torch.no_grad():
            for i in range(n):
                q = models[i](setup.probe_x).squeeze(0)
                q = q / (q.norm() + 1e-12)
                scores = (setup.docs @ q).numpy()
                ndcgs.append(ndcg_at_k(scores, setup.relevance, cfg.k))
                f1s.append(f1_at_k(scores, setup.relevance_binary, cfg.k))

        drift = float(np.linalg.norm(Theta - theta0, axis=1).mean())

        history.append(
            HopMetrics(
                hop=hop, lambda2=lam2,
                ndcg_at_k=float(np.mean(ndcgs)), f1_at_k=float(np.mean(f1s)),
                mean_param_drift=drift,
            )
        )
        logger.info(
            "[%-8s] hop %d/%d: lambda2=%.4f NDCG@%d=%.4f F1@%d=%.4f drift=%.4f edges=%d",
            strategy, hop, cfg.n_hops, lam2, cfg.k, history[-1].ndcg_at_k,
            cfg.k, history[-1].f1_at_k, drift, G.number_of_edges(),
        )

    return history


# --------------------------------------------------------------------------
# make plots
# --------------------------------------------------------------------------

IEEE_COLUMN_WIDTH_IN = 3.45   # single column
IEEE_PAGE_WIDTH_IN = 7.16     # double column


def _set_ieee_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "lines.linewidth": 1.3,
        "lines.markersize": 4,
        "axes.linewidth": 0.6,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "axes.grid": True,
        "pdf.fonttype": 42,   # embed fonts
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


BASELINE_STYLE = dict(color="#4C72B0", marker="o", linestyle="--", label="Baseline (cosine MP)")
SIR_STYLE = dict(color="#C44E52", marker="s", linestyle="-", label="Proposed (SIR)")


def plot_ieee_figures(
    baseline_hist: List[HopMetrics], sir_hist: List[HopMetrics], k: int, out_dir: str = "figures"
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    _set_ieee_style()

    hops = [m.hop for m in baseline_hist]
    series = {
        "lambda2": ([m.lambda2 for m in baseline_hist], [m.lambda2 for m in sir_hist], r"Algebraic connectivity $\lambda_2$"),
        "ndcg": ([m.ndcg_at_k for m in baseline_hist], [m.ndcg_at_k for m in sir_hist], f"NDCG@{k}"),
        "f1": ([m.f1_at_k for m in baseline_hist], [m.f1_at_k for m in sir_hist], f"F1@{k}"),
        "drift": (
            [m.mean_param_drift for m in baseline_hist], [m.mean_param_drift for m in sir_hist],
            r"Mean parameter drift $\|\theta_t-\theta_0\|_2$",
        ),
    }

    fig, axes = plt.subplots(2, 2, figsize=(IEEE_PAGE_WIDTH_IN, IEEE_PAGE_WIDTH_IN * 0.62))
    for ax, key in zip(axes.flat, ["lambda2", "ndcg", "f1", "drift"]):
        b_vals, s_vals, ylabel = series[key]
        ax.plot(hops, b_vals, **BASELINE_STYLE)
        ax.plot(hops, s_vals, **SIR_STYLE)
        ax.set_xlabel("Multi-hop reasoning step $t$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(hops)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext in ("pdf", "eps"):
        fig.savefig(os.path.join(out_dir, f"routing_comparison_trends.{ext}"))
    plt.close(fig)

    b_lam2, s_lam2, _ = series["lambda2"]
    b_ndcg, s_ndcg, _ = series["ndcg"]

    fig2, ax2 = plt.subplots(figsize=(IEEE_COLUMN_WIDTH_IN, IEEE_COLUMN_WIDTH_IN * 0.8))
    ax2.plot(b_lam2, b_ndcg, **BASELINE_STYLE)
    ax2.plot(s_lam2, s_ndcg, **SIR_STYLE)
    for xs, ys in ((b_lam2, b_ndcg), (s_lam2, s_ndcg)):
        for t, x, y in zip(hops, xs, ys):
            ax2.annotate(str(t), (x, y), fontsize=6, textcoords="offset points", xytext=(3, 3))
    ax2.set_xlabel(r"Algebraic connectivity $\lambda_2$")
    ax2.set_ylabel(f"NDCG@{k}")
    ax2.legend(frameon=False, loc="lower right")
    fig2.tight_layout()
    for ext in ("pdf", "eps"):
        fig2.savefig(os.path.join(out_dir, f"lambda2_vs_ndcg.{ext}"))
    plt.close(fig2)

    logger.info("Saved IEEE-formatted vector figures to %s/", out_dir)


# --------------------------------------------------------------------------
# print summary
# --------------------------------------------------------------------------

def save_results_csv(
    baseline_hist: List[HopMetrics], sir_hist: List[HopMetrics], out_dir: str = "results"
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "routing_comparison_metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["strategy", "hop", "lambda2", "ndcg_at_k", "f1_at_k", "mean_param_drift"])
        for label, hist in (("baseline", baseline_hist), ("sir", sir_hist)):
            for m in hist:
                writer.writerow([label, m.hop, m.lambda2, m.ndcg_at_k, m.f1_at_k, m.mean_param_drift])
    logger.info("Saved raw metrics to %s", path)
    return path


def print_summary(baseline_hist: List[HopMetrics], sir_hist: List[HopMetrics]) -> None:
    b_final, s_final = baseline_hist[-1], sir_hist[-1]
    print("\n=== Final-hop comparison (hop 5) ===")
    print(f"{'metric':<14}{'baseline':>12}{'SIR':>12}{'delta':>12}")
    for name, bv, sv in [
        ("lambda_2", b_final.lambda2, s_final.lambda2),
        ("NDCG@k", b_final.ndcg_at_k, s_final.ndcg_at_k),
        ("F1@k", b_final.f1_at_k, s_final.f1_at_k),
        ("param_drift", b_final.mean_param_drift, s_final.mean_param_drift),
    ]:
        print(f"{name:<14}{bv:>12.4f}{sv:>12.4f}{sv - bv:>12.4f}")

    all_lam2 = [m.lambda2 for m in baseline_hist] + [m.lambda2 for m in sir_hist]
    all_ndcg = [m.ndcg_at_k for m in baseline_hist] + [m.ndcg_at_k for m in sir_hist]
    corr = float(np.corrcoef(all_lam2, all_ndcg)[0, 1]) if np.std(all_lam2) > 0 else float("nan")
    print(f"\nPearson corr(lambda_2, NDCG@k) across both strategies x hops: {corr:.4f}")


# --------------------------------------------------------------------------
# hotpotqa pipeline
# --------------------------------------------------------------------------
#
# shard mapping
# context slots
# gold distractors
# frozen encoder
# fixed embeddings
# project embeddings
# trainable head
# own shard
# shared routing
# propagate geometry
# preserve routing

@dataclass
class RealDataConfig:
    n_agents: int = 10
    n_eval_queries: int = 40
    n_hops: int = 5
    k: int = 5                      # eval depth
    diffusion_alpha: float = 0.6
    baseline_top_k: int = 3
    noise_std: float = 0.005
    seed: int = 42
    lm_checkpoint: str = "distilgpt2"
    proj_dim: int = 32              # projection dim
    embed_batch_size: int = 16
    max_seq_len: int = 96
    device: str = "cpu"             # encoder device
    sir_gamma: Optional[float] = None  # kernel gamma
    sir_top_k: int = 3               # edge budget
    dataset_slice: str = "validation[:1000]"


@dataclass
class HotpotHopMetrics:
    hop: int
    lambda2: float
    ndcg_at_k: float
    f1_at_k: float
    cumulative_d_ig: float
    # theorem diagnostics
    delta_ig_sq: float = 0.0
    non_consensus_variance: Optional[float] = None
    theorem1_bound: float = 0.0
    # routing mass
    gold_inflow_mass: float = float("nan")
    distractor_inflow_mass: float = float("nan")
    gold_distractor_ratio: float = float("nan")
    mean_transition_entropy: float = 0.0
    # margin diagnostic
    mean_margin: float = float("nan")
    mean_hard_margin: float = float("nan")
    margin_violation_rate: float = float("nan")
    mean_margin_standardized: float = float("nan")
    mean_hard_margin_standardized: float = float("nan")


def _load_hotpotqa_shards(cfg: RealDataConfig) -> List[dict]:
    """load hotpotqa"""
    if not _DATASETS_AVAILABLE:
        raise ImportError("The `datasets` package is required for the HotpotQA pipeline (pip install datasets).")

    logger.info("Loading hotpot_qa/distractor %s ...", cfg.dataset_slice)
    ds = load_dataset("hotpot_qa", "distractor", split=cfg.dataset_slice)

    records: List[dict] = []
    for ex in ds:
        titles = ex["context"]["title"]
        sentences = ex["context"]["sentences"]
        if len(titles) < cfg.n_agents:
            continue
        supporting_titles = set(ex["supporting_facts"]["title"])
        paragraphs, relevance = [], []
        for j in range(cfg.n_agents):
            text = f"{titles[j]}: {' '.join(sentences[j])}".strip()
            paragraphs.append(text)
            relevance.append(1.0 if titles[j] in supporting_titles else 0.0)
        records.append({
            "question": ex["question"],
            "paragraphs": paragraphs,
            "relevance": np.array(relevance, dtype=np.float64),
        })
        if len(records) >= cfg.n_eval_queries:
            break

    if len(records) < cfg.n_eval_queries:
        logger.warning(
            "Only found %d usable examples (wanted %d) within %s.",
            len(records), cfg.n_eval_queries, cfg.dataset_slice,
        )
    logger.info("Selected %d HotpotQA queries, %d paragraphs each.", len(records), cfg.n_agents)
    return records


@torch.no_grad()
def _embed_texts(model, tokenizer, texts: List[str], cfg: RealDataConfig) -> torch.Tensor:
    """embed texts"""
    device = torch.device(cfg.device)
    all_embeds = []
    for start in range(0, len(texts), cfg.embed_batch_size):
        batch_texts = texts[start : start + cfg.embed_batch_size]
        enc = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_seq_len
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = model(**enc).last_hidden_state  # shape hint
        mask = enc["attention_mask"].unsqueeze(-1).float()  # shape hint
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        all_embeds.append(pooled.cpu())
    return torch.cat(all_embeds, dim=0)


def retrieval_log_likelihood(head: nn.Module, example: Dict[str, torch.Tensor]) -> torch.Tensor:
    """score likelihood"""
    q = head(example["e_q"])
    d = head(example["e_doc"])
    score = (q * d).sum(dim=-1)
    sign = 2.0 * example["y"] - 1.0
    return -nnf.softplus(-sign * score).squeeze()


def _build_agent_shards(
    cfg: RealDataConfig, records: List[dict], q_embeds: torch.Tensor, doc_embeds: torch.Tensor
) -> List[List[Dict[str, torch.Tensor]]]:
    """build shards"""
    shards: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(cfg.n_agents)]
    for i, rec in enumerate(records):
        for j in range(cfg.n_agents):
            shards[j].append({
                "e_q": q_embeds[i : i + 1],
                "e_doc": doc_embeds[i * cfg.n_agents + j : i * cfg.n_agents + j + 1],
                "y": torch.tensor([rec["relevance"][j]], dtype=torch.float32),
            })
    return shards


def run_hotpotqa_experiment(
    strategy: str,
    cfg: RealDataConfig,
    records: List[dict],
    shards: List[List[Dict[str, torch.Tensor]]],
    init_state_dicts: List[dict],
) -> List[HotpotHopMetrics]:
    assert strategy in ("baseline", "sir")
    nodes = [f"agent_{j}" for j in range(cfg.n_agents)]
    n = cfg.n_agents

    heads: List[nn.Linear] = []
    params_list: List[List[nn.Parameter]] = []
    for j in range(n):
        h = nn.Linear(cfg.proj_dim, cfg.proj_dim)
        h.load_state_dict({k: v.clone() for k, v in init_state_dicts[j].items()})
        heads.append(h)
        params_list.append([p for p in h.parameters() if p.requires_grad])

    theta0 = np.stack([flatten_params(params_list[j]).numpy() for j in range(n)], axis=0)
    Theta = theta0.copy()

    history: List[HotpotHopMetrics] = []
    cumulative_d_ig = 0.0
    theorem1_B = 0.0  # theorem recursion

    for hop in range(1, cfg.n_hops + 1):
        torch.manual_seed(cfg.seed * 7919 + hop)
        np.random.seed(cfg.seed * 7919 + hop)

        # local fisher
        # metric tensor
        # drift metric
        thetas: Dict[str, torch.Tensor] = {}
        fishers: Dict[str, torch.Tensor] = {}
        for j, name in enumerate(nodes):
            theta_j, fisher_j, _ = compute_empirical_fisher(
                heads[j], shards[j], retrieval_log_likelihood, params=params_list[j], diagonal=True
            )
            thetas[name], fishers[name] = theta_j, fisher_j

        if strategy == "baseline":
            G = build_baseline_graph(Theta, nodes, top_k=cfg.baseline_top_k)
        else:
            # calibrate kernel
            # avoid underflow
            # avoid edgeless
            # fixed gamma
            # less diffusion
            gamma = cfg.sir_gamma if cfg.sir_gamma is not None else auto_kernel_gamma(thetas, fishers, diagonal=True)
            W, node_order = fisher_rao_edge_weights(
                thetas, fishers, diagonal=True, symmetrize="average", zero_diagonal=True, gamma=gamma
            )
            # cap outgoing
            # top partners
            # limit spread
            W = sparsify_top_k(W, k=cfg.sir_top_k)
            G_full = weights_to_digraph(W, node_order, weight_floor=1e-9)
            # safety net
            # extra pruning
            # protect connectivity
            G = sir_prune_to_budget(G_full, target_edges=n * cfg.sir_top_k)

        lam2 = compute_pipeline_lambda2(G)

        # theorem lhs
        # hop distance
        # pre update
        # this hop
        delta_ig_sq = max_edge_geodesic_distance_sq(G, thetas, fishers, diagonal=True)
        contraction = max(0.0, 1.0 - lam2) ** 2
        theorem1_B = contraction * theorem1_B + delta_ig_sq

        A = weighted_adjacency(G, nodelist=nodes).toarray()
        A_sym = 0.5 * (A + A.T)
        routing_diag = routing_mass_diagnostic_batch(A, records, n_power=5)

        in_deg = A.sum(axis=0)
        P = np.zeros_like(A)
        receiving = in_deg > 1e-12
        P[:, receiving] = A[:, receiving] / in_deg[receiving]
        incoming_avg = P.T @ Theta

        Theta_prev = Theta.copy()
        Theta_new = Theta.copy()
        Theta_new[receiving] = (
            (1 - cfg.diffusion_alpha) * Theta[receiving] + cfg.diffusion_alpha * incoming_avg[receiving]
        )
        Theta_new += cfg.noise_std * np.random.randn(*Theta_new.shape)
        Theta = Theta_new

        # theorem check
        # post update
        # consensus subspace
        # edgeless caveat
        ncv = non_consensus_variance(Theta, A_sym)

        # cumulative drift
        # agent movement
        # accumulate hops
        hop_d_ig = 0.0
        for j, name in enumerate(nodes):
            d_ig = geodesic_distance(
                torch.from_numpy(Theta_prev[j]), torch.from_numpy(Theta[j]), fishers[name], diagonal=True
            )
            hop_d_ig += float(d_ig)
        hop_d_ig /= n
        cumulative_d_ig += hop_d_ig

        for j in range(n):
            unflatten_into(params_list[j], torch.from_numpy(Theta[j]))

        ndcgs, f1s, per_query_scores = [], [], []
        with torch.no_grad():
            for i, rec in enumerate(records):
                scores = np.array([
                    float((heads[j](shards[j][i]["e_q"]) * heads[j](shards[j][i]["e_doc"])).sum())
                    for j in range(n)
                ])
                ndcgs.append(ndcg_at_k(scores, rec["relevance"], cfg.k))
                f1s.append(f1_at_k(scores, (rec["relevance"] > 0).astype(np.float64), cfg.k))
                per_query_scores.append(scores)

        # margin diagnostic
        # reuse scores
        # ranking logits
        margin_diag = aggregate_margin_diagnostics(records, per_query_scores)

        history.append(HotpotHopMetrics(
            hop=hop, lambda2=lam2, ndcg_at_k=float(np.mean(ndcgs)), f1_at_k=float(np.mean(f1s)),
            cumulative_d_ig=cumulative_d_ig,
            delta_ig_sq=delta_ig_sq, non_consensus_variance=ncv, theorem1_bound=theorem1_B,
            gold_inflow_mass=routing_diag["gold_inflow_mass"],
            distractor_inflow_mass=routing_diag["distractor_inflow_mass"],
            gold_distractor_ratio=routing_diag["gold_to_distractor_ratio"],
            mean_transition_entropy=routing_diag["mean_transition_entropy"],
            mean_margin=margin_diag["mean_margin"],
            mean_hard_margin=margin_diag["mean_hard_margin"],
            margin_violation_rate=margin_diag["violation_rate"],
            mean_margin_standardized=margin_diag["mean_margin_standardized"],
            mean_hard_margin_standardized=margin_diag["mean_hard_margin_standardized"],
        ))
        logger.info(
            "[hotpotqa/%-8s] hop %d/%d: lambda2=%.4f NDCG@%d=%.4f F1@%d=%.4f cum_d_IG=%.4f edges=%d "
            "| Thm1: Delta_IG^2=%.4g B(h)=%.4g ncv=%s | gold/distractor=%.4g entropy=%.3f "
            "| margin: raw=%.4g raw_z=%.4g violation_rate=%.3f",
            strategy, hop, cfg.n_hops, lam2, cfg.k, history[-1].ndcg_at_k,
            cfg.k, history[-1].f1_at_k, cumulative_d_ig, G.number_of_edges(),
            delta_ig_sq, theorem1_B, f"{ncv:.4g}" if ncv is not None else "N/A (disconnected)",
            routing_diag["gold_to_distractor_ratio"], routing_diag["mean_transition_entropy"],
            margin_diag["mean_margin"], margin_diag["mean_margin_standardized"], margin_diag["violation_rate"],
        )

    return history


def _pearson_r(x: List[float], y: List[float]) -> float:
    """safe correlation"""
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0 else float("nan")


def save_hotpotqa_results_csv(
    baseline_hist: List[HotpotHopMetrics], sir_hist: List[HotpotHopMetrics], out_dir: str = "results"
) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "hotpotqa_real_metrics.csv")

    b_lam2 = [m.lambda2 for m in baseline_hist]
    b_ndcg = [m.ndcg_at_k for m in baseline_hist]
    s_lam2 = [m.lambda2 for m in sir_hist]
    s_ndcg = [m.ndcg_at_k for m in sir_hist]

    # pooled correlation
    # sign reversal
    # simpsons paradox
    # report all
    corr_pooled = _pearson_r(b_lam2 + s_lam2, b_ndcg + s_ndcg)
    corr_baseline = _pearson_r(b_lam2, b_ndcg)
    corr_sir = _pearson_r(s_lam2, s_ndcg)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy", "hop", "lambda2", "ndcg_at_5", "f1_at_5", "cumulative_d_ig",
            "pearson_r_lambda2_ndcg5_pooled", "pearson_r_lambda2_ndcg5_baseline", "pearson_r_lambda2_ndcg5_sir",
        ])
        for label, hist in (("baseline", baseline_hist), ("sir", sir_hist)):
            for m in hist:
                writer.writerow([
                    label, m.hop, m.lambda2, m.ndcg_at_k, m.f1_at_k, m.cumulative_d_ig,
                    corr_pooled, corr_baseline, corr_sir,
                ])
    logger.info("Saved HotpotQA metrics to %s", path)
    return path


def save_theorem1_diagnostics_csv(
    baseline_hist: List[HotpotHopMetrics], sir_hist: List[HotpotHopMetrics], out_dir: str = "results"
) -> str:
    """save theorem"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "hotpotqa_theorem1_diagnostics.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy", "hop", "lambda2", "delta_ig_sq", "theorem1_bound_B_h",
            "non_consensus_variance", "bound_satisfied",
            "gold_inflow_mass", "distractor_inflow_mass", "gold_distractor_ratio", "mean_transition_entropy",
        ])
        for label, hist in (("baseline", baseline_hist), ("sir", sir_hist)):
            for m in hist:
                satisfied = (
                    "N/A (disconnected)" if m.non_consensus_variance is None
                    else str(m.non_consensus_variance <= m.theorem1_bound)
                )
                writer.writerow([
                    label, m.hop, m.lambda2, m.delta_ig_sq, m.theorem1_bound,
                    m.non_consensus_variance if m.non_consensus_variance is not None else "",
                    satisfied,
                    m.gold_inflow_mass, m.distractor_inflow_mass, m.gold_distractor_ratio, m.mean_transition_entropy,
                ])
    logger.info("Saved Theorem 1 / routing-mass diagnostics to %s", path)
    return path


def save_margin_diagnostics_csv(
    baseline_hist: List[HotpotHopMetrics], sir_hist: List[HotpotHopMetrics], out_dir: str = "results"
) -> str:
    """save margins"""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "hotpotqa_margin_diagnostics.csv")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "strategy", "hop", "ndcg_at_5", "mean_margin", "mean_hard_margin", "margin_violation_rate",
            "mean_margin_standardized", "mean_hard_margin_standardized",
        ])
        for label, hist in (("baseline", baseline_hist), ("sir", sir_hist)):
            for m in hist:
                writer.writerow([
                    label, m.hop, m.ndcg_at_k, m.mean_margin, m.mean_hard_margin, m.margin_violation_rate,
                    m.mean_margin_standardized, m.mean_hard_margin_standardized,
                ])
    logger.info("Saved score-margin diagnostics to %s", path)
    return path


def plot_hotpotqa_figure(
    baseline_hist: List[HotpotHopMetrics], sir_hist: List[HotpotHopMetrics], k: int, out_dir: str = "figures"
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    _set_ieee_style()

    hops = [m.hop for m in baseline_hist]
    series = {
        "lambda2": ([m.lambda2 for m in baseline_hist], [m.lambda2 for m in sir_hist], r"Algebraic connectivity $\lambda_2$"),
        "ndcg": ([m.ndcg_at_k for m in baseline_hist], [m.ndcg_at_k for m in sir_hist], f"NDCG@{k}"),
        "f1": ([m.f1_at_k for m in baseline_hist], [m.f1_at_k for m in sir_hist], f"F1@{k}"),
        "drift": (
            [m.cumulative_d_ig for m in baseline_hist], [m.cumulative_d_ig for m in sir_hist],
            r"Cumulative Fisher drift $\sum_t d_{IG}$",
        ),
    }

    fig, axes = plt.subplots(2, 2, figsize=(IEEE_PAGE_WIDTH_IN, IEEE_PAGE_WIDTH_IN * 0.62))
    for ax, key in zip(axes.flat, ["lambda2", "ndcg", "f1", "drift"]):
        b_vals, s_vals, ylabel = series[key]
        ax.plot(hops, b_vals, **BASELINE_STYLE)
        ax.plot(hops, s_vals, **SIR_STYLE)
        ax.set_xlabel("Multi-hop reasoning step $t$")
        ax.set_ylabel(ylabel)
        ax.set_xticks(hops)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(out_dir, "hotpotqa_scaling.pdf"))
    fig.savefig(os.path.join(out_dir, "hotpotqa_scaling.eps"))
    plt.close(fig)
    logger.info("Saved HotpotQA scaling figure to %s/hotpotqa_scaling.pdf", out_dir)


def print_hotpotqa_summary(baseline_hist: List[HotpotHopMetrics], sir_hist: List[HotpotHopMetrics]) -> None:
    b_final, s_final = baseline_hist[-1], sir_hist[-1]
    print("\n=== HotpotQA final-hop comparison (hop 5) ===")
    print(f"{'metric':<16}{'baseline':>12}{'SIR':>12}{'delta':>12}")
    for name, bv, sv in [
        ("lambda_2", b_final.lambda2, s_final.lambda2),
        ("NDCG@5", b_final.ndcg_at_k, s_final.ndcg_at_k),
        ("F1@5", b_final.f1_at_k, s_final.f1_at_k),
        ("cum_d_IG", b_final.cumulative_d_ig, s_final.cumulative_d_ig),
    ]:
        print(f"{name:<16}{bv:>12.4f}{sv:>12.4f}{sv - bv:>12.4f}")

    b_lam2 = [m.lambda2 for m in baseline_hist]
    b_ndcg = [m.ndcg_at_k for m in baseline_hist]
    s_lam2 = [m.lambda2 for m in sir_hist]
    s_ndcg = [m.ndcg_at_k for m in sir_hist]
    print(f"\nPearson corr(lambda_2, NDCG@5), baseline only:  {_pearson_r(b_lam2, b_ndcg):.4f}")
    print(f"Pearson corr(lambda_2, NDCG@5), SIR only:       {_pearson_r(s_lam2, s_ndcg):.4f}")
    print(f"Pearson corr(lambda_2, NDCG@5), pooled (both):  {_pearson_r(b_lam2 + s_lam2, b_ndcg + s_ndcg):.4f}")


def main_hotpotqa(
    n_eval_queries: Optional[int] = None,
    device: Optional[str] = None,
    embed_batch_size: Optional[int] = None,
    gamma: Optional[float] = None,
    top_k: Optional[int] = None,
) -> None:
    if not (_DATASETS_AVAILABLE and _TRANSFORMERS_AVAILABLE):
        raise ImportError(
            "The HotpotQA pipeline requires `datasets` and `transformers` "
            "(pip install datasets transformers)."
        )

    cfg = RealDataConfig()
    if n_eval_queries is not None:
        cfg.n_eval_queries = n_eval_queries
    if device is not None:
        cfg.device = device
    if embed_batch_size is not None:
        cfg.embed_batch_size = embed_batch_size
    if gamma is not None:
        cfg.sir_gamma = gamma
    if top_k is not None:
        # shared budget
        # fair comparison
        cfg.sir_top_k = top_k
        cfg.baseline_top_k = top_k

    records = _load_hotpotqa_shards(cfg)
    if not records:
        raise RuntimeError("Not enough usable HotpotQA examples to run the pipeline.")

    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("cfg.device=%r but CUDA is not available; falling back to CPU.", cfg.device)
        device = torch.device("cpu")
        cfg.device = "cpu"

    logger.info("Loading frozen encoder %r for embeddings (device=%s) ...", cfg.lm_checkpoint, device)
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_checkpoint)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoder = AutoModel.from_pretrained(cfg.lm_checkpoint)
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    questions = [r["question"] for r in records]
    all_paragraphs = [p for r in records for p in r["paragraphs"]]

    q_hidden = _embed_texts(encoder, tokenizer, questions, cfg)        # shape hint
    doc_hidden = _embed_texts(encoder, tokenizer, all_paragraphs, cfg)  # shape hint

    hidden_dim = q_hidden.shape[1]
    proj_gen = torch.Generator().manual_seed(cfg.seed)
    projection = torch.randn(hidden_dim, cfg.proj_dim, generator=proj_gen) / math.sqrt(hidden_dim)

    q_embeds = q_hidden @ projection       # shape hint
    doc_embeds = doc_hidden @ projection   # shape hint

    shards = _build_agent_shards(cfg, records, q_embeds, doc_embeds)

    # shared init
    init_state_dicts = []
    for j in range(cfg.n_agents):
        gj = torch.Generator().manual_seed(cfg.seed * 1000 + j)
        h = nn.Linear(cfg.proj_dim, cfg.proj_dim)
        with torch.no_grad():
            h.weight.copy_(torch.randn(h.weight.shape, generator=gj) * 0.1)
            h.bias.zero_()
        init_state_dicts.append({k: v.clone() for k, v in h.state_dict().items()})

    baseline_hist = run_hotpotqa_experiment("baseline", cfg, records, shards, init_state_dicts)
    sir_hist = run_hotpotqa_experiment("sir", cfg, records, shards, init_state_dicts)

    save_hotpotqa_results_csv(baseline_hist, sir_hist)
    save_theorem1_diagnostics_csv(baseline_hist, sir_hist)
    save_margin_diagnostics_csv(baseline_hist, sir_hist)
    plot_hotpotqa_figure(baseline_hist, sir_hist, cfg.k)
    print_hotpotqa_summary(baseline_hist, sir_hist)


# --------------------------------------------------------------------------
# run pipeline
# --------------------------------------------------------------------------

def main() -> None:
    cfg = ExperimentConfig()
    setup = build_shared_setup(cfg)

    baseline_hist = run_experiment("baseline", cfg, setup)
    sir_hist = run_experiment("sir", cfg, setup)

    save_results_csv(baseline_hist, sir_hist)
    plot_ieee_figures(baseline_hist, sir_hist, cfg.k)
    print_summary(baseline_hist, sir_hist)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-agent routing evaluation harness.")
    parser.add_argument(
        "--pipeline", choices=["synthetic", "hotpotqa", "both"], default="synthetic",
        help="Which experiment to run (default: synthetic, matching prior behavior).",
    )
    parser.add_argument(
        "--n-eval-queries", "--samples", dest="n_eval_queries", type=int, default=None,
        help="Override RealDataConfig.n_eval_queries for the hotpotqa pipeline.",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="Device for the frozen encoder's forward pass (default: RealDataConfig.device='cpu').",
    )
    parser.add_argument(
        "--batch-size", dest="embed_batch_size", type=int, default=None,
        help="Override RealDataConfig.embed_batch_size (encoder embedding batch size).",
    )
    parser.add_argument(
        "--dataset", choices=["hotpotqa"], default=None,
        help="Accepted as an alias for --pipeline hotpotqa (only dataset currently supported).",
    )
    parser.add_argument(
        "--gamma", type=float, default=None,
        help="Fixed Fisher-Rao kernel bandwidth for SIR (W_ij = exp(-gamma * d_IG)); "
             "default is RealDataConfig.sir_gamma=None, i.e. auto_kernel_gamma's median heuristic.",
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Row-wise cap on outgoing routing edges per agent, applied to BOTH strategies "
             "(SIR: sparsify_top_k on the Fisher-Rao kernel; baseline: cosine top-k neighbors) "
             "so the two are compared under the same edge budget. Default: RealDataConfig.sir_top_k=3.",
    )
    args = parser.parse_args()

    pipeline = args.pipeline
    if args.dataset == "hotpotqa" and pipeline == "synthetic":
        pipeline = "hotpotqa"

    if pipeline in ("synthetic", "both"):
        main()
    if pipeline in ("hotpotqa", "both"):
        main_hotpotqa(
            n_eval_queries=args.n_eval_queries,
            device=args.device,
            embed_batch_size=args.embed_batch_size,
            gamma=args.gamma,
            top_k=args.top_k,
        )
