"""run evaluation"""

from __future__ import annotations

import csv
import json
import sys
import tempfile
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import logging
import math
import os
from dataclasses import asdict, dataclass, replace
from types import SimpleNamespace
from typing import Callable, Dict, List, Optional, Tuple

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
from topk_stability import (
    agent_disagreement,
    aggregate_full_stability,
    full_certificate,
    require_deterministic_noise,
    topk_indices,
    validate_certificate,
    param_change_norm,
)
from numerical_bounds import bilinear_score, certified_bounds, spectral_norm_upper, update_norm_upper
from run_provenance import (PROTOCOL_VERSION, start_run, finish_run, write_json, record_csv,
                            git_state)

from dataset_loaders import known_datasets, load_dataset_records
from model_adapters import format_passage, format_query, known_retrievers, pool_hidden_states

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


def recall_at_k(scores: np.ndarray, relevance_binary: np.ndarray, k: int) -> float:
    """compute recall"""
    k = min(k, len(scores))
    retrieved = set(np.argsort(-scores)[:k].tolist())
    relevant = set(np.where(relevance_binary > 0)[0].tolist())
    if not relevant:
        return 0.0
    return len(retrieved & relevant) / len(relevant)


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
        # Stable sort so equal similarities resolve by ascending agent index, the
        # same tie rule the retrieval ranking uses. Identical or fully converged
        # heads tie exactly, and an unstable sort made the graph depend on the
        # sort's internal pivot choice.
        top_u = np.argsort(-sim[v], kind="stable")[:top_k]
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


def legacy_output_dir(pipeline: str) -> str:
    """Fresh directory for a historical Fisher/SIR run.

    The historical `results/` and `figures/` files are the paper's retained
    evidence. These pipelines used to default to writing straight into them, so a
    rerun would silently replace them. Each run now gets its own directory and the
    writers refuse to overwrite a file that already exists.
    """
    root = os.path.join("runs", "legacy")
    os.makedirs(root, exist_ok=True)
    return tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime(f"{pipeline}_%Y%m%dT%H%M%S_"),
                            dir=root)


def _new_file(path: str):
    """Open for writing, refusing to replace retained evidence."""
    return open(path, "x", newline="")


def _new_figure_path(out_dir: str, name: str) -> str:
    path = os.path.join(out_dir, name)
    if os.path.exists(path):
        raise FileExistsError(f"Refusing to overwrite an existing figure: {path}")
    return path


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
    baseline_hist: List[HopMetrics], sir_hist: List[HopMetrics], k: int, out_dir: Optional[str] = None
) -> None:
    out_dir = out_dir or legacy_output_dir("synthetic")
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
        fig.savefig(_new_figure_path(out_dir, f"routing_comparison_trends.{ext}"))
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
        fig2.savefig(_new_figure_path(out_dir, f"lambda2_vs_ndcg.{ext}"))
    plt.close(fig2)

    logger.info("Saved IEEE-formatted vector figures to %s/", out_dir)


# --------------------------------------------------------------------------
# print summary
# --------------------------------------------------------------------------

def save_results_csv(
    baseline_hist: List[HopMetrics], sir_hist: List[HopMetrics], out_dir: Optional[str] = None
) -> str:
    out_dir = out_dir or legacy_output_dir("synthetic")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "routing_comparison_metrics.csv")
    with _new_file(path) as f:
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
class TopKStabilityConfig:
    dataset: str = "hotpotqa"       # hotpotqa, musique, 2wikimultihopqa
    n_agents: int = 10              # = candidate count, one agent per doc
    n_eval_queries: int = 40
    n_hops: int = 5
    k: int = 5                      # eval depth
    diffusion_alpha: float = 0.6    # blend rate
    baseline_top_k: int = 3
    noise_std: float = 0.0          # theory default
    seed: int = 42
    lm_checkpoint: str = "sentence-transformers/all-MiniLM-L6-v2"  # real retriever
    proj_dim: int = 384             # matches encoder hidden size, no lossy reduction
    embed_batch_size: int = 16
    max_seq_len: int = 96
    device: str = "cpu"             # encoder device
    dataset_slice: Optional[str] = None  # auto-sized scan window when unset
    run_dir: Optional[str] = None       # fresh output root, never historical results
    run_kind: str = "development"
    model_revision: Optional[str] = None
    dataset_revision: Optional[str] = None
    resolved_model_revision: Optional[str] = None


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


@dataclass
class TopKStabilityHopMetrics:
    hop: int
    lambda2: float
    ndcg_at_k: float
    f1_at_k: float
    recall_at_k: float
    param_change_norm: float           # theta jump
    param_disagreement: float          # agent spread
    mean_topk_margin: float            # rank gap
    mean_score_bound: float            # worst cap
    mean_pairwise_gap: float           # tightest pair
    mean_rank_gap: float               # ordered gap
    mean_empirical_score_delta: float  # actual jump
    mean_empirical_to_bound_ratio: float  # tightness ratio
    bound_holds_rate: float            # cap held
    global_certificate_rate: float     # global cert
    pairwise_certificate_rate: float   # pairwise cert
    ordered_certificate_rate: float    # ordered cert
    topk_changed_rate: float           # swap rate
    ordering_changed_rate: float       # order swap
    global_certificate_violations: int    # global breaks
    pairwise_certificate_violations: int  # pairwise breaks
    ordered_certificate_violations: int   # ordered breaks
    noise_std: float = 0.0
    retrieval_k: int = 5
    graph_neighbors: int = 3
    run_id: str = ""
    protocol_version: str = PROTOCOL_VERSION
    seed: int = 42
    candidate_slots: int = 10
    n_hops: int = 5
    dataset: str = ""
    retriever: str = ""
    diffusion_alpha: float = 0.6
    run_kind: str = "development"
    manifest_path: str = ""
    max_bound_holds_rate: float = float("nan")


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


@torch.no_grad()
def _embed_texts_retrieval(model, tokenizer, texts: List[str], cfg: "TopKStabilityConfig") -> torch.Tensor:
    """embed retrieval"""
    device = torch.device(cfg.device)
    all_embeds = []
    for start in range(0, len(texts), cfg.embed_batch_size):
        batch_texts = texts[start : start + cfg.embed_batch_size]
        enc = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_seq_len
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        hidden = model(**enc).last_hidden_state  # shape hint
        pooled = pool_hidden_states(hidden, enc["attention_mask"], cfg.lm_checkpoint)
        pooled = nnf.normalize(pooled, p=2, dim=-1)  # unit norm for dot-product retrieval
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
    """Certificate vectors have shape (dimension,); legacy shards retain a batch axis."""
    certificate = isinstance(cfg, TopKStabilityConfig)
    if certificate:
        for name, embeddings, count in (("query", q_embeds, len(records)),
                                        ("passage", doc_embeds, len(records) * cfg.n_agents)):
            if tuple(embeddings.shape) != (count, cfg.proj_dim):
                raise ValueError(f"{name} embedding batch shape must be {(count, cfg.proj_dim)}; "
                                 f"got {tuple(embeddings.shape)}")
            if not torch.isfinite(embeddings).all():
                raise ValueError(f"{name} embeddings must be finite")
    shards: List[List[Dict[str, torch.Tensor]]] = [[] for _ in range(cfg.n_agents)]
    for i, rec in enumerate(records):
        for j in range(cfg.n_agents):
            shards[j].append({
                "e_q": q_embeds[i] if certificate else q_embeds[i : i + 1],
                "e_doc": (doc_embeds[i * cfg.n_agents + j] if certificate else
                          doc_embeds[i * cfg.n_agents + j : i * cfg.n_agents + j + 1]),
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
    baseline_hist: List[HotpotHopMetrics], sir_hist: List[HotpotHopMetrics], out_dir: Optional[str] = None
) -> str:
    out_dir = out_dir or legacy_output_dir("hotpotqa")
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

    with _new_file(path) as f:
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
    baseline_hist: List[HotpotHopMetrics], sir_hist: List[HotpotHopMetrics], out_dir: Optional[str] = None
) -> str:
    """save theorem"""
    out_dir = out_dir or legacy_output_dir("hotpotqa")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "hotpotqa_theorem1_diagnostics.csv")

    with _new_file(path) as f:
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
    baseline_hist: List[HotpotHopMetrics], sir_hist: List[HotpotHopMetrics], out_dir: Optional[str] = None
) -> str:
    """save margins"""
    out_dir = out_dir or legacy_output_dir("hotpotqa")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "hotpotqa_margin_diagnostics.csv")

    with _new_file(path) as f:
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
    baseline_hist: List[HotpotHopMetrics], sir_hist: List[HotpotHopMetrics], k: int, out_dir: Optional[str] = None
) -> None:
    out_dir = out_dir or legacy_output_dir("hotpotqa")
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
    fig.savefig(_new_figure_path(out_dir, "hotpotqa_scaling.pdf"))
    fig.savefig(_new_figure_path(out_dir, "hotpotqa_scaling.eps"))
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


_IDENTITY_INIT_NOISE_STD = 0.02  # per-agent spread around identity


def _prepare_hotpotqa_run(
    cfg: RealDataConfig, bias: bool = True, identity_init: bool = False, dataset: Optional[str] = None,
) -> Tuple[List[dict], List[List[Dict[str, torch.Tensor]]], List[dict]]:
    """shared setup"""
    if dataset is not None:
        # common multi-dataset loader, one agent per candidate passage
        records = load_dataset_records(
            dataset, n_eval_queries=cfg.n_eval_queries, n_candidates=cfg.n_agents,
            seed=cfg.seed, dataset_slice=cfg.dataset_slice, revision=cfg.dataset_revision,
        )
    else:
        records = _load_hotpotqa_shards(cfg)
    if not records:
        raise RuntimeError("Not enough usable examples to run the pipeline.")

    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("cfg.device=%r but CUDA is not available; falling back to CPU.", cfg.device)
        device = torch.device("cpu")
        cfg.device = "cpu"

    logger.info("Loading frozen encoder %r for embeddings (device=%s) ...", cfg.lm_checkpoint, device)
    model_options = {"revision": cfg.model_revision} if dataset is not None else {}
    tokenizer = AutoTokenizer.from_pretrained(cfg.lm_checkpoint, **model_options)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoder = AutoModel.from_pretrained(cfg.lm_checkpoint, **model_options)
    if dataset is not None:
        cfg.resolved_model_revision = getattr(encoder.config, "_commit_hash", None)
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    questions = [r["question"] for r in records]
    all_paragraphs = [p for r in records for p in r["paragraphs"]]

    if dataset is not None:
        # respect each retriever's own query/passage formatting convention
        questions = [format_query(cfg.lm_checkpoint, q) for q in questions]
        all_paragraphs = [format_passage(cfg.lm_checkpoint, p) for p in all_paragraphs]

        # model-appropriate pooling + L2 normalization, used directly (no random
        # projection): each retriever's own embedding geometry is preserved,
        # rather than passed through an arbitrary fixed linear distortion
        q_embeds = _embed_texts_retrieval(encoder, tokenizer, questions, cfg)
        doc_embeds = _embed_texts_retrieval(encoder, tokenizer, all_paragraphs, cfg)
    else:
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
        h = nn.Linear(cfg.proj_dim, cfg.proj_dim, bias=bias)
        with torch.no_grad():
            if identity_init:
                # near identity: score stays close to the frozen embeddings' own
                # dot product, so ranking quality comes from the pretrained
                # representation, not a randomly initialized head
                eye = torch.eye(cfg.proj_dim)
                h.weight.copy_(eye + _IDENTITY_INIT_NOISE_STD * torch.randn(h.weight.shape, generator=gj))
            else:
                h.weight.copy_(torch.randn(h.weight.shape, generator=gj) * 0.1)
            if bias:
                h.bias.zero_()
        init_state_dicts.append({k: v.clone() for k, v in h.state_dict().items()})

    return records, shards, init_state_dicts


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

    records, shards, init_state_dicts = _prepare_hotpotqa_run(cfg)

    baseline_hist = run_hotpotqa_experiment("baseline", cfg, records, shards, init_state_dicts)
    sir_hist = run_hotpotqa_experiment("sir", cfg, records, shards, init_state_dicts)

    out_dir = legacy_output_dir("hotpotqa")
    save_hotpotqa_results_csv(baseline_hist, sir_hist, out_dir=out_dir)
    save_theorem1_diagnostics_csv(baseline_hist, sir_hist, out_dir=out_dir)
    save_margin_diagnostics_csv(baseline_hist, sir_hist, out_dir=out_dir)
    plot_hotpotqa_figure(baseline_hist, sir_hist, cfg.k, out_dir=out_dir)
    print_hotpotqa_summary(baseline_hist, sir_hist)
    logger.info("Legacy HotpotQA outputs written to %s", out_dir)


# --------------------------------------------------------------------------
# topk stability pipeline
# --------------------------------------------------------------------------
#
# fisher free
# cosine routing
# no sir
# separate run

def _validate_topk_config(cfg: TopKStabilityConfig) -> None:
    require_deterministic_noise(cfg.noise_std)
    if cfg.run_kind == "paper":
        if not cfg.model_revision or not cfg.dataset_revision:
            raise ValueError("Paper runs require explicit model_revision and dataset_revision")
        # A result produced from edits that exist only in this working tree cannot
        # be reproduced from any commit, so it is not paper evidence.
        state = git_state()
        if not state["commit"] or state["dirty"] is not False:
            raise ValueError("Paper runs require a clean working tree at a known commit; "
                             "commit the source first or use run_kind=development")
    if not 0 < cfg.k < cfg.n_agents:
        raise ValueError(f"retrieval_k must satisfy 0 < k < candidate_slots ({cfg.n_agents})")
    if not math.isfinite(cfg.diffusion_alpha):
        raise ValueError("diffusion_alpha must be finite; a nonfinite blend rate has no bounded update")
    if cfg.n_hops < 1 or cfg.n_eval_queries < 1:
        raise ValueError("hop_count and n_eval_queries must both be at least 1")
    # A receiving agent draws from its most similar peers; the count saturates at
    # n_agents-1 because self-similarity is excluded. A negative count silently
    # inverts the neighbour slice, so reject it here instead.
    if cfg.baseline_top_k < 0:
        raise ValueError("graph_neighbors must be nonnegative")


def _new_topk_run_dir(out_dir: Optional[str] = None) -> str:
    root = out_dir or "runs"
    os.makedirs(root, exist_ok=True)
    return tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_"), dir=root)


def _write_configuration(cfg: TopKStabilityConfig, out_dir: str, **extra) -> None:
    """Describe a CLI invocation that spans several runs.

    This is the parent of the per-run directories, not a run itself. Each child
    writes its own schema-4 manifest, and those manifests are the only provenance
    authority; this file records what one command asked for. It is named
    `sweep_configuration.json` so it is never mistaken for a run's
    `configuration.json`.
    """
    packages = {}
    for package in ("numpy", "torch", "scipy", "networkx", "matplotlib", "pandas", "datasets", "transformers"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    config = asdict(cfg)
    normalized = {**config, "graph_neighbors": cfg.baseline_top_k, "retrieval_k": cfg.k,
                  "candidate_slots": cfg.n_agents, "hop_count": cfg.n_hops,
                  "alpha": cfg.diffusion_alpha}
    with open(os.path.join(out_dir, "sweep_configuration.json"), "x") as f:
        json.dump({
            "document": "sweep_configuration", "schema_version": 1,
            "protocol_version": PROTOCOL_VERSION, "run_kind": cfg.run_kind,
            "configuration": normalized,
            "model_revision": cfg.model_revision, "dataset_revision": cfg.dataset_revision,
            "command": sys.argv, "python": sys.version, "packages": packages,
            "git": git_state(),
            "ranking_policy": "descending score, ascending candidate-slot ID (stable sort)",
            "authority": "per-run manifest.json files in this directory are the provenance record",
            **extra,
        }, f, indent=2)


def run_topk_stability_experiment(
    cfg: TopKStabilityConfig,
    records: List[dict],
    shards: List[List[Dict[str, torch.Tensor]]],
    init_state_dicts: List[dict],
    per_query_sink: Optional[List[dict]] = None,
) -> List[TopKStabilityHopMetrics]:
    """topk experiment"""
    _validate_topk_config(cfg)
    if not records or len(shards) != cfg.n_agents or len(init_state_dicts) != cfg.n_agents:
        raise ValueError("Nonempty records, shards, and initial heads must match candidate count")
    for j, shard in enumerate(shards):
        if len(shard) != len(records):
            raise ValueError("Every candidate shard must contain one vector per query")
        for i, example in enumerate(shard):
            for name in ("e_q", "e_doc"):
                vector = example[name]
                if tuple(vector.shape) != (cfg.proj_dim,) or not torch.isfinite(vector).all():
                    raise ValueError(f"{name} must be a finite vector of shape ({cfg.proj_dim},)")
            if j and not torch.equal(example["e_q"], shards[0][i]["e_q"]):
                raise ValueError("All candidate slots must use the same query embedding")
    if cfg.run_kind == "paper" and not cfg.resolved_model_revision:
        raise ValueError("Paper runs require a resolved encoder revision; the run must load the "
                         "retriever it names, not a handcrafted fixture")
    audit_dir, manifest = start_run(cfg, records)
    np.savez_compressed(audit_dir / "inputs.npz",
                        queries=np.array([x["e_q"].numpy() for x in shards[0]]),
                        documents=np.array([[x["e_doc"].numpy() for x in shard] for shard in shards]),
                        initial_heads=np.array([state["weight"].numpy() for state in init_state_dicts]))
    row_context = {
        "run_id": manifest["run_id"], "protocol_version": PROTOCOL_VERSION, "seed": cfg.seed,
        "candidate_slots": cfg.n_agents, "n_hops": cfg.n_hops, "dataset": cfg.dataset,
        "retriever": cfg.lm_checkpoint, "diffusion_alpha": cfg.diffusion_alpha,
        "run_kind": cfg.run_kind, "manifest_path": str((audit_dir / "manifest.json").resolve()),
    }
    all_query_rows = []
    nodes = [f"agent_{j}" for j in range(cfg.n_agents)]
    n = cfg.n_agents

    heads: List[nn.Linear] = []
    params_list: List[List[nn.Parameter]] = []
    for j in range(n):
        h = nn.Linear(cfg.proj_dim, cfg.proj_dim, bias=False).double()
        h.load_state_dict({k: v.clone() for k, v in init_state_dicts[j].items()})
        heads.append(h)
        params_list.append([p for p in h.parameters() if p.requires_grad])

    Theta = np.stack([flatten_params(params_list[j]).numpy() for j in range(n)], axis=0)

    def score_all() -> List[np.ndarray]:
        """Read current heads; scoring never mutates parameters."""
        out = []
        with torch.no_grad():
            for i, rec in enumerate(records):
                scores = np.array([
                    bilinear_score(heads[j].weight.detach().numpy(),
                                   shards[j][i]["e_q"].numpy(), shards[j][i]["e_doc"].numpy())
                    for j in range(n)
                ])
                out.append(scores)
        return out

    # embedding norms, fixed for the whole run
    q_norms = np.array([float(shards[0][i]["e_q"].norm()) for i in range(len(records))])
    doc_norms = np.array([
        [float(shards[j][i]["e_doc"].norm()) for j in range(n)]
        for i in range(len(records))
    ])

    history: List[TopKStabilityHopMetrics] = []

    for hop in range(1, cfg.n_hops + 1):
        torch.manual_seed(cfg.seed * 7919 + hop)
        np.random.seed(cfg.seed * 7919 + hop)

        G = build_baseline_graph(Theta, nodes, top_k=cfg.baseline_top_k)
        lam2 = compute_pipeline_lambda2(G)
        disagreement = agent_disagreement(Theta)

        scores_old = score_all()  # before scores

        A = weighted_adjacency(G, nodelist=nodes).toarray()
        in_deg = A.sum(axis=0)
        P = np.zeros_like(A)
        receiving = in_deg > 1e-12
        P[:, receiving] = A[:, receiving] / in_deg[receiving]
        incoming_avg = P.T @ Theta

        Theta_old = Theta.copy()
        Theta_consensus = Theta.copy()  # deterministic step
        Theta_consensus[receiving] = (
            (1 - cfg.diffusion_alpha) * Theta[receiving] + cfg.diffusion_alpha * incoming_avg[receiving]
        )

        # Certify exact stored-parameter endpoints using validated norm enclosures.
        old_heads = Theta_old.reshape(n, cfg.proj_dim, cfg.proj_dim)
        proposed_heads = Theta_consensus.reshape(n, cfg.proj_dim, cfg.proj_dim)
        norm_data = [(spectral_norm_upper(w), update_norm_upper(w, p))
                     for w, p in zip(old_heads, proposed_heads, strict=True)]
        bound_data = [
            certified_bounds(old_heads, proposed_heads, shards[0][i]["e_q"].numpy(),
                             np.array([shards[j][i]["e_doc"].numpy() for j in range(n)]), norm_data)
            for i in range(len(records))
        ]
        b_per_query = [bounds for real, errors, bounds in bound_data]
        pnorm = param_change_norm(Theta_old, Theta_consensus)

        # Snapshot every query decision before assigning any proposed parameter.
        decisions = tuple(
            full_certificate(scores_old[i], cfg.k, b_per_query[i], noise_std=cfg.noise_std,
                             real_bounds=bound_data[i][0], score_error_bounds=bound_data[i][1])
            for i in range(len(records))
        )
        if audit_dir is not None:
            np.savez_compressed(
                os.path.join(audit_dir, f"hop_{hop}_pre_update.npz"),
                theta_old=Theta_old, theta_proposed=Theta_consensus,
                scores_old=np.asarray(scores_old), bounds=np.asarray(b_per_query),
                q_norms=q_norms, doc_norms=doc_norms,
                adjacency=A, incoming_weights=P,
            )
            write_json(audit_dir / f"hop_{hop}_decisions.json", [asdict(d) for d in decisions])

        Theta = Theta_consensus
        for j in range(n):
            unflatten_into(params_list[j], torch.from_numpy(Theta[j]))

        scores_new = score_all()  # after scores

        diagnostics = [validate_certificate(d, s) for d, s in zip(decisions, scores_new, strict=True)]
        if audit_dir is not None:
            np.save(os.path.join(audit_dir, f"hop_{hop}_scores_new.npy"), np.asarray(scores_new))

        ndcgs, f1s, recalls = [], [], []
        for i, rec in enumerate(records):
            # Canonicalize tied scores by slot before invoking shared legacy metrics.
            order = topk_indices(scores_new[i], n)
            ranked_scores = np.arange(n, 0, -1, dtype=float)
            ranked_rel = np.asarray(rec["relevance"])[order]
            ndcgs.append(ndcg_at_k(ranked_scores, ranked_rel, cfg.k))
            rel_binary = (ranked_rel > 0).astype(np.float64)
            f1s.append(f1_at_k(ranked_scores, rel_binary, cfg.k))
            recalls.append(recall_at_k(ranked_scores, rel_binary, cfg.k))

        topk_diag = aggregate_full_stability(decisions, scores_new)

        history.append(TopKStabilityHopMetrics(
            **row_context,
            noise_std=cfg.noise_std, retrieval_k=cfg.k, graph_neighbors=cfg.baseline_top_k,
            hop=hop, lambda2=lam2, ndcg_at_k=float(np.mean(ndcgs)), f1_at_k=float(np.mean(f1s)),
            recall_at_k=float(np.mean(recalls)),
            param_change_norm=pnorm,
            param_disagreement=disagreement,
            mean_topk_margin=topk_diag["mean_topk_margin"],
            mean_score_bound=topk_diag["mean_score_bound"],
            mean_pairwise_gap=topk_diag["mean_pairwise_gap"],
            mean_rank_gap=topk_diag["mean_rank_gap"],
            mean_empirical_score_delta=topk_diag["mean_empirical_score_delta"],
            mean_empirical_to_bound_ratio=topk_diag["mean_empirical_to_bound_ratio"],
            bound_holds_rate=topk_diag["bound_holds_rate"],
            max_bound_holds_rate=topk_diag["max_bound_holds_rate"],
            global_certificate_rate=topk_diag["global_certificate_rate"],
            pairwise_certificate_rate=topk_diag["pairwise_certificate_rate"],
            ordered_certificate_rate=topk_diag["ordered_certificate_rate"],
            topk_changed_rate=topk_diag["topk_changed_rate"],
            ordering_changed_rate=topk_diag["ordering_changed_rate"],
            global_certificate_violations=topk_diag["global_certificate_violations"],
            pairwise_certificate_violations=topk_diag["pairwise_certificate_violations"],
            ordered_certificate_violations=topk_diag["ordered_certificate_violations"],
        ))
        logger.info(
            "[topk-stability] hop %d/%d alpha=%.2f: lambda2=%.4f NDCG@%d=%.4f F1@%d=%.4f Recall@%d=%.4f "
            "|dTheta|=%.4f disagree=%.4f margin_k=%.4f B_max=%.4f pair_gap=%.17e rank_gap=%.17e empirical=%.4f "
            "ratio=%.4f bound_holds=%.3f global_rate=%.3f pairwise_rate=%.3f ordered_rate=%.3f "
            "topk_changed_rate=%.3f global_viol=%d pairwise_viol=%d ordered_viol=%d",
            hop, cfg.n_hops, cfg.diffusion_alpha, lam2, cfg.k, history[-1].ndcg_at_k, cfg.k, history[-1].f1_at_k,
            cfg.k, history[-1].recall_at_k,
            pnorm, disagreement, topk_diag["mean_topk_margin"], topk_diag["mean_score_bound"],
            topk_diag["mean_pairwise_gap"], topk_diag["mean_rank_gap"], topk_diag["mean_empirical_score_delta"],
            topk_diag["mean_empirical_to_bound_ratio"], topk_diag["bound_holds_rate"],
            topk_diag["global_certificate_rate"], topk_diag["pairwise_certificate_rate"],
            topk_diag["ordered_certificate_rate"],
            topk_diag["topk_changed_rate"], topk_diag["global_certificate_violations"],
            topk_diag["pairwise_certificate_violations"], topk_diag["ordered_certificate_violations"],
        )

        for i in range(len(records)):
                d = diagnostics[i]
                all_query_rows.append({
                    **row_context,
                    "hop": hop, "query_index": i, "retrieval_k": cfg.k,
                    "graph_neighbors": cfg.baseline_top_k, "noise_std": cfg.noise_std,
                    "ndcg_at_k": ndcgs[i], "f1_at_k": f1s[i], "recall_at_k": recalls[i],
                    **d,
                })

    save_topk_stability_csv(history, out_dir=str(audit_dir), filename="metrics.csv")
    save_per_query_csv(all_query_rows, out_dir=str(audit_dir), filename="per_query.csv")
    finish_run(audit_dir)
    if per_query_sink is not None:
        per_query_sink.extend(all_query_rows)
    return history


_TOPK_METRIC_COLUMNS = [
    "run_id", "protocol_version", "seed", "candidate_slots", "n_hops",
    "dataset", "retriever", "diffusion_alpha", "run_kind", "manifest_path",
    "noise_std", "retrieval_k", "graph_neighbors",
    "hop", "lambda2", "ndcg_at_k", "f1_at_k", "recall_at_k",
    "param_change_norm", "param_disagreement",
    "topk_margin", "score_bound", "pairwise_gap", "rank_gap",
    "empirical_score_delta", "empirical_to_bound_ratio", "bound_holds_rate", "max_bound_holds_rate",
    "global_certificate_rate", "pairwise_certificate_rate", "ordered_certificate_rate",
    "topk_changed_rate", "ordering_changed_rate",
    "global_certificate_violations", "pairwise_certificate_violations", "ordered_certificate_violations",
]


def _topk_row(m: TopKStabilityHopMetrics) -> list:
    """metric row"""
    return [
        m.run_id, m.protocol_version, m.seed, m.candidate_slots, m.n_hops,
        m.dataset, m.retriever, m.diffusion_alpha, m.run_kind, m.manifest_path,
        m.noise_std, m.retrieval_k, m.graph_neighbors,
        m.hop, m.lambda2, m.ndcg_at_k, m.f1_at_k, m.recall_at_k,
        m.param_change_norm, m.param_disagreement,
        m.mean_topk_margin, m.mean_score_bound, m.mean_pairwise_gap, m.mean_rank_gap,
        m.mean_empirical_score_delta, m.mean_empirical_to_bound_ratio, m.bound_holds_rate, m.max_bound_holds_rate,
        m.global_certificate_rate, m.pairwise_certificate_rate, m.ordered_certificate_rate,
        m.topk_changed_rate, m.ordering_changed_rate,
        m.global_certificate_violations, m.pairwise_certificate_violations, m.ordered_certificate_violations,
    ]


def save_topk_stability_csv(
    history: List[TopKStabilityHopMetrics], out_dir: Optional[str] = None, filename: str = "hotpotqa_topk_stability.csv"
) -> str:
    """save topk"""
    if out_dir is None:
        out_dir = _new_topk_run_dir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    with open(path, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_TOPK_METRIC_COLUMNS)
        for m in history:
            writer.writerow(_topk_row(m))
    record_csv(path)
    logger.info("Saved top-k stability diagnostics to %s", path)
    return path


def run_alpha_sweep(
    cfg: TopKStabilityConfig,
    records: List[dict],
    shards: List[List[Dict[str, torch.Tensor]]],
    init_state_dicts: List[dict],
    alphas: List[float],
) -> Dict[float, List[TopKStabilityHopMetrics]]:
    """alpha sweep"""
    sweep: Dict[float, List[TopKStabilityHopMetrics]] = {}
    for a in alphas:
        cfg_a = replace(cfg, diffusion_alpha=a)
        sweep[a] = run_topk_stability_experiment(cfg_a, records, shards, init_state_dicts)
    return sweep


def save_alpha_sweep_csv(
    sweep: Dict[float, List[TopKStabilityHopMetrics]],
    out_dir: Optional[str] = None,
    filename: str = "hotpotqa_topk_stability_alpha_sweep.csv",
) -> str:
    """save sweep"""
    if out_dir is None:
        out_dir = _new_topk_run_dir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    with open(path, "x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_TOPK_METRIC_COLUMNS)
        for alpha, history in sweep.items():
            for m in history:
                writer.writerow(_topk_row(m))
    record_csv(path)
    logger.info("Saved diffusion-alpha sweep to %s", path)
    return path


def print_topk_stability_summary(history: List[TopKStabilityHopMetrics]) -> None:
    print("\n=== Top-k stability certificate, per hop ===")
    print(
        f"{'hop':<4}{'lambda_2':>9}{'NDCG@k':>9}{'F1@k':>9}{'Recall@k':>9}{'|dTheta|':>9}{'disagree':>9}"
        f"{'margin_k':>10}{'B_max':>10}{'pair_gap':>10}{'rank_gap':>10}{'empir.':>10}{'ratio':>8}"
        f"{'cap_held':>9}{'global':>7}{'pair':>7}{'ordr':>7}{'swap':>7}"
        f"{'g_viol':>7}{'p_viol':>7}{'o_viol':>7}"
    )
    for m in history:
        print(
            f"{m.hop:<4}{m.lambda2:>9.4f}{m.ndcg_at_k:>9.4f}{m.f1_at_k:>9.4f}{m.recall_at_k:>9.4f}"
            f"{m.param_change_norm:>9.4f}{m.param_disagreement:>9.4f}"
            f"{m.mean_topk_margin:>10.4f}{m.mean_score_bound:>10.4f} {m.mean_pairwise_gap:.17e}"
            f" {m.mean_rank_gap:.17e} {m.mean_empirical_score_delta:>10.4f}"
            f"{m.mean_empirical_to_bound_ratio:>8.4f}{m.bound_holds_rate:>9.3f}"
            f"{m.global_certificate_rate:>7.3f}{m.pairwise_certificate_rate:>7.3f}"
            f"{m.ordered_certificate_rate:>7.3f}{m.topk_changed_rate:>7.3f}"
            f"{m.global_certificate_violations:>7d}{m.pairwise_certificate_violations:>7d}"
            f"{m.ordered_certificate_violations:>7d}"
        )
    total_global = sum(m.global_certificate_violations for m in history)
    total_pairwise = sum(m.pairwise_certificate_violations for m in history)
    total_ordered = sum(m.ordered_certificate_violations for m in history)
    print(
        f"\nTotal violations across all hops -- global: {total_global}, pairwise: {total_pairwise}, "
        f"ordered: {total_ordered} (all should always be 0)"
    )


def _topk_dict(m: TopKStabilityHopMetrics) -> dict:
    """metric dict"""
    return dict(zip(_TOPK_METRIC_COLUMNS, _topk_row(m)))


def _run_tag(cfg: TopKStabilityConfig) -> str:
    """run tag"""
    model_tag = cfg.lm_checkpoint.split("/")[-1]
    return f"{cfg.dataset}_{model_tag}_c{cfg.n_agents}"


def _require_topk_deps() -> None:
    """dep check"""
    if not (_DATASETS_AVAILABLE and _TRANSFORMERS_AVAILABLE):
        raise ImportError(
            "The topk-stability pipeline requires `datasets` and `transformers` "
            "(pip install datasets transformers)."
        )


def main_topk_stability(
    n_eval_queries: Optional[int] = None,
    device: Optional[str] = None,
    embed_batch_size: Optional[int] = None,
    top_k: Optional[int] = None,
    diffusion_alpha: Optional[float] = None,
    noise_std: Optional[float] = None,
    alpha_sweep: Optional[List[float]] = None,
    dataset: Optional[str] = None,
    retriever: Optional[str] = None,
    candidate_count: Optional[int] = None,
    save_per_query: bool = False,
    retrieval_k: Optional[int] = None,
    out_dir: Optional[str] = None,
    run_options: Optional[dict] = None,
) -> None:
    """single dataset/retriever run, optionally swept over alpha"""

    cfg = TopKStabilityConfig(**(run_options or {}))
    if retrieval_k is not None:
        cfg.k = retrieval_k
    if dataset is not None:
        cfg.dataset = dataset
    if retriever is not None:
        cfg.lm_checkpoint = retriever
    if candidate_count is not None:
        cfg.n_agents = candidate_count
    if n_eval_queries is not None:
        cfg.n_eval_queries = n_eval_queries
    if device is not None:
        cfg.device = device
    if embed_batch_size is not None:
        cfg.embed_batch_size = embed_batch_size
    if top_k is not None:
        cfg.baseline_top_k = top_k
    if diffusion_alpha is not None:
        cfg.diffusion_alpha = diffusion_alpha
    if noise_std is not None:
        cfg.noise_std = noise_std

    _validate_topk_config(cfg)
    _require_topk_deps()
    out_dir = _new_topk_run_dir(out_dir)
    cfg.run_dir = out_dir
    _write_configuration(cfg, out_dir, alphas=alpha_sweep)

    records, shards, init_state_dicts = _prepare_hotpotqa_run(
        cfg, bias=False, identity_init=True, dataset=cfg.dataset,
    )
    tag = _run_tag(cfg)

    if alpha_sweep:
        sweep = run_alpha_sweep(cfg, records, shards, init_state_dicts, alpha_sweep)
        save_alpha_sweep_csv(sweep, out_dir=out_dir, filename=f"topk_stability_alpha_sweep_{tag}.csv")
        for a, history in sweep.items():
            print(f"\n--- dataset={cfg.dataset} retriever={cfg.lm_checkpoint} diffusion_alpha={a} ---")
            print_topk_stability_summary(history)
    else:
        per_query_sink: Optional[List[dict]] = [] if save_per_query else None
        history = run_topk_stability_experiment(
            cfg, records, shards, init_state_dicts, per_query_sink=per_query_sink,
        )
        save_topk_stability_csv(history, out_dir=out_dir, filename=f"topk_stability_{tag}.csv")
        print_topk_stability_summary(history)
        if per_query_sink:
            save_per_query_csv(per_query_sink, out_dir=out_dir, filename=f"per_query_{tag}.csv")


# --------------------------------------------------------------------------
# full sweep (datasets x retrievers x alphas)
# --------------------------------------------------------------------------

def run_full_sweep(
    datasets: List[str],
    retrievers: List[str],
    alphas: List[float],
    cfg_base: TopKStabilityConfig,
    per_query_sink: Optional[List[dict]] = None,
) -> List[dict]:
    """full sweep"""
    rows: List[dict] = []
    for dataset in datasets:
        for retriever in retrievers:
            cfg = replace(cfg_base, dataset=dataset, lm_checkpoint=retriever)
            _validate_topk_config(cfg)
            logger.info("=== dataset=%s retriever=%s ===", dataset, retriever)
            records, shards, init_state_dicts = _prepare_hotpotqa_run(
                cfg, bias=False, identity_init=True, dataset=dataset,
            )
            for alpha in alphas:
                cfg_a = replace(cfg, diffusion_alpha=alpha)
                local_sink: Optional[List[dict]] = [] if per_query_sink is not None else None
                history = run_topk_stability_experiment(
                    cfg_a, records, shards, init_state_dicts, per_query_sink=local_sink,
                )
                if per_query_sink is not None:
                    for prow in local_sink:
                        per_query_sink.append({
                            "dataset": dataset, "retriever": retriever, "diffusion_alpha": alpha, **prow,
                        })
                for m in history:
                    rows.append({
                        "dataset": dataset, "retriever": retriever, "diffusion_alpha": alpha,
                        **_topk_dict(m),
                    })
    return rows


def save_full_sweep_csv(
    rows: List[dict], out_dir: Optional[str] = None, filename: str = "full_sweep_results.csv"
) -> str:
    """save full sweep"""
    if out_dir is None:
        out_dir = _new_topk_run_dir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    fieldnames = _TOPK_METRIC_COLUMNS
    with open(path, "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    record_csv(path)
    logger.info("Saved full dataset/model/alpha/hop sweep to %s", path)
    return path


_CERT_COMPARISON_COLUMNS = [
    "run_id", "protocol_version", "seed", "candidate_slots", "n_hops", "run_kind", "manifest_path",
    "noise_std", "retrieval_k", "graph_neighbors",
    "dataset", "retriever", "diffusion_alpha", "hop",
    "global_certificate_rate", "pairwise_certificate_rate", "ordered_certificate_rate",
    "global_certificate_violations", "pairwise_certificate_violations", "ordered_certificate_violations",
    "topk_changed_rate", "ordering_changed_rate",
]


def save_certificate_comparison_csv(
    rows: List[dict], out_dir: Optional[str] = None, filename: str = "certificate_comparison.csv"
) -> str:
    """save comparison"""
    if out_dir is None:
        out_dir = _new_topk_run_dir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_CERT_COMPARISON_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in _CERT_COMPARISON_COLUMNS})
    record_csv(path)
    logger.info("Saved global vs pairwise vs ordered certificate comparison to %s", path)
    return path


_SUMMARY_TABLE_COLUMNS = [
    "run_id", "protocol_version", "seed", "candidate_slots", "n_hops", "run_kind", "manifest_path",
    "noise_std", "retrieval_k", "graph_neighbors",
    "dataset", "retriever", "diffusion_alpha", "hop", "ndcg_at_k", "f1_at_k", "recall_at_k",
    "global_certificate_rate", "pairwise_certificate_rate", "ordered_certificate_rate",
    "global_certificate_violations", "pairwise_certificate_violations", "ordered_certificate_violations",
]


def save_summary_table_csv(
    rows: List[dict], out_dir: Optional[str] = None, filename: str = "summary_table.csv"
) -> str:
    """save summary"""
    if out_dir is None:
        out_dir = _new_topk_run_dir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)

    max_hop: Dict[tuple, int] = {}
    for row in rows:
        key = row["run_id"]
        max_hop[key] = max(max_hop.get(key, 0), row["hop"])
    final_rows = [
        row for row in rows
        if row["hop"] == max_hop[row["run_id"]]
    ]

    with open(path, "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_TABLE_COLUMNS)
        writer.writeheader()
        for row in final_rows:
            writer.writerow({k: row[k] for k in _SUMMARY_TABLE_COLUMNS})
    record_csv(path)
    logger.info("Saved descriptive summary table to %s", path)
    return path


def save_per_query_csv(
    per_query_rows: List[dict], out_dir: Optional[str] = None, filename: str = "per_query_certificates.csv"
) -> str:
    """save per query"""
    if out_dir is None:
        out_dir = _new_topk_run_dir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    if not per_query_rows:
        logger.warning("No per-query rows to save.")
        return path

    all_keys = {k for row in per_query_rows for k in row.keys()}
    preferred = [
        "dataset", "retriever", "diffusion_alpha", "hop", "query_index",
        "ndcg_at_k", "f1_at_k", "recall_at_k",
    ]
    fieldnames = [c for c in preferred if c in all_keys] + sorted(all_keys - set(preferred))

    with open(path, "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in per_query_rows:
            writer.writerow(row)
    record_csv(path)
    logger.info("Saved per-query certificate details to %s (%d rows)", path, len(per_query_rows))
    return path


def _require_one_revision_per_repository(cfg: TopKStabilityConfig, datasets, retrievers) -> None:
    """A single revision string cannot pin several repositories at once.

    `--dataset-revision`/`--model-revision` hold one value, so a multi-dataset or
    multi-retriever paper sweep would record the same revision for repositories it
    does not identify. Paper runs must therefore pin one dataset and one retriever
    per invocation.
    """
    if cfg.run_kind != "paper":
        return
    if len(datasets) > 1:
        raise ValueError("A paper run pins one dataset revision, so run one dataset per "
                         f"invocation; got {sorted(datasets)}")
    if len(retrievers) > 1:
        raise ValueError("A paper run pins one model revision, so run one retriever per "
                         f"invocation; got {sorted(retrievers)}")


def main_full_sweep(
    datasets: Optional[List[str]] = None,
    retrievers: Optional[List[str]] = None,
    alphas: Optional[List[float]] = None,
    n_eval_queries: Optional[int] = None,
    candidate_count: Optional[int] = None,
    top_k: Optional[int] = None,
    noise_std: Optional[float] = None,
    device: Optional[str] = None,
    embed_batch_size: Optional[int] = None,
    save_per_query: bool = False,
    out_dir: Optional[str] = None,
    retrieval_k: Optional[int] = None,
    run_options: Optional[dict] = None,
) -> None:
    """Dataset x retriever x alpha sweep, the paper's main experiment.

    Defaults to all three datasets and all three retrievers. A paper run pins one
    revision per repository, so it covers one dataset and one retriever per
    invocation and the sweep is over alpha and hops.
    """

    datasets = datasets or list(known_datasets())
    retrievers = retrievers or list(known_retrievers())
    alphas = alphas or [0.1, 0.2, 0.4, 0.6, 0.8]

    cfg = TopKStabilityConfig(**(run_options or {}))
    if retrieval_k is not None:
        cfg.k = retrieval_k
    if n_eval_queries is not None:
        cfg.n_eval_queries = n_eval_queries
    if candidate_count is not None:
        cfg.n_agents = candidate_count
    if top_k is not None:
        cfg.baseline_top_k = top_k
    if noise_std is not None:
        cfg.noise_std = noise_std
    if device is not None:
        cfg.device = device
    if embed_batch_size is not None:
        cfg.embed_batch_size = embed_batch_size

    _validate_topk_config(cfg)
    _require_one_revision_per_repository(cfg, datasets, retrievers)
    _require_topk_deps()
    out_dir = _new_topk_run_dir(out_dir)
    cfg.run_dir = out_dir
    _write_configuration(cfg, out_dir, datasets=datasets, retrievers=retrievers, alphas=alphas)

    per_query_sink: Optional[List[dict]] = [] if save_per_query else None
    rows = run_full_sweep(datasets, retrievers, alphas, cfg, per_query_sink=per_query_sink)

    save_full_sweep_csv(rows, out_dir=out_dir)
    save_certificate_comparison_csv(rows, out_dir=out_dir)
    save_summary_table_csv(rows, out_dir=out_dir)
    if per_query_sink:
        save_per_query_csv(per_query_sink, out_dir=out_dir, filename="full_sweep_per_query.csv")

    print(
        f"\n=== Full sweep complete: {len(datasets)} datasets x {len(retrievers)} retrievers "
        f"x {len(alphas)} alphas ==="
    )
    print(f"Total rows: {len(rows)}")
    print(
        "Total violations -- global: %d, pairwise: %d, ordered: %d (all should always be 0)"
        % (
            sum(r["global_certificate_violations"] for r in rows),
            sum(r["pairwise_certificate_violations"] for r in rows),
            sum(r["ordered_certificate_violations"] for r in rows),
        )
    )


# --------------------------------------------------------------------------
# candidate-pool ablation (MiniLM only, separate from the full sweep)
# --------------------------------------------------------------------------

def run_candidate_pool_ablation(
    datasets: List[str],
    candidate_counts: List[int],
    retriever: str,
    cfg_base: TopKStabilityConfig,
    on_combo_done: Optional[Callable[[List[dict]], None]] = None,
) -> List[dict]:
    """pool ablation"""
    import gc

    rows: List[dict] = []
    for dataset in datasets:
        for n_candidates in candidate_counts:
            cfg = replace(cfg_base, dataset=dataset, lm_checkpoint=retriever, n_agents=n_candidates)
            _validate_topk_config(cfg)
            logger.info("=== ablation dataset=%s n_candidates=%d ===", dataset, n_candidates)
            records, shards, init_state_dicts = _prepare_hotpotqa_run(
                cfg, bias=False, identity_init=True, dataset=dataset,
            )
            history = run_topk_stability_experiment(cfg, records, shards, init_state_dicts)
            combo_rows = [
                {"dataset": dataset, "retriever": retriever, "n_candidates": n_candidates, **_topk_dict(m)}
                for m in history
            ]
            rows.extend(combo_rows)
            if on_combo_done is not None:
                on_combo_done(combo_rows)  # incremental save, survives a mid-run kill

            del records, shards, init_state_dicts, history  # release encoder/tensors promptly
            gc.collect()
    return rows


_ABLATION_COLUMNS = list(dict.fromkeys(["dataset", "retriever", "n_candidates"] + _TOPK_METRIC_COLUMNS))


def save_candidate_pool_ablation_csv(
    rows: List[dict], out_dir: Optional[str] = None, filename: str = "candidate_pool_ablation.csv"
) -> str:
    """save ablation"""
    if out_dir is None:
        out_dir = _new_topk_run_dir()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, filename)
    with open(path, "x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_ABLATION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    record_csv(path)
    logger.info("Saved candidate-pool-size ablation to %s", path)
    return path


def main_candidate_pool_ablation(
    datasets: Optional[List[str]] = None,
    candidate_counts: Optional[List[int]] = None,
    n_eval_queries: Optional[int] = None,
    diffusion_alpha: Optional[float] = None,
    noise_std: Optional[float] = None,
    device: Optional[str] = None,
    embed_batch_size: Optional[int] = None,
    retrieval_k: Optional[int] = None,
    out_dir: Optional[str] = None,
    top_k: Optional[int] = None,
    run_options: Optional[dict] = None,
) -> None:
    """10/20/50 candidate passages, MiniLM only, separate from the 3x3 sweep."""

    datasets = datasets or list(known_datasets())
    candidate_counts = candidate_counts or [10, 20, 50]
    retriever = "sentence-transformers/all-MiniLM-L6-v2"

    cfg = TopKStabilityConfig(**(run_options or {}))
    if retrieval_k is not None:
        cfg.k = retrieval_k
    if top_k is not None:
        cfg.baseline_top_k = top_k
    if n_eval_queries is not None:
        cfg.n_eval_queries = n_eval_queries
    if diffusion_alpha is not None:
        cfg.diffusion_alpha = diffusion_alpha
    if noise_std is not None:
        cfg.noise_std = noise_std
    if device is not None:
        cfg.device = device
    if embed_batch_size is not None:
        cfg.embed_batch_size = embed_batch_size

    _validate_topk_config(cfg)
    _require_one_revision_per_repository(cfg, datasets, [retriever])
    _require_topk_deps()
    out_dir = _new_topk_run_dir(out_dir)
    cfg.run_dir = out_dir
    _write_configuration(cfg, out_dir, datasets=datasets, candidate_counts=candidate_counts)

    path = os.path.join(out_dir, "candidate_pool_ablation.csv")
    os.makedirs(out_dir, exist_ok=True)
    f = open(path, "x", newline="")
    writer = csv.DictWriter(f, fieldnames=_ABLATION_COLUMNS)
    writer.writeheader()

    def _save_combo(combo_rows: List[dict]) -> None:
        """flush combo"""
        for row in combo_rows:
            writer.writerow(row)
        f.flush()

    try:
        rows = run_candidate_pool_ablation(
            datasets, candidate_counts, retriever, cfg, on_combo_done=_save_combo,
        )
    finally:
        f.close()
    record_csv(path)
    logger.info("Saved candidate-pool-size ablation to %s", path)

    print(f"\n=== Candidate-pool ablation complete: {len(datasets)} datasets x {candidate_counts} candidates ===")
    print(
        "Total violations -- global: %d, pairwise: %d, ordered: %d (all should always be 0)"
        % (
            sum(r["global_certificate_violations"] for r in rows),
            sum(r["pairwise_certificate_violations"] for r in rows),
            sum(r["ordered_certificate_violations"] for r in rows),
        )
    )


# --------------------------------------------------------------------------
# run pipeline
# --------------------------------------------------------------------------

def main(out_dir: Optional[str] = None) -> None:
    cfg = ExperimentConfig()
    setup = build_shared_setup(cfg)

    baseline_hist = run_experiment("baseline", cfg, setup)
    sir_hist = run_experiment("sir", cfg, setup)

    out_dir = out_dir or legacy_output_dir("synthetic")
    save_results_csv(baseline_hist, sir_hist, out_dir=out_dir)
    plot_ieee_figures(baseline_hist, sir_hist, cfg.k, out_dir=out_dir)
    print_summary(baseline_hist, sir_hist)
    logger.info("Legacy synthetic outputs written to %s", out_dir)


def parse_args(argv=None):
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description="Multi-agent routing evaluation harness.")
    parser.add_argument(
        "--pipeline",
        choices=["synthetic", "hotpotqa", "topk-stability", "full-sweep", "candidate-ablation", "both"],
        default="synthetic",
        help="Which experiment to run (default: synthetic). 'topk-stability' runs a single "
             "dataset/retriever/alpha(-sweep); 'full-sweep' runs the 3 datasets x 3 retrievers "
             "x alpha-sweep main experiment; 'candidate-ablation' runs the 10/20/50-candidate "
             "MiniLM-only ablation.",
    )
    parser.add_argument(
        "--n-eval-queries", "--samples", dest="n_eval_queries", type=int, default=None,
        help="Number of evaluation queries. Overrides RealDataConfig.n_eval_queries for the "
             "legacy hotpotqa pipeline and TopKStabilityConfig.n_eval_queries for the "
             "certificate pipelines (default: 40).",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="Device for the frozen encoder's forward pass (default: cpu). Certificate scoring "
             "and the certificate decision always run on CPU in float64.",
    )
    parser.add_argument(
        "--batch-size", dest="embed_batch_size", type=int, default=None,
        help="Encoder embedding batch size; affects throughput only, not any score or decision "
             "(default: 16).",
    )
    parser.add_argument(
        "--dataset", choices=["hotpotqa", "musique", "2wikimultihopqa"], default=None,
        help="For --pipeline hotpotqa: 'hotpotqa' is also accepted as an alias for that pipeline. "
             "For topk-stability/full-sweep/candidate-ablation: which dataset(s) to use "
             "(full-sweep/candidate-ablation default to all three if omitted).",
    )
    parser.add_argument(
        "--retriever",
        choices=["sentence-transformers/all-MiniLM-L6-v2", "intfloat/e5-small-v2", "BAAI/bge-small-en-v1.5"],
        default=None,
        help="Retrieval embedding model, topk-stability/full-sweep pipelines only "
             "(full-sweep defaults to all three if omitted).",
    )
    parser.add_argument(
        "--candidate-count", dest="candidate_count", type=int, default=None,
        help="Number of candidate passages per query (= number of agents), topk-stability/"
             "full-sweep/candidate-ablation pipelines only. Default: TopKStabilityConfig.n_agents=10.",
    )
    parser.add_argument(
        "--save-per-query", action="store_true",
        help="Also write a combined per-query-per-hop CSV next to the sweep exports, "
             "topk-stability/full-sweep pipelines only. Each run always writes its own "
             "per_query.csv regardless of this flag.",
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Parent for a fresh timestamped certificate run directory (default: runs). "
             "Certificate pipelines only; the historical synthetic/hotpotqa pipelines always "
             "write a fresh runs/legacy/ directory and never touch results/ or figures/.",
    )
    parser.add_argument(
        "--gamma", type=float, default=None,
        help="Fixed Fisher-Rao kernel bandwidth for SIR (W_ij = exp(-gamma * d_IG)); "
             "default is RealDataConfig.sir_gamma=None, i.e. auto_kernel_gamma's median heuristic.",
    )
    routing = parser.add_mutually_exclusive_group()
    routing.add_argument("--graph-neighbors", type=int, default=None,
                         help="Incoming graph-neighbor count: each receiving agent takes edges from its "
                              "this-many most similar agents. Legacy HotpotQA shares this edge budget. "
                              "Unrelated to retrieval depth; see --retrieval-k.")
    routing.add_argument("--top-k", type=int, default=None,
                         help="Deprecated alias for --graph-neighbors; does not set retrieval depth.")
    parser.add_argument("--retrieval-k", type=int, default=None,
                        help="Retrieval evaluation and certificate depth (certificate pipelines; default: 5).")
    parser.add_argument(
        "--diffusion-alpha", type=float, default=None,
        help="Consensus blend rate for the topk-stability pipeline. Default: TopKStabilityConfig.diffusion_alpha=0.6.",
    )
    parser.add_argument(
        "--noise-std", type=float, default=None,
        help="Certificate pipelines require zero; nonzero post-consensus noise is rejected.",
    )
    parser.add_argument(
        "--alpha-sweep", type=str, default=None,
        help="Comma-separated diffusion_alpha values to sweep, topk-stability and full-sweep "
             "pipelines (e.g. 0.2,0.4,0.6,0.8). Overrides --diffusion-alpha.",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for candidate sampling, head initialization and per-hop graph "
                             "construction, certificate pipelines only (default: 42). Retrieval "
                             "ties are broken by candidate slot, not by this seed.")
    parser.add_argument("--hops", type=int, default=5,
                        help="Number of consensus hops, certificate pipelines only (default: 5). "
                             "One certificate decision is taken per query per hop.")
    parser.add_argument("--run-kind", choices=["smoke", "development", "paper"], default="development")
    parser.add_argument("--model-revision")
    parser.add_argument("--dataset-revision")
    args = parser.parse_args(argv)
    if args.top_k is not None:
        warnings.warn("--top-k is deprecated; use --graph-neighbors. Retrieval depth uses --retrieval-k.",
                      FutureWarning, stacklevel=2)
    else:
        args.top_k = args.graph_neighbors
    if args.retrieval_k is not None and args.pipeline not in ("topk-stability", "full-sweep", "candidate-ablation"):
        parser.error("--retrieval-k applies only to certificate pipelines")
    return args


if __name__ == "__main__":
    args = parse_args()
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
    if pipeline == "topk-stability":
        alpha_sweep = (
            [float(a) for a in args.alpha_sweep.split(",")] if args.alpha_sweep else None
        )
        main_topk_stability(
            run_options={"seed": args.seed, "n_hops": args.hops, "run_kind": args.run_kind,
                         "model_revision": args.model_revision, "dataset_revision": args.dataset_revision},
            retrieval_k=args.retrieval_k,
            out_dir=args.out_dir,
            n_eval_queries=args.n_eval_queries,
            device=args.device,
            embed_batch_size=args.embed_batch_size,
            top_k=args.top_k,
            diffusion_alpha=args.diffusion_alpha,
            noise_std=args.noise_std,
            alpha_sweep=alpha_sweep,
            dataset=args.dataset,
            retriever=args.retriever,
            candidate_count=args.candidate_count,
            save_per_query=args.save_per_query,
        )
    if pipeline == "full-sweep":
        alphas = [float(a) for a in args.alpha_sweep.split(",")] if args.alpha_sweep else None
        main_full_sweep(
            run_options={"seed": args.seed, "n_hops": args.hops, "run_kind": args.run_kind,
                         "model_revision": args.model_revision, "dataset_revision": args.dataset_revision},
            retrieval_k=args.retrieval_k,
            datasets=[args.dataset] if args.dataset else None,
            retrievers=[args.retriever] if args.retriever else None,
            alphas=alphas,
            n_eval_queries=args.n_eval_queries,
            candidate_count=args.candidate_count,
            top_k=args.top_k,
            noise_std=args.noise_std,
            device=args.device,
            embed_batch_size=args.embed_batch_size,
            save_per_query=args.save_per_query,
            out_dir=args.out_dir,
        )
    if pipeline == "candidate-ablation":
        main_candidate_pool_ablation(
            run_options={"seed": args.seed, "n_hops": args.hops, "run_kind": args.run_kind,
                         "model_revision": args.model_revision, "dataset_revision": args.dataset_revision},
            retrieval_k=args.retrieval_k,
            out_dir=args.out_dir,
            top_k=args.top_k,
            datasets=[args.dataset] if args.dataset else None,
            candidate_counts=[args.candidate_count] if args.candidate_count else None,
            n_eval_queries=args.n_eval_queries,
            diffusion_alpha=args.diffusion_alpha,
            noise_std=args.noise_std,
            device=args.device,
            embed_batch_size=args.embed_batch_size,
        )
