"""
margin_diagnostics.py
========================

Per-query gold-vs-distractor score-margin diagnostic for the HotpotQA
pipeline in `evaluation_harness.py`: measures whether cross-agent Markov
averaging compresses the score gap between gold and distractor documents
enough to explain the NDCG@5 gap on its own, as distinct from the
gold/distractor *mass* allocation already checked in
`theorem1_diagnostics.py`.

    mean margin   m_bar^(h) = mean(gold scores) - mean(distractor scores)
    hard margin   Delta_m   = min(gold scores) - max(distractor scores)
    violated      Delta_m < 0   (a distractor outscores the worst gold doc
                                  -- a necessary condition for that
                                  distractor to bump a gold doc out of a
                                  top-k cutoff)

Both are computed per query, then averaged (mean margin) / rate-counted
(violation) across the query set for a single (strategy, hop).
"""

from __future__ import annotations

from typing import List

import numpy as np


def compute_score_margin_diagnostics(scores: np.ndarray, gold_indices: np.ndarray) -> dict:
    """
    scores: (n_agents,) retrieval scores at one hop, for a single query.
    gold_indices: indices into `scores` of that query's gold documents.

    Also returns a *standardized* margin (scores z-scored within the
    query, i.e. (scores - mean) / std, before computing the same two
    quantities) alongside the raw ones. The raw bilinear score
    q . doc is not scale-invariant -- if the diffusion dynamics push an
    agent's head-weight norm to a different regime than another
    strategy's, every score for that query inflates or shrinks together,
    changing the raw margin's magnitude for reasons that have nothing to
    do with gold/distractor separability. Standardizing per query removes
    that confound so margins are comparable across strategies/hops whose
    underlying head norms may differ; NDCG/F1 are themselves scale/shift
    invariant, so the standardized margin is the more relevant one for
    explaining rank-based metrics specifically.

    Returns None-valued dict (via the caller's aggregation skipping it) is
    not done here -- callers should skip degenerate queries (no gold docs,
    or no distractor docs) themselves, mirroring
    `theorem1_diagnostics.routing_mass_diagnostic_batch`.
    """
    n = len(scores)
    gold_set = set(int(i) for i in gold_indices)
    distractor_indices = np.array([i for i in range(n) if i not in gold_set], dtype=int)

    def _margins(s: np.ndarray) -> tuple:
        gold_scores = s[gold_indices]
        distractor_scores = s[distractor_indices]
        mean_margin = float(np.mean(gold_scores) - np.mean(distractor_scores))
        hard_margin = float(np.min(gold_scores) - np.max(distractor_scores))
        return mean_margin, hard_margin

    mean_margin, hard_margin = _margins(scores)

    std = float(np.std(scores))
    scores_z = (scores - np.mean(scores)) / std if std > 1e-12 else np.zeros_like(scores)
    mean_margin_z, hard_margin_z = _margins(scores_z)

    return {
        "mean_margin": mean_margin,
        "hard_margin": hard_margin,
        "margin_violated": bool(hard_margin < 0),
        "mean_margin_standardized": mean_margin_z,
        "hard_margin_standardized": hard_margin_z,
    }


def aggregate_margin_diagnostics(records: List[dict], per_query_scores: List[np.ndarray]) -> dict:
    """
    Batched wrapper: `per_query_scores[i]` is the (n_agents,) score array
    for `records[i]` at one (strategy, hop). Skips any query with zero
    gold or zero distractor documents (can't define a margin), matching
    the same convention used for the routing-mass diagnostic.
    """
    mean_margins, hard_margins, violations = [], [], []
    mean_margins_z, hard_margins_z = [], []
    for rec, scores in zip(records, per_query_scores):
        gold_indices = np.where(rec["relevance"] > 0)[0]
        if len(gold_indices) == 0 or len(gold_indices) == len(scores):
            continue  # degenerate: no gold, or every slot is gold (no distractors to compare against)
        d = compute_score_margin_diagnostics(scores, gold_indices)
        mean_margins.append(d["mean_margin"])
        hard_margins.append(d["hard_margin"])
        violations.append(d["margin_violated"])
        mean_margins_z.append(d["mean_margin_standardized"])
        hard_margins_z.append(d["hard_margin_standardized"])

    if not mean_margins:
        return {
            "mean_margin": float("nan"), "mean_hard_margin": float("nan"), "violation_rate": float("nan"),
            "mean_margin_standardized": float("nan"), "mean_hard_margin_standardized": float("nan"),
            "n_queries_used": 0,
        }
    return {
        "mean_margin": float(np.mean(mean_margins)),
        "mean_hard_margin": float(np.mean(hard_margins)),
        "violation_rate": float(np.mean(violations)),
        "mean_margin_standardized": float(np.mean(mean_margins_z)),
        "mean_hard_margin_standardized": float(np.mean(hard_margins_z)),
        "n_queries_used": len(mean_margins),
    }
