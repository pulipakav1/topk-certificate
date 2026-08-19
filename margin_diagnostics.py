"""margin diagnostics"""

from __future__ import annotations

from typing import List

import numpy as np


def compute_score_margin_diagnostics(scores: np.ndarray, gold_indices: np.ndarray) -> dict:
    """score margins"""
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
    """batch margins"""
    mean_margins, hard_margins, violations = [], [], []
    mean_margins_z, hard_margins_z = [], []
    for rec, scores in zip(records, per_query_scores):
        gold_indices = np.where(rec["relevance"] > 0)[0]
        if len(gold_indices) == 0 or len(gold_indices) == len(scores):
            continue  # skip degenerate
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
