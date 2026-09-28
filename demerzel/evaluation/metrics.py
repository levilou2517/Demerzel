"""Metrics computed over a run (V1.md §15).

At least: Answer Accuracy, Retrieval Recall@K, MRR, NDCG, Token Cost, Retrieval
Latency, End-to-End Latency, Memory Storage Size, Pointer Update Count, Cold
Recovery Rate, False Recall Rate, MPE distribution.

The evaluation helpers stay small and dependency-free; the benchmark adapter is
separate (V1.md §5.5: Demerzel does not embed an evaluation framework).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence


def recall_at_k(ranked_ids: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Recall@K: fraction of relevant ids found in the top-K."""
    top = set(ranked_ids[:k])
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    return len(top & relevant_set) / len(relevant_set)


def mrr(ranked_ids: Sequence[str], relevant: Sequence[str]) -> float:
    """Mean reciprocal rank over the first relevant hit."""
    relevant_set = set(relevant)
    for i, memory_id in enumerate(ranked_ids, start=1):
        if memory_id in relevant_set:
            return 1.0 / i
    return 0.0


def ndcg_at_k(
    ranked_ids: Sequence[str], relevance: Dict[str, float], k: int
) -> float:
    """nDCG@K with optionally binary relevance values."""
    ideal = sorted(relevance.values(), reverse=True)[:k]
    ideal_dcg = sum(
        (2 ** r - 1) / math.log2(i + 2)
        for i, r in enumerate(ideal)
    )
    if ideal_dcg == 0.0:
        return 0.0
    dcg = 0.0
    for i, memory_id in enumerate(ranked_ids[:k], start=1):
        r = relevance.get(memory_id, 0.0)
        dcg += (2 ** r - 1) / math.log2(i + 2)
    return dcg / ideal_dcg


def token_cost(built_contexts: List[int]) -> int:
    """Sum of token counts across assembled contexts."""
    return sum(built_contexts)


def latency_stats(latencies: Sequence[float]) -> Dict[str, float]:
    """min/mean/max of a latency list."""
    if not latencies:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": round(min(latencies), 4),
        "mean": round(sum(latencies) / len(latencies), 4),
        "max": round(max(latencies), 4),
    }


def accuracy(predicted: Sequence[str], expected: Sequence[str]) -> float:
    """Fraction of exact matches."""
    if not predicted:
        return 0.0
    return sum(1 for p, e in zip(predicted, expected) if p == e) / len(predicted)


def mpe_distribution(mpe_values: Sequence[float]) -> Dict[str, float]:
    """Shared stats of the MPE values; n/a when there are none."""
    if not mpe_values:
        return {"count": 0, "mean": None, "positive_frac": None}
    return {
        "count": len(mpe_values),
        "mean": round(sum(mpe_values) / len(mpe_values), 6),
        "positive_frac": round(
            sum(1 for v in mpe_values if v > 0) / len(mpe_values), 6
        ),
    }


__all__ = [
    "accuracy",
    "latency_stats",
    "mpe_distribution",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    "token_cost",
]