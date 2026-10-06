"""IR and RAG Evaluation Metrics.
Implements:
1. Precision@K
2. Recall@K
3. Mean Reciprocal Rank (MRR)
4. Mean Average Precision (MAP)
5. Faithfulness & Citation Validity metrics
"""
from typing import List, Set, Dict, Any
import numpy as np


def precision_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Compute Precision@K: proportion of top-k retrieved items that are relevant."""
    if k <= 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return float(hits / k)


def recall_at_k(retrieved_ids: List[str], relevant_ids: Set[str], k: int) -> float:
    """Compute Recall@K: proportion of relevant items retrieved in top-k."""
    if not relevant_ids or not retrieved_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for item in top_k if item in relevant_ids)
    return float(hits / len(relevant_ids))


def reciprocal_rank(retrieved_ids: List[str], relevant_ids: Set[str]) -> float:
    """Compute Reciprocal Rank (RR): 1 / rank of first relevant item."""
    for rank, item in enumerate(retrieved_ids, start=1):
        if item in relevant_ids:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    """Compute Average Precision (AP) at k."""
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    num_hits = 0
    sum_precisions = 0.0
    for idx, item in enumerate(top_k, start=1):
        if item in relevant_ids:
            num_hits += 1
            sum_precisions += num_hits / idx
    if num_hits == 0:
        return 0.0
    return sum_precisions / min(len(relevant_ids), k)
