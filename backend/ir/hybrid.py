"""
backend/ir/hybrid.py
Hybrid Retriever Module for HealthNest.
Implements:
1. Reciprocal Rank Fusion (RRF) with constant k=60.
2. Weighted Sum Fusion with Min-Max Score Normalization.
3. Full rank and score inspection payload for frontend UI inspector component.

Lecture Concepts: Hybrid Retrieval, Rank Aggregation, Score Normalization, Reciprocal Rank Fusion.
"""

from typing import List, Dict, Any, Optional
import config


def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalizes a dictionary of chunk_id -> raw_score into [0, 1]."""
    if not scores:
        return {}
    min_val = min(scores.values())
    max_val = max(scores.values())
    val_range = max_val - min_val
    if val_range <= 1e-9:
        return {cid: 1.0 for cid in scores}
    return {cid: (val - min_val) / val_range for cid, val in scores.items()}


class HybridRetriever:
    def __init__(self, rrf_k: int = config.RRF_K, alpha: float = config.HYBRID_ALPHA):
        self.rrf_k = rrf_k
        self.alpha = alpha

    def fuse(
        self,
        sparse_results: List[Dict[str, Any]],
        dense_results: List[Dict[str, Any]],
        top_k: int = 5,
        fusion_method: str = "rrf"  # "rrf" or "weighted"
    ) -> List[Dict[str, Any]]:
        """
        Fuses sparse (TF-IDF or BM25) and dense retrieval results.
        
        Returns:
            List of fused result dicts with full inspector breakdown:
            {chunk_id, score, rank, sparse_rank, sparse_score, dense_rank, dense_score, matched_terms, chunk}
        """
        all_chunks: Dict[str, Dict[str, Any]] = {}
        sparse_ranks: Dict[str, int] = {}
        sparse_scores: Dict[str, float] = {}
        dense_ranks: Dict[str, int] = {}
        dense_scores: Dict[str, float] = {}
        matched_terms_map: Dict[str, List[str]] = {}

        for item in sparse_results:
            cid = item["chunk_id"]
            all_chunks[cid] = item["chunk"]
            sparse_ranks[cid] = item["rank"]
            sparse_scores[cid] = item["score"]
            matched_terms_map[cid] = item.get("matched_terms", [])

        for item in dense_results:
            cid = item["chunk_id"]
            all_chunks[cid] = item["chunk"]
            dense_ranks[cid] = item["rank"]
            dense_scores[cid] = item["score"]

        fused_scores: Dict[str, float] = {}

        if fusion_method == "rrf":
            # Reciprocal Rank Fusion: sum( 1 / (k + rank) )
            for cid in all_chunks:
                score = 0.0
                if cid in sparse_ranks:
                    score += 1.0 / (self.rrf_k + sparse_ranks[cid])
                if cid in dense_ranks:
                    score += 1.0 / (self.rrf_k + dense_ranks[cid])
                fused_scores[cid] = score

        elif fusion_method == "weighted":
            norm_sparse = min_max_normalize(sparse_scores)
            norm_dense = min_max_normalize(dense_scores)
            for cid in all_chunks:
                s_score = norm_sparse.get(cid, 0.0)
                d_score = norm_dense.get(cid, 0.0)
                fused_scores[cid] = (self.alpha * s_score) + ((1.0 - self.alpha) * d_score)

        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Sort combined results by fused score descending
        sorted_cids = sorted(fused_scores.keys(), key=lambda cid: fused_scores[cid], reverse=True)[:top_k]

        results = []
        for rank, cid in enumerate(sorted_cids, start=1):
            chunk_data = all_chunks[cid]
            results.append({
                "chunk_id": cid,
                "score": round(fused_scores[cid], 5),
                "rank": rank,
                "sparse_rank": sparse_ranks.get(cid, None),
                "sparse_score": round(sparse_scores.get(cid, 0.0), 4),
                "dense_rank": dense_ranks.get(cid, None),
                "dense_score": round(dense_scores.get(cid, 0.0), 4),
                "zone_scores": {
                    "sparse": round(sparse_scores.get(cid, 0.0), 4),
                    "dense": round(dense_scores.get(cid, 0.0), 4)
                },
                "matched_terms": matched_terms_map.get(cid, []),
                "chunk": chunk_data
            })

        return results
