"""Hybrid Retriever implementing Reciprocal Rank Fusion (RRF).
Fuses sparse (Inverted Index / BM25) and dense (SentenceTransformers) rankings.
"""
import logging
from typing import List, Dict, Any, Tuple
from collections import defaultdict

from src.retrievers.base import BaseRetriever
from src.config import RRF_K

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """Combines sparse and dense retrievers via Reciprocal Rank Fusion (RRF)."""
    
    def __init__(self, sparse_retriever: BaseRetriever, dense_retriever: BaseRetriever, rrf_k: int = RRF_K):
        self.sparse_retriever = sparse_retriever
        self.dense_retriever = dense_retriever
        self.rrf_k = rrf_k
        self.chunks: List[Dict[str, Any]] = []
        
    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Build underlying sparse and dense indexes."""
        self.chunks = chunks
        if not getattr(self.sparse_retriever, "is_built", False):
            self.sparse_retriever.build_index(chunks)
        if not getattr(self.dense_retriever, "is_built", False):
            self.dense_retriever.build_index(chunks)
            
    def retrieve(self, query: str, top_k: int = 5, candidate_pool_size: int = 50) -> List[Dict[str, Any]]:
        """Retrieve using RRF fusion over sparse and dense candidate lists."""
        sparse_results = self.sparse_retriever.retrieve(query, top_k=candidate_pool_size)
        dense_results = self.dense_retriever.retrieve(query, top_k=candidate_pool_size)
        
        # rrf_scores: chunk_id -> combined score
        rrf_scores: Dict[str, float] = defaultdict(float)
        chunk_map: Dict[str, Dict[str, Any]] = {}
        
        # Add sparse ranks
        for item in sparse_results:
            cid = item["chunk_id"]
            rank = item["rank"]
            rrf_scores[cid] += 1.0 / (self.rrf_k + rank)
            chunk_map[cid] = item
            
        # Add dense ranks
        for item in dense_results:
            cid = item["chunk_id"]
            rank = item["rank"]
            rrf_scores[cid] += 1.0 / (self.rrf_k + rank)
            if cid not in chunk_map:
                chunk_map[cid] = item
                
        # Sort by RRF score descending
        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (cid, score) in enumerate(sorted_chunks[:top_k], start=1):
            original = chunk_map[cid]
            results.append({
                "chunk_id": cid,
                "doc_id": original["doc_id"],
                "qtype": original["qtype"],
                "chunk_index": original["chunk_index"],
                "score": float(round(score, 6)),
                "text": original["text"],
                "rank": rank
            })
            
        return results

    def explain(self, query: str, top_k: int = 5, candidate_pool_size: int = 20) -> Dict[str, Any]:
        """Explain RRF calculation with per-retriever rank breakdown."""
        sparse_results = self.sparse_retriever.retrieve(query, top_k=candidate_pool_size)
        dense_results = self.dense_retriever.retrieve(query, top_k=candidate_pool_size)
        
        sparse_ranks = {item["chunk_id"]: item["rank"] for item in sparse_results}
        sparse_scores = {item["chunk_id"]: item["score"] for item in sparse_results}
        
        dense_ranks = {item["chunk_id"]: item["rank"] for item in dense_results}
        dense_scores = {item["chunk_id"]: item["score"] for item in dense_results}
        
        fused_results = self.retrieve(query, top_k=top_k, candidate_pool_size=candidate_pool_size)
        
        fused_breakdown = []
        for item in fused_results:
            cid = item["chunk_id"]
            s_rank = sparse_ranks.get(cid, None)
            d_rank = dense_ranks.get(cid, None)
            
            s_contrib = (1.0 / (self.rrf_k + s_rank)) if s_rank else 0.0
            d_contrib = (1.0 / (self.rrf_k + d_rank)) if d_rank else 0.0
            
            fused_breakdown.append({
                "chunk_id": cid,
                "fused_rank": item["rank"],
                "fused_rrf_score": item["score"],
                "sparse_rank": s_rank,
                "sparse_score": sparse_scores.get(cid, None),
                "sparse_rrf_contrib": round(s_contrib, 6),
                "dense_rank": d_rank,
                "dense_score": dense_scores.get(cid, None),
                "dense_rrf_contrib": round(d_contrib, 6),
                "text_preview": item["text"][:120] + "..."
            })
            
        return {
            "retriever": "Hybrid (Reciprocal Rank Fusion)",
            "rrf_k_constant": self.rrf_k,
            "sparse_retriever_name": self.sparse_retriever.__class__.__name__,
            "dense_retriever_name": self.dense_retriever.__class__.__name__,
            "query": query,
            "top_fused_results": fused_breakdown
        }
