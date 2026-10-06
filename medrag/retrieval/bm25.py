"""BM25 (Best Matching 25) retriever implementation.
Implements Okapi BM25 with k1 and b tuning parameters and intermediate inspectability.
"""
import math
import heapq
import logging
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple

from medrag.retrieval.base import BaseRetriever
from medrag.nlp.text import tokenize_and_stem, tokenize
from medrag.config import BM25_K1, BM25_B, PROCESSED_DIR

logger = logging.getLogger(__name__)

BM25_CACHE_PATH = PROCESSED_DIR / "bm25_index.pkl"


class BM25Retriever(BaseRetriever):
    """Okapi BM25 retriever with configurable k1 and b, and per-term score explanations."""
    
    def __init__(self, k1: float = BM25_K1, b: float = BM25_B):
        self.k1 = k1
        self.b = b
        self.chunks: List[Dict[str, Any]] = []
        self.N: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lengths: List[int] = []
        # term -> list of (c_idx, tf)
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        # term -> df
        self.df: Dict[str, int] = {}
        # term -> idf
        self.idf: Dict[str, float] = {}
        self.is_built: bool = False
        
    def build_index(self, chunks: List[Dict[str, Any]], force_rebuild: bool = False) -> None:
        """Build BM25 index over collection chunks or load from cache."""
        import pickle
        self.chunks = chunks
        self.N = len(chunks)

        if not force_rebuild and BM25_CACHE_PATH.exists():
            try:
                logger.info(f"Loading cached BM25 index from {BM25_CACHE_PATH}...")
                with open(BM25_CACHE_PATH, "rb") as f:
                    cache_data = pickle.load(f)
                if cache_data.get("N") == self.N:
                    self.postings = cache_data["postings"]
                    self.df = cache_data["df"]
                    self.idf = cache_data["idf"]
                    self.doc_lengths = cache_data["doc_lengths"]
                    self.avg_doc_len = cache_data["avg_doc_len"]
                    self.is_built = True
                    logger.info(f"Loaded BM25 index with avg_doc_len={self.avg_doc_len:.2f}")
                    return
            except Exception as e:
                logger.warning(f"Error loading BM25 cache: {e}. Rebuilding...")

        self.postings = defaultdict(list)
        self.df = defaultdict(int)
        self.doc_lengths = [0] * self.N
        
        total_len = 0
        for c_idx, chunk in enumerate(chunks):
            stems = tokenize_and_stem(chunk["text"])
            doc_len = len(stems)
            self.doc_lengths[c_idx] = doc_len
            total_len += doc_len
            
            tf_counts = Counter(stems)
            for term, tf in tf_counts.items():
                self.postings[term].append((c_idx, tf))
                self.df[term] += 1
                
        self.postings = dict(self.postings)
        self.df = dict(self.df)
        self.avg_doc_len = total_len / max(1, self.N)
        
        # Compute BM25 Robertson-Spärck Jones IDF
        self.idf = {}
        for term, df_val in self.df.items():
            self.idf[term] = math.log(((self.N - df_val + 0.5) / (df_val + 0.5)) + 1.0)
            
        self.is_built = True
        logger.info(f"Built BM25 index over {self.N} chunks. avg_doc_len={self.avg_doc_len:.2f}")

        # Save cache
        try:
            with open(BM25_CACHE_PATH, "wb") as f:
                pickle.dump({
                    "N": self.N,
                    "postings": self.postings,
                    "df": self.df,
                    "idf": self.idf,
                    "doc_lengths": self.doc_lengths,
                    "avg_doc_len": self.avg_doc_len
                }, f)
            logger.info("BM25 index cached successfully.")
        except Exception as e:
            logger.warning(f"Failed to cache BM25 index: {e}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Score candidate documents using Okapi BM25 formula."""
        if not self.is_built:
            raise RuntimeError("BM25 index has not been built.")
            
        query_stems = tokenize_and_stem(query)
        if not query_stems:
            return []
            
        q_tf = Counter(query_stems)
        accumulators: Dict[int, float] = defaultdict(float)
        
        for term, _ in q_tf.items():
            if term not in self.idf:
                continue
            idf_val = self.idf[term]
            postings_list = self.postings.get(term, [])
            
            for c_idx, tf in postings_list:
                doc_len = self.doc_lengths[c_idx]
                len_norm = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len)
                num = tf * (self.k1 + 1.0)
                den = tf + self.k1 * len_norm
                accumulators[c_idx] += idf_val * (num / den)
                
        if not accumulators:
            return []
            
        # Min-heap Top-K
        min_heap: List[Tuple[float, int]] = []
        for c_idx, score in accumulators.items():
            if len(min_heap) < top_k:
                heapq.heappush(min_heap, (score, c_idx))
            else:
                if score > min_heap[0][0]:
                    heapq.heappushpop(min_heap, (score, c_idx))
                    
        ranked = sorted(min_heap, key=lambda x: x[0], reverse=True)
        results = []
        for rank, (score, c_idx) in enumerate(ranked, start=1):
            chunk = self.chunks[c_idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "qtype": chunk["qtype"],
                "chunk_index": chunk["chunk_index"],
                "score": float(round(score, 4)),
                "text": chunk["text"],
                "rank": rank
            })
            
        return results

    def explain(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Explain BM25 scoring breakdown for each retrieved document."""
        query_stems = tokenize_and_stem(query)
        top_results = self.retrieve(query, top_k=top_k)
        
        term_diagnostics = {}
        for term in set(query_stems):
            df_val = self.df.get(term, 0)
            idf_val = self.idf.get(term, 0.0)
            term_diagnostics[term] = {
                "df": df_val,
                "idf": round(idf_val, 4),
                "in_vocab": term in self.idf
            }
            
        # Detailed score composition for top results
        results_breakdown = []
        for res in top_results:
            c_idx = next(i for i, c in enumerate(self.chunks) if c["chunk_id"] == res["chunk_id"])
            doc_len = self.doc_lengths[c_idx]
            len_norm = 1.0 - self.b + self.b * (doc_len / self.avg_doc_len)
            
            chunk_stems = tokenize_and_stem(res["text"])
            tf_counts = Counter(chunk_stems)
            
            per_term_scores = {}
            for term in set(query_stems):
                if term in self.idf:
                    tf = tf_counts.get(term, 0)
                    if tf > 0:
                        score_part = self.idf[term] * (tf * (self.k1 + 1.0)) / (tf + self.k1 * len_norm)
                        per_term_scores[term] = {
                            "tf": tf,
                            "bm25_term_score": round(score_part, 4)
                        }
                        
            results_breakdown.append({
                "chunk_id": res["chunk_id"],
                "rank": res["rank"],
                "total_score": res["score"],
                "doc_length": doc_len,
                "length_normalization_factor": round(len_norm, 4),
                "term_contributions": per_term_scores,
                "text_preview": res["text"][:120] + "..."
            })
            
        return {
            "retriever": "Okapi BM25",
            "k1": self.k1,
            "b": self.b,
            "avg_doc_len": round(self.avg_doc_len, 2),
            "query_stems": query_stems,
            "term_diagnostics": term_diagnostics,
            "top_results_breakdown": results_breakdown
        }
