"""Inverted Index retriever using SMART lnc.ltc weighting and min-heap Top-K.
Implements:
1. Positional / frequency postings list with Porter Stemmer
2. SMART weighting scheme:
   - Document: lnc (log tf, no idf, cosine normalization)
   - Query: ltc (log tf, idf, cosine normalization)
3. Term-At-A-Time (TAAT) scoring accumulator
4. Top-K min-heap ranking
5. Full inspectability API for intermediate IR states
"""
import math
import heapq
import pickle
import logging
from collections import defaultdict, Counter
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from src.retrievers.base import BaseRetriever
from src.text_processing import tokenize_and_stem, tokenize
from src.config import INDEX_CACHE_PATH

logger = logging.getLogger(__name__)


class InvertedIndexRetriever(BaseRetriever):
    """Inspectable Inverted Index with SMART lnc.ltc Vector Space Model."""
    
    def __init__(self):
        self.chunks: List[Dict[str, Any]] = []
        # term -> list of (chunk_idx, raw_tf)
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        # term -> document frequency (df)
        self.df: Dict[str, int] = {}
        # term -> idf = log(N / df)
        self.idf: Dict[str, float] = {}
        # chunk_idx -> Euclidean norm of document lnc vector
        self.doc_norms: List[float] = []
        # Total number of chunks N
        self.N: int = 0
        self.is_built: bool = False
        
    def build_index(self, chunks: List[Dict[str, Any]], force_rebuild: bool = False) -> None:
        """Build the inverted index from chunks or load from cache."""
        self.chunks = chunks
        self.N = len(chunks)
        
        if not force_rebuild and INDEX_CACHE_PATH.exists():
            try:
                logger.info(f"Loading cached inverted index from {INDEX_CACHE_PATH}...")
                with open(INDEX_CACHE_PATH, "rb") as f:
                    cache_data = pickle.load(f)
                if cache_data.get("N") == self.N:
                    self.postings = cache_data["postings"]
                    self.df = cache_data["df"]
                    self.idf = cache_data["idf"]
                    self.doc_norms = cache_data["doc_norms"]
                    self.is_built = True
                    logger.info(f"Loaded inverted index with {len(self.df)} unique vocabulary terms.")
                    return
            except Exception as e:
                logger.warning(f"Could not load cache, rebuilding: {e}")
                
        logger.info(f"Building inverted index over {self.N} chunks...")
        self.postings = defaultdict(list)
        self.df = defaultdict(int)
        self.doc_norms = [0.0] * self.N
        
        for c_idx, chunk in enumerate(chunks):
            text = chunk["text"]
            stems = tokenize_and_stem(text)
            tf_counts = Counter(stems)
            
            # Compute document lnc weights: w = 1 + ln(tf)
            sum_sq = 0.0
            for term, tf in tf_counts.items():
                self.postings[term].append((c_idx, tf))
                self.df[term] += 1
                w_d = 1.0 + math.log(tf)
                sum_sq += w_d * w_d
                
            self.doc_norms[c_idx] = math.sqrt(sum_sq) if sum_sq > 0 else 1.0
            
        # Convert defaultdict to normal dict
        self.postings = dict(self.postings)
        self.df = dict(self.df)
        
        # Compute IDF for all terms: idf = log(N / df)
        logger.info(f"Computing IDF weights for {len(self.df)} terms (N={self.N})...")
        self.idf = {}
        for term, doc_freq in self.df.items():
            # Standard IR natural log idf
            self.idf[term] = math.log((self.N + 1.0) / (doc_freq + 0.5))
            
        self.is_built = True
        
        # Save cache
        try:
            INDEX_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(INDEX_CACHE_PATH, "wb") as f:
                pickle.dump({
                    "N": self.N,
                    "postings": self.postings,
                    "df": self.df,
                    "idf": self.idf,
                    "doc_norms": self.doc_norms
                }, f)
            logger.info("Inverted index cached successfully.")
        except Exception as e:
            logger.warning(f"Failed to cache inverted index: {e}")

    def get_postings(self, term: str) -> Optional[List[Tuple[int, int]]]:
        """Return postings list for a given term."""
        return self.postings.get(term, None)

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Execute SMART lnc.ltc retrieval with TAAT accumulator and min-heap Top-K."""
        if not self.is_built:
            raise RuntimeError("Index has not been built yet.")
            
        query_stems = tokenize_and_stem(query)
        if not query_stems:
            return []
            
        q_tf = Counter(query_stems)
        
        # 1. Compute query vector in ltc scheme
        # l = 1 + ln(tf), t = idf, c = cosine norm
        q_weights: Dict[str, float] = {}
        q_sum_sq = 0.0
        for term, tf in q_tf.items():
            if term in self.idf:
                w_q = (1.0 + math.log(tf)) * self.idf[term]
                q_weights[term] = w_q
                q_sum_sq += w_q * w_q
                
        if q_sum_sq == 0.0:
            return []
            
        q_norm = math.sqrt(q_sum_sq)
        # Normalized query weights
        norm_q_weights = {term: w / q_norm for term, w in q_weights.items()}
        
        # 2. Term-At-A-Time (TAAT) score accumulation
        accumulators: Dict[int, float] = defaultdict(float)
        for term, q_w in norm_q_weights.items():
            postings_list = self.postings.get(term, [])
            for c_idx, doc_tf in postings_list:
                doc_norm = self.doc_norms[c_idx]
                if doc_norm > 0:
                    # Document lnc weight: (1 + ln(tf)) / doc_norm
                    w_d_norm = (1.0 + math.log(doc_tf)) / doc_norm
                    accumulators[c_idx] += q_w * w_d_norm
                    
        if not accumulators:
            return []
            
        # 3. Top-K selection using a min-heap of size K
        # Heap elements: (score, c_idx)
        min_heap: List[Tuple[float, int]] = []
        for c_idx, score in accumulators.items():
            if len(min_heap) < top_k:
                heapq.heappush(min_heap, (score, c_idx))
            else:
                if score > min_heap[0][0]:
                    heapq.heappushpop(min_heap, (score, c_idx))
                    
        # Sort descending by score
        ranked_items = sorted(min_heap, key=lambda x: x[0], reverse=True)
        
        results = []
        for rank, (score, c_idx) in enumerate(ranked_items, start=1):
            chunk = self.chunks[c_idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "qtype": chunk["qtype"],
                "chunk_index": chunk["chunk_index"],
                "score": float(round(score, 5)),
                "text": chunk["text"],
                "rank": rank
            })
            
        return results

    def explain(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Provide detailed, inspectable intermediate IR diagnostics."""
        tokens = tokenize(query)
        stems = tokenize_and_stem(query)
        q_tf = Counter(stems)
        
        query_terms_diagnostic = {}
        q_sum_sq = 0.0
        
        for term, tf in q_tf.items():
            in_vocab = term in self.idf
            doc_freq = self.df.get(term, 0)
            idf_val = self.idf.get(term, 0.0)
            unnorm_w = (1.0 + math.log(tf)) * idf_val if in_vocab else 0.0
            q_sum_sq += unnorm_w * unnorm_w
            
            # Sample postings (first 5)
            sample_postings = self.postings.get(term, [])[:5]
            
            query_terms_diagnostic[term] = {
                "raw_tf": tf,
                "in_vocabulary": in_vocab,
                "df": doc_freq,
                "idf": round(idf_val, 4),
                "unnormalized_ltc_weight": round(unnorm_w, 4),
                "postings_count": len(self.postings.get(term, [])),
                "postings_preview": [{"chunk_idx": c, "tf": t} for c, t in sample_postings]
            }
            
        q_norm = math.sqrt(q_sum_sq) if q_sum_sq > 0 else 1.0
        for term in query_terms_diagnostic:
            unnorm_w = query_terms_diagnostic[term]["unnormalized_ltc_weight"]
            query_terms_diagnostic[term]["normalized_ltc_weight"] = round(unnorm_w / q_norm, 4)
            
        retrieved_results = self.retrieve(query, top_k=top_k)
        
        return {
            "retriever": "InvertedIndex (lnc.ltc + Min-Heap)",
            "query": query,
            "tokens": tokens,
            "stems": stems,
            "query_norm": round(q_norm, 4),
            "term_diagnostics": query_terms_diagnostic,
            "results_count": len(retrieved_results),
            "top_results": retrieved_results
        }
