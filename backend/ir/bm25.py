"""
backend/ir/bm25.py
Okapi BM25 Probabilistic Retrieval Model for HealthNest.
Operates over the 'body' zone using the hand-written Inverted Index structure.
Parameters:
  k1: Term frequency saturation non-linear scaling constant (default 1.2)
  b:  Document length normalization weighting constant (default 0.75)

Lecture Concepts: Probabilistic IR, Okapi BM25, Document Length Normalization.
"""

import math
import heapq
from collections import defaultdict
from typing import Dict, List, Any
import config
from backend.ir.inverted_index import InvertedIndex
from backend.ir.text import tokenize


class BM25Retriever:
    def __init__(self, index: InvertedIndex, k1: float = config.BM25_K1, b: float = config.BM25_B):
        self.index = index
        self.k1 = k1
        self.b = b
        self.body_zone = index.zones["body"]
        
        # Calculate document lengths & average document length for body zone
        self.doc_lens: Dict[str, int] = defaultdict(int)
        total_len = 0
        for chunk_id, chunk in index.chunks_map.items():
            # Estimate doc length by word count of body text
            l = len(chunk["text"].split())
            self.doc_lens[chunk_id] = l
            total_len += l
        self.avg_doc_len: float = (total_len / float(max(1, index.N))) if index.N > 0 else 1.0

    def search(
        self,
        query: str,
        top_k: int = 5,
        use_stopwords: bool = config.USE_STOP_WORDS,
        use_stemmer: bool = config.USE_PORTER_STEMMER
    ) -> List[Dict[str, Any]]:
        """
        Executes Okapi BM25 search over body zone.
        """
        q_tokens = tokenize(query, use_stopwords=use_stopwords, use_stemmer=use_stemmer)
        if not q_tokens or self.index.N == 0:
            return []

        q_tf_map = defaultdict(int)
        for tok in q_tokens:
            q_tf_map[tok] += 1

        scores: Dict[str, float] = defaultdict(float)
        matched_terms_map: Dict[str, List[str]] = defaultdict(list)

        for term in q_tf_map:
            df = self.body_zone.df.get(term, 0)
            if df == 0:
                continue

            # Standard Robertson-Spärck Jones BM25 IDF formula with smoothing
            idf = math.log((self.index.N - df + 0.5) / (df + 0.5) + 1.0)
            if idf < 0:
                idf = 0.0

            postings = self.body_zone.postings.get(term, [])
            for chunk_id, d_tf in postings:
                doc_len = self.doc_lens.get(chunk_id, int(self.avg_doc_len))
                # BM25 TF saturation component with doc length normalization
                denom = d_tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                term_score = idf * ((d_tf * (self.k1 + 1.0)) / denom)
                
                scores[chunk_id] += term_score
                matched_terms_map[chunk_id].append(term)

        # Min-heap top-K
        top_chunk_pairs = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])

        results = []
        for rank, (chunk_id, score) in enumerate(top_chunk_pairs, start=1):
            if score <= 0:
                continue
            chunk_data = self.index.chunks_map[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "score": round(score, 4),
                "rank": rank,
                "zone_scores": {"body": round(score, 4)},
                "matched_terms": sorted(matched_terms_map[chunk_id]),
                "chunk": chunk_data
            })

        return results
