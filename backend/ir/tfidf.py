"""
backend/ir/tfidf.py
SMART lnc.ltc TF-IDF Cosine Vector Space Model Retriever for HealthNest.
Implements:
1. SMART lnc.ltc weighting scheme:
   - Document (lnc): log tf, no idf, cosine normalization.
   - Query (ltc): log tf, idf (ln(N/df)), cosine normalization.
2. Accumulator-based Term-at-a-time (TAAT) scoring algorithm.
3. Zone weighting: final score = sum(zone_weight * cosine_zone).
4. Optional Champion Lists and Index Elimination query processing optimizations.
5. Per-chunk score breakdown and term matching explanation for UI inspector.

Lecture Concepts: Vector Space Model, SMART notation (lnc.ltc), Term-at-a-time Scoring, Heap Top-K.
Strict Compliance: Built from scratch with math & heapq!
"""

import math
import heapq
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set
import config
from backend.ir.inverted_index import InvertedIndex
from backend.ir.text import tokenize


class TfIdfRetriever:
    def __init__(self, index: InvertedIndex):
        self.index = index

    def search(
        self,
        query: str,
        top_k: int = 5,
        zone_weights: Dict[str, float] = config.ZONE_WEIGHTS,
        use_champion_lists: bool = False,
        use_index_elimination: bool = False,
        use_stopwords: bool = config.USE_STOP_WORDS,
        use_stemmer: bool = config.USE_PORTER_STEMMER
    ) -> List[Dict[str, Any]]:
        """
        Executes TF-IDF Vector Space search across question and body zones.
        
        Returns:
            List of result dicts: {chunk_id, score, zone_scores, matched_terms, chunk}
        """
        q_tokens = tokenize(query, use_stopwords=use_stopwords, use_stemmer=use_stemmer)
        if not q_tokens or self.index.N == 0:
            return []

        # Count query term frequencies
        q_tf_map = defaultdict(int)
        for tok in q_tokens:
            q_tf_map[tok] += 1

        # Accumulated scores per zone: zone_accumulators[zone_name][chunk_id] = dot_product
        zone_accumulators: Dict[str, Dict[str, float]] = {z: defaultdict(float) for z in self.index.zones}
        matched_terms_map: Dict[str, Set[str]] = defaultdict(set)

        for zone_name, zone_index in self.index.zones.items():
            accumulator = zone_accumulators[zone_name]
            q_norm_sq = 0.0

            for term, q_tf in q_tf_map.items():
                df = zone_index.df.get(term, 0)
                if df == 0:
                    continue

                # Query IDF: ln(N / df)
                idf = math.log(self.index.N / float(df))
                
                # Index Elimination optimization: skip low-IDF terms if requested
                if use_index_elimination and idf < config.INDEX_ELIMINATION_IDF_THRESHOLD:
                    continue

                # Query Weight (ltc): (1 + log(tf)) * idf
                q_wt = (1.0 + math.log(q_tf)) * idf
                q_norm_sq += q_wt * q_wt

                # Select postings source (Champion List vs Full Postings List)
                if use_champion_lists and term in zone_index.champion_lists:
                    postings = zone_index.champion_lists[term]
                else:
                    postings = zone_index.postings.get(term, [])

                # Accumulate doc dot products (lnc doc weight: 1 + log(tf))
                for chunk_id, d_tf in postings:
                    d_wt = 1.0 + math.log(d_tf)
                    accumulator[chunk_id] += q_wt * d_wt
                    matched_terms_map[chunk_id].add(term)

            # Cosine normalize zone scores
            q_norm = math.sqrt(q_norm_sq)
            if q_norm > 0:
                for chunk_id, dot_prod in accumulator.items():
                    d_norm = zone_index.doc_norms.get(chunk_id, 1.0)
                    if d_norm > 0:
                        accumulator[chunk_id] = dot_prod / (q_norm * d_norm)
                    else:
                        accumulator[chunk_id] = 0.0

        # Combine zone scores into final score using zone weights
        combined_scores: Dict[str, float] = {}
        zone_breakdowns: Dict[str, Dict[str, float]] = defaultdict(dict)

        all_candidate_chunk_ids = set()
        for zone_name, acc in zone_accumulators.items():
            all_candidate_chunk_ids.update(acc.keys())

        for chunk_id in all_candidate_chunk_ids:
            total_score = 0.0
            for zone_name, weight in zone_weights.items():
                z_score = zone_accumulators[zone_name].get(chunk_id, 0.0)
                zone_breakdowns[chunk_id][zone_name] = round(z_score, 4)
                total_score += weight * z_score
            combined_scores[chunk_id] = total_score

        # Select top-K using min-heap
        top_chunk_pairs = heapq.nlargest(top_k, combined_scores.items(), key=lambda x: x[1])

        results = []
        for rank, (chunk_id, score) in enumerate(top_chunk_pairs, start=1):
            if score <= 0:
                continue
            chunk_data = self.index.chunks_map[chunk_id]
            results.append({
                "chunk_id": chunk_id,
                "score": round(score, 4),
                "rank": rank,
                "zone_scores": zone_breakdowns[chunk_id],
                "matched_terms": sorted(list(matched_terms_map[chunk_id])),
                "chunk": chunk_data
            })

        return results
