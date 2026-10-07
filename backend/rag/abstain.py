"""
backend/rag/abstain.py
Abstention & Confidence Decision Module for HealthNest.
Implements evidence strength thresholding to abstain ("I don't know") when:
1. Top retrieval score is below calibrated threshold.
2. Query term IDF coverage in top-K retrieved chunks is too low.
3. LLM output indicates insufficient evidence.

Lecture Concept: Selective Answering / Abstention in QA Systems.
"""

from typing import List, Dict, Any, Tuple, Set
import config
from backend.ir.text import tokenize, HINGLISH_STOP_WORDS


def calculate_query_term_coverage(
    query: str,
    top_chunks: List[Dict[str, Any]],
    added_terms: List[str] = None,
    concept_groups: List[Set[str]] = None
) -> float:
    """
    Calculates proportion of query concepts present across top retrieved chunks.
    For Hinglish / expanded queries, evaluates concept coverage across synonym groups.
    """
    if not top_chunks:
        return 0.0

    all_chunk_tokens = set()
    for item in top_chunks:
        chunk_data = item.get("chunk", item) if isinstance(item, dict) else {}
        chunk_text = f"{chunk_data.get('text', '')} {chunk_data.get('question', '')}"
        all_chunk_tokens.update(tokenize(chunk_text, use_stopwords=True, use_stemmer=True))

    # Evaluate concept groups (each group is a set of synonym tokens; any match satisfies the concept)
    if concept_groups:
        satisfied = sum(1 for grp in concept_groups if len(grp.intersection(all_chunk_tokens)) > 0)
        coverage = satisfied / float(len(concept_groups))
        return round(coverage, 4)

    # For queries expanded from Hinglish / synonyms without explicit concept_groups:
    if added_terms:
        groups = []
        for term in added_terms:
            t_tokens = set(tokenize(term, use_stopwords=True, use_stemmer=True))
            if t_tokens:
                groups.append(t_tokens)
        if groups:
            satisfied = sum(1 for grp in groups if len(grp.intersection(all_chunk_tokens)) > 0)
            coverage = satisfied / float(len(groups))
            return round(coverage, 4)

    # Standard query coverage:
    q_tokens = set(tokenize(query, use_stopwords=True, use_stemmer=True))
    if not q_tokens:
        return 1.0

    found_tokens = q_tokens.intersection(all_chunk_tokens)
    coverage = len(found_tokens) / float(len(q_tokens))
    return round(coverage, 4)


def evaluate_abstention(
    query: str,
    retrieval_results: List[Dict[str, Any]],
    llm_text: str = "",
    score_threshold: float = config.ABSTAIN_SCORE_THRESHOLD,
    coverage_threshold: float = config.ABSTAIN_QUERY_COVERAGE_THRESHOLD,
    added_terms: List[str] = None,
    concept_groups: List[Set[str]] = None
) -> Dict[str, Any]:
    """
    Evaluates evidence quality and returns abstention decision dict.
    """
    raw_top_score = retrieval_results[0]["score"] if retrieval_results else 0.0
    
    # RRF scores max out around 2/(60+1) = 0.0327. Scale RRF scores to [0, 1] range for threshold check
    if raw_top_score < 0.05 and raw_top_score > 0:
        scaled_top_score = raw_top_score * 30.0  # Maps 0.0327 -> ~0.98
    else:
        scaled_top_score = raw_top_score

    coverage = calculate_query_term_coverage(
        query, retrieval_results, added_terms=added_terms, concept_groups=concept_groups
    )

    metrics = {
        "raw_top_score": raw_top_score,
        "scaled_top_score": round(scaled_top_score, 4),
        "query_coverage": coverage,
        "score_threshold": score_threshold,
        "coverage_threshold": coverage_threshold
    }

    # Condition 1: No retrieval results or score below score threshold
    if not retrieval_results or scaled_top_score < score_threshold:
        return {
            "should_abstain": True,
            "reason": f"Low retrieval confidence score ({scaled_top_score:.4f} < threshold {score_threshold:.4f})",
            "metrics": metrics
        }

    # Condition 2: Top retrieved document has zero matching query terms
    if retrieval_results:
        top_data = retrieval_results[0].get("chunk", retrieval_results[0])
        top_text = f"{top_data.get('text', '')} {top_data.get('question', '')}"
        top_tokens = set(tokenize(top_text, use_stopwords=True, use_stemmer=True))

        if concept_groups:
            has_overlap = any(len(grp.intersection(top_tokens)) > 0 for grp in concept_groups)
            if not has_overlap:
                return {
                    "should_abstain": True,
                    "reason": "Top retrieved reference contains none of the queried medical terms.",
                    "metrics": metrics
                }
        else:
            if added_terms:
                target_tokens = set()
                for t in added_terms:
                    target_tokens.update(tokenize(t, use_stopwords=True, use_stemmer=True))
            else:
                target_tokens = set(tokenize(query, use_stopwords=True, use_stemmer=True))

            if target_tokens and len(target_tokens.intersection(top_tokens)) == 0:
                return {
                    "should_abstain": True,
                    "reason": "Top retrieved reference contains none of the queried medical terms.",
                    "metrics": metrics
                }

    # Condition 3: Query term coverage in retrieved documents is too low
    if coverage < coverage_threshold:
        return {
            "should_abstain": True,
            "reason": f"Insufficient query keyword coverage ({coverage:.2f} < threshold {coverage_threshold:.2f})",
            "metrics": metrics
        }

    # Condition 4: LLM explicitly declared lack of evidence
    if "don't know based on the provided pages" in llm_text.lower():
        return {
            "should_abstain": True,
            "reason": "LLM model determined provided context pages were insufficient to answer.",
            "metrics": metrics
        }

    return {
        "should_abstain": False,
        "reason": "Sufficient retrieval score and term coverage.",
        "metrics": metrics
    }
