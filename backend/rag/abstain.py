"""
backend/rag/abstain.py
Abstention & Confidence Decision Module for HealthNest.
Implements evidence strength thresholding to abstain ("I don't know") when:
1. Top retrieval score is below calibrated threshold.
2. Query term IDF coverage in top-K retrieved chunks is too low.
3. LLM output indicates insufficient evidence.

Lecture Concept: Selective Answering / Abstention in QA Systems.
"""

from typing import List, Dict, Any, Tuple
import config
from backend.ir.text import tokenize


def calculate_query_term_coverage(query: str, top_chunks: List[Dict[str, Any]]) -> float:
    """
    Calculates proportion of query terms present across top retrieved chunks.
    """
    q_tokens = set(tokenize(query, use_stopwords=True, use_stemmer=True))
    if not q_tokens:
        return 1.0

    found_tokens = set()
    for item in top_chunks:
        chunk_text = item.get("chunk", {}).get("text", "") + " " + item.get("chunk", {}).get("question", "")
        c_tokens = set(tokenize(chunk_text, use_stopwords=True, use_stemmer=True))
        found_tokens.update(q_tokens.intersection(c_tokens))

    coverage = len(found_tokens) / float(len(q_tokens))
    return round(coverage, 4)


def evaluate_abstention(
    query: str,
    retrieval_results: List[Dict[str, Any]],
    llm_text: str = "",
    score_threshold: float = config.ABSTAIN_SCORE_THRESHOLD,
    coverage_threshold: float = config.ABSTAIN_QUERY_COVERAGE_THRESHOLD
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

    coverage = calculate_query_term_coverage(query, retrieval_results)

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
    q_tokens = set(tokenize(query, use_stopwords=True, use_stemmer=True))
    if q_tokens and retrieval_results:
        top_data = retrieval_results[0].get("chunk", retrieval_results[0])
        top_text = f"{top_data.get('text', '')} {top_data.get('question', '')}"
        top_tokens = set(tokenize(top_text, use_stopwords=True, use_stemmer=True))
        if len(q_tokens.intersection(top_tokens)) == 0:
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

    # Condition 3: LLM explicitly declared lack of evidence
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
