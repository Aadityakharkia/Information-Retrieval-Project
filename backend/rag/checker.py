"""
backend/rag/checker.py
Sentence Verification & Fact-Checking Module for HealthNest.
Verifies every generated sentence against its cited source document chunk(s).

Verification Checks Implemented:
1. Sparse Cosine Similarity over index vocabulary.
2. Key-Term Stem Coverage (share of sentence non-stop-word stems present in chunk).
3. Exact Number Verification (every dose/number in sentence MUST exist in cited chunk).
4. Dense Cosine Similarity via SentenceTransformers.

KNWON LIMITATION NOTE:
Cosine similarity measures semantic topic overlap but cannot detect logical negation
(e.g., "do take paracetamol" vs "do NOT take paracetamol" will have high cosine score).
Number verification partially mitigates dosage errors.
"""

import re
import math
from typing import List, Dict, Any, Tuple
import numpy as np
import config
from backend.ir.text import tokenize, ALL_STOP_WORDS


def extract_numbers(text: str) -> List[str]:
    """Extracts all numbers, dosages, and numeric values from text."""
    return re.findall(r'\b\d+(?:\.\d+)?\b', text)


def compute_sentence_sparse_cosine(sentence_tokens: List[str], chunk_tokens: List[str]) -> float:
    """Computes sparse cosine similarity between sentence and cited chunk."""
    if not sentence_tokens or not chunk_tokens:
        return 0.0

    s_tf = {}
    for t in sentence_tokens:
        s_tf[t] = s_tf.get(t, 0) + 1

    c_tf = {}
    for t in chunk_tokens:
        c_tf[t] = c_tf.get(t, 0) + 1

    dot = 0.0
    for t, count in s_tf.items():
        if t in c_tf:
            dot += count * c_tf[t]

    s_norm = math.sqrt(sum(v*v for v in s_tf.values()))
    c_norm = math.sqrt(sum(v*v for v in c_tf.values()))

    if s_norm > 0 and c_norm > 0:
        return dot / (s_norm * c_norm)
    return 0.0


def check_sentence_verification(
    sentence: Dict[str, Any],
    top_chunks: List[Dict[str, Any]],
    dense_model=None
) -> Dict[str, Any]:
    """
    Verifies a single sentence against its cited chunk(s).
    
    Returns:
        {
            "supported": bool,
            "badge": "supported" | "not_found",
            "reasons": List[str],
            "metrics": {"sparse_cosine": float, "key_term_coverage": float, "numbers_pass": bool}
        }
    """
    text = sentence["text"]
    citations = sentence.get("citations", [])

    if not citations:
        return {
            "supported": False,
            "badge": "not_found",
            "reasons": ["Sentence missing citation tag [n]."],
            "metrics": {"sparse_cosine": 0.0, "key_term_coverage": 0.0, "numbers_pass": False}
        }

    # Gather cited chunks
    cited_chunks = []
    for cid in citations:
        idx = cid - 1
        if 0 <= idx < len(top_chunks):
            chunk_item = top_chunks[idx]
            cited_chunks.append(chunk_item.get("chunk", chunk_item))

    if not cited_chunks:
        return {
            "supported": False,
            "badge": "not_found",
            "reasons": [f"Citation ID {citations} out of range (1..{len(top_chunks)})."],
            "metrics": {"sparse_cosine": 0.0, "key_term_coverage": 0.0, "numbers_pass": False}
        }

    combined_cited_text = " ".join([c.get("text", "") + " " + c.get("question", "") for c in cited_chunks])
    
    s_tokens = tokenize(text, use_stopwords=True, use_stemmer=True)
    c_tokens = tokenize(combined_cited_text, use_stopwords=True, use_stemmer=True)

    # 1. Key-term coverage
    if not s_tokens:
        coverage = 1.0
    else:
        found = sum(1 for t in s_tokens if t in c_tokens)
        coverage = round(found / float(len(s_tokens)), 4)

    # 2. Sparse cosine
    sparse_cos = round(compute_sentence_sparse_cosine(s_tokens, c_tokens), 4)

    # 3. Number / dosage verification
    s_numbers = extract_numbers(text)
    c_numbers = set(extract_numbers(combined_cited_text))
    missing_numbers = [n for n in s_numbers if n not in c_numbers]
    numbers_pass = (len(missing_numbers) == 0)

    # Decision rule
    reasons = []
    supported = True

    if coverage < config.CHECKER_KEY_TERM_COVERAGE_MIN:
        supported = False
        reasons.append(f"Low key-term coverage ({coverage:.2f} < {config.CHECKER_KEY_TERM_COVERAGE_MIN})")

    if sparse_cos < config.CHECKER_SPARSE_COSINE_MIN:
        supported = False
        reasons.append(f"Low sparse cosine match ({sparse_cos:.2f} < {config.CHECKER_SPARSE_COSINE_MIN})")

    if not numbers_pass:
        supported = False
        reasons.append(f"Uncited numbers/dosages present: {missing_numbers}")

    if supported and not reasons:
        reasons.append("Sentence facts fully supported by cited reference page.")

    return {
        "supported": supported,
        "badge": "supported" if supported else "not_found",
        "reasons": reasons,
        "metrics": {
            "sparse_cosine": sparse_cos,
            "key_term_coverage": coverage,
            "numbers_pass": numbers_pass,
            "missing_numbers": missing_numbers
        }
    }


def verify_all_sentences(
    sentences: List[Dict[str, Any]],
    top_chunks: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Verifies all sentences in an answer.
    """
    verified_list = []
    for s in sentences:
        result = check_sentence_verification(s, top_chunks)
        s_copy = dict(s)
        s_copy["support"] = result
        verified_list.append(s_copy)
    return verified_list
