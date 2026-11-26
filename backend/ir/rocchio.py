"""
backend/ir/rocchio.py
Rocchio relevance feedback for HealthNest, in both vector spaces the system uses.

    q_new = alpha * q  +  beta * mean(relevant)  -  gamma * mean(non_relevant)

  Sparse: vectors are ltc-weighted terms ((1 + log tf) * idf, cosine-normalised, body-zone idf).
          Negative weights are dropped, and only the original terms plus the strongest
          ROCCHIO_MAX_NEW_TERMS new terms are kept (a short query stays fast and readable).
          New terms must be ordinary vocabulary (3+ letters, in >= ROCCHIO_MIN_NEW_TERM_DF passages).
  Dense:  the same formula on the normalised MiniLM embeddings, then re-normalised.

Two ways feedback reaches here (see HealthNestPipeline._retrieve):
  * explicit - the patient marks sources "More like this" / "Not relevant";
  * pseudo   - the top ROCCHIO_PRF_DOCS results of the first search are assumed relevant
               (only when that search was confident, to avoid query drift).

Lecture Concepts: Relevance Feedback, Rocchio Algorithm, Pseudo-Relevance Feedback, Query Expansion
(Manning, Raghavan & Schuetze, Introduction to Information Retrieval, ch. 9).
Strict Compliance: hand-written; numpy only for the dense vector arithmetic.
"""

import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

import config
from backend.ir.inverted_index import InvertedIndex
from backend.ir.text import tokenize


def ltc_vector(tokens: Iterable[str], index: InvertedIndex) -> Dict[str, float]:
    """(1 + log tf) * idf, cosine-normalised. Terms unknown to the index get no weight."""
    df = index.zones["body"].df
    vec = {}
    for term, tf in Counter(tokens).items():
        d = df.get(term, 0)
        if d:
            w = (1.0 + math.log(tf)) * math.log(index.N / d)
            if w > 0:
                vec[term] = w
    norm = math.sqrt(sum(w * w for w in vec.values()))
    return {t: w / norm for t, w in vec.items()} if norm else {}


def _params(alpha, beta, gamma):
    return (config.ROCCHIO_ALPHA if alpha is None else alpha, config.ROCCHIO_BETA if beta is None else beta,
            config.ROCCHIO_GAMMA if gamma is None else gamma)


def _mean(vectors: List[Dict[str, float]]) -> Dict[str, float]:
    total: Dict[str, float] = {}
    for v in vectors:
        for t, w in v.items():
            total[t] = total.get(t, 0.0) + w
    return {t: w / len(vectors) for t, w in total.items()} if vectors else {}


def sparse_query(query: str, relevant: List[str], non_relevant: List[str], index: InvertedIndex,
                 alpha: Optional[float] = None, beta: Optional[float] = None, gamma: Optional[float] = None,
                 max_new_terms: Optional[int] = None) -> Tuple[Dict[str, float], List[str]]:
    """Returns (new term weights, the new terms that were added, strongest first).
    `relevant` / `non_relevant` are passage texts. Unset parameters are read from config at call time."""
    alpha, beta, gamma = _params(alpha, beta, gamma)
    max_new_terms = config.ROCCHIO_MAX_NEW_TERMS if max_new_terms is None else max_new_terms
    q = ltc_vector(tokenize(query), index)
    rel = _mean([ltc_vector(tokenize(t), index) for t in relevant])
    non = _mean([ltc_vector(tokenize(t), index) for t in non_relevant])
    new = {t: alpha * q.get(t, 0.0) + beta * rel.get(t, 0.0) - gamma * non.get(t, 0.0) for t in set(q) | set(rel) | set(non)}
    kept = {t: w for t, w in new.items() if w > 0 and t in q}
    df = index.zones["body"].df
    # New words must be real vocabulary: 3+ letters and seen in several passages. Rare fragments such as
    # pronunciation guides ("ah-RITH-me-ahs" -> "rith") have a high idf and would otherwise dominate.
    usable = lambda t: len(t) >= 3 and t.isalpha() and df.get(t, 0) >= config.ROCCHIO_MIN_NEW_TERM_DF
    added = sorted((t for t, w in new.items() if w > 0 and t not in q and usable(t)), key=lambda t: (-new[t], t))[:max_new_terms]
    kept.update({t: new[t] for t in added})
    return kept, added


def dense_query(q: np.ndarray, relevant: List[np.ndarray], non_relevant: List[np.ndarray],
                alpha: Optional[float] = None, beta: Optional[float] = None, gamma: Optional[float] = None) -> np.ndarray:
    alpha, beta, gamma = _params(alpha, beta, gamma)
    v = alpha * q
    if relevant:
        v = v + beta * np.mean(relevant, axis=0)
    if non_relevant:
        v = v - gamma * np.mean(non_relevant, axis=0)
    n = float(np.linalg.norm(v))
    return (v / n).astype(np.float32) if n else q
