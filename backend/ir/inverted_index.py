"""
backend/ir/inverted_index.py
Hand-written Inverted Index with Multi-Zone Support for HealthNest.
Implements:
1. Zone-based Inverted Index ("question" zone and "body" zone).
2. Dictionary + Postings List structure: term -> list of (chunk_id, term_frequency).
3. Document Frequency (df), Total Chunks (N), and Vector Euclidean Norms per zone.
4. Pre-computed Champion Lists (top-r chunks per term by TF).
5. Serialization (save/load) using pickle.

Lecture Concepts: Inverted Index, Postings List, Zones & Fields, Champion Lists.
Strict Compliance: No external IR/search libraries used!
"""

import math
import pickle
import logging
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Any
import config
from backend.ir.text import tokenize

logger = logging.getLogger(__name__)


class ZoneIndex:
    """Inverted index for a single document zone (e.g. 'question' or 'body')."""
    def __init__(self, zone_name: str):
        self.zone_name = zone_name
        # term -> list of (chunk_id, tf)
        self.postings: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
        # term -> document frequency (df)
        self.df: Dict[str, int] = defaultdict(int)
        # chunk_id -> lnc document vector norm sqrt(sum((1 + log(tf))^2))
        self.doc_norms: Dict[str, float] = defaultdict(float)
        # term -> top-r (chunk_id, tf) for Champion List optimization
        self.champion_lists: Dict[str, List[Tuple[str, int]]] = {}

    def add_document(self, chunk_id: str, tokens: List[str]) -> None:
        """Indexes a single document chunk's tokens into postings and computes TF."""
        if not tokens:
            return

        # Compute Term Frequencies (tf)
        tf_map = defaultdict(int)
        for token in tokens:
            tf_map[token] += 1

        norm_sq = 0.0
        for term, tf in tf_map.items():
            self.postings[term].append((chunk_id, tf))
            self.df[term] += 1
            # lnc scheme log-tf weight: 1 + log(tf)
            weight = 1.0 + math.log(tf)
            norm_sq += weight * weight

        self.doc_norms[chunk_id] = math.sqrt(norm_sq)

    def build_champion_lists(self, r: int = config.CHAMPION_LIST_R) -> None:
        """Pre-computes top-r highest TF chunks for each term."""
        for term, posting_list in self.postings.items():
            sorted_postings = sorted(posting_list, key=lambda x: x[1], reverse=True)
            self.champion_lists[term] = sorted_postings[:r]


class InvertedIndex:
    """Multi-zone inverted index containing 'question' and 'body' zone indexes."""
    def __init__(self):
        self.zones: Dict[str, ZoneIndex] = {
            "question": ZoneIndex("question"),
            "body": ZoneIndex("body")
        }
        self.N: int = 0  # Total number of chunks indexed
        self.chunks_map: Dict[str, Dict[str, Any]] = {}  # chunk_id -> chunk dict

    def index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        use_stopwords: bool = config.USE_STOP_WORDS,
        use_stemmer: bool = config.USE_PORTER_STEMMER
    ) -> None:
        """Indexes a list of chunk dicts into question and body zones."""
        self.N = len(chunks)
        self.chunks_map = {c["chunk_id"]: c for c in chunks}

        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            q_text = chunk["question"]
            body_text = chunk["text"]

            q_tokens = tokenize(q_text, use_stopwords=use_stopwords, use_stemmer=use_stemmer)
            body_tokens = tokenize(body_text, use_stopwords=use_stopwords, use_stemmer=use_stemmer)

            self.zones["question"].add_document(chunk_id, q_tokens)
            self.zones["body"].add_document(chunk_id, body_tokens)

        for zone in self.zones.values():
            zone.build_champion_lists()

        logger.info(f"Indexed {self.N} document chunks into InvertedIndex across {len(self.zones)} zones.")

    def save(self, filepath: Path = config.INDEX_CACHE_PATH) -> None:
        """Saves inverted index to disk via pickle."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"Saved InvertedIndex to {filepath}")

    @classmethod
    def load(cls, filepath: Path = config.INDEX_CACHE_PATH) -> "InvertedIndex":
        """Loads inverted index from disk via pickle."""
        with open(filepath, "rb") as f:
            idx = pickle.load(f)
        logger.info(f"Loaded InvertedIndex with N={idx.N} chunks from {filepath}")
        return idx
