"""
backend/ir/hinglish.py
Lexicon-based Query Expansion & Vocabulary Normalization Module.
Implements synonym expansion for Hinglish/Hindi medical queries by looking up
terms in 'lexicon_hinglish_health.tsv' and appending equivalent English terms.

Lecture Concept: Synonymy & Vocabulary Mismatch Mitigation in Information Retrieval.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, List, Set, Tuple
import config
from backend.ir.text import normalize_text, tokenize, HINGLISH_STOP_WORDS

logger = logging.getLogger(__name__)


class HinglishExpander:
    def __init__(self, lexicon_path: Path = config.LEXICON_PATH):
        self.lexicon_path = lexicon_path
        self.term_map: Dict[str, List[str]] = {}
        self.load_lexicon()

    def load_lexicon(self) -> None:
        """Loads TSV lexicon file into a dictionary."""
        if not self.lexicon_path.exists():
            logger.warning(f"Lexicon file missing at {self.lexicon_path}. Hinglish expansion will be disabled.")
            return

        with open(self.lexicon_path, mode="r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    src_term = normalize_text(parts[0].strip())
                    eng_terms = [normalize_text(t.strip()) for t in parts[1].split(",") if t.strip()]
                    if src_term and eng_terms:
                        self.term_map[src_term] = eng_terms

        logger.info(f"Loaded Hinglish lexicon with {len(self.term_map)} phrase mappings.")

    def expand_query(self, query: str) -> Tuple[str, List[str]]:
        """
        Expands query string with English synonym equivalents.
        
        Returns:
            (expanded_query_string, expanded_terms_added)
        """
        if not self.term_map or not query:
            return query, []

        norm_query = normalize_text(query)
        added_terms: Set[str] = set()

        # Check multi-word phrase matches first, then single words
        # Sort terms by length descending so longer phrases match first
        sorted_keys = sorted(self.term_map.keys(), key=lambda k: len(k.split()), reverse=True)

        query_words = set(norm_query.split())
        for src_phrase in sorted_keys:
            phrase_matched = False
            if src_phrase in norm_query:
                phrase_matched = True
            else:
                src_words = src_phrase.split()
                if len(src_words) > 1 and all(w in query_words for w in src_words):
                    phrase_matched = True

            if phrase_matched:
                for eng_term in self.term_map[src_phrase]:
                    if eng_term not in norm_query:
                        added_terms.add(eng_term)

        if added_terms:
            expanded_suffix = " " + " ".join(sorted(added_terms))
            expanded_query = query + expanded_suffix
            return expanded_query, sorted(list(added_terms))

        return query, []

    def get_concept_groups(self, query: str) -> List[Set[str]]:
        """
        Extracts semantic concept groups for the query.
        For Hinglish/medical queries, each matched term produces one concept group
        containing all of its English synonym tokens (any of which satisfies the concept).
        Non-stopword English tokens from the original query that were not part of
        a matched Hinglish term form additional single-token concept groups.
        """
        if not query:
            return []

        norm_query = normalize_text(query)
        concept_groups: List[Set[str]] = []
        matched_hindi_words: Set[str] = set()

        sorted_keys = sorted(self.term_map.keys(), key=lambda k: len(k.split()), reverse=True)
        query_words = set(norm_query.split())

        for src_phrase in sorted_keys:
            phrase_matched = False
            if src_phrase in norm_query:
                phrase_matched = True
            else:
                src_words = src_phrase.split()
                if len(src_words) > 1 and all(w in query_words for w in src_words):
                    phrase_matched = True

            if phrase_matched:
                src_w_set = set(src_phrase.split())
                if src_w_set.issubset(matched_hindi_words):
                    continue

                group_tokens = set()
                for eng_term in self.term_map[src_phrase]:
                    for tok in tokenize(eng_term, use_stopwords=True, use_stemmer=True):
                        group_tokens.add(tok)

                if group_tokens:
                    concept_groups.append(group_tokens)
                    matched_hindi_words.update(src_w_set)

        # Include remaining English words from original query
        orig_tokens = tokenize(query, use_stopwords=True, use_stemmer=True)
        for t in orig_tokens:
            if t not in matched_hindi_words and t not in HINGLISH_STOP_WORDS and len(t) > 2:
                concept_groups.append({t})

        return concept_groups

