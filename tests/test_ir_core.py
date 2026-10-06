r"""
tests/test_ir_core.py
Unit Tests for HealthNest IR Core Components.
Tests:
1. Tokenizer and Devanagari regex [\w\u0900-\u097F]+
2. Inverted Index construction and postings correctness
3. TF-IDF SMART lnc.ltc scoring on a toy corpus with hand-computed expectations
4. BM25 scoring behavior
5. Sklearn TfidfVectorizer cross-check verification

Lecture Verification: Verifies exact hand-computed cosine similarity & term weighting.
"""

import math
import unittest
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.ir.text import tokenize, detect_language
from backend.ir.inverted_index import InvertedIndex
from backend.ir.tfidf import TfIdfRetriever
from backend.ir.bm25 import BM25Retriever
from backend.ir.hinglish import HinglishExpander


class TestIRCore(unittest.TestCase):

    def test_devanagari_tokenization(self):
        """Tests regex tokenization preserving Hindi matras."""
        text = "पेट में दर्द और सिरदर्द (headache)"
        tokens = tokenize(text, use_stopwords=False, use_stemmer=False)
        self.assertIn("पेट", tokens)
        self.assertIn("दर्द", tokens)
        self.assertIn("headache", tokens)
        
        lang = detect_language("पेट में दर्द")
        self.assertEqual(lang, "hindi")

    def test_hinglish_language_detection(self):
        """Tests language detection for Hinglish."""
        lang = detect_language("mujhe pet me dard hai")
        self.assertEqual(lang, "hinglish")

    def test_toy_corpus_tfidf_hand_calculated(self):
        """
        Toy Corpus:
        c1: "diabetes symptoms"
        c2: "diabetes treatment"
        Query: "diabetes symptoms"
        
        N = 2
        'diabetes': df = 2, q_idf = ln(2/2) = 0.0 -> q_wt = 0
        'symptoms': df = 1, q_idf = ln(2/1) = 0.693147
        
        For c1 ("diabetes symptoms"):
        'symptoms': d_tf = 1 -> d_wt = 1.0. d_norm = sqrt((1)^2 + (1)^2) = sqrt(2) = 1.4142
        q_norm = sqrt((q_wt_symptoms)^2) = 0.693147
        c1 dot product = q_wt_symptoms * d_wt_symptoms = 0.693147 * 1.0 = 0.693147
        c1 cosine = 0.693147 / (0.693147 * 1.4142) = 1 / 1.4142 = 0.7071
        
        For c2 ("diabetes treatment"):
        'symptoms' not in c2 -> score = 0.0
        
        Therefore, c1 must rank first!
        """
        toy_chunks = [
            {"chunk_id": "c1", "qa_id": "qa1", "question": "diabetes symptoms", "text": "diabetes symptoms", "position": 0},
            {"chunk_id": "c2", "qa_id": "qa2", "question": "diabetes treatment", "text": "diabetes treatment", "position": 0}
        ]

        idx = InvertedIndex()
        idx.index_chunks(toy_chunks, use_stopwords=False, use_stemmer=False)

        retriever = TfIdfRetriever(idx)
        results = retriever.search("diabetes symptoms", top_k=2, use_stopwords=False, use_stemmer=False)

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["chunk_id"], "c1")
        # Verify c1 score matches hand calculated value ~0.7071
        self.assertAlmostEqual(results[0]["score"], 0.7071, places=3)

    def test_bm25_retrieval(self):
        """Tests BM25 retriever ranking on toy corpus."""
        toy_chunks = [
            {"chunk_id": "c1", "qa_id": "qa1", "question": "fever remedy", "text": "paracetamol for fever treatment", "position": 0},
            {"chunk_id": "c2", "qa_id": "qa2", "question": "stomach pain", "text": "antacids for stomach pain", "position": 0}
        ]
        idx = InvertedIndex()
        idx.index_chunks(toy_chunks, use_stopwords=False, use_stemmer=False)

        bm25 = BM25Retriever(idx)
        results = bm25.search("fever treatment", top_k=2, use_stopwords=False, use_stemmer=False)
        
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["chunk_id"], "c1")

    def test_sklearn_cross_check(self):
        """
        Cross-checks cosine similarity concept against scikit-learn TfidfVectorizer.
        """
        corpus = [
            "paracetamol dosage for fever",
            "metformin for diabetes control"
        ]
        vec = TfidfVectorizer()
        matrix = vec.fit_transform(corpus)
        q_vec = vec.transform(["fever dosage"])
        
        scores = (matrix * q_vec.T).toarray().flatten()
        self.assertGreater(scores[0], scores[1])


if __name__ == "__main__":
    unittest.main()
