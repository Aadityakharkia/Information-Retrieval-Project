"""Unit tests for Information Retrieval and Citation Verification components.
"""
import pytest
from src.text_processing import tokenize, tokenize_and_stem, compute_tf, split_into_sentences
from src.data_processor import chunk_text
from src.retrievers.inverted_index import InvertedIndexRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.citation_checker import SentenceCitationChecker
from src.generator import MedicalAnswerGenerator


def test_chunking_sliding_window():
    text = " ".join([f"word{i}" for i in range(250)])
    chunks = chunk_text(text, chunk_size=200, overlap=30)
    assert len(chunks) == 2
    words_c1 = chunks[0].split()
    words_c2 = chunks[1].split()
    assert len(words_c1) == 200
    # Overlap check: last 30 words of c1 should match first 30 of c2
    assert words_c1[-30:] == words_c2[:30]


def test_text_processing_stemmer():
    text = "Doctors are treating infections with medications"
    stems = tokenize_and_stem(text)
    assert "doctor" in stems
    assert "treat" in stems
    assert "infect" in stems
    assert "medic" in stems


def test_sentence_splitter():
    text = "Patient was diagnosed with LCM. Dr. Smith prescribed fluids! Is there fever?"
    sents = split_into_sentences(text)
    assert len(sents) == 3
    assert sents[0] == "Patient was diagnosed with LCM."
    assert "Dr. Smith prescribed fluids!" in sents[1]
    assert sents[2] == "Is there fever?"


def test_inverted_index_lnc_ltc():
    toy_chunks = [
        {"chunk_id": "c1", "doc_id": "d1", "qtype": "symptoms", "chunk_index": 0, "text": "Fever, headache and nausea are common symptoms."},
        {"chunk_id": "c2", "doc_id": "d2", "qtype": "treatment", "chunk_index": 0, "text": "Treatment consists of oral rehydration and rest."},
        {"chunk_id": "c3", "doc_id": "d3", "qtype": "causes", "chunk_index": 0, "text": "The disease is caused by rodent exposure and bites."}
    ]
    retriever = InvertedIndexRetriever()
    retriever.build_index(toy_chunks, force_rebuild=True)
    
    # Check postings
    assert "fever" in retriever.df
    assert len(retriever.postings["fever"]) == 1
    
    # Query matching c1
    results = retriever.retrieve("What causes fever and headache?", top_k=2)
    assert len(results) > 0
    assert results[0]["chunk_id"] == "c1"
    
    # Diagnostics check
    exp = retriever.explain("fever", top_k=1)
    assert "term_diagnostics" in exp
    assert "fever" in exp["term_diagnostics"]
    assert exp["term_diagnostics"]["fever"]["in_vocabulary"] is True


def test_bm25_retrieval():
    toy_chunks = [
        {"chunk_id": "c1", "doc_id": "d1", "qtype": "symptoms", "chunk_index": 0, "text": "Fever and severe muscle pain."},
        {"chunk_id": "c2", "doc_id": "d2", "qtype": "treatment", "chunk_index": 0, "text": "Antibiotics are used for bacterial infections."},
    ]
    retriever = BM25Retriever()
    retriever.build_index(toy_chunks)
    results = retriever.retrieve("muscle pain", top_k=1)
    assert len(results) == 1
    assert results[0]["chunk_id"] == "c1"


def test_citation_checker_supported_vs_unsupported():
    checker = SentenceCitationChecker()
    citation_map = {
        "[C1]": {
            "chunk_id": "c1",
            "text": "LCMV infection symptoms include high fever, severe headache, and persistent vomiting in the first stage."
        }
    }
    
    # Case 1: Supported statement
    answer_supported = "The initial symptoms of LCMV include high fever and severe headache. [C1]"
    audit_sup = checker.audit_response(answer_supported, citation_map)
    assert audit_sup["sentence_audits"][0]["verdict"] in ["SUPPORTED", "PARTIALLY_SUPPORTED"]
    assert audit_sup["faithfulness_score"] >= 0.5
    
    # Case 2: Hallucinated / Unsupported statement
    answer_unsupported = "Patients usually experience purple discoloration of their toenails. [C1]"
    audit_unsup = checker.audit_response(answer_unsupported, citation_map)
    assert audit_unsup["sentence_audits"][0]["verdict"] == "UNSUPPORTED"
    
    # Case 3: Uncited statement
    answer_uncited = "Patients usually develop cough."
    audit_uncited = checker.audit_response(answer_uncited, citation_map)
    assert audit_uncited["sentence_audits"][0]["verdict"] == "UNCITED"


def test_refusal_threshold():
    generator = MedicalAnswerGenerator(model_name="mock-offline")
    
    # Fake chunks with very low score
    low_chunks = [{"chunk_id": "c1", "text": "Some text", "score": 0.05}]
    res = generator.generate("Quantum physics question?", low_chunks, threshold=0.30)
    assert res["refused"] is True
    assert "refusal_reason" in res
