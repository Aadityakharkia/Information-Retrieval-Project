"""
backend/rag/generator.py
Answer Generation & Citation Parsing Module for HealthNest.
Calls LLM client, parses answer into discrete sentences, extracts citation tags [n],
and flags invalid/missing citation IDs.

Inputs: Prompt messages payload, LLM model selection, max context page count K.
Outputs: List of sentence dicts: {text, citations, is_valid_citation, raw_sentence}.
"""

import re
from typing import List, Dict, Any, Tuple
import config
from backend.rag.llm_client import generate_llm_response
from backend.rag.prompt import format_rag_prompt
from backend.data.chunker import split_into_sentences


def parse_citations(sentence_text: str, max_k: int = 5) -> Tuple[str, List[int], bool]:
    """
    Extracts citation numbers like [1], [2] from a sentence string.
    Returns:
        (clean_sentence_text, citation_ids_list, is_valid_citation)
    """
    # Find all bracketed digits like [1], [2], [1, 2]
    matches = re.findall(r'\[(\d+(?:\s*,\s*\d+)*)\]', sentence_text)
    citation_ids = []
    
    for match in matches:
        parts = match.split(",")
        for p in parts:
            try:
                cid = int(p.strip())
                citation_ids.append(cid)
            except ValueError:
                pass

    # Clean citation markers from text for clean display
    clean_text = re.sub(r'\s*\[\d+(?:\s*,\s*\d+)*\]', '', sentence_text).strip()

    # Check if citations are valid (within 1..max_k)
    is_valid = True
    if not citation_ids:
        is_valid = False
    else:
        for cid in citation_ids:
            if cid < 1 or cid > max_k:
                is_valid = False
                break

    return clean_text, citation_ids, is_valid


def generate_answer(
    question: str,
    top_chunks: List[Dict[str, Any]],
    model: str = config.DEFAULT_MODEL,
    added_terms: List[str] = None
) -> Dict[str, Any]:
    """
    Executes grounded RAG generation using LLM.
    
    Returns:
        {
            "raw_answer": str,
            "sentences": List[Dict[str, Any]],
            "has_uncited": bool,
            "has_invalid_citations": bool
        }
    """
    messages = format_rag_prompt(question, top_chunks, added_terms=added_terms)
    raw_response, provider = generate_llm_response(messages, model=model)
    # Some models emit full-width brackets (e.g. 【1】); normalise to [1].
    raw_response = re.sub(r"[【\[]\s*(\d+(?:\s*,\s*\d+)*)\s*[】\]]", r"[\1]", raw_response)

    if "don't know based on the provided pages" in raw_response.lower():
        return {
            "raw_answer": "I don't know based on the provided pages.",
            "sentences": [],
            "has_uncited": False,
            "has_invalid_citations": False,
            "provider": provider
        }

    raw_sentences = split_into_sentences(raw_response)
    if not raw_sentences:
        raw_sentences = [raw_response]

    parsed_sentences = []
    has_uncited = False
    has_invalid_citations = False
    max_k = len(top_chunks)

    for s_idx, raw_s in enumerate(raw_sentences):
        clean_text, citations, is_valid = parse_citations(raw_s, max_k=max_k)
        if not citations:
            has_uncited = True
        if not is_valid:
            has_invalid_citations = True

        parsed_sentences.append({
            "sentence_id": s_idx + 1,
            "text": clean_text if clean_text else raw_s,
            "raw_text": raw_s,
            "citations": citations,
            "is_valid_citation": is_valid
        })

    return {
        "raw_answer": raw_response,
        "sentences": parsed_sentences,
        "has_uncited": has_uncited,
        "has_invalid_citations": has_invalid_citations,
        "provider": provider
    }
