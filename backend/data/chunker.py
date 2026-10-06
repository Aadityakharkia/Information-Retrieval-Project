"""
backend/data/chunker.py
Document Chunking Module for HealthNest.
Implements 3 chunking strategies ('what is a document' in IR terms):
1. whole_qa: entire Q&A pair as a single document chunk.
2. fixed_words: fixed word window with sliding overlap.
3. sentence_window: sentence-based grouping up to max words with sentence overlap.

Input: Raw Q&A record dict.
Output: List of chunk dicts {chunk_id, qa_id, question, text, position}.
"""

import re
from typing import List, Dict, Any
import config


def split_into_sentences(text: str) -> List[str]:
    """
    Splits text into sentences using sentence-ending punctuation,
    preserving trailing citation markers like [1].
    """
    if not text:
        return []
    # Split on period, exclamation, or question mark (and optional bracketed citation) followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s*(?=\[?\d+\]?\s*|$|\b[A-Z])', text.strip())
    # Clean up empty splits
    result = []
    for s in sentences:
        s_clean = s.strip()
        if s_clean:
            # If a split is just a citation tag like "[1]", append to previous sentence
            if re.match(r'^\[\d+\]$', s_clean) and result:
                result[-1] = f"{result[-1]} {s_clean}"
            else:
                result.append(s_clean)
    return result


def create_chunks_for_record(
    record: Dict[str, Any],
    strategy: str = config.DEFAULT_CHUNKER_STRATEGY,
    fixed_words: int = config.FIXED_WORDS_SIZE,
    fixed_overlap: int = config.FIXED_WORDS_OVERLAP,
    window_max_words: int = config.SENTENCE_WINDOW_MAX_WORDS
) -> List[Dict[str, Any]]:
    """
    Chunks a single Q&A record based on chosen strategy.
    """
    qa_id = record["qa_id"]
    question = record["question"]
    answer = record["answer"]

    chunks = []

    if strategy == "whole_qa":
        full_text = f"{question} {answer}"
        chunks.append({
            "chunk_id": f"{qa_id}_c0",
            "qa_id": qa_id,
            "question": question,
            "text": full_text.strip(),
            "position": 0
        })

    elif strategy == "fixed_words":
        words = answer.split()
        if not words:
            words = [answer]
        
        pos = 0
        idx = 0
        while pos < len(words):
            chunk_words = words[pos : pos + fixed_words]
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "chunk_id": f"{qa_id}_c{idx}",
                "qa_id": qa_id,
                "question": question,
                "text": chunk_text,
                "position": idx
            })
            idx += 1
            pos += (fixed_words - fixed_overlap)
            if pos <= 0:  # Prevent infinite loop if overlap >= size
                break

    elif strategy == "sentence_window":
        sentences = split_into_sentences(answer)
        if not sentences:
            sentences = [answer]

        pos = 0
        idx = 0
        while pos < len(sentences):
            current_group = []
            word_count = 0
            curr_pos = pos
            while curr_pos < len(sentences):
                s = sentences[curr_pos]
                s_words = len(s.split())
                if current_group and (word_count + s_words > window_max_words):
                    break
                current_group.append(s)
                word_count += s_words
                curr_pos += 1

            chunk_text = " ".join(current_group)
            chunks.append({
                "chunk_id": f"{qa_id}_c{idx}",
                "qa_id": qa_id,
                "question": question,
                "text": chunk_text,
                "position": idx
            })
            idx += 1

            # Advance pos with 1-sentence overlap
            advance = max(1, len(current_group) - config.SENTENCE_WINDOW_OVERLAP_SENTENCES)
            pos += advance

    else:
        raise ValueError(f"Unknown chunker strategy: {strategy}")

    return chunks


def chunk_dataset(
    records: List[Dict[str, Any]],
    strategy: str = config.DEFAULT_CHUNKER_STRATEGY
) -> List[Dict[str, Any]]:
    """
    Chunks an entire list of Q&A records into document chunks.
    """
    all_chunks = []
    for rec in records:
        all_chunks.extend(create_chunks_for_record(rec, strategy=strategy))
    return all_chunks
