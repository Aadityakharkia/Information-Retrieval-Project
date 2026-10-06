"""Dataset preprocessing and chunking module for MedQuAD.
Implements:
1. Deduplication and question grouping
2. 200-word sliding window chunking with 30-word overlap (Fedallah & Inik, 2026)
3. Stratified test set partitioning by qtype
4. Ground-truth relevance mapping for evaluation
"""
import json
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from medrag.config import (
    RAW_DATA_PATH,
    PROCESSED_DIR,
    CHUNKS_PATH,
    TEST_QUERIES_PATH,
    CHUNK_WORD_SIZE,
    CHUNK_OVERLAP_WORDS,
    TEST_SET_SIZE,
    RANDOM_SEED
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = CHUNK_WORD_SIZE, overlap: int = CHUNK_OVERLAP_WORDS) -> List[str]:
    """Split text into chunks of `chunk_size` words with `overlap` words sliding window.
    
    Matches Fedallah & Inik (2026) ISADES configuration.
    """
    words = text.strip().split()
    if not words:
        return []
    
    if len(words) <= chunk_size:
        return [" ".join(words)]
    
    chunks = []
    step = chunk_size - overlap
    if step <= 0:
        step = chunk_size // 2
        
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start += step
        
    return chunks


class MedQuADProcessor:
    """Processes MedQuAD dataset into clean chunks and stratified evaluation splits."""
    
    def __init__(self, raw_path: str = str(RAW_DATA_PATH)):
        self.raw_path = raw_path
        
    def process_and_save(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Execute full preprocessing pipeline and save to disk.
        
        Returns:
            Tuple of (chunks_list, test_queries_list)
        """
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading raw dataset from {self.raw_path}...")
        df = pd.read_csv(self.raw_path)
        logger.info(f"Loaded {len(df)} initial rows.")
        
        # 1. Drop exact duplicates
        exact_dups = df.duplicated().sum()
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info(f"Dropped {exact_dups} exact duplicate rows. Remaining: {len(df)}")
        
        # Strip whitespace
        df["Question"] = df["Question"].astype(str).str.strip()
        df["Answer"] = df["Answer"].astype(str).str.strip()
        df["qtype"] = df["qtype"].astype(str).str.strip()
        
        # 2. Assign unique doc_id to each unique answer
        # If multiple rows share the same answer, they share doc_id
        unique_answers = df["Answer"].unique()
        answer_to_doc_id = {ans: f"doc_{i:05d}" for i, ans in enumerate(unique_answers)}
        df["doc_id"] = df["Answer"].map(answer_to_doc_id)
        logger.info(f"Identified {len(unique_answers)} unique answers/documents.")
        
        # 3. Stratified test query selection
        # Group by Question to collect all relevant doc_ids per question
        q_groups = df.groupby("Question").agg({
            "qtype": "first",
            "doc_id": lambda s: sorted(list(set(s))),
            "Answer": "first"
        }).reset_index()
        
        logger.info(f"Found {len(q_groups)} unique questions.")
        
        # Sample TEST_SET_SIZE questions stratified by qtype
        np.random.seed(RANDOM_SEED)
        
        # Filter qtypes that have at least 2 questions
        qtype_counts = q_groups["qtype"].value_counts()
        eligible_q = q_groups[q_groups["qtype"].isin(qtype_counts[qtype_counts >= 5].index)]
        
        # Perform stratified sampling
        test_queries_list = []
        train_q_set = set()
        
        # Compute sample proportions
        total_eligible = len(eligible_q)
        test_df_parts = []
        for qtype, group in eligible_q.groupby("qtype"):
            n_sample = max(1, int(round(len(group) / total_eligible * TEST_SET_SIZE)))
            sample_part = group.sample(n=min(n_sample, len(group)), random_state=RANDOM_SEED)
            test_df_parts.append(sample_part)
            
        test_df = pd.concat(test_df_parts).sample(frac=1.0, random_state=RANDOM_SEED)
        if len(test_df) > TEST_SET_SIZE:
            test_df = test_df.iloc[:TEST_SET_SIZE]
            
        test_questions_set = set(test_df["Question"])
        logger.info(f"Sampled {len(test_df)} stratified test questions across {test_df['qtype'].nunique()} qtypes.")
        
        # 4. Generate Chunks for all unique documents in the corpus
        logger.info("Chunking answers using 200-word window and 30-word overlap...")
        chunks: List[Dict[str, Any]] = []
        doc_to_chunk_ids: Dict[str, List[str]] = {}
        
        # Map doc_id to its qtype and representative question
        doc_metadata = df.groupby("doc_id").agg({
            "qtype": "first",
            "Question": "first",
            "Answer": "first"
        }).reset_index()
        
        for _, row in doc_metadata.iterrows():
            d_id = row["doc_id"]
            d_qtype = row["qtype"]
            d_question = row["Question"]
            d_text = row["Answer"]
            
            raw_chunks = chunk_text(d_text, CHUNK_WORD_SIZE, CHUNK_OVERLAP_WORDS)
            doc_to_chunk_ids[d_id] = []
            
            for c_idx, c_text in enumerate(raw_chunks):
                chunk_id = f"{d_id}_c{c_idx}"
                doc_to_chunk_ids[d_id].append(chunk_id)
                chunks.append({
                    "chunk_id": chunk_id,
                    "doc_id": d_id,
                    "qtype": d_qtype,
                    "chunk_index": c_idx,
                    "total_chunks": len(raw_chunks),
                    "text": c_text,
                    "word_count": len(c_text.split()),
                    "source_question": d_question
                })
                
        logger.info(f"Generated {len(chunks)} total chunks from {len(doc_metadata)} documents.")
        
        # 5. Format test queries with ground-truth relevant doc_ids and chunk_ids
        for _, row in test_df.iterrows():
            q_text = row["Question"]
            rel_doc_ids = row["doc_id"]
            rel_chunk_ids = []
            for did in rel_doc_ids:
                rel_chunk_ids.extend(doc_to_chunk_ids.get(did, []))
                
            test_queries_list.append({
                "query_id": f"q_{len(test_queries_list):04d}",
                "question": q_text,
                "qtype": row["qtype"],
                "relevant_doc_ids": rel_doc_ids,
                "relevant_chunk_ids": rel_chunk_ids,
                "gold_answer": row["Answer"]
            })
            
        # 6. Save chunks and test queries to disk
        logger.info(f"Saving {len(chunks)} chunks to {CHUNKS_PATH}...")
        with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
            
        logger.info(f"Saving {len(test_queries_list)} test queries to {TEST_QUERIES_PATH}...")
        with open(TEST_QUERIES_PATH, "w", encoding="utf-8") as f:
            json.dump(test_queries_list, f, indent=2)
            
        logger.info("Dataset preprocessing complete!")
        return chunks, test_queries_list


def load_chunks() -> List[Dict[str, Any]]:
    """Load preprocessed chunks from disk, creating them if not found."""
    if not CHUNKS_PATH.exists():
        processor = MedQuADProcessor()
        chunks, _ = processor.process_and_save()
        return chunks
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_test_queries() -> List[Dict[str, Any]]:
    """Load test queries from disk, creating them if not found."""
    if not TEST_QUERIES_PATH.exists():
        processor = MedQuADProcessor()
        _, queries = processor.process_and_save()
        return queries
    with open(TEST_QUERIES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    processor = MedQuADProcessor()
    chunks, test_queries = processor.process_and_save()
    print(f"Summary: {len(chunks)} chunks, {len(test_queries)} test queries.")
