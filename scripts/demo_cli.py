"""
scripts/demo_cli.py
Phase 1 CLI Demo for HealthNest.
Loads Q&A dataset, builds chunks & inverted index, runs TF-IDF and BM25 retrievers,
and prints top-5 ranked chunks with detailed score breakdown.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import config
from backend.data.loader import load_dataset
from backend.data.chunker import chunk_dataset
from backend.ir.inverted_index import InvertedIndex
from backend.ir.tfidf import TfIdfRetriever
from backend.ir.bm25 import BM25Retriever
from backend.ir.hinglish import HinglishExpander


def run_demo(query: str = "pet me dard and fever remedy"):
    print("=" * 70)
    print("HEALTHNEST - PHASE 1 IR CORE DEMO")
    print("=" * 70)

    # 1. Load dataset & chunk
    print("\n[1] Loading dataset...")
    records, stats = load_dataset()
    print(f"Loaded {stats['valid_records']} Q&A records.")

    print("\n[2] Chunking documents (strategy: sentence_window)...")
    chunks = chunk_dataset(records, strategy=config.DEFAULT_CHUNKER_STRATEGY)
    print(f"Generated {len(chunks)} document chunks.")

    # 2. Build Inverted Index
    print("\n[3] Building Multi-Zone Inverted Index...")
    index = InvertedIndex()
    index.index_chunks(chunks)

    # 3. Query expansion
    expander = HinglishExpander()
    expanded_q, added_terms = expander.expand_query(query)
    print(f"\n[4] Query: '{query}'")
    if added_terms:
        print(f"    Hinglish Lexicon Expanded Query: '{expanded_q}' (Added: {added_terms})")

    # 4. TF-IDF Search
    print("\n" + "-" * 50)
    print("TOP-5 TF-IDF (lnc.ltc) RESULTS:")
    print("-" * 50)
    tfidf_retriever = TfIdfRetriever(index)
    results = tfidf_retriever.search(expanded_q, top_k=5)

    for res in results:
        print(f"Rank {res['rank']} | Score: {res['score']} | Chunk ID: {res['chunk_id']}")
        print(f"  Zone Scores: {res['zone_scores']}")
        print(f"  Matched Terms: {res['matched_terms']}")
        print(f"  Question: {res['chunk']['question']}")
        print(f"  Snippet: {res['chunk']['text'][:120]}...\n")

    # 5. BM25 Search
    print("-" * 50)
    print("TOP-5 BM25 RESULTS:")
    print("-" * 50)
    bm25_retriever = BM25Retriever(index)
    bm25_results = bm25_retriever.search(expanded_q, top_k=5)

    for res in bm25_results:
        print(f"Rank {res['rank']} | Score: {res['score']} | Chunk ID: {res['chunk_id']}")
        print(f"  Matched Terms: {res['matched_terms']}")
        print(f"  Question: {res['chunk']['question']}")
        print(f"  Snippet: {res['chunk']['text'][:120]}...\n")

    print("=" * 70)


if __name__ == "__main__":
    test_q = sys.argv[1] if len(sys.argv) > 1 else "pet me dard and fever remedy"
    run_demo(test_q)
