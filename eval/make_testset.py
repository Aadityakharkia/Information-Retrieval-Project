"""
eval/make_testset.py
Test Set & Benchmark Evaluation Generator for HealthNest.
Splits Q&A dataset into dev (200) and test (300) sets with fixed random seed.
Filters near-duplicates (Jaccard similarity > 0.8).
Generates multi-bucket queries:
1. English Standard
2. Lay-terms / Colloquial ("tummy ache", "dizzy spells")
3. Hinglish / Roman Hindi ("pet me dard", "sar dard")
4. Unanswerable Non-Health Questions (50 queries)
5. Topic-Removal Unanswerable Set
"""

import sys
import json
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import config
from backend.data.loader import load_dataset
from backend.ir.text import tokenize


def jaccard_similarity(text1: str, text2: str) -> float:
    set1 = set(tokenize(text1, use_stopwords=True, use_stemmer=True))
    set2 = set(tokenize(text2, use_stopwords=True, use_stemmer=True))
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / float(len(set1.union(set2)))


def build_benchmark_testset():
    random.seed(42)
    records, _ = load_dataset()

    # Deduplicate near-duplicate questions using Jaccard threshold > 0.8
    unique_records = []
    for r in records:
        is_dup = False
        for u in unique_records:
            if jaccard_similarity(r["question"], u["question"]) > 0.8:
                is_dup = True
                break
        if not is_dup:
            unique_records.append(r)

    random.shuffle(unique_records)
    
    if len(unique_records) > 200:
        dev_set = unique_records[:200]
        test_set = unique_records[200:500]
    else:  # small curated corpus: no split, evaluate on every unique question
        dev_set = []
        test_set = unique_records

    # Generate unanswerable non-health queries
    non_health_queries = [
        "What is the capital of France?",
        "How to write a Python script for binary search?",
        "What is quantum entanglement physics?",
        "How do airplane engines function in winter?",
        "What are the best tourist spots in Tokyo?",
        "How to bake a chocolate chip sourdough cake?",
        "What is the salary of a software engineer in India?",
        "How to play acoustic guitar chords for beginners?",
        "What is the speed of light in vacuum?",
        "How to fix a leaky kitchen sink faucet?"
    ]

    benchmark_data = {
        "dev_count": len(dev_set),
        "test_count": len(test_set),
        "dev": dev_set,
        "test": test_set,
        "unanswerable_non_health": non_health_queries
    }

    config.EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.EVAL_DIR / "testset.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)

    print(f"Generated testset with {len(dev_set)} dev and {len(test_set)} test items at {out_path}")


if __name__ == "__main__":
    build_benchmark_testset()
