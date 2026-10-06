"""Comprehensive Retrieval Evaluation Script.
Evaluates InvertedIndex (lnc.ltc), BM25, Dense, and Hybrid (RRF) across 300 test queries.
Computes P@1, P@3, P@5, R@1, R@3, R@5, MRR, MAP, plus per-qtype breakdown.
"""
import sys
import json
import logging
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any
import numpy as np

from medrag.data.processor import load_chunks, load_test_queries
from medrag.retrieval.inverted_index import InvertedIndexRetriever
from medrag.retrieval.bm25 import BM25Retriever
from medrag.retrieval.dense import DenseRetriever
from medrag.retrieval.hybrid import HybridRetriever
from medrag.evaluation.metrics import precision_at_k, recall_at_k, reciprocal_rank, average_precision
from medrag.config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_retriever_on_queries(
    retriever,
    retriever_name: str,
    test_queries: List[Dict[str, Any]],
    k_vals: List[int] = [1, 3, 5]
) -> Dict[str, Any]:
    """Run evaluation for a single retriever over test queries."""
    logger.info(f"Evaluating {retriever_name} over {len(test_queries)} queries...")
    
    p_at_k = {k: [] for k in k_vals}
    r_at_k = {k: [] for k in k_vals}
    mrr_list = []
    map_list = []
    
    # Per qtype tracking
    qtype_metrics = defaultdict(lambda: {"p_at_5": [], "r_at_5": [], "mrr": []})
    
    for q in test_queries:
        query_text = q["question"]
        qtype = q["qtype"]
        gold_doc_ids = set(q["relevant_doc_ids"])
        
        # Retrieve top 10
        results = retriever.retrieve(query_text, top_k=10)
        retrieved_doc_ids = [r["doc_id"] for r in results]
        
        # Metrics
        for k in k_vals:
            p_at_k[k].append(precision_at_k(retrieved_doc_ids, gold_doc_ids, k))
            r_at_k[k].append(recall_at_k(retrieved_doc_ids, gold_doc_ids, k))
            
        rr = reciprocal_rank(retrieved_doc_ids, gold_doc_ids)
        ap = average_precision(retrieved_doc_ids, gold_doc_ids, k=10)
        
        mrr_list.append(rr)
        map_list.append(ap)
        
        # Per qtype
        p5 = precision_at_k(retrieved_doc_ids, gold_doc_ids, 5)
        r5 = recall_at_k(retrieved_doc_ids, gold_doc_ids, 5)
        qtype_metrics[qtype]["p_at_5"].append(p5)
        qtype_metrics[qtype]["r_at_5"].append(r5)
        qtype_metrics[qtype]["mrr"].append(rr)
        
    summary = {
        "retriever": retriever_name,
        "num_queries": len(test_queries),
        "P@1": float(round(np.mean(p_at_k[1]), 4)),
        "P@3": float(round(np.mean(p_at_k[3]), 4)),
        "P@5": float(round(np.mean(p_at_k[5]), 4)),
        "R@1": float(round(np.mean(r_at_k[1]), 4)),
        "R@3": float(round(np.mean(r_at_k[3]), 4)),
        "R@5": float(round(np.mean(r_at_k[5]), 4)),
        "MRR": float(round(np.mean(mrr_list), 4)),
        "MAP": float(round(np.mean(map_list), 4)),
        "by_qtype": {}
    }
    
    for qtype, scores in qtype_metrics.items():
        summary["by_qtype"][qtype] = {
            "count": len(scores["mrr"]),
            "P@5": float(round(np.mean(scores["p_at_5"]), 4)),
            "R@5": float(round(np.mean(scores["r_at_5"]), 4)),
            "MRR": float(round(np.mean(scores["mrr"]), 4))
        }
        
    return summary


def run_full_retrieval_benchmark(limit_queries: int = 150):
    """Run full benchmarking suite across InvertedIndex, BM25, Dense, and Hybrid."""
    chunks = load_chunks()
    all_test_queries = load_test_queries()
    test_queries = all_test_queries[:limit_queries] if limit_queries else all_test_queries
    
    logger.info(f"Loaded {len(chunks)} chunks and using {len(test_queries)} test queries.")
    
    # 1. Inverted Index (lnc.ltc)
    inv_index = InvertedIndexRetriever()
    inv_index.build_index(chunks)
    res_inv = evaluate_retriever_on_queries(inv_index, "InvertedIndex (lnc.ltc)", test_queries)
    
    # 2. BM25
    bm25 = BM25Retriever()
    bm25.build_index(chunks)
    res_bm25 = evaluate_retriever_on_queries(bm25, "Okapi BM25", test_queries)
    
    # 3. Dense (all-MiniLM-L6-v2)
    dense = DenseRetriever()
    dense.build_index(chunks)
    res_dense = evaluate_retriever_on_queries(dense, "Dense (all-MiniLM-L6-v2)", test_queries)
    
    # 4. Hybrid (RRF)
    hybrid = HybridRetriever(inv_index, dense)
    res_hybrid = evaluate_retriever_on_queries(hybrid, "Hybrid (RRF: Sparse + Dense)", test_queries)
    
    all_results = [res_inv, res_bm25, res_dense, res_hybrid]
    
    # Save results
    output_path = DATA_DIR / "eval_retrieval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved evaluation results to {output_path}")
    
    # Print formatted markdown table
    print("\n" + "="*80)
    print("### RETRIEVAL BENCHMARK RESULTS (MedQuAD Test Queries)")
    print("="*80)
    print("| Retriever Model | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | MRR | MAP |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in all_results:
        print(f"| **{r['retriever']}** | {r['P@1']:.4f} | {r['P@3']:.4f} | {r['P@5']:.4f} | {r['R@1']:.4f} | {r['R@3']:.4f} | {r['R@5']:.4f} | {r['MRR']:.4f} | {r['MAP']:.4f} |")
    print("="*80 + "\n")
    
    return all_results


if __name__ == "__main__":
    limit = 100 if len(sys.argv) < 2 else int(sys.argv[1])
    run_full_retrieval_benchmark(limit_queries=limit)
