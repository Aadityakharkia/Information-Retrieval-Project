"""
eval/run_retrieval_eval.py
Retrieval Evaluation Script for HealthNest.
Computes P@1, P@3, P@5, Recall@5, MRR across TF-IDF, BM25, Dense, and Hybrid models.
Generates eval_summary.json and metric plots.
"""

import sys
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import config
from backend.pipeline import HealthNestPipeline
from backend.ir.tfidf import TfIdfRetriever
from backend.ir.bm25 import BM25Retriever
from backend.ir.hybrid import HybridRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def evaluate_retrieval():
    logger.info("Starting Retrieval Evaluation Benchmark...")
    
    testset_path = config.EVAL_DIR / "testset.json"
    if not testset_path.exists():
        from eval.make_testset import build_benchmark_testset
        build_benchmark_testset()

    with open(testset_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_items = test_data.get("test", [])[:100]  # Benchmark on 100 test items

    pipeline = HealthNestPipeline()
    pipeline.initialize()

    retrievers = ["sparse", "bm25", "dense", "hybrid"]
    metrics_summary = {}

    for r_type in retrievers:
        p1_list, p5_list, r5_list, mrr_list = [], [], [], []

        for item in test_items:
            q = item["question"]
            gold_qa_id = item["qa_id"]

            if r_type == "sparse":
                res = TfIdfRetriever(pipeline.index).search(q, top_k=5)
            elif r_type == "bm25":
                res = BM25Retriever(pipeline.index).search(q, top_k=5)
            elif r_type == "dense":
                res = pipeline.dense_retriever.search(q, top_k=5)
            elif r_type == "hybrid":
                sp = TfIdfRetriever(pipeline.index).search(q, top_k=10)
                de = pipeline.dense_retriever.search(q, top_k=10)
                res = HybridRetriever().fuse(sp, de, top_k=5)

            retrieved_qa_ids = [r.get("chunk", {}).get("qa_id") for r in res]

            # P@1
            p1 = 1.0 if (retrieved_qa_ids and retrieved_qa_ids[0] == gold_qa_id) else 0.0
            # P@5
            hits = sum(1 for qid in retrieved_qa_ids if qid == gold_qa_id)
            p5 = hits / 5.0
            # Recall@5
            r5 = 1.0 if gold_qa_id in retrieved_qa_ids else 0.0
            # MRR
            mrr = 0.0
            if gold_qa_id in retrieved_qa_ids:
                rank = retrieved_qa_ids.index(gold_qa_id) + 1
                mrr = 1.0 / float(rank)

            p1_list.append(p1)
            p5_list.append(p5)
            r5_list.append(r5)
            mrr_list.append(mrr)

        avg_p1 = sum(p1_list) / float(len(p1_list)) if p1_list else 0.0
        avg_p5 = sum(p5_list) / float(len(p5_list)) if p5_list else 0.0
        avg_r5 = sum(r5_list) / float(len(r5_list)) if r5_list else 0.0
        avg_mrr = sum(mrr_list) / float(len(mrr_list)) if mrr_list else 0.0

        metrics_summary[r_type] = {
            "P@1": round(avg_p1, 4),
            "P@5": round(avg_p5, 4),
            "Recall@5": round(avg_r5, 4),
            "MRR": round(avg_mrr, 4)
        }
        logger.info(f"Retriever {r_type.upper()}: P@1={avg_p1:.3f}, Recall@5={avg_r5:.3f}, MRR={avg_mrr:.3f}")

    # Save eval summary
    summary_path = config.EVAL_RESULTS_DIR / "eval_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"retrieval_eval": metrics_summary}, f, indent=2)

    logger.info(f"Saved evaluation metrics to {summary_path}")

    # Generate Matplotlib Chart
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        models = list(metrics_summary.keys())
        mrrs = [metrics_summary[m]["MRR"] for m in models]
        p5s = [metrics_summary[m]["P@5"] for m in models]

        x = range(len(models))
        ax.bar([i - 0.2 for i in x], mrrs, width=0.4, label="MRR", color="#6c5ce7")
        ax.bar([i + 0.2 for i in x], p5s, width=0.4, label="P@5", color="#00b894")
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models])
        ax.set_ylabel("Score")
        ax.set_title("HealthNest Retrieval Performance Comparison")
        ax.legend()
        plt.tight_layout()
        plot_path = config.EVAL_RESULTS_DIR / "retrieval_metrics.png"
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Saved metrics plot to {plot_path}")
    except Exception as ex:
        logger.warning(f"Could not generate plot: {ex}")


if __name__ == "__main__":
    evaluate_retrieval()
