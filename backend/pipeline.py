"""
backend/pipeline.py
Unified HealthNest End-to-End Pipeline Controller.
Connects safety check -> language guess -> Hinglish expansion -> multi-retriever ->
abstention decision -> LLM generator -> sentence fact checker -> assembled response payload.
"""

import time
import logging
from typing import Dict, Any, List, Optional
import config

from backend.rag.safety import check_safety_guards
from backend.ir.text import detect_language
from backend.ir.hinglish import HinglishExpander
from backend.ir.inverted_index import InvertedIndex
from backend.ir.tfidf import TfIdfRetriever
from backend.ir.bm25 import BM25Retriever
from backend.ir.dense import DenseRetriever
from backend.ir.hybrid import HybridRetriever
from backend.rag.abstain import evaluate_abstention
from backend.rag.generator import generate_answer
from backend.rag.checker import verify_all_sentences
from backend.data.loader import load_dataset
from backend.data.chunker import chunk_dataset

logger = logging.getLogger(__name__)


class HealthNestPipeline:
    def __init__(self):
        self.index: Optional[InvertedIndex] = None
        self.dense_retriever: Optional[DenseRetriever] = None
        self.hinglish_expander = HinglishExpander()
        self.chunks: List[Dict[str, Any]] = []
        self.is_initialized = False

    def initialize(self):
        """Loads cached indexes or builds them if missing."""
        if self.is_initialized:
            return

        logger.info("Initializing HealthNest Pipeline...")

        # 1. Try loading inverted index
        if config.INDEX_CACHE_PATH.exists():
            self.index = InvertedIndex.load(config.INDEX_CACHE_PATH)
            self.chunks = list(self.index.chunks_map.values())
        else:
            logger.info("Index cache missing. Building on the fly...")
            records, _ = load_dataset()
            self.chunks = chunk_dataset(records, strategy=config.DEFAULT_CHUNKER_STRATEGY)
            self.index = InvertedIndex()
            self.index.index_chunks(self.chunks)
            self.index.save(config.INDEX_CACHE_PATH)

        # 2. Init Dense Retriever
        self.dense_retriever = DenseRetriever()
        if not self.dense_retriever.load_cache(self.chunks):
            logger.info("Dense embeddings cache missing. Building on the fly...")
            self.dense_retriever.build_and_cache(self.chunks)

        self.is_initialized = True
        logger.info("HealthNest Pipeline Initialization Complete!")

    def ask(
        self,
        question: str,
        retriever_type: str = "hybrid",  # "sparse", "dense", "hybrid", "bm25"
        top_k: int = 5,
        model: str = config.DEFAULT_MODEL,
        enable_abstain: bool = True,
        enable_checker: bool = True,
        use_champion_lists: bool = False
    ) -> Dict[str, Any]:
        """
        Executes complete RAG Q&A pipeline.
        """
        start_time = time.time()
        self.initialize()

        timings = {}

        # Step 1: Safety Guard Check
        t0 = time.time()
        safety = check_safety_guards(question)
        timings["safety_ms"] = round((time.time() - t0) * 1000, 2)

        if safety["is_self_harm"]:
            return {
                "status": "self_harm",
                "question": question,
                "disclaimer": config.DISCLAIMER_TEXT,
                "self_harm_message": safety["self_harm_message"],
                "emergency_banner": None,
                "sentences": [],
                "sources": [],
                "abstain": {"should_abstain": False, "reason": "Self-harm safety protocol triggered."},
                "timings": timings,
                "retriever": retriever_type,
                "model": model
            }

        # Step 2: Language Guess & Hinglish Expansion
        t0 = time.time()
        lang = detect_language(question)
        expanded_q, added_terms = self.hinglish_expander.expand_query(question)
        timings["preproc_ms"] = round((time.time() - t0) * 1000, 2)

        # Step 3: Retrieval
        t0 = time.time()
        retrieval_results = []
        sparse_res = []
        dense_res = []

        if retriever_type == "sparse":
            tfidf_r = TfIdfRetriever(self.index)
            retrieval_results = tfidf_r.search(expanded_q, top_k=top_k, use_champion_lists=use_champion_lists)
            sparse_res = retrieval_results

        elif retriever_type == "bm25":
            bm25_r = BM25Retriever(self.index)
            retrieval_results = bm25_r.search(expanded_q, top_k=top_k)
            sparse_res = retrieval_results

        elif retriever_type == "dense":
            retrieval_results = self.dense_retriever.search(expanded_q, top_k=top_k)
            dense_res = retrieval_results

        elif retriever_type == "hybrid":
            tfidf_r = TfIdfRetriever(self.index)
            sparse_res = tfidf_r.search(expanded_q, top_k=top_k*2, use_champion_lists=use_champion_lists)
            dense_res = self.dense_retriever.search(expanded_q, top_k=top_k*2)
            
            hybrid_r = HybridRetriever()
            retrieval_results = hybrid_r.fuse(sparse_res, dense_res, top_k=top_k, fusion_method="rrf")

        else:
            raise ValueError(f"Unknown retriever_type: {retriever_type}")

        timings["retrieval_ms"] = round((time.time() - t0) * 1000, 2)

        # Format sources for inspector UI
        sources = []
        for idx, item in enumerate(retrieval_results, start=1):
            chunk_data = item.get("chunk", item)
            sources.append({
                "n": idx,
                "chunk_id": chunk_data.get("chunk_id"),
                "qa_id": chunk_data.get("qa_id"),
                "question": chunk_data.get("question"),
                "text": chunk_data.get("text"),
                "score": item.get("score", 0.0),
                "zone_scores": item.get("zone_scores", {}),
                "matched_terms": item.get("matched_terms", []),
                "sparse_rank": item.get("sparse_rank"),
                "dense_rank": item.get("dense_rank")
            })

        # Inspector comparison data (Top-5 side-by-side)
        inspector_data = {
            "sparse_top5": [s["chunk"]["chunk_id"] for s in sparse_res[:5]],
            "dense_top5": [d["chunk"]["chunk_id"] for d in dense_res[:5]],
            "fused_top5": [r["chunk_id"] for r in sources[:5]]
        }

        # Step 4: Abstention Check
        t0 = time.time()
        abstain_decision = evaluate_abstention(expanded_q, retrieval_results)
        timings["abstain_ms"] = round((time.time() - t0) * 1000, 2)

        if enable_abstain and abstain_decision["should_abstain"]:
            total_time = round((time.time() - start_time) * 1000, 2)
            timings["total_ms"] = total_time
            return {
                "status": "abstained",
                "question": question,
                "language": lang,
                "expanded_query": expanded_q,
                "disclaimer": config.DISCLAIMER_TEXT,
                "emergency_banner": safety["emergency_banner"],
                "self_harm_message": None,
                "sentences": [],
                "sources": sources,
                "inspector": inspector_data,
                "abstain": abstain_decision,
                "timings": timings,
                "retriever": retriever_type,
                "model": model
            }

        # Step 5: Answer Generation
        t0 = time.time()
        gen_result = generate_answer(question, retrieval_results, model=model)
        timings["generation_ms"] = round((time.time() - t0) * 1000, 2)

        # Check if LLM output answered "I don't know"
        if gen_result["raw_answer"] == "I don't know based on the provided pages.":
            abstain_decision["should_abstain"] = True
            abstain_decision["reason"] = "LLM indicated insufficient reference evidence."
            total_time = round((time.time() - start_time) * 1000, 2)
            timings["total_ms"] = total_time
            return {
                "status": "abstained",
                "question": question,
                "language": lang,
                "expanded_query": expanded_q,
                "disclaimer": config.DISCLAIMER_TEXT,
                "emergency_banner": safety["emergency_banner"],
                "self_harm_message": None,
                "sentences": [],
                "sources": sources,
                "inspector": inspector_data,
                "abstain": abstain_decision,
                "timings": timings,
                "retriever": retriever_type,
                "model": model
            }

        # Step 6: Fact Verification Checking
        t0 = time.time()
        if enable_checker:
            verified_sentences = verify_all_sentences(gen_result["sentences"], retrieval_results)
        else:
            verified_sentences = gen_result["sentences"]
            for s in verified_sentences:
                s["support"] = {"supported": True, "badge": "supported", "reasons": ["Fact checker disabled."]}
        timings["checker_ms"] = round((time.time() - t0) * 1000, 2)

        total_time = round((time.time() - start_time) * 1000, 2)
        timings["total_ms"] = total_time

        status = "emergency" if safety["is_emergency"] else "answered"

        return {
            "status": status,
            "question": question,
            "language": lang,
            "expanded_query": expanded_q,
            "added_synonyms": added_terms,
            "disclaimer": config.DISCLAIMER_TEXT,
            "emergency_banner": safety["emergency_banner"],
            "self_harm_message": None,
            "sentences": verified_sentences,
            "sources": sources,
            "inspector": inspector_data,
            "abstain": abstain_decision,
            "timings": timings,
            "retriever": retriever_type,
            "model": model
        }


# Singleton pipeline instance
_pipeline_instance = HealthNestPipeline()

def ask(question: str, **kwargs) -> Dict[str, Any]:
    return _pipeline_instance.ask(question, **kwargs)
