"""End-to-End Trustworthy Medical RAG Pipeline on MedQuAD.
Orchestrates:
1. Retriever Selection: Inverted Index (lnc.ltc), BM25, Dense, or Hybrid (RRF)
2. Confidence / Refusal Threshold checking
3. Labeled [C1]..[Ck] context formatting
4. LLM Generation (Groq Llama 3.3 / Llama 3.1 / Mistral / Offline Fallback)
5. Sentence-Level Citation Verification
6. Complete IR Inspectability Diagnostics
"""
import logging
from typing import List, Dict, Any, Optional

from src.data_processor import load_chunks
from src.retrievers.inverted_index import InvertedIndexRetriever
from src.retrievers.bm25_retriever import BM25Retriever
from src.retrievers.dense_retriever import DenseRetriever
from src.retrievers.hybrid_retriever import HybridRetriever
from src.generator import MedicalAnswerGenerator, get_default_threshold_for_retriever
from src.citation_checker import SentenceCitationChecker
from src.config import DEFAULT_TOP_K

logger = logging.getLogger(__name__)


class TrustworthyMedicalRAGPipeline:
    """Production-grade, inspectable RAG pipeline for medical questions."""
    
    def __init__(self, init_dense: bool = False):
        self.chunks: List[Dict[str, Any]] = []
        self.inverted_index = InvertedIndexRetriever()
        self.bm25 = BM25Retriever()
        self.dense = DenseRetriever()
        self.hybrid: Optional[HybridRetriever] = None
        self.generator = MedicalAnswerGenerator()
        self.citation_checker = SentenceCitationChecker()
        self.is_initialized = False
        self.init_dense = init_dense
        
    def initialize(self, force_rebuild_sparse: bool = False, force_rebuild_dense: bool = False):
        """Load corpus chunks and build indices."""
        logger.info("Initializing Trustworthy Medical RAG Pipeline...")
        self.chunks = load_chunks()
        logger.info(f"Loaded {len(self.chunks)} corpus chunks.")
        
        # Build Inverted Index (lnc.ltc)
        self.inverted_index.build_index(self.chunks, force_rebuild=force_rebuild_sparse)
        
        # Build BM25
        self.bm25.build_index(self.chunks)
        
        # Build Dense if requested
        if self.init_dense:
            self.dense.build_index(self.chunks, force_recompute=force_rebuild_dense)
            self.hybrid = HybridRetriever(self.inverted_index, self.dense)
            
        self.is_initialized = True
        logger.info("Pipeline initialization complete!")

    def ensure_dense_built(self):
        """Lazy builder for dense and hybrid components."""
        if not getattr(self.dense, "is_built", False):
            logger.info("Building dense index on demand...")
            self.dense.build_index(self.chunks)
            self.hybrid = HybridRetriever(self.inverted_index, self.dense)

    def get_retriever(self, retriever_type: str):
        """Retrieve appropriate retriever instance based on selection."""
        r_type = retriever_type.lower().strip()
        if r_type in ["inverted_index", "tfidf", "lnc_ltc", "sparse"]:
            return self.inverted_index
        elif r_type in ["bm25"]:
            return self.bm25
        elif r_type in ["dense"]:
            self.ensure_dense_built()
            return self.dense
        elif r_type in ["hybrid", "rrf"]:
            self.ensure_dense_built()
            return self.hybrid
        else:
            logger.warning(f"Unknown retriever type '{retriever_type}'. Defaulting to Inverted Index.")
            return self.inverted_index

    def answer_question(
        self,
        question: str,
        retriever_type: str = "inverted_index",
        top_k: int = DEFAULT_TOP_K,
        threshold: Optional[float] = None,
        generator_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute end-to-end question answering with IR inspection and citation verification.
        
        Returns:
            Dict containing:
            - question: str
            - retriever_type: str
            - retrieved_chunks: list
            - answer: str
            - refused: bool
            - ir_diagnostics: dict
            - citation_audit: dict
            - performance_metadata: dict
        """
        if not self.is_initialized:
            self.initialize()
            
        retriever = self.get_retriever(retriever_type)
        actual_retriever_name = retriever.__class__.__name__
        
        # 1. IR Diagnostics & Retrieval
        ir_diagnostics = retriever.explain(question, top_k=top_k)
        retrieved_chunks = retriever.retrieve(question, top_k=top_k)
        
        # 2. Generation with Refusal Gate
        if generator_model and generator_model != self.generator.model_name:
            active_gen = MedicalAnswerGenerator(model_name=generator_model)
        else:
            active_gen = self.generator
            
        gen_result = active_gen.generate(
            query=question,
            retrieved_chunks=retrieved_chunks,
            threshold=threshold,
            retriever_name=actual_retriever_name
        )
        
        answer_text = gen_result["answer"]
        refused = gen_result["refused"]
        citation_map = gen_result["citation_map"]
        
        # 3. Citation Audit
        if not refused and citation_map:
            citation_audit = self.citation_checker.audit_response(
                generated_answer=answer_text,
                citation_map=citation_map,
                idf_dict=self.inverted_index.idf
            )
        else:
            citation_audit = {
                "verdict": "REFUSED" if refused else "NO_CONTEXT",
                "faithfulness_score": 0.0,
                "total_sentences": 0,
                "supported_count": 0,
                "partially_supported_count": 0,
                "unsupported_count": 0,
                "uncited_count": 0,
                "sentence_audits": []
            }
            
        return {
            "question": question,
            "retriever_type": actual_retriever_name,
            "top_k": top_k,
            "refused": refused,
            "refusal_reason": gen_result.get("refusal_reason", None),
            "top_retrieval_score": gen_result.get("top_retrieval_score", 0.0),
            "threshold_used": gen_result.get("threshold_used", 0.0),
            "generator_model": gen_result.get("model_used", ""),
            "answer": answer_text,
            "retrieved_chunks": retrieved_chunks,
            "ir_diagnostics": ir_diagnostics,
            "citation_audit": citation_audit
        }
