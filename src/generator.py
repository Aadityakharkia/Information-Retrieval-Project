"""Trustworthy Generator module with Refusal Threshold and Groq LLM integration.
Implements:
1. Retrieval confidence threshold check (refusal when confidence is too low)
2. Strict context formatting with [C1]..[Ck] citation prompt
3. Groq API integration (Llama 3.3 70B, Llama 3.1 8B, Mistral)
4. Deterministic offline fallback generator (guarantees zero-API-key execution)
5. Disk caching of LLM answers
"""
import os
import json
import hashlib
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from src.config import (
    GROQ_API_KEY,
    DEFAULT_GENERATOR_MODEL,
    SUPPORTED_GENERATOR_MODELS,
    GENERATOR_TEMPERATURE,
    GENERATOR_MAX_TOKENS,
    COSINE_REFUSAL_THRESHOLD,
    BM25_REFUSAL_THRESHOLD,
    DENSE_REFUSAL_THRESHOLD,
    RRF_REFUSAL_THRESHOLD,
    REFUSAL_MESSAGE,
    DATA_DIR
)

logger = logging.getLogger(__name__)

CACHE_FILE = DATA_DIR / "llm_cache.json"


SYSTEM_PROMPT = """You are a trustworthy medical question-answering assistant for healthcare platforms.
Your task is to answer the user's medical question using ONLY the provided context chunks.

RULES:
1. Base your answer strictly on the provided context chunks. Do NOT introduce external medical knowledge or speculate.
2. Every claim, statement, or fact you make MUST be followed immediately by its citation label, e.g. [C1], [C2], etc.
3. If the context does not contain sufficient information to answer the question accurately, you MUST reply:
   "I don't have enough information about this in the provided context."
4. Maintain a clear, objective, professional tone suitable for medical information.
"""


def get_default_threshold_for_retriever(retriever_name: str) -> float:
    """Return default score threshold for a specific retriever type."""
    r_lower = retriever_name.lower()
    if "inverted" in r_lower or "lnc" in r_lower:
        return COSINE_REFUSAL_THRESHOLD
    elif "bm25" in r_lower:
        return BM25_REFUSAL_THRESHOLD
    elif "dense" in r_lower:
        return DENSE_REFUSAL_THRESHOLD
    elif "hybrid" in r_lower or "rrf" in r_lower:
        return RRF_REFUSAL_THRESHOLD
    return 0.1


class MedicalAnswerGenerator:
    """Orchestrates prompt formatting, refusal gating, and LLM answer generation."""
    
    def __init__(self, model_name: str = DEFAULT_GENERATOR_MODEL, api_key: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key or GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
        self.client = None
        self._init_groq_client()
        self._load_cache()
        
    def _init_groq_client(self):
        if self.api_key and self.model_name != "mock-offline":
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
                logger.info(f"Initialized Groq client for model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Could not initialize Groq client: {e}. Will use offline fallback.")
                self.client = None
        else:
            self.client = None

    def _load_cache(self):
        self.cache: Dict[str, str] = {}
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}

    def _save_cache(self):
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save LLM cache: {e}")

    def _make_cache_key(self, query: str, context_str: str) -> str:
        raw = f"{self.model_name}::{query.strip().lower()}::{context_str}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> Tuple[str, Dict[str, Dict[str, Any]]]:
        """Format top-K chunks into labeled blocks [C1]..[Ck] and return citation map."""
        context_blocks = []
        citation_map = {}
        
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            label = f"[C{idx}]"
            context_blocks.append(f"{label} {chunk['text'].strip()}")
            citation_map[label] = {
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "qtype": chunk.get("qtype", ""),
                "rank": chunk.get("rank", idx),
                "score": chunk.get("score", 0.0),
                "text": chunk["text"]
            }
            
        full_context_str = "\n\n".join(context_blocks)
        return full_context_str, citation_map

    def check_refusal(self, retrieved_chunks: List[Dict[str, Any]], threshold: float) -> Tuple[bool, float]:
        """Determine whether the retrieval confidence is sufficient to answer."""
        if not retrieved_chunks:
            return True, 0.0
        top_score = retrieved_chunks[0].get("score", 0.0)
        should_refuse = top_score < threshold
        return should_refuse, top_score

    def _generate_offline_fallback(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Deterministic extractor used when offline or without API keys."""
        if not retrieved_chunks:
            return "I don't have enough information about this in the provided context."
            
        # Select salient sentences from [C1] and [C2]
        c1 = retrieved_chunks[0]
        c1_text = c1["text"].strip()
        sentences = [s.strip() for s in c1_text.split(". ") if len(s.strip()) > 15]
        
        lines = []
        if sentences:
            lines.append(f"{sentences[0]}. [C1]")
            if len(sentences) > 1:
                lines.append(f"{sentences[1]}. [C1]")
                
        if len(retrieved_chunks) > 1:
            c2_text = retrieved_chunks[1]["text"].strip()
            c2_sentences = [s.strip() for s in c2_text.split(". ") if len(s.strip()) > 15]
            if c2_sentences:
                lines.append(f"Additionally, {c2_sentences[0].lower()}. [C2]")
                
        if not lines:
            return f"According to the medical records, {c1_text[:200]}... [C1]"
            
        return " ".join(lines)

    def generate(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        threshold: Optional[float] = None,
        retriever_name: str = "InvertedIndex"
    ) -> Dict[str, Any]:
        """Generate cited answer with refusal gating.
        
        Returns:
            Dict containing:
            - answer: str
            - refused: bool
            - top_retrieval_score: float
            - threshold_used: float
            - model_used: str
            - citation_map: dict
        """
        if threshold is None:
            threshold = get_default_threshold_for_retriever(retriever_name)
            
        should_refuse, top_score = self.check_refusal(retrieved_chunks, threshold)
        
        if should_refuse:
            return {
                "answer": REFUSAL_MESSAGE,
                "refused": True,
                "refusal_reason": f"Top retrieval score ({top_score:.4f}) is below confidence threshold ({threshold:.4f}).",
                "top_retrieval_score": top_score,
                "threshold_used": threshold,
                "model_used": "refusal_gate",
                "citation_map": {}
            }
            
        context_str, citation_map = self.format_context(retrieved_chunks)
        cache_key = self._make_cache_key(query, context_str)
        
        # Check cache
        if cache_key in self.cache:
            logger.info("Serving answer from LLM cache.")
            return {
                "answer": self.cache[cache_key],
                "refused": False,
                "top_retrieval_score": top_score,
                "threshold_used": threshold,
                "model_used": f"{self.model_name} (cached)",
                "citation_map": citation_map
            }
            
        # Attempt Groq API generation
        if self.client is not None:
            try:
                user_content = f"CONTEXT:\n{context_str}\n\nQUESTION: {query}\n\nANSWER:"
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=GENERATOR_TEMPERATURE,
                    max_tokens=GENERATOR_MAX_TOKENS
                )
                answer_text = response.choices[0].message.content.strip()
                self.cache[cache_key] = answer_text
                self._save_cache()
                
                return {
                    "answer": answer_text,
                    "refused": False,
                    "top_retrieval_score": top_score,
                    "threshold_used": threshold,
                    "model_used": self.model_name,
                    "citation_map": citation_map
                }
            except Exception as e:
                logger.warning(f"Groq generation failed ({e}). Falling back to deterministic generator.")
                
        # Deterministic fallback
        fallback_answer = self._generate_offline_fallback(query, retrieved_chunks)
        return {
            "answer": fallback_answer,
            "refused": False,
            "top_retrieval_score": top_score,
            "threshold_used": threshold,
            "model_used": "deterministic-offline-fallback",
            "citation_map": citation_map
        }
