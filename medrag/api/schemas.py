"""Pydantic schemas: the public API contract consumed by the frontend."""
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

RetrieverType = Literal["inverted_index", "bm25", "dense", "hybrid"]


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Medical question")
    retriever_type: RetrieverType = "inverted_index"
    top_k: int = Field(5, ge=1, le=20)
    threshold: Optional[float] = Field(
        None, description="Refusal score threshold; retriever default if omitted"
    )
    generator_model: Optional[str] = None


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    qtype: str
    chunk_index: int
    score: float
    text: str
    rank: int


class SentenceAudit(BaseModel):
    sentence_index: int
    sentence: str
    clean_text: str
    citations: List[str]
    verdict: str
    confidence_score: float
    cosine_similarity: float
    term_coverage: float
    matched_terms: List[str]
    explanation: str
    primary_citation: Optional[str] = None


class CitationAudit(BaseModel):
    verdict: str
    faithfulness_score: float
    total_sentences: int
    supported_count: int
    partially_supported_count: int = 0
    unsupported_count: int
    uncited_count: int
    sentence_audits: List[SentenceAudit]


class QueryResponse(BaseModel):
    question: str
    retriever_type: str
    top_k: int
    refused: bool
    refusal_reason: Optional[str] = None
    top_retrieval_score: float
    threshold_used: float
    generator_model: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    ir_diagnostics: Dict[str, Any]
    citation_audit: CitationAudit


class HealthResponse(BaseModel):
    status: str
    chunks_indexed: int
    is_initialized: bool


class ConfigResponse(BaseModel):
    models: List[str]
    default_model: str
    retrievers: List[str]
    default_top_k: int
    total_chunks: int


class TermPostingsResponse(BaseModel):
    term: str
    df: int
    idf: float
    total_postings: int
    postings_snippet: List[Any]
