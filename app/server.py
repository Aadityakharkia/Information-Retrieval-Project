"""FastAPI Server exposing Trustworthy Medical RAG & IR Inspection endpoints.
"""
import os
import logging
from typing import Optional, List
from fastapi import FastAPI, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.pipeline import TrustworthyMedicalRAGPipeline
from src.config import SUPPORTED_GENERATOR_MODELS, DEFAULT_GENERATOR_MODEL

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Trustworthy Medical RAG API",
    description="Inspectable Information Retrieval & Verification System on MedQuAD",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline singleton
pipeline = TrustworthyMedicalRAGPipeline()


@app.on_event("startup")
def startup_event():
    logger.info("Initializing RAG pipeline on server startup...")
    pipeline.initialize()


class QueryRequest(BaseModel):
    question: str
    retriever_type: str = "inverted_index"
    top_k: int = 5
    threshold: Optional[float] = None
    generator_model: Optional[str] = DEFAULT_GENERATOR_MODEL


@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "chunks_indexed": len(pipeline.chunks),
        "is_initialized": pipeline.is_initialized
    }


@app.get("/api/config")
def get_config():
    return {
        "models": SUPPORTED_GENERATOR_MODELS,
        "default_model": DEFAULT_GENERATOR_MODEL,
        "retrievers": ["inverted_index", "bm25", "dense", "hybrid"],
        "default_top_k": 5,
        "total_chunks": len(pipeline.chunks)
    }


@app.post("/api/query")
def process_query(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
        
    try:
        response = pipeline.answer_question(
            question=req.question,
            retriever_type=req.retriever_type,
            top_k=req.top_k,
            threshold=req.threshold,
            generator_model=req.generator_model
        )
        return response
    except Exception as e:
        logger.error(f"Error handling query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/explain_term")
def explain_term(term: str = Query(...)):
    """Inspect the inverted index postings list and IDF for a single term."""
    stems = pipeline.inverted_index.postings.get(term.lower().strip(), [])
    df_val = pipeline.inverted_index.df.get(term.lower().strip(), 0)
    idf_val = pipeline.inverted_index.idf.get(term.lower().strip(), 0.0)
    
    return {
        "term": term,
        "df": df_val,
        "idf": idf_val,
        "total_postings": len(stems),
        "postings_snippet": stems[:10]
    }


# Mount static assets
static_path = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_path):
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")
