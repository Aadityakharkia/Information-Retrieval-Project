"""FastAPI application exposing the Trustworthy Medical RAG pipeline.

Run:  uvicorn medrag.api.main:app --reload
Docs: http://127.0.0.1:8000/docs  (OpenAPI schema at /openapi.json)
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from medrag.api.schemas import (
    ConfigResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    TermPostingsResponse,
)
from medrag.config import (
    CORS_ORIGINS,
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_TOP_K,
    FRONTEND_DIR,
    SUPPORTED_GENERATOR_MODELS,
)
from medrag.pipeline import TrustworthyMedicalRAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

pipeline = TrustworthyMedicalRAGPipeline()


@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Initializing RAG pipeline...")
    pipeline.initialize()
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Trustworthy Medical RAG API",
        description="Inspectable IR + citation-verified answers over MedQuAD",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthResponse, tags=["system"])
    def health():
        return HealthResponse(
            status="healthy",
            chunks_indexed=len(pipeline.chunks),
            is_initialized=pipeline.is_initialized,
        )

    @app.get("/api/config", response_model=ConfigResponse, tags=["system"])
    def get_config():
        return ConfigResponse(
            models=SUPPORTED_GENERATOR_MODELS,
            default_model=DEFAULT_GENERATOR_MODEL,
            retrievers=["inverted_index", "bm25", "dense", "hybrid"],
            default_top_k=DEFAULT_TOP_K,
            total_chunks=len(pipeline.chunks),
        )

    @app.post("/api/query", response_model=QueryResponse, tags=["rag"])
    def query(req: QueryRequest):
        try:
            return pipeline.answer_question(
                question=req.question,
                retriever_type=req.retriever_type,
                top_k=req.top_k,
                threshold=req.threshold,
                generator_model=req.generator_model or DEFAULT_GENERATOR_MODEL,
            )
        except Exception as exc:  # surface as clean 500 for the client
            logger.exception("Query failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.get("/api/explain_term", response_model=TermPostingsResponse, tags=["inspection"])
    def explain_term(term: str = Query(..., min_length=1)):
        key = term.lower().strip()
        index = pipeline.inverted_index
        postings = index.postings.get(key, [])
        return TermPostingsResponse(
            term=term,
            df=index.df.get(key, 0),
            idf=index.idf.get(key, 0.0),
            total_postings=len(postings),
            postings_snippet=postings[:10],
        )

    # Optional bundled UI; the standalone frontend can ignore/replace this.
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    return app


app = create_app()
