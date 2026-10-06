"""
backend/app.py
Flask API Web Server for HealthNest.
Serves REST API routes for retrieval, RAG QA, library browsing, model selection,
and evaluation summaries. Serves frontend static files directly from 'frontend/' directory.

Lecture Concepts: Web IR Interface, REST API for RAG, End-to-End Pipeline Service.
"""

import sys
import json
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import config
from backend.pipeline import ask, HealthNestPipeline
from backend.data.loader import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Static frontend folder
FRONTEND_DIR = config.BASE_DIR / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
pipeline_instance = HealthNestPipeline()


# -----------------------------------------------------------------------------
# Static Frontend Serving Routes
# -----------------------------------------------------------------------------
@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/answer")
@app.route("/answer.html")
def serve_answer():
    return send_from_directory(FRONTEND_DIR, "answer.html")

@app.route("/library")
def serve_library():
    return send_from_directory(FRONTEND_DIR, "library.html")

@app.route("/about")
def serve_about():
    return send_from_directory(FRONTEND_DIR, "about.html")


# -----------------------------------------------------------------------------
# REST API Endpoints
# -----------------------------------------------------------------------------
@app.route("/api/ask", methods=["POST"])
def api_ask():
    """
    Main RAG Q&A Endpoint.
    Body JSON:
      {
        "question": str,
        "retriever": "sparse"|"dense"|"hybrid"|"bm25",
        "top_k": int,
        "model": str,
        "enable_abstain": bool,
        "enable_checker": bool,
        "use_champion_lists": bool
      }
    """
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Question parameter is required"}), 400

    retriever = data.get("retriever", "hybrid")
    top_k = int(data.get("top_k", 5))
    model = data.get("model", config.DEFAULT_MODEL)
    enable_abstain = bool(data.get("enable_abstain", True))
    enable_checker = bool(data.get("enable_checker", True))
    use_champion_lists = bool(data.get("use_champion_lists", False))

    try:
        result = pipeline_instance.ask(
            question=question,
            retriever_type=retriever,
            top_k=top_k,
            model=model,
            enable_abstain=enable_abstain,
            enable_checker=enable_checker,
            use_champion_lists=use_champion_lists
        )
        return jsonify(result)
    except Exception as e:
        logger.exception(f"Error processing /api/ask: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/search", methods=["GET"])
def api_search():
    """Retrieval-only Search Endpoint (no LLM generation)."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"error": "Query parameter 'q' is required"}), 400

    retriever = request.args.get("retriever", "hybrid")
    top_k = int(request.args.get("top_k", 5))

    pipeline_instance.initialize()
    
    if retriever == "sparse":
        from backend.ir.tfidf import TfIdfRetriever
        r = TfIdfRetriever(pipeline_instance.index)
        results = r.search(q, top_k=top_k)
    elif retriever == "bm25":
        from backend.ir.bm25 import BM25Retriever
        r = BM25Retriever(pipeline_instance.index)
        results = r.search(q, top_k=top_k)
    elif retriever == "dense":
        results = pipeline_instance.dense_retriever.search(q, top_k=top_k)
    else:
        from backend.ir.tfidf import TfIdfRetriever
        from backend.ir.hybrid import HybridRetriever
        sparse_res = TfIdfRetriever(pipeline_instance.index).search(q, top_k=top_k*2)
        dense_res = pipeline_instance.dense_retriever.search(q, top_k=top_k*2)
        results = HybridRetriever().fuse(sparse_res, dense_res, top_k=top_k)

    return jsonify({"query": q, "count": len(results), "results": results})


@app.route("/api/library", methods=["GET"])
def api_library():
    """Browse, filter, and paginate dataset Q&A chunks."""
    pipeline_instance.initialize()
    page = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 12))
    category = request.args.get("category", "").strip().lower()
    search = request.args.get("search", "").strip().lower()

    chunks = pipeline_instance.chunks

    filtered = []
    for c in chunks:
        q_text = c.get("question", "")
        body_text = c.get("text", "")
        # Filter by search term if provided
        if search and (search not in q_text.lower() and search not in body_text.lower()):
            continue
        filtered.append(c)

    total_records = len(filtered)
    start_idx = (page - 1) * limit
    end_idx = start_idx + limit
    page_chunks = filtered[start_idx:end_idx]

    return jsonify({
        "page": page,
        "limit": limit,
        "total_records": total_records,
        "total_pages": math.ceil(total_records / float(limit)) if limit > 0 else 1,
        "chunks": page_chunks
    })


@app.route("/api/models", methods=["GET"])
def api_models():
    """Returns list of supported LLM models."""
    return jsonify({
        "models": config.AVAILABLE_MODELS,
        "default": config.DEFAULT_MODEL
    })


@app.route("/api/eval/summary", methods=["GET"])
def api_eval_summary():
    """Serves report-card evaluation summary JSON."""
    summary_path = config.EVAL_RESULTS_DIR / "eval_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)
    
    # Return placeholder summary if eval scripts haven't run yet
    return jsonify({
        "status": "pending",
        "message": "Evaluation script results are generating or pending."
    })


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health status endpoint."""
    pipeline_instance.initialize()
    return jsonify({
        "status": "ok",
        "chunks_count": len(pipeline_instance.chunks),
        "index_loaded": pipeline_instance.index is not None,
        "dense_loaded": pipeline_instance.dense_retriever.embeddings is not None
    })


import math

if __name__ == "__main__":
    logger.info("Starting HealthNest Flask Server on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
