"""
backend/app.py
Flask server for HealthNest: REST API + static frontend on one origin.

Run:  python -m backend.app        (http://127.0.0.1:5000)

Endpoints
  POST /api/ask           full RAG answer (retrieve -> abstain? -> generate -> verify)
  GET  /api/suggest?q=    type-ahead suggestions
  GET  /api/search?q=     retrieval-only results
  GET  /api/library       paginated corpus browser
  GET  /api/models        selectable LLM models
  GET  /api/eval/summary  retrieval benchmark report card
  GET  /api/health        liveness + index status
"""

import json
import logging
import math
import os

from flask import Flask, jsonify, request, send_from_directory

import config
from backend.pipeline import RETRIEVER_TYPES, get_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FRONTEND_DIR = config.BASE_DIR / "frontend"
PAGES = {"": "index.html", "answer": "answer.html", "library": "library.html", "about": "about.html"}

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
pipeline = get_pipeline()


def _int_arg(value, default, lo, hi):
    try:
        return max(lo, min(int(value), hi))
    except (TypeError, ValueError):
        return default


def _error(message, status=400):
    return jsonify({"error": message}), status


# ----------------------------------------------------------------------------
# Frontend pages (also reachable as /answer.html etc. through the static folder)
# ----------------------------------------------------------------------------
@app.route("/", defaults={"page": ""})
@app.route("/<any(answer, library, about):page>")
def serve_page(page):
    return send_from_directory(FRONTEND_DIR, PAGES[page])


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
@app.post("/api/ask")
def api_ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return _error("Field 'question' is required.")
    if len(question) > 500:
        return _error("Question is too long (max 500 characters).")

    retriever = data.get("retriever") or "hybrid"
    if retriever not in RETRIEVER_TYPES:
        return _error(f"Unknown retriever '{retriever}'. Choose from {list(RETRIEVER_TYPES)}.")

    model = data.get("model") or config.DEFAULT_MODEL
    try:
        result = pipeline.ask(
            question=question,
            retriever_type=retriever,
            top_k=_int_arg(data.get("top_k"), 5, 1, 10),
            model=model,
            enable_abstain=bool(data.get("enable_abstain", True)),
            enable_checker=bool(data.get("enable_checker", True)),
            use_champion_lists=bool(data.get("use_champion_lists", False)),
        )
        return jsonify(result)
    except Exception:
        logger.exception("/api/ask failed")
        return _error("Internal error while answering the question.", 500)


@app.get("/api/suggest")
def api_suggest():
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify({"prefix": "", "suggestions": []})
    limit = _int_arg(request.args.get("limit"), 6, 1, 20)
    return jsonify({"prefix": q, "suggestions": pipeline.suggest_queries(q, limit=limit)})


@app.get("/api/search")
def api_search():
    q = request.args.get("q", "").strip()
    if not q:
        return _error("Query parameter 'q' is required.")
    retriever = request.args.get("retriever", "hybrid")
    if retriever not in RETRIEVER_TYPES:
        return _error(f"Unknown retriever '{retriever}'. Choose from {list(RETRIEVER_TYPES)}.")
    results = pipeline.search(q, retriever_type=retriever, top_k=_int_arg(request.args.get("top_k"), 5, 1, 10))
    return jsonify({"query": q, "count": len(results), "results": results})


@app.get("/api/library")
def api_library():
    pipeline.initialize()
    page = _int_arg(request.args.get("page"), 1, 1, 10**6)
    limit = _int_arg(request.args.get("limit"), 12, 1, 50)
    search = request.args.get("search", "").strip().lower()

    chunks = [
        c for c in pipeline.chunks
        if not search or search in c.get("question", "").lower() or search in c.get("text", "").lower()
    ]
    start = (page - 1) * limit
    return jsonify({
        "page": page,
        "limit": limit,
        "total_records": len(chunks),
        "total_pages": max(1, math.ceil(len(chunks) / limit)),
        "chunks": chunks[start:start + limit],
    })


@app.get("/api/models")
def api_models():
    return jsonify({"models": config.AVAILABLE_MODELS, "default": config.DEFAULT_MODEL})


@app.get("/api/eval/summary")
def api_eval_summary():
    summary_path = config.EVAL_RESULTS_DIR / "eval_summary.json"
    if summary_path.exists():
        with open(summary_path, "r", encoding="utf-8") as f:
            return jsonify(json.load(f))
    return jsonify({"status": "pending", "message": "Run `make eval` to generate the report card."})


@app.get("/api/health")
def api_health():
    pipeline.initialize()
    return jsonify({
        "status": "ok",
        "chunks_count": len(pipeline.chunks),
        "index_loaded": pipeline.index is not None,
        "dense_loaded": pipeline.dense_retriever is not None and pipeline.dense_retriever.embeddings is not None,
        "default_model": config.DEFAULT_MODEL,
        "groq_configured": bool(config.GROQ_API_KEY),
    })


if __name__ == "__main__":
    pipeline.initialize()  # warm indexes before serving the first request
    port = int(os.getenv("PORT", "5000"))
    logger.info("HealthNest running on http://127.0.0.1:%d", port)
    app.run(host="127.0.0.1", port=port, debug=False)
