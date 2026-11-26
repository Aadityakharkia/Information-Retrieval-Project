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
  GET/POST /api/documents, DELETE /api/documents/<id>   the patient's prescription PDFs
  GET/DELETE /api/history, DELETE /api/history/<id>      the patient's search history
  DELETE /api/me          delete everything stored for this patient
"""

import json
import logging
import math
import os
import secrets
from urllib.parse import urlparse

from flask import Flask, g, jsonify, request, send_from_directory

import config
from backend.memory import report
from backend.memory.prescription import PdfError, extract_pdf, medicines
from backend.memory.store import PatientStore, patient_id
from backend.pipeline import RETRIEVER_TYPES, get_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

FRONTEND_DIR = config.BASE_DIR / "frontend"
PAGES = {"": "index.html", "answer": "answer.html", "library": "library.html", "about": "about.html", "my": "my.html"}

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_BYTES + 64 * 1024  # PDF + multipart overhead
pipeline = get_pipeline()
store = PatientStore()


def _int_arg(value, default, lo, hi):
    try:
        return max(lo, min(int(value), hi))
    except (TypeError, ValueError):
        return default


def _error(message, status=400):
    return jsonify({"error": message}), status


# ----------------------------------------------------------------------------
# Patient identity: anonymous random token in an HttpOnly, SameSite=Strict cookie
# ----------------------------------------------------------------------------
@app.before_request
def _block_cross_site_writes():
    origin = request.headers.get("Origin")
    if request.method not in ("GET", "HEAD", "OPTIONS") and origin and urlparse(origin).netloc != request.host:
        return _error("Cross-site requests are not allowed.", 403)


def _current_patient(create: bool = False):
    """Patient id for this browser, or None. With create=True a new identity is issued."""
    token = request.cookies.get(config.PATIENT_COOKIE, "")
    if len(token) < 32:
        if not create:
            return None
        token = secrets.token_urlsafe(32)
        g.new_patient_token = token
    return patient_id(token)


@app.after_request
def _finish_response(response):
    token = g.pop("new_patient_token", None)
    if token:
        response.set_cookie(config.PATIENT_COOKIE, token, max_age=config.PATIENT_COOKIE_DAYS * 86400,
                            httponly=True, samesite="Strict", secure=request.is_secure)
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.errorhandler(413)
def _too_large(_):
    return _error(f"File is too large (maximum {config.MAX_UPLOAD_BYTES // (1024 * 1024)} MB).", 413)


# ----------------------------------------------------------------------------
# Frontend pages (also reachable as /answer.html etc. through the static folder)
# ----------------------------------------------------------------------------
@app.route("/", defaults={"page": ""})
@app.route("/<any(answer, library, about, my):page>")
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

    link_mode = data.get("link_mode") or config.LINK_DEFAULT_MODE
    if link_mode not in config.LINK_MODES:
        return _error(f"Unknown link_mode '{link_mode}'. Choose from {list(config.LINK_MODES)}.")

    # Wikipedia is contacted only with explicit permission for one exact search term.
    wiki_query = None
    if data.get("allow_wikipedia") is True:
        wiki_query = data.get("wikipedia_query")
        wiki_query = " ".join(wiki_query.split()) if isinstance(wiki_query, str) else ""
        if not wiki_query or len(wiki_query) > 120:
            return _error("Field 'wikipedia_query' (1-120 characters) is required when 'allow_wikipedia' is true.")

    # Rocchio relevance feedback: chunk ids the user marked on the previous results of this question.
    feedback = data.get("feedback")
    if feedback is not None:
        marks = [] if not isinstance(feedback, dict) else [feedback.get(k, []) for k in ("relevant", "non_relevant")]
        pipeline.initialize()
        if (len(marks) != 2 or not all(isinstance(m, list) and all(isinstance(c, str) for c in m) for m in marks)
                or sum(map(len, marks)) > config.ROCCHIO_MAX_MARKS or set(marks[0]) & set(marks[1])
                or any(c not in pipeline.index.chunks_map for m in marks for c in m)):
            return _error(f"Field 'feedback' must be {{relevant: [...], non_relevant: [...]}} with up to "
                          f"{config.ROCCHIO_MAX_MARKS} known passage ids, none in both lists.")
        feedback = {"relevant": marks[0], "non_relevant": marks[1]}

    model = data.get("model") or config.DEFAULT_MODEL
    pid = _current_patient(create=True)
    try:
        result = pipeline.ask(
            question=question,
            retriever_type=retriever,
            top_k=_int_arg(data.get("top_k"), 5, 1, 10),
            model=model,
            enable_abstain=bool(data.get("enable_abstain", True)),
            enable_checker=bool(data.get("enable_checker", True)),
            use_champion_lists=bool(data.get("use_champion_lists", False)),
            link_mode=link_mode,
            patient_docs=store.documents(pid) if data.get("use_memory", True) else None,
            wiki_query=wiki_query,
            feedback=feedback,
            pseudo_feedback=bool(data.get("pseudo_feedback", config.PRF_DEFAULT)),
        )
        # Crisis searches are never kept in the history.
        if result["status"] != "self_harm" and data.get("save_history", True):
            store.add_search(pid, question, result["status"])
        return jsonify(result)
    except Exception:
        logger.exception("/api/ask failed")
        return _error("Internal error while answering the question.", 500)


# ----------------------------------------------------------------------------
# Private patient memory
# ----------------------------------------------------------------------------
def _doc_json(d):
    return {"id": d["id"], "filename": d["filename"], "pages": d["pages"], "created": d["created"],
            "kind": "report" if report.is_report(d["text"]) else "prescription",
            "medicines": medicines(d["text"]), "tests": [r["name"] for r in report.rows(d["text"])], "text": d["text"]}


@app.get("/api/documents")
def api_documents():
    pid = _current_patient()
    return jsonify({"documents": [_doc_json(d) for d in store.documents(pid)] if pid else []})


@app.post("/api/documents")
def api_upload_document():
    upload = request.files.get("file")
    if upload is None or not upload.filename:
        return _error("Please choose a PDF file.")
    data = upload.read(config.MAX_UPLOAD_BYTES + 1)
    if len(data) > config.MAX_UPLOAD_BYTES:
        return _too_large(None)
    pid = _current_patient(create=True)
    if len(store.documents(pid)) >= config.MAX_DOCS_PER_PATIENT:
        return _error(f"You can keep up to {config.MAX_DOCS_PER_PATIENT} documents. Delete one first.", 409)
    try:
        text, pages = extract_pdf(data)
    except PdfError as exc:
        return _error(str(exc), 422)
    name = os.path.basename(upload.filename.replace("\\", "/"))[:120] or "document.pdf"
    doc_id = store.add_document(pid, name, pages, text)
    doc = next(d for d in store.documents(pid) if d["id"] == doc_id)
    return jsonify(_doc_json(doc)), 201


@app.delete("/api/documents/<int:doc_id>")
def api_delete_document(doc_id):
    pid = _current_patient()
    if not pid or not store.delete_document(pid, doc_id):
        return _error("Document not found.", 404)
    return jsonify({"deleted": doc_id})


@app.get("/api/history")
def api_history():
    pid = _current_patient()
    limit = _int_arg(request.args.get("limit"), config.HISTORY_LIMIT, 1, config.HISTORY_LIMIT)
    return jsonify({"history": store.history(pid, limit) if pid else []})


@app.delete("/api/history")
def api_clear_history():
    pid = _current_patient()
    return jsonify({"deleted": store.clear_history(pid) if pid else 0})


@app.delete("/api/history/<int:item_id>")
def api_delete_search(item_id):
    pid = _current_patient()
    if not pid or not store.delete_search(pid, item_id):
        return _error("Search not found.", 404)
    return jsonify({"deleted": item_id})


@app.delete("/api/me")
def api_delete_me():
    pid = _current_patient()
    if pid:
        store.delete_patient(pid)
    response = jsonify({"deleted": bool(pid)})
    response.delete_cookie(config.PATIENT_COOKIE, httponly=True, samesite="Strict")
    return response


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
    category = request.args.get("category", "").strip()

    if search:
        # Use pipeline search to get ranked results
        # top_k=500 is large enough to get all relevant chunks for pagination
        results = pipeline.search(search, retriever_type="hybrid", top_k=500)
        chunks = [r["chunk"] for r in results]
        
        # Filter by category if provided
        if category:
            chunks = [
                c for c in chunks 
                if c.get("metadata", {}).get("category", "") == category or c.get("category", "") == category
            ]
    else:
        # Fallback to category filtering
        chunks = [
            c for c in pipeline.chunks
            if not category or c.get("metadata", {}).get("category", "") == category or c.get("category", "") == category
        ]

    start = (page - 1) * limit
    return jsonify({
        "page": page,
        "limit": limit,
        "total_records": len(chunks),
        "total_pages": max(1, math.ceil(len(chunks) / limit)),
        "chunks": chunks[start:start + limit],
    })

@app.get("/api/library/categories")
def api_library_categories():
    pipeline.initialize()
    categories = set()
    for c in pipeline.chunks:
        cat = c.get("metadata", {}).get("category") or c.get("category")
        if cat:
            categories.add(cat)
    return jsonify({"categories": sorted(list(categories))})


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
    port = int(os.getenv("PORT", "5001"))
    logger.info("HealthNest running on http://127.0.0.1:%d", port)
    app.run(host="127.0.0.1", port=port, debug=False)
