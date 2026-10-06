# Backend API Contract (for frontend integration)

Base URL (dev): `http://127.0.0.1:8000` — interactive docs at `/docs`, schema at `/openapi.json`.
CORS origins are set via `CORS_ORIGINS` (see `.env.example`). Types are defined in
[`medrag/api/schemas.py`](../medrag/api/schemas.py).

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | Liveness + index status |
| GET | `/api/config` | Available retrievers, generator models, defaults |
| POST | `/api/query` | Run retrieval → refusal gate → generation → citation audit |
| GET | `/api/explain_term?term=` | Inspect inverted-index postings/DF/IDF for a term |

## POST `/api/query`

Request:
```json
{
  "question": "What are the symptoms of glaucoma?",
  "retriever_type": "hybrid",
  "top_k": 5,
  "threshold": null,
  "generator_model": "llama-3.3-70b-versatile"
}
```
- `retriever_type`: `inverted_index` | `bm25` | `dense` | `hybrid`
- `threshold`: optional; omitted → per-retriever default (inverted_index 0.28, bm25 8.0, dense 0.35, hybrid 0.015)
- `generator_model`: use `mock-offline` to run without a Groq key

Response (abridged):
```json
{
  "refused": false,
  "refusal_reason": null,
  "answer": "… [C1] … [C2]",
  "top_retrieval_score": 0.377,
  "threshold_used": 0.28,
  "generator_model": "llama-3.3-70b-versatile",
  "retrieved_chunks": [{"chunk_id": "doc_00012_c0", "doc_id": "doc_00012", "qtype": "symptoms", "chunk_index": 0, "score": 0.377, "text": "…", "rank": 1}],
  "citation_audit": {
    "verdict": "TRUSTWORTHY | CAUTION | HIGH_RISK | REFUSED",
    "faithfulness_score": 1.0,
    "sentence_audits": [{"sentence": "…", "verdict": "SUPPORTED | PARTIALLY_SUPPORTED | UNSUPPORTED | UNCITED | DISCLAIMER", "confidence_score": 0.68, "cosine_similarity": 0.68, "term_coverage": 0.68, "citations": ["[C1]"]}]
  },
  "ir_diagnostics": { "…": "retriever-specific; shape varies (see below)" }
}
```

Notes for UI:
- `[C#]` in `answer` maps to `retrieved_chunks[#-1]`.
- When `refused` is true, show `answer` as the refusal text; `citation_audit.verdict` is `REFUSED`.
- `ir_diagnostics` is retriever-specific: `inverted_index` has `stems`, `query_norm`, `term_diagnostics`;
  `hybrid` has `top_fused_results`; `bm25` has `top_results_breakdown`; `dense` has `model_name`, `top_results`.
- Errors: HTTP 422 for invalid body, 500 with `{"detail": "..."}` on pipeline failure.
