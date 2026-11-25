# HealthNest

Evidence-grounded health Q&A built on classic Information Retrieval.
Every answer sentence is traced to a retrieved passage, verified against it,
and the system says *"I don't know"* when the evidence is weak.

```
question ─► safety guard ─► patient's own prescription? ─► Hinglish expansion ─► retrieval (TF-IDF · BM25 · dense · hybrid RRF)
         ─► PageRank + HITS re-ranking
         ─► abstention gate ─► LLM answer with [n] citations ─► sentence-level fact check ─► UI
              └─► no evidence: ask permission ─► Wikipedia
```

## Quick start

```bash
make install            # venv + dependencies
cp .env.example .env    # optional: add GROQ_API_KEY for hosted LLM answers
make serve              # http://127.0.0.1:5001  (UI + API on one origin)
```

The first start builds the indexes from `data/raw/health_qa.csv` (a few seconds) and caches them
in `data/processed/` (git-ignored). `make build` forces a rebuild.

### LLM providers (first that works wins)

| Priority | Provider | When |
|---|---|---|
| 1 | Groq (`openai/gpt-oss-120b` default) | `GROQ_API_KEY` set |
| 2 | Ollama (`llama3.2:1b` …) | local server at `OLLAMA_BASE_URL` |
| 3 | Offline extractive generator | always available; no network needed |

The answer page shows which engine actually produced the answer.

## Project layout

```
backend/
  app.py            Flask server: REST API + static frontend
  pipeline.py       End-to-end orchestration (HealthNestPipeline)
  data/             loader.py (CSV → records) · chunker.py (sentence-window chunks)
  ir/               text.py · inverted_index.py · tfidf.py (lnc.ltc) · bm25.py · dense.py · hybrid.py (RRF) · linkgraph.py (PageRank, HITS) · hinglish.py
  rag/              safety.py · abstain.py · prompt.py · llm_client.py · generator.py · checker.py
  memory/           store.py (SQLite patient memory) · prescription.py (PDF reading, verbatim answers) · report.py (test reports)
  web/              wikipedia.py (permission-based fallback)
frontend/           index · answer · library · my · about  (+ js/, css/, assets/)
data/               raw/health_qa.csv · lexicon_hinglish_health.tsv · processed/ (generated cache)
eval/               make_testset.py · run_retrieval_eval.py · results/
tests/              test_ir_core.py (IR maths) · test_api.py (API contract)
docs/               IR_CONCEPT_MAP.md · AI_USAGE_LOG.md
config.py           all tunable parameters and paths
```

## API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/ask` | `{question, retriever?, top_k?, model?, enable_abstain?, enable_checker?, link_mode?, use_memory?, allow_wikipedia?, wikipedia_query?}` → answer, sentences + citations + support badges, sources, retrieval inspector, timings |
| GET | `/api/suggest?q=` | type-ahead suggestions |
| GET | `/api/search?q=&retriever=&top_k=` | retrieval only |
| GET | `/api/library?page=&limit=&search=` | browse the indexed corpus |
| GET | `/api/models` | selectable models + server default |
| GET | `/api/eval/summary` | retrieval benchmark report card |
| GET | `/api/health` | index status |
| GET · POST | `/api/documents` | list / upload prescription PDFs (`DELETE /api/documents/<id>`) |
| GET · DELETE | `/api/history` | list / clear search history (`DELETE /api/history/<id>`) |
| DELETE | `/api/me` | delete everything stored for this patient |

`retriever` ∈ `sparse` · `bm25` · `dense` · `hybrid` (default). Invalid input returns `400 {"error": …}`.

`link_mode` ∈ `off` · `pagerank` · `hits` · `both` (default).

## Private patient memory

- **No login.** A random token in an HttpOnly, SameSite=Strict cookie identifies the browser; only its hash is stored. People sharing a browser share one memory, and clearing cookies loses it.
- **Text PDFs only** (10 MB, 30 pages, 10 files). Scanned or handwritten prescriptions are refused with a message.
- **Test reports** (blood tests, scans) work too. Lines are quoted exactly; a value is compared only with the range printed on the same line, and unclear lines are never compared.
- The PDF is not kept, only its text, in `data/private/` (git-ignored, not encrypted).
- **Prescription lines are shown word for word**, never rewritten by a model, and questions that use them are answered on this computer only (local Ollama or the offline generator, never Groq).
- **Wikipedia** is contacted only after the user approves the exact words. Pages are not added to the corpus.
- History keeps the last 100 searches; self-harm searches are not saved.

## IR components

- **Link analysis**: topics link to the topics their text mentions (4,774 topics, 36,791 links). PageRank is global; HITS runs on a query-specific base set. On held-out questions nDCG@10 went from 0.316 to 0.330 (p = 0.11, not significant); see `eval/model_comparison/link_results.json`.

- **Sparse**: inverted index with positional-free postings, SMART `lnc.ltc` cosine scoring, optional champion lists.
- **BM25**: Okapi BM25 (`k1=1.2`, `b=0.75`) over the same index.
- **Dense**: `paraphrase-multilingual-MiniLM-L12-v2` embeddings (handles Hindi/Devanagari).
- **Hybrid**: Reciprocal Rank Fusion (`k=60`) of sparse + dense.
- **Hinglish**: lexicon-based query expansion (`pet dard` → stomach pain).
- **Abstention**: refuses when top score or query-term coverage is too low.
- **Fact checker**: per-sentence support via TF-IDF cosine + key-term coverage against the cited chunk.
- **Safety**: emergency banner and self-harm support message before any retrieval.

See [docs/IR_CONCEPT_MAP.md](docs/IR_CONCEPT_MAP.md) for the mapping to course concepts.

## Development

```bash
make test    # IR maths + API contract tests
make eval    # P@k / Recall@5 / MRR for all retrievers → eval/results/
```

> **Corpus size.** `data/raw/health_qa.csv` is a small curated set (27 Q&As → 54 passages).
> Evaluation queries are the corpus questions themselves, so the scores
> measure self-retrieval, not generalisation. Replace the CSV with a larger dataset
> (columns `qa_id, question, answer, category, source` — other common names are auto-detected)
> and run `make build eval` for meaningful numbers.

> Information from a Q&A dataset, not medical advice.
