# Trustworthy Medical RAG on MedQuAD

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-teal.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

> **CSD358 Information Retrieval Hackathon (Midsem)**  
> **Track 1: Retrieval-Augmented Generation (RAG) and Trustworthy Answers**  
> **Base Paper:** *"RAG-Based Knowledge Retrieval Architecture for Medical Web Platforms"*, Bara Fedallah & Özkan İnik (2026), IEEE ISADES. DOI: [10.1109/ISADES69945.2026.11608242](https://doi.org/10.1109/ISADES69945.2026.11608242).

---

## 1. Executive Summary & Core IR Contributions

Modern clinical question-answering systems often suffer from hallucinations, opacity, and overconfidence on out-of-distribution queries. While Fedallah & İnik (ISADES 2026) demonstrated dense-only retrieval over a tiny 56-chunk FAQ dataset with an uncalibrated top-5 pass-through, this project scales the architecture to **MedQuAD** (NIH clinical question-answer corpus) across **25,643 chunks** and introduces **four foundational Information Retrieval innovations**:

1. **Real, Inspectable Vector Space Engine (`lnc.ltc`)**:
   - Built from scratch: Porter Stemming, stopword removal, dictionary mappings, inverted index with frequency postings.
   - SMART weighting: Document normalized under `lnc` (log tf, no idf, cosine norm); Query normalized under `ltc` (log tf, idf, cosine norm).
   - Term-at-a-time (TAAT) accumulator scoring coupled with a **min-heap top-K** selector.
   - Deep inspection API exposing exact postings, document frequencies, idf weights, and vector norms.
2. **Multi-Model Retrieval Suite**:
   - Custom **Okapi BM25** ($k_1=1.5, b=0.75$) with document length normalization and term score decomposition.
   - Dense Neural Retrieval using **`all-MiniLM-L6-v2`** with FAISS L2-normalized inner product search.
   - **Reciprocal Rank Fusion (RRF)** hybrid retriever combining sparse lexical precision and dense semantic recall ($k=60$).
3. **Safety Refusal Gate (Addressing Base Paper Gap)**:
   - Evaluates retrieval confidence against calibrated threshold $\tau$.
   - Automatically refuses irrelevant and non-medical queries (`"I do not have enough verified medical information..."`), dropping Out-Of-Distribution (OOD) false acceptance from 100% to 0%.
4. **Local Sentence-Level Citation Checker**:
   - Every generated claim is linked to context chunks `[C1]..[Ck]`.
   - Re-evaluates each generated sentence against its cited source using **TF-IDF cosine similarity** and **IDF-weighted concept coverage**.
   - Assigns independent, deterministic verdicts: `[SUPPORTED]`, `[PARTIALLY_SUPPORTED]`, `[UNSUPPORTED]`, or `[UNCITED]` with zero dependency on external LLM judges.
5. **Multilingual Diagnostic (Hindi / Hinglish)**:
   - Proves vocabulary mismatch failures of English sparse retrieval on Hinglish queries and validates dense multilingual bridging.

---

## 2. Key Extensions Over Base Paper

| Feature | Base Paper (Fedallah & İnik, 2026) | Our Extended System |
|---|---|---|
| **Knowledge Base** | 56 chunks (single FAQ section) | **25,643 chunks** (Full NIH MedQuAD answers) |
| **Retrieval Architecture** | Dense only (`all-MiniLM-L6-v2`) | **Inverted Index (`lnc.ltc`) + BM25 + Dense + RRF Hybrid** |
| **Top-K Strategy** | Fixed top-5 (no confidence check) | **Calibrated Refusal Threshold ($\tau$) + Top-K min-heap** |
| **Out-of-Distribution Handling** | Always answers (hallucination risk) | **Guaranteed refusal on low confidence / OOD** |
| **Faithfulness Check** | LLM judge (Llama 3.3 70B via Groq) | **Local Sentence-Level TF-IDF Cosine & Concept Checker** |
| **Languages Tested** | English only | **English + Hindi / Hinglish cross-lingual queries** |
| **Evaluation Metrics** | RAGAS Faithfulness only | **P@1, P@3, P@5, R@1, R@3, R@5, MRR, MAP, Threshold Curves** |

---

## 3. System Architecture & Pipeline

```
                                  [ User Query ]
                                        │
           ┌────────────────────────────┼────────────────────────────┐
           ▼                            ▼                            ▼
  [Porter Stemmer & Tokenizer]    [BM25 Analyzer]          [Dense Transformer]
           │                            │                            │
           ▼                            ▼                            ▼
 [Inverted Index (lnc.ltc)]       [Okapi BM25]             [all-MiniLM-L6-v2]
           │                            │                            │
           └────────────────────────────┼────────────────────────────┘
                                        ▼
                         [Reciprocal Rank Fusion (RRF)]
                                        │
                                        ▼
                        [Confidence Refusal Threshold (τ)]
                                ├── Below τ ──► [Safe Medical Refusal Response]
                                │
                                └── Above τ ──► [Top-K Chunks: [C1] ... [Ck]]
                                                       │
                                                       ▼
                                            [Generator: Llama/Mistral]
                                                       │
                                                       ▼
                                       [Generated Answer with [C#] Citations]
                                                       │
                                                       ▼
                                      [Local Sentence-Level Citation Checker]
                                           - TF-IDF Cosine Similarity
                                           - IDF-Weighted Concept Coverage
                                                       │
                                                       ▼
                                  [Audit Badges: SUPPORTED / UNSUPPORTED]
```

---

## 4. Mathematical Formulations

### 4.1 SMART Vector Space Weighting (`lnc.ltc`)
- **Document Vector (`lnc`)**:
  $$w_{t, d} = \frac{1 + \ln(\text{tf}_{t, d})}{\sqrt{\sum_{t'} (1 + \ln(\text{tf}_{t', d}))^2}}$$
- **Query Vector (`ltc`)**:
  $$w_{t, q} = \frac{(1 + \ln(\text{tf}_{t, q})) \cdot \ln\left(\frac{N+1}{\text{df}_t + 0.5}\right)}{\sqrt{\sum_{t'} \left((1 + \ln(\text{tf}_{t', q})) \cdot \ln\left(\frac{N+1}{\text{df}_{t'} + 0.5}\right)\right)^2}}$$
- **Cosine Score**:
  $$\text{Score}(q, d) = \sum_{t \in q \cap d} w_{t, q} \cdot w_{t, d}$$

### 4.2 Okapi BM25
$$\text{BM25}(q, d) = \sum_{t \in q} \ln\left(\frac{N - \text{df}_t + 0.5}{\text{df}_t + 0.5} + 1\right) \cdot \frac{\text{tf}_{t, d} \cdot (k_1 + 1)}{\text{tf}_{t, d} + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$
with $k_1 = 1.5, b = 0.75$.

### 4.3 Reciprocal Rank Fusion (RRF)
$$\text{RRF\_Score}(d) = \sum_{r \in \{\text{Sparse}, \text{Dense}\}} \frac{1}{k_{\text{rrf}} + \text{rank}_r(d)}, \quad k_{\text{rrf}} = 60$$

### 4.4 Local Citation Support Score
For sentence $S$ citing chunk $C$:
$$\text{Support}(S, C) = 0.65 \cdot \text{Cosine}_{\text{TF-IDF}}(S, C) + 0.35 \cdot \frac{\sum_{t \in S \cap C} \text{IDF}(t)}{\sum_{t \in S} \text{IDF}(t)}$$
- If $\text{Support}(S, C) \ge 0.25$: **`[SUPPORTED]`**
- If $0.12 \le \text{Support}(S, C) < 0.25$: **`[PARTIALLY_SUPPORTED]`**
- If $\text{Support}(S, C) < 0.12$: **`[UNSUPPORTED]`** (Flagged as potential hallucination)

---

## 5. Experimental Results

### 5.1 Retrieval Performance (Held-out MedQuAD Test Queries)
Evaluated on stratified test queries across 15 medical query types (`qtype`):

| Retriever Model | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | MRR | MAP |
|---|---|---|---|---|---|---|---|---|
| **Inverted Index (`lnc.ltc`)** | 0.2800 | 0.1867 | 0.1520 | 0.2800 | 0.5500 | 0.7400 | 0.4553 | 0.4669 |
| **Okapi BM25** | 0.3600 | 0.1800 | 0.1400 | 0.3500 | 0.5300 | 0.6800 | 0.4893 | 0.5054 |
| **Dense (`all-MiniLM-L6-v2`)** | 0.5600 | 0.2800 | 0.1880 | 0.5400 | 0.8000 | 0.8900 | **0.6792** | **0.7156** |
| **Hybrid (RRF: Sparse + Dense)** | 0.3600 | 0.2533 | 0.1760 | 0.3500 | 0.7400 | 0.8500 | 0.5687 | 0.5888 |

### 5.2 Refusal Threshold Tradeoff & Out-Of-Distribution Safety

| Threshold ($\tau$) | In-Dist Answered (%) | In-Dist Refused (%) | OOD Refused (Safety %) | OOD False Acceptance (%) |
|---|---|---|---|---|
| 0.10 | 100.0% | 0.0% | 0.0% | 100.0% |
| 0.15 | 100.0% | 0.0% | 40.0% | 60.0% |
| 0.20 | 100.0% | 0.0% | 90.0% | 10.0% |
| 0.25 | 100.0% | 0.0% | 100.0% | 0.0% |
| **0.28 (Optimal)** | **96.0%** | **4.0%** | **100.0%** | **0.0%** |
| 0.32 | 92.0% | 8.0% | 100.0% | 0.0% |
| 0.36 | 80.0% | 20.0% | 100.0% | 0.0% |

*Finding:* At $\tau = 0.28$, the system retains 96% coverage on legitimate medical queries while completely eliminating false acceptance on out-of-distribution prompts.

### 5.3 Multilingual Hinglish vs English Queries
| Query Language | Sparse Retriever Match Score | Dense Retriever Match Score | Keyword Retention |
|---|---|---|---|
| **English** | 0.35 - 0.46 | 0.52 - 0.65 | 100% |
| **Hinglish** | 0.08 - 0.15 (Drop of ~70%) | 0.42 - 0.58 (Retained) | 90% |

*Finding:* English sparse lexical retrievers experience acute vocabulary mismatch on Hinglish medical queries, whereas dense semantic encoders successfully bridge linguistic variations.

---

## 6. Installation & Quickstart

Requires Python 3.10+ (tested on 3.11).

```bash
make install        # venv + editable install with dev extras
make data           # chunk MedQuAD (needs data/train.csv) -> data/processed/
make test           # 7 unit tests
make serve          # API + bundled UI at http://127.0.0.1:8000  (API docs: /docs)
make cli            # interactive terminal client with IR inspection
```

Optional: `export GROQ_API_KEY=...` for LLM generation (otherwise a deterministic offline generator is used; see `.env.example`).

Reproduce benchmarks:
```bash
make eval-retrieval      # P@k, R@k, MRR, MAP
make eval-threshold      # refusal threshold sweep
make eval-multilingual   # English vs Hinglish
```

### Frontend integration
The backend is a standalone REST API; the UI in `frontend/` is just a bundled demo and can be replaced.
See **[docs/API.md](docs/API.md)** for the request/response contract (typed in `medrag/api/schemas.py`).
Set `CORS_ORIGINS` to your frontend's dev origin (e.g. `http://localhost:5173`).

---

## 7. Project Structure

```
IR_midsem/
├── medrag/                      # Python package (all backend logic)
│   ├── config.py                # Paths, hyperparameters, thresholds, env settings
│   ├── pipeline.py              # Orchestrates retrieve -> refuse -> generate -> verify
│   ├── nlp/text.py              # Tokenizer, Porter stemmer, stopwords, sentence splitter
│   ├── data/processor.py        # Dedup, 200w/30w chunking, stratified test split
│   ├── retrieval/               # base interface + inverted_index, bm25, dense, hybrid (RRF)
│   ├── generation/generator.py  # Refusal gate, [C#] prompting, Groq + offline fallback
│   ├── verification/            # Sentence-level citation checker
│   ├── evaluation/metrics.py    # P@k, R@k, MRR, MAP
│   └── api/                     # FastAPI app (main.py) + typed schemas.py
├── scripts/                     # CLI + evaluation entry points
├── frontend/                    # Bundled static demo UI (replaceable)
├── tests/                       # Unit tests
├── docs/                        # API contract, report draft, demo script
├── data/                        # Dataset + generated artifacts (large files git-ignored)
├── Makefile  pyproject.toml  requirements.txt  .env.example
```

---

## 8. AI-Use Declaration
In compliance with the hackathon rules:
- Code architecture and implementation drafted with AI coding agent assistance.
- All IR algorithms (Inverted Index, postings structures, SMART `lnc.ltc` weighting, BM25, RRF, sentence citation verification) were implemented in pure Python/NumPy with zero black-box retrieval libraries.
- Generative models utilized: Llama 3.3 70B and Llama 3.1 8B via Groq API.
- Dense embedding representations: `all-MiniLM-L6-v2` (Sentence-Transformers).

---

## 9. Citation & Credits
- **MedQuAD Dataset**: Asma Ben Abacha and Dina Demner-Fushman, *"A Question-Entailment Approach for Question Answering in the Medical Domain"*, BMC Medical Informatics and Decision Making, 2019.
- **Base Architecture Paper**: Bara Fedallah and Özkan İnik, *"RAG-Based Knowledge Retrieval Architecture for Medical Web Platforms"*, IEEE ISADES, 2026.
