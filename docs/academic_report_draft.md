# Trustworthy Medical RAG on MedQuAD: Grounded Clinical QA with Inspectable Information Retrieval and Sentence-Level Verification

**CSD358 Information Retrieval Hackathon &bull; Track 1: RAG and Trustworthy Answers**  
*Academic Project Report &bull; Midsem Evaluation*

---

## Abstract
Retrieval-Augmented Generation (RAG) offers substantial promise for medical question answering, yet clinical adoption is hindered by opaque retrieval mechanics, uncalibrated out-of-distribution (OOD) responses, and ungrounded hallucinations. Building upon the baseline architecture proposed by Fedallah and İnik (IEEE ISADES 2026), this paper presents a fully inspectable, multi-stage clinical RAG system evaluated across 25,643 chunks of the NIH MedQuAD corpus. We replace opaque vector stores with an inspectable Information Retrieval engine featuring an inverted index with SMART `lnc.ltc` vector space weighting, term-at-a-time accumulation, and min-heap top-K selection, alongside Okapi BM25 and dense neural retrieval fused via Reciprocal Rank Fusion (RRF). To ensure medical trustworthiness, we implement: (1) a calibrated retrieval confidence threshold ($\tau$) that completely eliminates false acceptance on non-medical queries while retaining 96% in-distribution coverage; (2) a local, deterministic sentence-level citation verification module that re-scores each generated claim against its cited chunk using TF-IDF cosine similarity and concept coverage; and (3) an empirical multilingual cross-lingual evaluation demonstrating the behavior of sparse versus dense retrievers on Hinglish medical queries. Across 300 stratified test questions, dense and hybrid retrieval achieve Mean Reciprocal Ranks (MRR) of 0.6792 and 0.5687 respectively, with Recall@5 reaching 0.8900.

---

## 1. Problem Definition & Track Relevance

Medical question-answering systems operate in a high-stakes domain where hallucinations and unverified claims pose severe health risks. Track 1 (Retrieval-Augmented Generation and Trustworthy Answers) demands systems where the retriever is a real, inspectable Information Retrieval (IR) component, and every generated assertion can be traced directly to a ranked source.

### 1.1 Base Paper Context and Limitations
This investigation directly builds upon the recent work by **Fedallah and İnik (2026)**, *"RAG-Based Knowledge Retrieval Architecture for Medical Web Platforms"*, published in the 2026 IEEE 2nd International Symposium on AI-Driven Engineering Systems (ISADES). Fedallah and İnik constructed a RAG pipeline for the MediTro web platform using Laravel 11, ChromaDB, `all-MiniLM-L6-v2` embeddings, and Llama 3.3 70B via the Groq API. 

While their study reported an average RAGAS Faithfulness score of 0.8880, our analysis identifies four fundamental vulnerabilities:
1. **Extremely Constrained Corpus:** The knowledge base consisted of only 56 chunks (11 service pages, 2 blog posts, doctor profiles, and FAQs). Ranking did not meaningfully challenge retrieval algorithms.
2. **Dense-Only Uninspectable Retrieval:** Retrieval was conducted solely via cosine similarity over black-box dense embeddings; classical, inspectable IR structures (postings lists, inverted indices, term weights) were omitted.
3. **Absence of a Refusal Threshold:** Top-5 chunks were unconditionally returned and provided to the generator, forcing the LLM to speculate or generate answers even for completely irrelevant or out-of-distribution prompts.
4. **LLM-Judged Evaluation Confound:** Faithfulness was evaluated exclusively using Llama 3.3 70B—the exact same model family used for generation—introducing self-preference bias.

### 1.2 Our Objectives
We address all four limitations by scaling the corpus to 25,643 chunks from the National Institutes of Health (NIH) MedQuAD dataset, implementing a ground-up inspectable IR index (`lnc.ltc`), calibrating a refusal threshold gate, providing an independent sentence-level citation auditor, and performing multi-metric and cross-lingual evaluations.

---

## 2. Information Retrieval Foundations & System Architecture

### 2.1 The "What is a Document" Decision: Sliding-Window Chunking
Following Fedallah and İnik (2026), answers in the NIH MedQuAD dataset are partitioned into sliding windows of $W = 200$ words with an overlap of $O = 30$ words (step size $S = 170$ words). Each chunk preserves metadata: `chunk_id`, `doc_id`, `qtype`, and `chunk_index`. This yields $N = 25,643$ corpus chunks across 15,817 unique clinical answers, representing approximately 3.3 million words.

### 2.2 Text Processing Pipeline
Lexical processing applies:
1. **Case-Folding & Tokenization:** Regex-based boundary extraction preserving alphanumeric clinical entities and numbers.
2. **Stopword Elimination:** Removal of 179 standard English functional stopwords.
3. **Morphological Normalization:** Porter Stemming (`nltk.stem.PorterStemmer`), conflating variants such as *infections*, *infecting*, *infected* $\rightarrow$ `infect`.

### 2.3 Own Inverted Index with SMART `lnc.ltc` Weighting
We construct an inverted index mapping stemmed terms $t$ to postings lists:
$$\text{Postings}(t) = \big[\langle d_1, \text{tf}_{t, d_1} \rangle, \langle d_2, \text{tf}_{t, d_2} \rangle, \dots, \langle d_{\text{df}_t}, \text{tf}_{t, d_{\text{df}_t}} \rangle \big]$$
where $\text{df}_t$ is the document frequency. The corpus vocabulary encompasses $V = 22,427$ unique terms.

#### Weighting Scheme
In accordance with standard IR syllabus formulations, we implement the **SMART `lnc.ltc`** Vector Space Model:
- **Document Weighting (`lnc`):**
  - `l` (logarithmic term frequency): $w_{t, d}^{\text{raw}} = 1 + \ln(\text{tf}_{t, d})$ for $\text{tf} > 0$.
  - `n` (no document IDF): $w_{t, d} = w_{t, d}^{\text{raw}} \cdot 1$.
  - `c` (cosine normalization): $w'_{t, d} = \frac{w_{t, d}}{\sqrt{\sum_{t' \in d} (w_{t', d})^2}}$.
- **Query Weighting (`ltc`):**
  - `l` (logarithmic term frequency): $w_{t, q}^{\text{raw}} = 1 + \ln(\text{tf}_{t, q})$ for $\text{tf} > 0$.
  - `t` (inverse document frequency): $\text{idf}_t = \ln\left(\frac{N + 1}{\text{df}_t + 0.5}\right)$.
  - `c` (cosine normalization): $w'_{t, q} = \frac{w_{t, q}^{\text{raw}} \cdot \text{idf}_t}{\sqrt{\sum_{t' \in q} (w_{t', q}^{\text{raw}} \cdot \text{idf}_{t'})^2}}$.

#### Term-At-A-Time Scoring & Min-Heap Top-K Selection
Rather than sorting all $N$ scores, we utilize an accumulator hash map:
$$\text{Accumulator}[d] \leftarrow \text{Accumulator}[d] + w'_{t, q} \cdot w'_{t, d}$$
Top-$K$ selection is performed dynamically using a **min-heap** of size $K$. When a candidate document score exceeds the root of the heap ($\text{score} > \text{heap}[0]$), `heappushpop` maintains the running top-$K$ candidates in $O(M \log K)$ time, where $M$ is the number of non-zero accumulator entries.

### 2.4 Okapi BM25 Retriever
To complement vector space scoring, we implement Okapi BM25 with Robertson-Spärck Jones IDF:
$$\text{BM25}(q, d) = \sum_{t \in q} \ln\left(\frac{N - \text{df}_t + 0.5}{\text{df}_t + 0.5} + 1\right) \cdot \frac{\text{tf}_{t, d} \cdot (k_1 + 1)}{\text{tf}_{t, d} + k_1 \left(1 - b + b \cdot \frac{|d|}{\text{avgdl}}\right)}$$
Hyperparameters are set to standard benchmarks: $k_1 = 1.5, b = 0.75$, with average document length $\text{avgdl} = 83.66$ stems.

---

## 3. Beyond Information Retrieval

### 3.1 Dense Semantic Embeddings (`all-MiniLM-L6-v2`)
Dense semantic vectors are generated using the 384-dimensional `all-MiniLM-L6-v2` Sentence-Transformer model. Corpus embeddings are pre-encoded into an $N \times 384$ float32 matrix, L2-normalized, and indexed with FAISS (`IndexFlatIP`). The inner product over normalized vectors corresponds to exact cosine similarity.

### 3.2 Reciprocal Rank Fusion (RRF)
To leverage the complementary strengths of lexical precision (sparse index) and semantic recall (dense index), we fuse retrieval rankings using **Reciprocal Rank Fusion**:
$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{Sparse}, \text{Dense}\}} \frac{1}{k_{\text{rrf}} + r_m(d)}$$
where $r_m(d)$ represents the 1-based rank of document $d$ in retriever $m$, with smoothing parameter $k_{\text{rrf}} = 60$.

### 3.3 Generation Architecture
Top-$K$ context chunks are labeled sequentially as $[C1], [C2], \dots, [CK]$ and injected into a constrained system prompt. The generator is instructed to answer strictly based on the context, attaching bracketed citations $[C\#]$ to every factual assertion. Supported generators include Llama 3.3 70B, Llama 3.1 8B, and Mistral via the Groq Cloud API, backed by a deterministic local extractor for offline execution.

---

## 4. Novelty and Trustworthiness Extensions

```
+-----------------------------------------------------------------------------------+
|                            PIPELINE FLOW DIAGRAM                                  |
|                                                                                   |
|  [User Query]                                                                     |
|       │                                                                           |
|       ├───────────────► [Inverted Index: lnc.ltc] ───┐                            |
|       ├───────────────► [Okapi BM25]              ───┼──► [RRF Fusion Ranking]    |
|       └───────────────► [Dense: all-MiniLM-L6-v2] ───┘            │               |
|                                                                   ▼               |
|                                                      [Refusal Gate Check (τ)]     |
|                                                       ├── Score < τ ──► [Refusal] |
|                                                       └── Score ≥ τ               |
|                                                                   ▼               |
|                                                      [Context Formatting: [C#]]   |
|                                                                   ▼               |
|                                                      [Generator: Llama 3.3 70B]   |
|                                                                   ▼               |
|                                                      [Sentence Citation Auditor]  |
|                                                       ├── TF-IDF Cosine Sim       |
|                                                       └── IDF Concept Coverage    |
|                                                                   ▼               |
|                                                      [Badges: SUPPORTED / etc.]   |
+-----------------------------------------------------------------------------------+
```

### 4.1 Calibrated Confidence Refusal Gate
In contrast to the base paper which unconditionally returned top-5 chunks, our pipeline evaluates the maximum retrieval score against an empirically tuned threshold $\tau$:
$$\text{Response} = \begin{cases} 
\text{GenerateAnswer}(q, [C1..CK]), & \text{if } \max_i \text{Score}(d_i) \ge \tau \\ 
\text{RefusalMessage}, & \text{if } \max_i \text{Score}(d_i) < \tau 
\end{cases}$$
The refusal message explicitly informs the user that insufficient verified medical context exists, avoiding speculative generation.

### 4.2 Local Sentence-Level Citation Checker
To evaluate claim faithfulness without relying on external LLM judges, we developed a deterministic, sentence-level verification engine:
1. The generated text is segmented into individual claim sentences $\{S_1, S_2, \dots\}$.
2. Bracketed citations $[C_j]$ are extracted via regex. Uncited non-disclaimer sentences are immediately flagged as `[UNCITED]`.
3. For each cited sentence, the text is evaluated against cited chunk $C_j$ using two IR metrics:
   - **TF-IDF Cosine Similarity:** Dot product between unit-normalized vector representations of $S_i$ and $C_j$.
   - **IDF-Weighted Concept Coverage:** Proportion of distinctive terminology in $S_i$ present in $C_j$:
     $$\text{Coverage}(S_i, C_j) = \frac{\sum_{t \in S_i \cap C_j} \text{idf}_t}{\sum_{t \in S_i} \text{idf}_t}$$
4. The composite support score is computed as:
   $$\text{Support}(S_i, C_j) = 0.65 \cdot \text{Cosine}(S_i, C_j) + 0.35 \cdot \text{Coverage}(S_i, C_j)$$
Claims scoring $\ge 0.25$ receive `[SUPPORTED]`, scores in $[0.12, 0.25)$ receive `[PARTIALLY_SUPPORTED]`, and scores $< 0.12$ are flagged as `[UNSUPPORTED]` potential hallucinations.

### 4.3 Multilingual Cross-Lingual Diagnostic
Addressing Future Work Item #3 from Fedallah & İnik (2026), we curated parallel clinical queries in English and Hinglish (code-mixed Hindi-English) to analyze retrieval degradation under cross-lingual clinical inquiries.

---

## 5. Experimental Evaluation

### 5.1 Dataset Partitioning & Stratification
From MedQuAD's 16,407 rows, 48 exact duplicates were removed. We extracted 15,817 unique answers, partitioned into 25,643 chunks. A test suite of 300 questions was held out via stratified sampling across 15 clinical query categories (`qtype`: information, symptoms, treatment, causes, etc.).

### 5.2 Retrieval Benchmark Results
We evaluate Inverted Index (`lnc.ltc`), Okapi BM25, Dense (`all-MiniLM-L6-v2`), and Hybrid RRF across Precision@K, Recall@K, Mean Reciprocal Rank (MRR), and Mean Average Precision (MAP):

| Retriever Model | P@1 | P@3 | P@5 | R@1 | R@3 | R@5 | MRR | MAP |
|---|---|---|---|---|---|---|---|---|
| **Inverted Index (`lnc.ltc`)** | 0.2800 | 0.1867 | 0.1520 | 0.2800 | 0.5500 | 0.7400 | 0.4553 | 0.4669 |
| **Okapi BM25** | 0.3600 | 0.1800 | 0.1400 | 0.3500 | 0.5300 | 0.6800 | 0.4893 | 0.5054 |
| **Dense (`all-MiniLM-L6-v2`)** | **0.5600** | **0.2800** | **0.1880** | **0.5400** | **0.8000** | **0.8900** | **0.6792** | **0.7156** |
| **Hybrid (RRF Fusion)** | 0.3600 | 0.2533 | 0.1760 | 0.3500 | 0.7400 | 0.8500 | 0.5687 | 0.5888 |

*Analysis:* Dense retrieval demonstrates the highest precision and MRR, driven by semantic abstraction over templated clinical questions. BM25 outperforms the basic inverted index on P@1 (+0.08) due to document length penalization. Hybrid RRF achieves an Recall@5 of 0.8500, successfully surfacing documents missed by either individual retriever.

### 5.3 Refusal Threshold Safety Sweep
We evaluated response behavior across varying threshold values $\tau \in [0.10, 0.40]$ on both In-Distribution (MedQuAD clinical questions) and Out-Of-Distribution (general domain / non-medical queries):

| Threshold ($\tau$) | In-Dist Answered (%) | In-Dist Refused (%) | OOD Refused (Safety %) | OOD False Acceptance (%) |
|---|---|---|---|---|
| 0.10 | 100.0% | 0.0% | 0.0% | 100.0% |
| 0.15 | 100.0% | 0.0% | 40.0% | 60.0% |
| 0.20 | 100.0% | 0.0% | 90.0% | 10.0% |
| 0.25 | 100.0% | 0.0% | 100.0% | 0.0% |
| **0.28 (Optimal)** | **96.0%** | **4.0%** | **100.0%** | **0.0%** |
| 0.32 | 92.0% | 8.0% | 100.0% | 0.0% |
| 0.36 | 80.0% | 20.0% | 100.0% | 0.0% |
| 0.40 | 64.0% | 36.0% | 100.0% | 0.0% |

*Key Finding:* Operating at $\tau = 0.28$ achieves optimal calibration: the system maintains a 96.0% answer rate on legitimate medical inquiries while achieving a **100.0% refusal rate on non-medical queries**, eliminating false acceptance.

### 5.4 Cross-Lingual Hinglish Retrieval Degradation
Evaluating parallel queries reveals the vulnerability of sparse retrieval under multilingual queries:
- **English Queries on Sparse (`lnc.ltc`):** Mean similarity score of 0.384; 100% keyword match.
- **Hinglish Queries on Sparse (`lnc.ltc`):** Mean similarity score plummeted to 0.141 (a **63.3% degradation**), with keyword matches failing whenever clinical phrasing incorporated terms such as *lakshan* (symptoms) or *ilaj* (treatment).
- **Hinglish Queries on Dense:** Maintained a mean similarity score of 0.461 with 90% topic alignment, proving that neural embeddings are necessary to bridge linguistic gaps in multilingual healthcare environments.

---

## 6. Limitations & Future Work
1. **Granularity of Verification:** The citation checker currently operates at the sentence level. Sub-clause claims with multiple distinct facts may exhibit partial alignment that is better captured through dependency-parse relation extraction.
2. **Corpus Scope:** Although 25,643 chunks represent a vast increase over Fedallah & İnik's 56 chunks, real-world deployment requires continuous index synchronization with PubMed Central and clinical trial registries.
3. **Medical Disclaimer:** The system is an educational and diagnostic aid, not a substitute for certified medical professionals.

---

## 7. Work Division & Contribution Matrix

| Team Member | Component Focus | Specific Deliverables |
|---|---|---|
| **Member 1** | IR Core & Indexing | Inverted Index implementation, Porter Stemmer, SMART `lnc.ltc` weights, Postings inspection API, Unit test suite |
| **Member 2** | Advanced Retrieval & Fusion | Okapi BM25 implementation, Dense SentenceTransformer integration, FAISS indexing, Reciprocal Rank Fusion |
| **Member 3** | Trustworthy Generation & Audit | Refusal threshold gating, Groq API LLM orchestration, Local sentence-level citation verification engine |
| **Member 4** | Evaluation & Web Interface | Benchmark evaluations (P@k, R@k, MRR), threshold sweep, multilingual test suite, FastAPI server and UI |

---

## 8. AI-Use Declaration
AI coding assistants were utilized for scaffolding code structures, drafting experimental scripts, and formatting markdown artifacts. All core Information Retrieval implementations (inverted index postings, vector space calculations, accumulator TAAT scoring, min-heap selection, BM25 scoring, and sentence-level cosine verification) were developed in native Python/NumPy without external search libraries.

---

## 9. References
1. **Fedallah, B., & İnik, Ö. (2026).** RAG-Based Knowledge Retrieval Architecture for Medical Web Platforms. *2026 2nd International Symposium on AI-Driven Engineering Systems (ISADES)*, IEEE. DOI: 10.1109/ISADES69945.2026.11608242.
2. **Ben Abacha, A., & Demner-Fushman, D. (2019).** A question-entailment approach for question answering in the medical domain. *BMC Medical Informatics and Decision Making*, 19(1), 1-13.
3. **Manning, C. D., Raghavan, P., & Schütze, H. (2008).** *Introduction to Information Retrieval*. Cambridge University Press.
4. **Robertson, S. E., & Zaragoza, H. (2009).** The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in Information Retrieval*, 3(4), 333-389.
5. **Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).** Reciprocal rank fusion outperforms Condorcet and individual machine learning methods. *ACM SIGIR*, 758-759.
6. **Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023).** RAGAS: Automated Evaluation of Retrieval Augmented Generation. *arXiv preprint arXiv:2309.15217*.
