# 5-8 Minute Live Demo Presentation Script
**CSD358 IR Hackathon (Midsem) &bull; Track 1: Trustworthy Medical RAG on MedQuAD**

> **Format:** Live recording, NO slides, real working code, intermediate outputs shown, each team member speaks for their component.
> **Total Time:** ~6 to 7 minutes.

---

### Segment 1: Introduction & Architecture Overview (0:00 - 1:15)
**Presenter: Member 1**
- **Action:** Show the running Web UI at `http://localhost:8000`.
- **Talking Points:**
  - "Hello, we are presenting our project for Track 1: Retrieval-Augmented Generation (RAG) and Trustworthy Answers."
  - "Our base paper is Fedallah & İnik (IEEE ISADES 2026), which implemented a RAG layer for medical platforms. However, their paper had four major limitations: it only indexed 56 chunks, used dense-only retrieval without an inspectable IR index, had no confidence threshold (always returning top-5 even if irrelevant), and relied entirely on an LLM judge for evaluation."
  - "We extended this across **25,643 chunks** of the NIH MedQuAD corpus with a real, inspectable Information Retrieval engine built from scratch."
  - "Our pipeline combines an inverted index with SMART `lnc.ltc` vector space weighting, Okapi BM25, dense retrieval, and Reciprocal Rank Fusion, guarded by a confidence refusal gate and a local sentence-level citation checker."

---

### Segment 2: Deep IR Inspectability & Intermediate Outputs (1:15 - 3:00)
**Presenter: Member 2**
- **Action:** Type query: `"What are the symptoms of Lymphocytic Choriomeningitis?"` with retriever set to `Inverted Index (SMART lnc.ltc)`. Click `Execute RAG & IR Inspection`.
- **Talking Points:**
  - "Here is our core IR component in action. When the query is submitted, we inspect every intermediate step in the Vector Space Model."
  - *Click 'Vector Space (lnc.ltc)' Tab:*
    - "First, the query is tokenized, stripped of stopwords, and stemmed using the Porter Stemmer into `['symptom', 'lymphocyt', 'choriomening']`."
    - "Notice the Euclidean norm of the query vector $|q| = 5.09$."
    - "In the term diagnostics table, you can see the Document Frequency (DF), natural log IDF, and normalized `ltc` query weight for each stem."
  - *Click 'View Postings' on `lymphocyt`:*
    - "We can directly inspect the underlying inverted index postings list: document frequency is 167, and here are the exact chunk IDs and raw term frequencies."
  - *Click 'Rank Fusion / Scoring' Tab:*
    - "Scoring uses term-at-a-time accumulation followed by a min-heap of size K to efficiently track the top candidates without full array sorting."

---

### Segment 3: Refusal Threshold Gate & Safety Against Hallucination (3:00 - 4:15)
**Presenter: Member 3**
- **Action 1:** Select preset chip `"OOD Refusal Test"` (`"How do I bake chocolate chip cookies?"`). Click `Execute`.
- **Talking Points:**
  - "One of the most dangerous vulnerabilities in medical RAG is out-of-distribution queries, where models hallucinate medical answers for non-medical questions because a top-K context is forced."
  - "Notice what happens here: the top retrieval score is only 0.1678, which is well below our calibrated refusal threshold $\tau = 0.28$."
  - "The system immediately triggers a safe medical refusal banner: *'I do not have enough verified medical information in the knowledge base to answer this question reliably.'*"
  - "In our experimental sweep, this threshold achieved a 100% rejection rate of non-medical queries while retaining 96% coverage on valid medical questions."

---

### Segment 4: Answer Generation & Sentence-Level Citation Verification (4:15 - 5:30)
**Presenter: Member 4**
- **Action:** Switch back to a clinical query: `"What is the treatment for diabetes mellitus?"`. Run with `Hybrid (RRF)` retriever and `Llama 3.3 70B`.
- **Talking Points:**
  - "Here the system retrieved relevant chunks and passed them to the generator with strict citation prompts."
  - "The generated answer attributes each claim with `[C1]`, `[C2]` tags."
  - "Now look at our Local Sentence-Level Citation Checker below the answer. This is an independent, IR-based auditor that doesn't rely on costly or noisy LLM judges."
  - "For each generated sentence, it extracts the cited chunk and calculates the TF-IDF cosine similarity and IDF-weighted concept coverage."
  - "Every claim is assigned a badge: `[SUPPORTED]` (high semantic and term agreement), `[PARTIALLY_SUPPORTED]`, or `[UNSUPPORTED]`."
  - "This guarantees that clinicians can trace every statement directly to a ranked source chunk."

---

### Segment 5: Evaluation Results, Limitations & Conclusion (5:30 - 6:45)
**Presenter: All Members / Lead**
- **Action:** Show the benchmark summary table in the terminal or report.
- **Talking Points:**
  - "We evaluated all four retrievers on 300 held-out test queries stratified across 15 medical query types."
  - "Dense retrieval achieved the highest MRR (0.6792), while our own Inverted Index achieved P@1 of 0.2800 and Recall@5 of 0.7400. Hybrid RRF achieved a balanced Recall@5 of 0.8500."
  - "In multilingual testing with Hinglish queries, sparse retrieval dropped by ~70% due to vocabulary mismatch, highlighting where dense embeddings bridge the gap."
  - **Limitations:**
    - "Currently, the citation checker evaluates at sentence granularity; multi-sentence complex reasoning claims may benefit from phrase-level parsing."
    - "While our dataset has 25,643 chunks, real clinical settings require live integration with PubMed and EHR databases."
  - "Thank you!"
