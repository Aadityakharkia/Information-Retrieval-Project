# HealthNest - IR Concept Map

This document maps core Information Retrieval (IR) lecture concepts to their exact implementation files, classes, and functions in the HealthNest codebase.

| Lecture Concept | Description | Codebase Location | Function / Class |
| :--- | :--- | :--- | :--- |
| **Document Definition & Chunking** | Defining what constitutes a document unit (whole Q&A vs sentence windowing) | [`backend/data/chunker.py`](file:///c:/Users/Aditya%20Chauhan/Desktop/Information-Retrieval-Project/backend/data/chunker.py) | `create_chunks_for_record()` |
| **Unicode Normalization** | NFKC normalization for multi-script handling | [`backend/ir/text.py`](file:///c:/Users/Aditya%20Chauhan/Desktop/Information-Retrieval-Project/backend/ir/text.py) | `normalize_text()` |
| **Devanagari Regex Tokenization** | Token splitting with Devanagari vowel sign support `[\w\u0900-\u097F]+` | [`backend/ir/text.py`](file:///c:/Users/Aditya%20Chauhan/Desktop/Information-Retrieval-Project/backend/ir/text.py) | `tokenize()` |
| **Stop-words & Stemming** | English & Hinglish stop-words removal; Porter Stemmer | [`backend/ir/text.py`](file:///c:/Users/Aditya%20Chauhan/Desktop/Information-Retrieval-Project/backend/ir/text.py) | `ALL_STOP_WORDS`, `tokenize()` |
| **Synonym Expansion & Vocab Normalization** | Lexicon-based query expansion for Roman Hindi / Hinglish medical terms | [`backend/ir/hinglish.py`](file:///c:/Users/Aditya%20Chauhan/Desktop/Information-Retrieval-Project/backend/ir/hinglish.py) | `HinglishExpander.expand_query()` |
| **Multi-Zone Inverted Index** | Dictionary & Postings Lists separated into `question` and `body` zones | [`backend/ir/inverted_index.py`](file:///c:/Users/Aditya%20Chauhan/Desktop/Information-Retrieval-Project/backend/ir/inverted_index.py) | `ZoneIndex`, `InvertedIndex` |
| **SMART lnc.ltc Vector Space Model** | Document: log tf, no idf, cosine norm. Query: log tf, idf, cosine norm | [`backend/ir/tfidf.py`](file:///c:/Users/Aditya%20Chauhan/Desktop/Information-Retrieval-Project/backend/ir/tfidf.py) | `TfIdfRetriever.search()` |
| **Term-at-a-Time (TAAT) Scoring** | Efficient accumulator-based cosine scoring | [`backend/ir/tfidf.py`](file:///c:/Users/Aditya%20Chauhan/Desktop/Information-Retrieval-Project/backend/ir/tfidf.py) | `TfIdfRetriever.search()` |
| **Champion Lists & Index Elimination** | Query optimization via top-r posting lists and low-IDF term pruning | [`backend/ir/tfidf.py`](file:///c:/Users/Aditya%20Chauhan/Desktop/Information-Retrieval-Project/backend/ir/tfidf.py) | `use_champion_lists`, `use_index_elimination` |
| **Okapi BM25 Probabilistic IR** | BM25 term saturation & document length normalization ($k_1=1.2, b=0.75$) | [`backend/ir/bm25.py`](file:///c:/Users/Aditya%20Chauhan/Desktop/Information-Retrieval-Project/backend/ir/bm25.py) | `BM25Retriever.search()` |
| **Dense Semantic Vector Search** | Multilingual sentence embeddings & numpy matrix dot product cosine | [`backend/ir/dense.py`](file:///c:/Users/Aditya%20Chauhan/Desktop/Information-Retrieval-Project/backend/ir/dense.py) | `DenseRetriever.search()` |
| **Reciprocal Rank Fusion (RRF)** | Hybrid rank aggregation combining sparse and dense rankings ($k=60$) | [`backend/ir/hybrid.py`](file:///c:/Users/Aditya%20Chauhan/Desktop/Information-Retrieval-Project/backend/ir/hybrid.py) | `HybridRetriever.fuse()` |
| **Selective Answering & Abstention** | Abstaining ("I don't know") on weak evidence / low term coverage | [`backend/rag/abstain.py`](file:///c:/Users/Aditya%20Chauhan/Desktop/Information-Retrieval-Project/backend/rag/abstain.py) | `evaluate_abstention()` |
| **Sentence Fact Verification** | Sentence-level sparse cosine, key-term coverage, & dosage number check | [`backend/rag/checker.py`](file:///c:/Users/Aditya%20Chauhan/Desktop/Information-Retrieval-Project/backend/rag/checker.py) | `check_sentence_verification()` |
