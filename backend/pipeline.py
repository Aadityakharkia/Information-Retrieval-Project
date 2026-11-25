"""
backend/pipeline.py
Unified HealthNest End-to-End Pipeline Controller.
Connects safety check -> language guess -> Hinglish expansion -> multi-retriever ->
abstention decision -> LLM generator -> sentence fact checker -> assembled response payload.
"""

import re
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import config

from backend.rag.safety import check_safety_guards
from backend.ir.text import ALL_STOP_WORDS, detect_language, tokenize
from backend.ir.hinglish import HinglishExpander
from backend.ir.inverted_index import InvertedIndex
from backend.ir.tfidf import TfIdfRetriever
from backend.ir.bm25 import BM25Retriever
from backend.ir.dense import DenseRetriever
from backend.ir.hybrid import HybridRetriever
from backend.ir.linkgraph import LinkGraph, topic_of
from backend.rag.abstain import evaluate_abstention
from backend.rag.generator import generate_answer
from backend.rag.checker import verify_all_sentences
from backend.data.loader import load_dataset
from backend.data.chunker import chunk_dataset
from backend.memory import prescription, report
from backend.web import wikipedia

logger = logging.getLogger(__name__)

RETRIEVER_TYPES = ("sparse", "bm25", "dense", "hybrid")
MAX_TOP_K = 10


class HealthNestPipeline:
    def __init__(self):
        self.index: Optional[InvertedIndex] = None
        self.dense_retriever: Optional[DenseRetriever] = None
        self.hinglish_expander = HinglishExpander()
        self.chunks: List[Dict[str, Any]] = []
        self.link_graph: Optional[LinkGraph] = None
        self.is_initialized = False

    def initialize(self):
        """Loads cached indexes or builds them if missing."""
        if self.is_initialized:
            return

        logger.info("Initializing HealthNest Pipeline...")

        # 1. Try loading inverted index
        if config.INDEX_CACHE_PATH.exists():
            self.index = InvertedIndex.load(config.INDEX_CACHE_PATH)
            self.chunks = list(self.index.chunks_map.values())
        else:
            logger.info("Index cache missing. Building on the fly...")
            records, _ = load_dataset()
            self.chunks = chunk_dataset(records, strategy=config.DEFAULT_CHUNKER_STRATEGY)
            self.index = InvertedIndex()
            self.index.index_chunks(self.chunks)
            self.index.save(config.INDEX_CACHE_PATH)

        # 2. Init Dense Retriever
        self.dense_retriever = DenseRetriever()
        if not self.dense_retriever.load_cache(self.chunks):
            logger.info("Dense embeddings cache missing. Building on the fly...")
            self.dense_retriever.build_and_cache(self.chunks)

        # 3. Topic link graph + PageRank (about a second to build, so it is not cached)
        self.link_graph = LinkGraph(self.chunks)
        logger.info("Link graph: %d topics, %d links", len(self.link_graph.nodes), self.link_graph.n_edges)

        self.is_initialized = True
        logger.info("HealthNest Pipeline Initialization Complete!")

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------
    def _retrieve(self, expanded_q: str, retriever_type: str, top_k: int, link_mode: str,
                  use_champion_lists: bool = False) -> Tuple[list, list, list, dict]:
        """Returns (results, sparse_results, dense_results, link_info); results are link re-ranked and cut to top_k."""
        depth = top_k if link_mode == "off" else max(config.LINK_POOL, top_k)
        sparse_res, dense_res = [], []
        if retriever_type == "sparse":
            sparse_res = results = TfIdfRetriever(self.index).search(expanded_q, top_k=depth, use_champion_lists=use_champion_lists)
        elif retriever_type == "bm25":
            sparse_res = results = BM25Retriever(self.index).search(expanded_q, top_k=depth)
        elif retriever_type == "dense":
            dense_res = results = self.dense_retriever.search(expanded_q, top_k=depth)
        else:
            fuse_depth = max(top_k * 2, depth)
            sparse_res = TfIdfRetriever(self.index).search(expanded_q, top_k=fuse_depth, use_champion_lists=use_champion_lists)
            dense_res = self.dense_retriever.search(expanded_q, top_k=fuse_depth)
            results = HybridRetriever().fuse(sparse_res, dense_res, top_k=depth, fusion_method="rrf")
        results, link_info = self.link_graph.rerank(results, link_mode)
        return results[:top_k], sparse_res, dense_res, link_info

    @staticmethod
    def _source(n: int, item: Dict[str, Any], origin: str = "database") -> Dict[str, Any]:
        chunk = item.get("chunk", item)
        return {
            "n": n,
            "origin": origin,
            "url": chunk.get("url"),
            "chunk_id": chunk.get("chunk_id"),
            "qa_id": chunk.get("qa_id"),
            "question": chunk.get("question"),
            "text": chunk.get("text"),
            "score": item.get("score", 0.0),
            "zone_scores": item.get("zone_scores", {}),
            "matched_terms": item.get("matched_terms", []),
            "sparse_rank": item.get("sparse_rank"),
            "dense_rank": item.get("dense_rank"),
            "link": item.get("link"),
        }

    @staticmethod
    def _wikipedia_offer(term: str) -> Optional[Dict[str, str]]:
        """What the user is asked to approve. Nothing is sent to Wikipedia until they do."""
        term = " ".join(term.split())[:120]
        return {"query": term, "host": wikipedia.host_for(term)} if term else None

    @staticmethod
    def _search_term(question: str) -> str:
        """Readable default term for the permission prompt ('what is the capital of France?' -> 'capital of France')."""
        q = question.strip(" ?.!।")
        if detect_language(question) == "english":
            term = re.sub(r"^(?:please\s+)?(?:what|who|when|where|why|which|how|can|could|should|do|does|is|are|tell me about)\b"
                          r"(?:\s+(?:is|are|was|were|do|does|did|can|should|to|i|you|we|a|an|the))*\s+", "", q, flags=re.I)
            return term or q
        words = re.findall(r"[\w\u0900-\u097F'-]+", q)
        return " ".join([w for w in words if w.lower() not in ALL_STOP_WORDS] or words)

    def _generate(self, question: str, items: List[Dict[str, Any]], model: str, origin: str, offset: int = 0,
                  added_terms: Optional[List[str]] = None, enable_checker: bool = True,
                  local_only: bool = False) -> Tuple[Optional[list], str]:
        """Generate + verify. Returns (sentences or None when the generator finds no answer, provider).
        Citations are shifted by `offset` so that sources from several origins share one numbering."""
        gen = generate_answer(question, items, model=model, added_terms=added_terms, local_only=local_only)
        if gen["raw_answer"] == "I don't know based on the provided pages.":
            return None, gen["provider"]
        if enable_checker:
            sentences = verify_all_sentences(gen["sentences"], items)
        else:
            sentences = [dict(s, support={"supported": True, "badge": "supported", "reasons": ["Fact checker disabled."]}) for s in gen["sentences"]]
        for s in sentences:
            s["citations"] = [c + offset for c in s["citations"] if 1 <= c <= len(items)]
            s["origin"] = origin
        return sentences, gen["provider"]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def ask(
        self,
        question: str,
        retriever_type: str = "hybrid",  # "sparse", "dense", "hybrid", "bm25"
        top_k: int = 5,
        model: str = config.DEFAULT_MODEL,
        enable_abstain: bool = True,
        enable_checker: bool = True,
        use_champion_lists: bool = False,
        link_mode: str = config.LINK_DEFAULT_MODE,  # "off", "pagerank", "hits", "both"
        patient_docs: Optional[List[Dict[str, Any]]] = None,
        wiki_query: Optional[str] = None,  # set only after the user approved this exact term
    ) -> Dict[str, Any]:
        """
        Executes the complete RAG Q&A pipeline:
        safety -> patient's prescription -> HealthNest database -> (with permission) Wikipedia.
        """
        start_time = time.time()
        self.initialize()

        if retriever_type not in RETRIEVER_TYPES:
            raise ValueError(f"Unknown retriever_type '{retriever_type}'. Choose from {RETRIEVER_TYPES}.")
        if link_mode not in config.LINK_MODES:
            raise ValueError(f"Unknown link_mode '{link_mode}'. Choose from {config.LINK_MODES}.")
        top_k = max(1, min(int(top_k), MAX_TOP_K))
        timings = {}

        def lap(name, since):
            timings[name] = round((time.time() - since) * 1000, 2)

        t0 = time.time()
        safety = check_safety_guards(question)
        lap("safety_ms", t0)
        base = {
            "question": question,
            "disclaimer": config.DISCLAIMER_TEXT,
            "emergency_banner": safety["emergency_banner"],
            "self_harm_message": None,
            "sentences": [],
            "sources": [],
            "timings": timings,
            "retriever": retriever_type,
            "link_mode": link_mode,
            "model": model,
            "memory_used": False,
            "wikipedia_offer": None,
        }

        if safety["is_self_harm"]:
            return {**base, "status": "self_harm", "self_harm_message": safety["self_harm_message"], "emergency_banner": None,
                    "abstain": {"should_abstain": False, "reason": "Self-harm safety protocol triggered."}}

        answered = "emergency" if safety["is_emergency"] else "answered"
        t0 = time.time()
        base["language"] = detect_language(question)
        memory = prescription.analyse(question, patient_docs or [])
        base["memory_used"] = memory["used"]
        base["local_only"] = memory["used"]
        lap("preproc_ms", t0)

        if wiki_query:
            result = self._ask_wikipedia(question, wiki_query, base, answered, model, top_k, enable_checker, memory["used"])
        elif memory["used"]:
            result = self._ask_memory(question, memory, base, answered, retriever_type, model, link_mode, enable_checker)
        else:
            result = self._ask_database(question, base, answered, retriever_type, top_k, model, link_mode,
                                        enable_abstain, enable_checker, use_champion_lists, lap)
        lap("total_ms", start_time)
        return result

    def _ask_database(self, question, base, answered, retriever_type, top_k, model, link_mode,
                      enable_abstain, enable_checker, use_champion_lists, lap) -> Dict[str, Any]:
        expanded_q, added_terms = self.hinglish_expander.expand_query(question)
        concept_groups = self.hinglish_expander.get_concept_groups(question)

        t0 = time.time()
        results, sparse_res, dense_res, link_info = self._retrieve(expanded_q, retriever_type, top_k, link_mode, use_champion_lists)
        lap("retrieval_ms", t0)
        sources = [self._source(i, item) for i, item in enumerate(results, 1)]
        out = {
            **base,
            "expanded_query": expanded_q,
            "added_synonyms": added_terms,
            "sources": sources,
            "inspector": {
                "sparse_top5": [s["chunk"]["chunk_id"] for s in sparse_res[:5]],
                "dense_top5": [d["chunk"]["chunk_id"] for d in dense_res[:5]],
                "fused_top5": [s["chunk_id"] for s in sources[:5]],
                "link_analysis": link_info,
            },
        }

        t0 = time.time()
        decision = evaluate_abstention(question, results, added_terms=added_terms, concept_groups=concept_groups,
                                       dense_top_score=dense_res[0]["score"] if dense_res else None)
        lap("abstain_ms", t0)
        out["abstain"] = decision
        offer = self._wikipedia_offer(self._search_term(question))
        if enable_abstain and decision["should_abstain"]:
            return {**out, "status": "abstained", "wikipedia_offer": offer}

        t0 = time.time()
        sentences, provider = self._generate(question, results, model, "database", added_terms=added_terms, enable_checker=enable_checker)
        lap("generation_ms", t0)
        out["generator"] = provider
        if sentences is None:
            out["abstain"] = {**decision, "should_abstain": True, "reason": "LLM indicated insufficient reference evidence."}
            return {**out, "status": "abstained", "wikipedia_offer": offer}
        return {**out, "status": answered, "sentences": sentences}

    def _ask_memory(self, question, memory, base, answered, retriever_type, model, link_mode, enable_checker) -> Dict[str, Any]:
        """Prescription lines verbatim, then (if general knowledge is asked for) what the database knows about each medicine."""
        sources, sentences = [], []
        for doc in memory["sources"]:
            n = len(sources) + 1
            label = "report" if doc["kind"] == "report" else "prescription"
            text = "\n".join(line["text"] for line in doc["lines"])
            if doc["hidden"]:
                text += f"\n({doc['hidden']} more lines in this PDF are not shown)"
            sources.append({"n": n, "origin": doc["kind"], "url": None, "chunk_id": f"{label}_{doc['doc_id']}", "qa_id": None,
                            "question": f"Your {label}: {doc['filename']}", "text": text, "score": None,
                            "zone_scores": {}, "matched_terms": [], "sparse_rank": None, "dense_rank": None, "link": None})
            for line in doc["lines"]:
                sentences.append({"sentence_id": len(sentences) + 1, "text": line["text"], "raw_text": line["text"], "citations": [n],
                                  "is_valid_citation": True, "origin": doc["kind"], "note": line["note"],
                                  "support": {"supported": True, "badge": "verbatim", "reasons": ["Copied word for word from your PDF."]}})

        missing, provider, link_info = [], None, {}
        if memory["knowledge"]:
            for med in memory["topics"][:5]:
                stems = set(tokenize(med))
                is_test = memory["topic_kind"] == "test"
                lookup = med if is_test else prescription.lookup_query(question, med)
                expanded, _ = self.hinglish_expander.expand_query(lookup)
                results, _, _, med_links = self._retrieve(expanded, retriever_type, MAX_TOP_K, link_mode)
                # A passage counts only if it actually names this medicine or test.
                hits_ = [r for r in results if stems & set(tokenize(f"{r['chunk']['question']} {r['chunk']['text']}"))]
                if is_test:
                    hits_ = [r for r in hits_ if report.page_is_about(med, topic_of(r["chunk"]["question"]))]
                hits_ = hits_[:2]
                found = None
                if hits_:
                    found, provider = self._generate(lookup, hits_, model, "database", offset=len(sources),
                                                     enable_checker=enable_checker, local_only=True)
                if found:
                    link_info = med_links
                    for r in hits_:
                        sources.append(self._source(len(sources) + 1, r))
                    sentences.extend(found)
                else:
                    missing.append(med)

        offer = None
        if missing:
            offer = self._wikipedia_offer(missing[0] if memory["topic_kind"] == "test" else prescription.lookup_query(question, missing[0]))
        return {**base, "status": answered, "sentences": sentences, "sources": sources, "generator": provider or "verbatim",
                "missing_medicines": missing, "remarks": memory.get("remarks", []), "wikipedia_offer": offer, "expanded_query": question, "added_synonyms": [],
                "inspector": {"sparse_top5": [], "dense_top5": [], "fused_top5": [s["chunk_id"] for s in sources][:5], "link_analysis": link_info},
                "abstain": {"should_abstain": False, "reason": "Answered from the patient's own documents.", "metrics": {}}}

    def _ask_wikipedia(self, question, wiki_query, base, answered, model, top_k, enable_checker, local_only) -> Dict[str, Any]:
        expanded, added = self.hinglish_expander.expand_query(wiki_query)
        out = {**base, "expanded_query": expanded, "added_synonyms": added, "wikipedia_query": wiki_query,
               "inspector": {"sparse_top5": [], "dense_top5": [], "fused_top5": [], "link_analysis": {}}}
        try:
            found = wikipedia.search(wiki_query, expanded, top_k)
        except wikipedia.WikipediaError as exc:
            return {**out, "status": "abstained", "abstain": {"should_abstain": True, "reason": str(exc), "metrics": {}}}
        results = found["results"]
        sources = [self._source(i, r, "wikipedia") for i, r in enumerate(results, 1)]
        out.update(sources=sources, wikipedia_pages=found["pages"])
        out["inspector"]["fused_top5"] = [s["chunk_id"] for s in sources[:5]]
        # BM25 scores over a handful of pages are tiny (every page mentions the topic), so the corpus score
        # threshold does not apply here; term coverage still does.
        decision = evaluate_abstention(wiki_query, results, score_threshold=0.0,
                                       concept_groups=self.hinglish_expander.get_concept_groups(wiki_query))
        out["abstain"] = decision
        if decision["should_abstain"]:
            return {**out, "status": "abstained"}
        # With a prescription involved only the approved term is used, never the personal question.
        sentences, provider = self._generate(wiki_query if local_only else question, results, model, "wikipedia",
                                             added_terms=added, enable_checker=enable_checker, local_only=local_only)
        out["generator"] = provider
        if sentences is None:
            out["abstain"] = {**decision, "should_abstain": True, "reason": "The Wikipedia pages do not answer this question."}
            return {**out, "status": "abstained"}
        return {**out, "status": answered, "sentences": sentences}

    def search(self, query: str, retriever_type: str = "hybrid", top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieval-only search (no generation / abstention)."""
        self.initialize()
        if retriever_type not in RETRIEVER_TYPES:
            raise ValueError(f"Unknown retriever_type '{retriever_type}'. Choose from {RETRIEVER_TYPES}.")
        top_k = max(1, min(int(top_k), MAX_TOP_K))
        expanded_q, _ = self.hinglish_expander.expand_query(query)

        if retriever_type == "sparse":
            return TfIdfRetriever(self.index).search(expanded_q, top_k=top_k)
        if retriever_type == "bm25":
            return BM25Retriever(self.index).search(expanded_q, top_k=top_k)
        if retriever_type == "dense":
            return self.dense_retriever.search(expanded_q, top_k=top_k)
        sparse_res = TfIdfRetriever(self.index).search(expanded_q, top_k=top_k * 2)
        dense_res = self.dense_retriever.search(expanded_q, top_k=top_k * 2)
        return HybridRetriever().fuse(sparse_res, dense_res, top_k=top_k, fusion_method="rrf")


    def suggest_queries(self, prefix: str, limit: int = 8) -> List[str]:
        """
        Comprehensive IR Auto-Suggest Engine using Prefix Match & IR Term Vocabulary.
        Matches query prefixes against:
        1. Concise health search terms & completions (e.g., diabetes, fever, stomach ache)
        2. Core medical Q&A database questions
        3. Hinglish lexicon mappings (e.g. pet me dard, bukhar, sir dard)
        4. Dataset document vocabulary terms
        """
        if not prefix or len(prefix.strip()) < 1:
            return []

        self.initialize()
        clean_pfx = prefix.strip().lower()
        suggestions = []
        seen = set()

        import re

        # Concise Health Search Terms & Standard Completions
        short_completions = [
            "diabetes",
            "diabetes symptoms",
            "what is diabetes",
            "diabetes treatment & diet",
            "fever remedies and treatment",
            "fever dosage for adults",
            "bukhar (fever)",
            "stomach ache and acidity remedies",
            "pet me dard (stomach pain)",
            "headache relief and causes",
            "sir me dard (headache)",
            "high blood pressure management",
            "insomnia and sleep remedies",
            "skin rash and itching relief",
            "paracetamol dosage",
            "amoxicillin side effects",
            "asthma symptoms and treatment",
            "stroke warning signs"
        ]

        # Preset Core Medical Questions
        core_medical_queries = [
            "What are the early symptoms of diabetes?",
            "What are the common remedies for fever and body ache?",
            "What is the recommended dosage for Paracetamol in adults?",
            "How to manage high blood pressure naturally?",
            "What are common remedies for severe stomach ache and gastritis?",
            "How to improve sleep quality and overcome insomnia?",
            "What are the side effects of Amoxicillin antibiotic?",
            "What are the warning signs of a stroke?",
            "How to treat skin rashes and itching at home?",
            "How to reduce high cholesterol levels through diet?",
            "What causes joint pain in fingers and knees?",
            "What are the primary causes and treatments for asthma attacks?"
        ]

        # 1. Match short query completions where a word starts with prefix
        for item in short_completions:
            words = item.lower().split()
            if any(w.startswith(clean_pfx) for w in words) or item.lower().startswith(clean_pfx):
                if item not in seen:
                    seen.add(item)
                    suggestions.append(item)
                    if len(suggestions) >= limit:
                        return suggestions

        # 2. Match core medical questions
        for q_item in core_medical_queries:
            q_words = q_item.lower().split()
            if any(w.startswith(clean_pfx) for w in q_words) or clean_pfx in q_item.lower():
                if q_item not in seen:
                    seen.add(q_item)
                    suggestions.append(q_item)
                    if len(suggestions) >= limit:
                        return suggestions

        # 3. Match Hinglish lexicon terms & translations
        if len(suggestions) < limit and hasattr(self, 'hinglish_expander'):
            for src_phrase, eng_terms in self.hinglish_expander.term_map.items():
                if src_phrase.startswith(clean_pfx) or any(t.startswith(clean_pfx) for t in eng_terms):
                    phrase_label = f"{src_phrase} ({', '.join(eng_terms)})"
                    if src_phrase not in seen:
                        seen.add(src_phrase)
                        suggestions.append(src_phrase)
                        if len(suggestions) >= limit:
                            return suggestions

        # 4. Match dataset question titles
        if len(suggestions) < limit:
            for chunk in self.chunks:
                q_title = chunk.get("question", "")
                clean_title = re.sub(r'\s*\(Case\s*#\d+\)', '', q_title).strip()
                t_words = clean_title.lower().split()
                if any(w.startswith(clean_pfx) for w in t_words):
                    if clean_title not in seen:
                        seen.add(clean_title)
                        suggestions.append(clean_title)
                        if len(suggestions) >= limit:
                            return suggestions

        # 5. Match vocabulary terms across corpus
        if len(suggestions) < limit:
            for chunk in self.chunks:
                words = chunk.get("text", "").split() + chunk.get("question", "").split()
                for w in words:
                    clean_w = "".join(c for c in w.lower() if c.isalnum())
                    if clean_w.startswith(clean_pfx) and len(clean_w) >= len(clean_pfx) and clean_w not in seen:
                        seen.add(clean_w)
                        suggestions.append(clean_w)
                        if len(suggestions) >= limit:
                            return suggestions

        return suggestions[:limit]


# Singleton pipeline instance
_pipeline_instance = HealthNestPipeline()


def get_pipeline() -> HealthNestPipeline:
    return _pipeline_instance


def suggest(prefix: str, limit: int = 6) -> List[str]:
    return _pipeline_instance.suggest_queries(prefix, limit=limit)
