"""
HealthNest - Global Configuration File
All settings, parameters, thresholds, and paths are centrally controlled here.
No magic numbers allowed in other files!
"""

import os
from pathlib import Path


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (KEY=VALUE per line); real environment variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(Path(__file__).parent.resolve() / ".env")

# Base Paths
BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LEXICON_PATH = DATA_DIR / "lexicon_hinglish_health.tsv"

RAW_CSV_PATH = RAW_DATA_DIR / "health_qa.csv"
INDEX_CACHE_PATH = PROCESSED_DATA_DIR / "inverted_index.pkl"
EMBEDDINGS_CACHE_PATH = PROCESSED_DATA_DIR / "dense_embeddings.npy"
CHUNKS_CACHE_PATH = PROCESSED_DATA_DIR / "chunks.json"

EVAL_DIR = BASE_DIR / "eval"
EVAL_RESULTS_DIR = EVAL_DIR / "results"

# Data Loader Mapping (Flexible for various Kaggle Health Q&A datasets)
CSV_COLUMN_MAPPING = {
    "qa_id": ["id", "qa_id", "question_id", "index"],
    "question": ["question", "Query", "Question", "q", "title"],
    "answer": ["answer", "Response", "Answer", "a", "description"],
    "category": ["category", "Category", "tag", "topic", "department"],
    "source": ["source", "Source", "url", "doctor"]
}

# Chunking Configuration
# Options: 'whole_qa', 'fixed_words', 'sentence_window'
DEFAULT_CHUNKER_STRATEGY = "sentence_window"
FIXED_WORDS_SIZE = 100
FIXED_WORDS_OVERLAP = 20
SENTENCE_WINDOW_MAX_WORDS = 80
SENTENCE_WINDOW_OVERLAP_SENTENCES = 1

# Text Processing & IR Parameters
USE_PORTER_STEMMER = True
USE_STOP_WORDS = True

# Zone Weights for Multi-zone IR
ZONE_WEIGHTS = {
    "question": 0.6,
    "body": 0.4
}

# TF-IDF SMART Scheme: lnc.ltc
# Document: l (log tf), n (no idf), c (cosine norm)
# Query: l (log tf), t (idf), c (cosine norm)
CHAMPION_LIST_R = 50
INDEX_ELIMINATION_IDF_THRESHOLD = 0.5  # Skip terms with idf below this when enabled

# BM25 Parameters
BM25_K1 = 1.2
BM25_B = 0.75

# Dense Retrieval Parameters
# paraphrase-multilingual-MiniLM-L12-v2 supports 50+ languages including Hindi/Devanagari
DENSE_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DENSE_BATCH_SIZE = 32

# Hybrid Fusion Parameters
RRF_K = 60
HYBRID_ALPHA = 0.5  # Weight for sparse in weighted sum fusion (1-alpha for dense)

# RAG & LLM Configuration
# Provider order: Groq (if GROQ_API_KEY set) -> local Ollama -> offline extractive mock.
# Ollama model tags contain ':' (e.g. llama3.2:1b); Groq model ids do not.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]  # verify via GET /openai/v1/models
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODELS = ["llama3.2:1b", "llama3.2:3b", "llama3.1:8b"]
MOCK_MODEL = "mock-demo"
AVAILABLE_MODELS = (GROQ_MODELS if GROQ_API_KEY else []) + OLLAMA_MODELS + [MOCK_MODEL]
DEFAULT_MODEL = GROQ_MODELS[0] if GROQ_API_KEY else OLLAMA_MODELS[0]
LLM_TEMPERATURE = 0.0
LLM_TIMEOUT_SECONDS = 20

# Abstention Thresholds (Tuned on dev set)
ABSTAIN_SCORE_THRESHOLD = 0.12
ABSTAIN_QUERY_COVERAGE_THRESHOLD = 0.40

# Answer Verification Checker Thresholds
CHECKER_SPARSE_COSINE_MIN = 0.20
CHECKER_KEY_TERM_COVERAGE_MIN = 0.35
CHECKER_DENSE_COSINE_MIN = 0.40

# Health Safety & Emergency Keywords
DISCLAIMER_TEXT = "Information from a Q&A dataset, not medical advice."

EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "can't breathe", "cannot breathe",
    "shortness of breath", "severe bleeding", "unconscious", "stroke",
    "poisoning", "seizure", "anaphylaxis", "severe burn", "head injury",
    "dilated pupils", "coughing blood"
]
EMERGENCY_BANNER_TEXT = "⚠️ EMERGENCY WARNING: If this is a medical emergency, immediately contact local emergency services (112 in India) or visit the nearest hospital emergency room."

SELF_HARM_KEYWORDS = [
    "suicide", "kill myself", "end my life", "self harm", "want to die",
    "cutting myself", "overdose"
]
SELF_HARM_SAFETY_MESSAGE = "💙 Help is available. You are not alone. Please reach out to emergency services (112 in India), a trusted person, or a crisis helpline immediately."
