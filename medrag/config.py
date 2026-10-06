"""Configuration file for Trustworthy Medical RAG on MedQuAD.
CSD358 IR Hackathon - Track 1.
"""
from pathlib import Path
import os

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Serving / API settings (override via environment variables)
FRONTEND_DIR = Path(os.environ.get("FRONTEND_DIR", BASE_DIR / "frontend"))
CORS_ORIGINS = [
    o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()
]

# Data paths
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "train.csv"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_PATH = PROCESSED_DIR / "medquad_chunks.json"
TEST_QUERIES_PATH = PROCESSED_DIR / "test_queries.json"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
EMBEDDINGS_CACHE_PATH = EMBEDDINGS_DIR / "dense_embeddings.npy"
CHUNK_IDS_PATH = EMBEDDINGS_DIR / "chunk_ids.json"
INDEX_CACHE_PATH = PROCESSED_DIR / "inverted_index.pkl"

# Chunking settings (matching Fedallah & Inik, 2026 base paper)
CHUNK_WORD_SIZE = 200
CHUNK_OVERLAP_WORDS = 30

# Evaluation settings
TEST_SET_SIZE = 300
RANDOM_SEED = 42

# IR Settings
DEFAULT_TOP_K = 5
BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 60

# Dense Embedding Models
DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MULTILINGUAL_DENSE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Refusal Thresholds (normalized similarity or min score below which system refuses)
# In RRF or Cosine space, minimum threshold for answering
COSINE_REFUSAL_THRESHOLD = 0.28
BM25_REFUSAL_THRESHOLD = 8.0
DENSE_REFUSAL_THRESHOLD = 0.35
RRF_REFUSAL_THRESHOLD = 0.015

REFUSAL_MESSAGE = (
    "I do not have enough verified medical information in the knowledge base "
    "to answer this question reliably. Please consult a qualified healthcare professional."
)

# LLM Generator Settings
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
DEFAULT_GENERATOR_MODEL = "llama-3.3-70b-versatile"
SUPPORTED_GENERATOR_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mistral-saba-24b",
    "mock-offline"
]
GENERATOR_TEMPERATURE = 0.3
GENERATOR_MAX_TOKENS = 500

# Citation Verification Thresholds
CITATION_COSINE_SUPPORT_THRESHOLD = 0.25
CITATION_UNSUPPORTED_THRESHOLD = 0.12
