"""
scripts/build_index.py
Index Builder & Artifact Cache Script for HealthNest.
Executes end-to-end indexing pipeline:
1. Loads dataset CSV
2. Chunks documents
3. Builds & caches Inverted Index (.pkl)
4. Encodes & caches Dense Embeddings (.npy)
5. Saves chunks JSON metadata
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import config
from backend.data.loader import load_dataset
from backend.data.chunker import chunk_dataset
from backend.ir.inverted_index import InvertedIndex
from backend.ir.dense import DenseRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_all_indexes(strategy: str = config.DEFAULT_CHUNKER_STRATEGY):
    logger.info("=" * 60)
    logger.info("BUILDING HEALTHNEST INDEXES AND ARTIFACTS")
    logger.info("=" * 60)

    # 1. Load dataset
    records, stats = load_dataset()
    logger.info(f"Loaded {stats['valid_records']} valid records.")

    # 2. Chunk dataset
    chunks = chunk_dataset(records, strategy=strategy)
    logger.info(f"Generated {len(chunks)} chunks using strategy '{strategy}'.")

    # Cache chunks JSON
    config.CHUNKS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(config.CHUNKS_CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved chunks metadata to {config.CHUNKS_CACHE_PATH}")

    # 3. Build & save Inverted Index
    index = InvertedIndex()
    index.index_chunks(chunks)
    index.save(config.INDEX_CACHE_PATH)

    # 4. Encode & save Dense Embeddings
    dense_retriever = DenseRetriever()
    dense_retriever.build_and_cache(chunks)

    logger.info("=" * 60)
    logger.info("ALL INDEXES AND EMBEDDINGS BUILT SUCCESSFULLY!")
    logger.info("=" * 60)


if __name__ == "__main__":
    strategy = sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_CHUNKER_STRATEGY
    build_all_indexes(strategy)
