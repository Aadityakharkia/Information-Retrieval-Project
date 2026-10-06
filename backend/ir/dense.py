"""
backend/ir/dense.py
Dense Embedding Retriever for HealthNest.
Implements:
1. Dense vector encoding using sentence-transformers (default: paraphrase-multilingual-MiniLM-L12-v2).
2. Question-prepended text embedding: f"{chunk['question']} {chunk['text']}".
3. Vector L2 normalization for fast cosine similarity via numpy matmul.
4. Top-K search via numpy argpartition/argsort.
5. Caching & loading of pre-computed embeddings (.npy file).

Lecture Concepts: Semantic Search, Dense Vector Space, Cosine Similarity via Matrix Operations.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import config

logger = logging.getLogger(__name__)


class DenseRetriever:
    def __init__(
        self,
        model_name: str = config.DENSE_MODEL_NAME,
        cache_path: Path = config.EMBEDDINGS_CACHE_PATH
    ):
        self.model_name = model_name
        self.cache_path = cache_path
        self._model: Optional[SentenceTransformer] = None
        self.embeddings: Optional[np.ndarray] = None
        self.chunk_ids: List[str] = []
        self.chunks_map: Dict[str, Dict[str, Any]] = {}

    def _get_model(self) -> SentenceTransformer:
        """Lazy loader for SentenceTransformer model."""
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model '{self.model_name}'...")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def build_and_cache(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: int = config.DENSE_BATCH_SIZE
    ) -> None:
        """Encodes all document chunks, normalizes, and saves to disk."""
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self.chunks_map = {c["chunk_id"]: c for c in chunks}

        texts = [f"{c['question']} {c['text']}" for c in chunks]
        logger.info(f"Encoding {len(texts)} document chunks with {self.model_name}...")

        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # Ensures L2 norm = 1.0 for cosine similarity
        )

        self.embeddings = embeddings.astype(np.float32)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.cache_path, self.embeddings)
        logger.info(f"Saved dense embeddings shape {self.embeddings.shape} to {self.cache_path}")

    def load_cache(self, chunks: List[Dict[str, Any]]) -> bool:
        """Loads pre-computed dense embeddings from .npy cache file."""
        if not self.cache_path.exists():
            return False

        try:
            self.embeddings = np.load(self.cache_path)
            self.chunk_ids = [c["chunk_id"] for c in chunks]
            self.chunks_map = {c["chunk_id"]: c for c in chunks}
            
            if len(self.chunk_ids) != self.embeddings.shape[0]:
                logger.warning("Cache size mismatch with chunks count! Rebuilding needed.")
                return False

            logger.info(f"Loaded dense embeddings shape {self.embeddings.shape} from {self.cache_path}")
            return True
        except Exception as e:
            logger.warning(f"Failed to load dense embeddings cache: {e}")
            return False

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Executes Dense Cosine Similarity search using matrix multiplication.
        """
        if self.embeddings is None or len(self.chunk_ids) == 0:
            return []

        model = self._get_model()
        q_emb = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)[0]

        # Dot product of normalized vectors = Cosine Similarity
        cosine_scores = np.matmul(self.embeddings, q_emb)

        # Top-K indices via argpartition
        top_k = min(top_k, len(cosine_scores))
        if top_k <= 0:
            return []

        partition_idx = np.argpartition(cosine_scores, -top_k)[-top_k:]
        sorted_indices = partition_idx[np.argsort(-cosine_scores[partition_idx])]

        results = []
        for rank, idx in enumerate(sorted_indices, start=1):
            score = float(cosine_scores[idx])
            chunk_id = self.chunk_ids[idx]
            chunk_data = self.chunks_map[chunk_id]

            results.append({
                "chunk_id": chunk_id,
                "score": round(score, 4),
                "rank": rank,
                "zone_scores": {"dense": round(score, 4)},
                "matched_terms": [],  # Semantic search covers full context
                "chunk": chunk_data
            })

        return results
