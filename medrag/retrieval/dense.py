"""Dense Neural Retriever using Sentence-Transformers and FAISS.
Implements:
1. all-MiniLM-L6-v2 embedding model (Fedallah & Inik, 2026 base paper)
2. Multilingual embedding support for Hindi/Hinglish queries
3. Disk caching of corpus dense embeddings
4. Cosine similarity search via FAISS IndexFlatIP
"""
import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

import faiss
from sentence_transformers import SentenceTransformer

from medrag.retrieval.base import BaseRetriever
from medrag.config import (
    DEFAULT_DENSE_MODEL,
    MULTILINGUAL_DENSE_MODEL,
    EMBEDDINGS_DIR,
    EMBEDDINGS_CACHE_PATH,
    CHUNK_IDS_PATH
)

logger = logging.getLogger(__name__)


class DenseRetriever(BaseRetriever):
    """Dense retriever using pretrained SentenceTransformer embeddings and FAISS index."""
    
    def __init__(self, model_name: str = DEFAULT_DENSE_MODEL):
        self.model_name = model_name
        self.model: Optional[SentenceTransformer] = None
        self.chunks: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.IndexFlatIP] = None
        self.is_built: bool = False
        
    def _load_model(self) -> None:
        if self.model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            
    def build_index(self, chunks: List[Dict[str, Any]], force_recompute: bool = False) -> None:
        """Build or load dense index from disk."""
        self.chunks = chunks
        EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Check cache
        if not force_recompute and EMBEDDINGS_CACHE_PATH.exists() and CHUNK_IDS_PATH.exists():
            try:
                with open(CHUNK_IDS_PATH, "r", encoding="utf-8") as f:
                    cached_chunk_ids = json.load(f)
                current_chunk_ids = [c["chunk_id"] for c in chunks]
                
                if cached_chunk_ids == current_chunk_ids:
                    logger.info(f"Loading cached embeddings from {EMBEDDINGS_CACHE_PATH}...")
                    self.embeddings = np.load(str(EMBEDDINGS_CACHE_PATH))
                    dim = self.embeddings.shape[1]
                    self.index = faiss.IndexFlatIP(dim)
                    self.index.add(self.embeddings)
                    self.is_built = True
                    logger.info(f"Loaded {self.embeddings.shape[0]} cached embeddings (dim={dim}).")
                    return
            except Exception as e:
                logger.warning(f"Error loading dense cache: {e}. Recomputing...")
                
        self._load_model()
        logger.info(f"Encoding {len(chunks)} chunks with {self.model_name}...")
        texts = [c["text"] for c in chunks]
        
        # Encode with batching and progress bar
        # sentence-transformers encodes and normalizes embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")
        
        self.embeddings = embeddings
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        self.is_built = True
        
        # Save cache
        try:
            logger.info(f"Saving embeddings cache to {EMBEDDINGS_CACHE_PATH}...")
            np.save(str(EMBEDDINGS_CACHE_PATH), self.embeddings)
            chunk_ids = [c["chunk_id"] for c in chunks]
            with open(CHUNK_IDS_PATH, "w", encoding="utf-8") as f:
                json.dump(chunk_ids, f)
            logger.info("Embeddings successfully cached.")
        except Exception as e:
            logger.warning(f"Failed to cache embeddings: {e}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top_k chunks using cosine similarity."""
        if not self.is_built:
            raise RuntimeError("Dense index has not been built.")
            
        self._load_model()
        query_vec = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        
        # Search FAISS index
        scores, indices = self.index.search(query_vec, top_k)
        
        results = []
        for rank, (score, c_idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if c_idx < 0 or c_idx >= len(self.chunks):
                continue
            chunk = self.chunks[c_idx]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "doc_id": chunk["doc_id"],
                "qtype": chunk["qtype"],
                "chunk_index": chunk["chunk_index"],
                "score": float(round(float(score), 4)),
                "text": chunk["text"],
                "rank": rank
            })
            
        return results

    def explain(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Explain dense retrieval results."""
        top_results = self.retrieve(query, top_k=top_k)
        dim = self.embeddings.shape[1] if self.embeddings is not None else 384
        
        return {
            "retriever": "Dense (SentenceTransformer + FAISS IndexFlatIP)",
            "model_name": self.model_name,
            "embedding_dimension": dim,
            "metric": "Cosine Similarity (Inner Product of L2-normalized vectors)",
            "query": query,
            "top_results": top_results
        }
