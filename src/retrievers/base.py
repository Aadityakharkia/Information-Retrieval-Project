"""Abstract base class for all retrievers.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple


class BaseRetriever(ABC):
    """Abstract interface for all IR retrieval algorithms."""
    
    @abstractmethod
    def build_index(self, chunks: List[Dict[str, Any]]) -> None:
        """Build or load the retrieval index over corpus chunks."""
        pass
        
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top_k relevant chunks for a given query.
        
        Returns:
            List of dictionaries containing:
            - chunk_id: str
            - score: float
            - text: str
            - doc_id: str
            - qtype: str
            - rank: int
        """
        pass

    @abstractmethod
    def explain(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Provide detailed, inspectable intermediate IR diagnostics for the query.
        
        Returns:
            Dictionary containing intermediate weights, postings, accumulator state, etc.
        """
        pass
