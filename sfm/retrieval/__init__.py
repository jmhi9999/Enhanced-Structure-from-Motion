"""
Image retrieval utilities built on top of DINO CLS embeddings.
"""

from .dino_retriever import DINOCLSRetriever, RetrievalResult

__all__ = ["DINOCLSRetriever", "RetrievalResult"]
