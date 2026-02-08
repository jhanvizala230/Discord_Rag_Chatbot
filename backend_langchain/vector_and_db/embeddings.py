from __future__ import annotations

from urllib.parse import urlparse

import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

try:  # Optional dependency
    from langchain_ollama import OllamaEmbeddings
except ImportError:  # pragma: no cover
    OllamaEmbeddings = None

from ..core.config import EMBEDDING_MODEL, EMBEDDING_PROVIDER, OLLAMA_URL
from backend_langchain.core.logger import setup_logger

logger = setup_logger(__name__)
_lc_embeddings: Embeddings | None = None


def _normalize_ollama_base_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.scheme}://{parsed.netloc}"
    return url


def _get_langchain_embeddings() -> Embeddings:
    """Get or initialize the LangChain embeddings model (internal function)."""
    global _lc_embeddings
    if _lc_embeddings is None:
        logger.info(
            "initializing_langchain_embeddings | provider=%s | model_name=%s",
            EMBEDDING_PROVIDER,
            EMBEDDING_MODEL,
        )
        if EMBEDDING_PROVIDER.lower() == "ollama" and OllamaEmbeddings is not None:
            base_url = _normalize_ollama_base_url(OLLAMA_URL)
            _lc_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=base_url)
        else:
            _lc_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        logger.info("langchain_embeddings_initialized")
    return _lc_embeddings


class Embedder:
    """Main embedding manager for handling text, chunks, and flexible input formats"""

    def __init__(self):
        """Initialize embedder with lazy-loaded model"""
        self._embeddings: Embeddings | None = None

    def _ensure_model_loaded(self):
        """Lazy load the model on first use"""
        if self._embeddings is None:
            self._embeddings = _get_langchain_embeddings()

    def embed_texts(self, texts):
        """
        Embed multiple texts using the sentence transformer model.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            numpy array of shape (len(texts), embedding_dim) with dtype float32
        """
        self._ensure_model_loaded()
        logger.info(f"embedding_texts | num_texts={len(texts)}")
        try:
            embeddings = self._embeddings.embed_documents(texts)
            embs = np.array(embeddings, dtype="float32")
            avg_len = sum(len(t) for t in texts) / len(texts) if texts else 0
            logger.debug(
                "texts_embedded | embeddings_shape=%s | avg_text_length=%.2f",
                str(embs.shape),
                avg_len,
            )
            return embs
        except Exception as e:
            logger.error(f"embedding_failed | error={str(e)}")
            raise

    def embed_text(self, text):
        """
        Embed a single text.
        
        Args:
            text: String to embed
            
        Returns:
            numpy array of shape (embedding_dim,) with dtype float32
        """
        logger.info(f"embedding_single_text | text_length={len(text)}")
        self._ensure_model_loaded()
        embedding = self._embeddings.embed_query(text)
        return np.array(embedding, dtype="float32")

    def embed_chunks(self, chunks):
        """
        Embed chunks with flexible input handling.
        Supports both:
        - List of strings: ["text1", "text2", ...]
        - List of dicts with 'text' field: [{"text": "...", "metadata": ...}, ...]
        
        Args:
            chunks: List of strings or list of dicts with 'text' field
            
        Returns:
            numpy array of embeddings with shape (len(chunks), embedding_dim)
        """
        logger.info(f"Embedder.embed_chunks | num_chunks={len(chunks)}")
        try:
            # Extract text from dict chunks or use as-is for string chunks
            if chunks and isinstance(chunks[0], dict) and "text" in chunks[0]:
                texts = [c["text"] for c in chunks]
            else:
                texts = chunks

            embeddings = self.embed_texts(texts)
            logger.info(f"Embedder.embed_chunks | completed | shape={embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Embedder.embed_chunks failed | error={str(e)}")
            raise


_embedder_instance = Embedder()


def get_langchain_embeddings() -> Embeddings:
    """Return the singleton LangChain embeddings instance."""
    return _get_langchain_embeddings()


def embed_texts(texts):
    """Convenience function - embed multiple texts."""
    return _embedder_instance.embed_texts(texts)


def embed_text(text):
    """Convenience function - embed single text."""
    return _embedder_instance.embed_text(text)


# Global embedder instance for convenience
# _embedder_instance = None


# def get_embedder():
#     """Get singleton embedder instance"""
#     global _embedder_instance
#     if _embedder_instance is None:
#         _embedder_instance = Embedder()
#     return _embedder_instance


# # Convenience functions for backward compatibility
# def embed_texts(texts):
#     """Convenience function - embed multiple texts"""
#     return get_embedder().embed_texts(texts)


# def embed_text(text):
#     """Convenience function - embed single text"""
#     return get_embedder().embed_text(text)


# def embed_chunks(chunks):
#     """Convenience function - embed chunks"""
#     return get_embedder().embed_chunks(chunks)
