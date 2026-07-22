"""
Local embedding model wrapper.

Why wrap sentence-transformers instead of calling it directly everywhere:
    Every module that needs vectors (ingestion, retriever agent, query-time
    search) should go through one place. If we ever swap the embedding
    model (e.g. a larger one for better recall, or a domain-specific one),
    only this file changes -- nothing that calls encode() needs to know
    the model name or its output dimension.

Why sentence-transformers + local model instead of an embedding API:
    No per-call cost, no network dependency at embed time, and no rate
    limits during bulk ingestion of many PDF chunks. The tradeoff is
    embedding quality is capped by the local model's size -- fine for a
    portfolio project, and swappable later (see above) if needed.
"""

from functools import lru_cache
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings
from app.logging_config import logger


class EmbeddingModel:
    """Thin wrapper exposing embed_texts() / embed_query() with a stable
    dimension property, so FAISS index creation (Module 3's vector_store.py)
    doesn't hardcode a dimension number anywhere."""

    def __init__(self, model_name: str):
        logger.info("Loading embedding model '{model_name}'...", model_name=model_name)
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model loaded (dimension={dim})", dim=self._dimension
        )

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of document chunks for indexing.
        normalize_embeddings=True makes cosine similarity equivalent to a
        plain dot product, which lets us use FAISS's faster IndexFlatIP
        instead of needing a separate cosine-specific index type."""
        return self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string at retrieval time."""
        return self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0]


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    """Cached singleton -- the model is loaded from disk/HF cache once per
    process, not once per request. Loading it repeatedly would add
    noticeable latency to every ingestion or search call."""
    settings = get_settings()
    return EmbeddingModel(settings.embedding_model_name)
