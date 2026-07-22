"""
FAISS vector store wrapper.

Why we need a parallel metadata store:
    FAISS indexes only store vectors and return integer positions on
    search -- it has no concept of "which document/chunk was this."
    We maintain a Python list `self._metadata` where metadata[i]
    describes the vector at FAISS internal id i. This is the standard
    way to bridge FAISS back to your real data (here: document_id,
    chunk text, page number), and it's what lets retrieved chunks be
    cited back to a specific uploaded PDF stored in MySQL (Module 2).

Why IndexFlatIP instead of an approximate index (e.g. IVF, HNSW):
    IndexFlatIP does exact nearest-neighbor search via inner product.
    Since embeddings are normalized (see embeddings.py), inner product
    equals cosine similarity. Exact search is slower at very large scale
    but perfectly correct and simple -- appropriate for a project-scale
    corpus (hundreds to low thousands of chunks), and easy to explain and
    defend in an interview versus a tuned approximate index whose
    recall/speed tradeoffs would need justification.

Persistence:
    We save the FAISS index and metadata to disk under FAISS_INDEX_DIR
    so ingested documents survive process restarts -- the vector data
    doesn't live only in memory.
"""

import json
import os
import threading
from dataclasses import dataclass, asdict
from typing import List, Optional

import faiss
import numpy as np

from app.config import get_settings
from app.logging_config import logger


@dataclass
class ChunkMetadata:
    document_id: int
    session_id: int
    chunk_index: int
    page_number: int
    text: str


@dataclass
class SearchResult:
    metadata: ChunkMetadata
    score: float


class FaissVectorStore:
    """Wraps a single FAISS IndexFlatIP + its metadata sidecar list.

    Thread-safety: a lock guards add/search since FastAPI can serve
    concurrent requests and multiple agents could read/write the same
    session's index during a graph run.
    """

    def __init__(self, dimension: int, index_dir: str):
        self._dimension = dimension
        self._index_dir = index_dir
        self._index = faiss.IndexFlatIP(dimension)
        self._metadata: List[ChunkMetadata] = []
        self._lock = threading.Lock()
        os.makedirs(index_dir, exist_ok=True)

    @property
    def index_path(self) -> str:
        return os.path.join(self._index_dir, "index.faiss")

    @property
    def metadata_path(self) -> str:
        return os.path.join(self._index_dir, "metadata.json")

    def add(self, vectors: np.ndarray, metadata: List[ChunkMetadata]) -> None:
        if vectors.shape[0] != len(metadata):
            raise ValueError("vectors and metadata must have the same length")
        with self._lock:
            self._index.add(vectors.astype("float32"))
            self._metadata.extend(metadata)

    def search(
        self, query_vector: np.ndarray, top_k: int = 5, session_id: Optional[int] = None
    ) -> List[SearchResult]:
        """Search for the top_k most similar chunks. If session_id is
        given, results are filtered to that session -- since IndexFlatIP
        has no native filtering, we over-fetch and filter in Python,
        which is fine at this scale."""
        with self._lock:
            if self._index.ntotal == 0:
                return []
            fetch_k = min(self._index.ntotal, top_k * 5 if session_id else top_k)
            scores, indices = self._index.search(
                query_vector.astype("float32").reshape(1, -1), fetch_k
            )

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            if session_id is not None and meta.session_id != session_id:
                continue
            results.append(SearchResult(metadata=meta, score=float(score)))
            if len(results) >= top_k:
                break
        return results

    def save(self) -> None:
        with self._lock:
            faiss.write_index(self._index, self.index_path)
            with open(self.metadata_path, "w") as f:
                json.dump([asdict(m) for m in self._metadata], f)
        logger.info("Saved FAISS index ({n} vectors) to {path}", n=self._index.ntotal, path=self.index_path)

    def load(self) -> bool:
        """Returns True if an existing index was found and loaded, False
        if there was nothing on disk yet (fresh start)."""
        if not (os.path.exists(self.index_path) and os.path.exists(self.metadata_path)):
            return False
        with self._lock:
            self._index = faiss.read_index(self.index_path)
            with open(self.metadata_path) as f:
                raw = json.load(f)
            self._metadata = [ChunkMetadata(**m) for m in raw]
        logger.info("Loaded FAISS index ({n} vectors) from {path}", n=self._index.ntotal, path=self.index_path)
        return True

    @property
    def size(self) -> int:
        return self._index.ntotal


_store_instance: Optional[FaissVectorStore] = None
_store_lock = threading.Lock()


def get_vector_store() -> FaissVectorStore:
    """Process-wide singleton, lazily created and loaded from disk if a
    prior index exists. Using a module-level singleton (rather than
    re-instantiating per request) means we don't reload the whole index
    from disk on every API call."""
    global _store_instance
    if _store_instance is None:
        with _store_lock:
            if _store_instance is None:
                from app.rag.embeddings import get_embedding_model

                settings = get_settings()
                dimension = get_embedding_model().dimension
                store = FaissVectorStore(dimension, settings.faiss_index_dir)
                store.load()
                _store_instance = store
    return _store_instance
