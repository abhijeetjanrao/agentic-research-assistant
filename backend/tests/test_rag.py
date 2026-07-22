"""
Tests for Module 3: PDF ingestion, embeddings, and FAISS vector store.

These tests exercise the real pipeline against a real (small) generated
PDF and the real sentence-transformers model -- not mocks -- because the
whole point of this module is text extraction quality, chunk boundaries,
and retrieval relevance, none of which a mock would actually validate.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from app.rag.ingestion import chunk_pdf, extract_text_by_page, ingest_pdf
from app.rag.vector_store import ChunkMetadata, FaissVectorStore

FIXTURE_PDF = str(Path(__file__).parent / "fixtures" / "sample_paper.pdf")


def test_extract_text_by_page_finds_both_pages():
    pages = extract_text_by_page(FIXTURE_PDF)
    assert len(pages) == 2
    assert "retrieval" in pages[0].lower()
    assert "faiss" in pages[1].lower()


def test_chunk_pdf_produces_sentence_aware_chunks():
    chunks = chunk_pdf(FIXTURE_PDF, chunk_size=300, overlap=50)
    assert len(chunks) > 0
    # every chunk should end on a sentence boundary (period), not mid-word
    for chunk in chunks:
        assert chunk.text.strip()[-1] in ".!?" or chunk == chunks[-1]
    # page numbers should be tracked correctly
    page_numbers = {c.page_number for c in chunks}
    assert page_numbers == {1, 2}


def test_chunk_pdf_respects_size_roughly():
    chunks = chunk_pdf(FIXTURE_PDF, chunk_size=200, overlap=30)
    # allow some slack since we don't cut mid-sentence
    oversized = [c for c in chunks if len(c.text) > 400]
    assert not oversized, f"found chunks far exceeding chunk_size: {[len(c.text) for c in oversized]}"


@pytest.fixture()
def temp_vector_store():
    tmp_dir = tempfile.mkdtemp()
    from app.rag.embeddings import get_embedding_model

    model = get_embedding_model()
    store = FaissVectorStore(dimension=model.dimension, index_dir=tmp_dir)
    yield store
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_vector_store_add_and_search_semantic_relevance(temp_vector_store):
    from app.rag.embeddings import get_embedding_model

    model = get_embedding_model()
    texts = [
        "The cat sat on the mat and purred softly in the sun.",
        "FAISS is a library for efficient similarity search of vectors.",
        "Bananas are a good source of potassium and fiber.",
    ]
    vectors = model.embed_texts(texts)
    metadata = [
        ChunkMetadata(document_id=1, session_id=1, chunk_index=i, page_number=1, text=t)
        for i, t in enumerate(texts)
    ]
    temp_vector_store.add(vectors, metadata)

    query_vector = model.embed_query("What library can I use for vector similarity search?")
    results = temp_vector_store.search(query_vector, top_k=1)

    assert len(results) == 1
    assert "FAISS" in results[0].metadata.text


def test_vector_store_session_filtering(temp_vector_store):
    from app.rag.embeddings import get_embedding_model

    model = get_embedding_model()
    texts = ["Session one document about llamas.", "Session two document about llamas."]
    vectors = model.embed_texts(texts)
    metadata = [
        ChunkMetadata(document_id=1, session_id=1, chunk_index=0, page_number=1, text=texts[0]),
        ChunkMetadata(document_id=2, session_id=2, chunk_index=0, page_number=1, text=texts[1]),
    ]
    temp_vector_store.add(vectors, metadata)

    query_vector = model.embed_query("Tell me about llamas")
    results = temp_vector_store.search(query_vector, top_k=5, session_id=1)

    assert len(results) == 1
    assert results[0].metadata.session_id == 1


def test_vector_store_save_and_load_roundtrip(temp_vector_store):
    from app.rag.embeddings import get_embedding_model

    model = get_embedding_model()
    texts = ["Persisted chunk about mountains."]
    vectors = model.embed_texts(texts)
    metadata = [ChunkMetadata(document_id=1, session_id=1, chunk_index=0, page_number=1, text=texts[0])]
    temp_vector_store.add(vectors, metadata)
    temp_vector_store.save()

    reloaded = FaissVectorStore(dimension=model.dimension, index_dir=temp_vector_store._index_dir)
    found = reloaded.load()

    assert found is True
    assert reloaded.size == 1


def test_full_ingest_pdf_pipeline(monkeypatch, temp_vector_store):
    """End-to-end: real PDF -> real chunks -> real embeddings -> real
    FAISS index -> real semantic search returns the right chunk."""
    import app.rag.ingestion as ingestion_module

    monkeypatch.setattr(ingestion_module, "get_vector_store", lambda: temp_vector_store)

    num_chunks = ingest_pdf(FIXTURE_PDF, document_id=42, session_id=7)
    assert num_chunks > 0
    assert temp_vector_store.size == num_chunks

    from app.rag.embeddings import get_embedding_model

    model = get_embedding_model()
    query_vector = model.embed_query("How does FAISS support similarity search?")
    results = temp_vector_store.search(query_vector, top_k=1, session_id=7)

    assert len(results) == 1
    assert results[0].metadata.document_id == 42
    assert "faiss" in results[0].metadata.text.lower()
