"""
PDF ingestion: text extraction + chunking.

Chunking strategy rationale:
    A naive "split every N characters" chunker frequently cuts sentences
    or even words in half, which hurts embedding quality (the chunk no
    longer represents one coherent idea). We instead:
      1. Split text into sentences (simple heuristic split, no heavy NLP
         dependency needed for this).
      2. Greedily pack sentences into chunks up to `chunk_size` characters.
      3. Carry the last `overlap` characters of each chunk into the start
         of the next chunk, so a fact split across a chunk boundary is
         still fully present in at least one chunk. This overlap-window
         approach is a standard, well-documented RAG chunking pattern.
"""

import re
from dataclasses import dataclass
from typing import List

from pypdf import PdfReader

from app.logging_config import logger


@dataclass
class Chunk:
    text: str
    chunk_index: int
    page_number: int


def extract_text_by_page(pdf_path: str) -> List[str]:
    """Extract raw text per page. Kept separate from chunking so we can
    still report which page a chunk roughly came from -- useful for the
    Citation Agent later, which needs to point back to a source location."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append(text)
        if not text.strip():
            logger.warning(
                "Page {page_num} of {path} extracted no text (likely scanned/image-only)",
                page_num=i + 1,
                path=pdf_path,
            )
    return pages


from app.rag.vector_store import ChunkMetadata, get_vector_store

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return _SENTENCE_SPLIT_RE.split(text)


def chunk_pdf(
    pdf_path: str, chunk_size: int = 1000, overlap: int = 150
) -> List[Chunk]:
    """Extract and chunk a PDF into overlapping, sentence-aware chunks.

    Args:
        pdf_path: path to the PDF file.
        chunk_size: target max characters per chunk.
        overlap: characters of trailing context carried into the next chunk.
    """
    pages = extract_text_by_page(pdf_path)
    chunks: List[Chunk] = []
    chunk_index = 0

    for page_number, page_text in enumerate(pages, start=1):
        sentences = _split_sentences(page_text)
        if not sentences:
            continue

        current = ""
        for sentence in sentences:
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > chunk_size and current:
                chunks.append(Chunk(text=current, chunk_index=chunk_index, page_number=page_number))
                chunk_index += 1
                # carry trailing `overlap` chars forward as context
                current = current[-overlap:] + " " + sentence
            else:
                current = candidate

        if current:
            chunks.append(Chunk(text=current, chunk_index=chunk_index, page_number=page_number))
            chunk_index += 1

    logger.info(
        "Chunked {path} into {n} chunks across {pages} pages",
        path=pdf_path,
        n=len(chunks),
        pages=len(pages),
    )
    return chunks


def ingest_pdf(
    pdf_path: str,
    document_id: int,
    session_id: int,
    chunk_size: int = 1000,
    overlap: int = 150,
) -> int:
    """Full ingestion pipeline for one uploaded PDF: extract -> chunk ->
    embed -> add to the FAISS store. Returns the number of chunks added.

    This is the function the future document-upload API route (Module 8)
    and the Retriever Agent's ingestion step will call. Kept as one
    function (rather than scattering these steps across route handlers)
    so ingestion behaves identically regardless of caller.
    """
    from app.rag.embeddings import get_embedding_model

    chunks = chunk_pdf(pdf_path, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        logger.warning("No extractable text found in {path}", path=pdf_path)
        return 0

    model = get_embedding_model()
    vectors = model.embed_texts([c.text for c in chunks])

    metadata = [
        ChunkMetadata(
            document_id=document_id,
            session_id=session_id,
            chunk_index=c.chunk_index,
            page_number=c.page_number,
            text=c.text,
        )
        for c in chunks
    ]

    store = get_vector_store()
    store.add(vectors, metadata)
    store.save()

    return len(chunks)
