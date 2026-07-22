"""
Document upload and ingestion routes.
"""

import os
import shutil

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import crud
from app.db.models import DocumentStatus
from app.db.session import get_db
from app.logging_config import logger
from app.rag.ingestion import ingest_pdf
from app.schemas.schemas import DocumentResponse

router = APIRouter(tags=["documents"])


@router.post("/sessions/{session_id}/documents", response_model=DocumentResponse)
async def upload_document(session_id: int, file: UploadFile, db: Session = Depends(get_db)):
    settings = get_settings()

    session = crud.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    os.makedirs(settings.upload_dir, exist_ok=True)
    dest_path = os.path.join(settings.upload_dir, f"session_{session_id}_{file.filename}")

    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    document = crud.create_document(db, session_id, file.filename, dest_path)

    try:
        crud.update_document_status(db, document.id, DocumentStatus.INGESTING)
        num_chunks = ingest_pdf(dest_path, document_id=document.id, session_id=session_id)
        document = crud.update_document_status(
            db, document.id, DocumentStatus.INGESTED, num_chunks=num_chunks
        )
    except Exception as e:
        logger.error("Ingestion failed for document {doc_id}: {error}", doc_id=document.id, error=str(e))
        document = crud.update_document_status(
            db, document.id, DocumentStatus.FAILED, error_message=str(e)
        )

    return document


@router.get("/sessions/{session_id}/documents", response_model=list[DocumentResponse])
def list_documents(session_id: int, db: Session = Depends(get_db)):
    session = crud.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return crud.get_session_documents(db, session_id)
