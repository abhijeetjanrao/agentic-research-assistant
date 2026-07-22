"""
CRUD helper functions.

Why a dedicated crud.py instead of querying the DB directly in routes/agents:
    Keeps SQLAlchemy query logic in one testable place. Routes (Module 8)
    and agents (Modules 4-7) call these functions instead of constructing
    queries inline, so a schema change only requires updating this file.
"""

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import (
    Document,
    DocumentStatus,
    Message,
    MessageRole,
    Report,
    ResearchSession,
    SessionStatus,
)


# --- ResearchSession ---

def create_session(db: Session, title: str) -> ResearchSession:
    session = ResearchSession(title=title, status=SessionStatus.ACTIVE)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session(db: Session, session_id: int) -> Optional[ResearchSession]:
    return db.get(ResearchSession, session_id)


def list_sessions(db: Session, limit: int = 50) -> List[ResearchSession]:
    stmt = select(ResearchSession).order_by(ResearchSession.created_at.desc()).limit(limit)
    return list(db.scalars(stmt))


def update_session_status(db: Session, session_id: int, status: SessionStatus) -> Optional[ResearchSession]:
    session = db.get(ResearchSession, session_id)
    if session is None:
        return None
    session.status = status
    db.commit()
    db.refresh(session)
    return session


# --- Message ---

def add_message(
    db: Session,
    session_id: int,
    role: MessageRole,
    content: str,
    agent_name: Optional[str] = None,
) -> Message:
    message = Message(session_id=session_id, role=role, content=content, agent_name=agent_name)
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def get_session_messages(db: Session, session_id: int) -> List[Message]:
    stmt = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
    )
    return list(db.scalars(stmt))


# --- Document ---

def create_document(db: Session, session_id: int, filename: str, file_path: str) -> Document:
    document = Document(
        session_id=session_id,
        filename=filename,
        file_path=file_path,
        status=DocumentStatus.UPLOADED,
    )
    db.add(document)
    db.commit()
    db.refresh(document)
    return document


def update_document_status(
    db: Session,
    document_id: int,
    status: DocumentStatus,
    num_chunks: Optional[int] = None,
    error_message: Optional[str] = None,
) -> Optional[Document]:
    document = db.get(Document, document_id)
    if document is None:
        return None
    document.status = status
    if num_chunks is not None:
        document.num_chunks = num_chunks
    if error_message is not None:
        document.error_message = error_message
    db.commit()
    db.refresh(document)
    return document


def get_session_documents(db: Session, session_id: int) -> List[Document]:
    stmt = select(Document).where(Document.session_id == session_id)
    return list(db.scalars(stmt))


# --- Report ---

def create_report(
    db: Session,
    session_id: int,
    title: str,
    content_markdown: str,
    citations_json: Optional[str] = None,
) -> Report:
    report = Report(
        session_id=session_id,
        title=title,
        content_markdown=content_markdown,
        citations_json=citations_json,
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    return report


def get_session_reports(db: Session, session_id: int) -> List[Report]:
    stmt = select(Report).where(Report.session_id == session_id)
    return list(db.scalars(stmt))
