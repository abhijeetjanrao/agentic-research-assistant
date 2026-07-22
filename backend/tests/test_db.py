"""
Tests for Module 2: DB models and CRUD functions.

We use an in-memory SQLite engine here instead of a real MySQL server.
This is standard practice for unit tests -- SQLAlchemy's ORM layer and
our CRUD functions are database-agnostic (we avoid MySQL-specific SQL),
so SQLite is a fast, dependency-free stand-in for testing logic. The
actual MySQL dialect/connection is exercised separately via Alembic
migrations against a real instance in staging/CI, not here.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db import crud
from app.db.models import Base, DocumentStatus, MessageRole, SessionStatus


@pytest.fixture()
def db() -> Session:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_create_and_get_session(db):
    session = crud.create_session(db, title="Transformers vs Mamba")
    assert session.id is not None
    assert session.status == SessionStatus.ACTIVE

    fetched = crud.get_session(db, session.id)
    assert fetched.title == "Transformers vs Mamba"


def test_update_session_status(db):
    session = crud.create_session(db, title="Test session")
    updated = crud.update_session_status(db, session.id, SessionStatus.COMPLETED)
    assert updated.status == SessionStatus.COMPLETED


def test_add_and_list_messages(db):
    session = crud.create_session(db, title="Test session")
    crud.add_message(db, session.id, MessageRole.USER, "What is RAG?")
    crud.add_message(
        db, session.id, MessageRole.AGENT, "Plan: retrieve, summarize, cite.",
        agent_name="planner_agent",
    )

    messages = crud.get_session_messages(db, session.id)
    assert len(messages) == 2
    assert messages[0].role == MessageRole.USER
    assert messages[1].agent_name == "planner_agent"


def test_document_lifecycle(db):
    session = crud.create_session(db, title="Test session")
    doc = crud.create_document(db, session.id, "paper.pdf", "/data/uploads/paper.pdf")
    assert doc.status == DocumentStatus.UPLOADED

    updated = crud.update_document_status(
        db, doc.id, DocumentStatus.INGESTED, num_chunks=42
    )
    assert updated.status == DocumentStatus.INGESTED
    assert updated.num_chunks == 42


def test_report_creation(db):
    session = crud.create_session(db, title="Test session")
    report = crud.create_report(
        db,
        session.id,
        title="RAG Literature Review",
        content_markdown="# Findings\n...",
        citations_json='[{"source": "arxiv.org/abs/1234"}]',
    )
    reports = crud.get_session_reports(db, session.id)
    assert len(reports) == 1
    assert reports[0].title == "RAG Literature Review"


def test_cascade_delete_removes_children(db):
    """Deleting a session should cascade-delete its messages/documents/reports
    -- important so orphaned rows don't accumulate as sessions get cleaned up."""
    session = crud.create_session(db, title="Test session")
    crud.add_message(db, session.id, MessageRole.USER, "hello")
    crud.create_document(db, session.id, "a.pdf", "/tmp/a.pdf")

    db.delete(session)
    db.commit()

    assert crud.get_session_messages(db, session.id) == []
    assert crud.get_session_documents(db, session.id) == []
