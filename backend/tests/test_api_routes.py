"""
Tests for Module 8: FastAPI routes.

Uses a real in-memory SQLite DB (via dependency override of get_db) so
these are true integration tests of the route -> crud -> DB path, not
mocks all the way down. The research graph and PDF ingestion pipeline
ARE mocked, since exercising real Gemini/FAISS calls is already covered
by the agent-level tests in tests/test_module5-7_agents.py -- these
tests are specifically about routing, request/response shapes, and
error handling (404s, etc), not agent behavior.
"""

import io
import os
from unittest.mock import MagicMock, patch

import pytest

# Must be set before any `app.*` import -- app.db.session reads Settings
# at import time (module-level `settings = get_settings()`), which runs
# during test collection, before any fixture has a chance to monkeypatch.
os.environ.setdefault("GOOGLE_API_KEY", "test")
os.environ.setdefault("TAVILY_API_KEY", "test")
os.environ.setdefault("MYSQL_USER", "test")
os.environ.setdefault("MYSQL_PASSWORD", "test")
os.environ.setdefault("MYSQL_DATABASE", "test")

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.models import Base
from app.db.session import get_db


@pytest.fixture()
def client():
    from app.main import app

    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    yield TestClient(app)
    app.dependency_overrides.clear()


# --- Session routes ---

def test_create_session(client):
    response = client.post("/api/v1/sessions", json={"title": "Test Session"})
    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "Test Session"
    assert body["status"] == "active"
    assert "id" in body


def test_get_session_not_found(client):
    response = client.get("/api/v1/sessions/999")
    assert response.status_code == 404


def test_get_session_found(client):
    created = client.post("/api/v1/sessions", json={"title": "Findable"}).json()
    response = client.get(f"/api/v1/sessions/{created['id']}")
    assert response.status_code == 200
    assert response.json()["title"] == "Findable"


def test_list_sessions(client):
    client.post("/api/v1/sessions", json={"title": "One"})
    client.post("/api/v1/sessions", json={"title": "Two"})
    response = client.get("/api/v1/sessions")
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_create_session_rejects_empty_title(client):
    response = client.post("/api/v1/sessions", json={"title": ""})
    assert response.status_code == 422  # pydantic validation error


# --- Document routes ---

@patch("app.api.routes_documents.ingest_pdf")
def test_upload_document_success(mock_ingest_pdf, client, tmp_path):
    mock_ingest_pdf.return_value = 5  # pretend 5 chunks were created

    session = client.post("/api/v1/sessions", json={"title": "Doc test"}).json()

    fake_pdf_bytes = b"%PDF-1.4 fake content"
    response = client.post(
        f"/api/v1/sessions/{session['id']}/documents",
        files={"file": ("paper.pdf", io.BytesIO(fake_pdf_bytes), "application/pdf")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ingested"
    assert body["num_chunks"] == 5


@patch("app.api.routes_documents.ingest_pdf")
def test_upload_document_ingestion_failure_recorded(mock_ingest_pdf, client):
    mock_ingest_pdf.side_effect = Exception("corrupt PDF")

    session = client.post("/api/v1/sessions", json={"title": "Doc fail test"}).json()
    response = client.post(
        f"/api/v1/sessions/{session['id']}/documents",
        files={"file": ("bad.pdf", io.BytesIO(b"not a real pdf"), "application/pdf")},
    )

    assert response.status_code == 200  # request succeeds; failure is recorded on the document
    body = response.json()
    assert body["status"] == "failed"
    assert "corrupt PDF" in body["error_message"]


def test_upload_document_rejects_non_pdf(client):
    session = client.post("/api/v1/sessions", json={"title": "Reject test"}).json()
    response = client.post(
        f"/api/v1/sessions/{session['id']}/documents",
        files={"file": ("notes.txt", io.BytesIO(b"plain text"), "text/plain")},
    )
    assert response.status_code == 400


def test_upload_document_session_not_found(client):
    response = client.post(
        "/api/v1/sessions/999/documents",
        files={"file": ("paper.pdf", io.BytesIO(b"%PDF"), "application/pdf")},
    )
    assert response.status_code == 404


# --- Research (graph) route ---

@patch("app.api.routes_chat.get_research_graph")
def test_run_research_query_success(mock_get_graph, client):
    mock_graph = MagicMock()
    mock_graph.invoke.return_value = {
        "final_report": "# Research Report\n...",
        "citations": [{"index": 1, "source_type": "local_document"}],
        "gaps": [],
        "agent_trace": [{"agent_name": "manager_agent", "summary": "routed", "data": {}}],
    }
    mock_get_graph.return_value = mock_graph

    session = client.post("/api/v1/sessions", json={"title": "Research test"}).json()
    response = client.post(
        "/api/v1/research", json={"session_id": session["id"], "query": "What is RAG?"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["final_report"] == "# Research Report\n..."
    assert body["gaps"] == []
    mock_graph.invoke.assert_called_once()


def test_run_research_query_session_not_found(client):
    response = client.post("/api/v1/research", json={"session_id": 999, "query": "test"})
    assert response.status_code == 404


@patch("app.api.routes_chat.get_research_graph")
def test_run_research_query_graph_failure_returns_500(mock_get_graph, client):
    mock_graph = MagicMock()
    mock_graph.invoke.side_effect = Exception("graph exploded")
    mock_get_graph.return_value = mock_graph

    session = client.post("/api/v1/sessions", json={"title": "Failure test"}).json()
    response = client.post(
        "/api/v1/research", json={"session_id": session["id"], "query": "test"}
    )
    assert response.status_code == 500


# --- Report routes ---

def test_list_reports_empty(client):
    session = client.post("/api/v1/sessions", json={"title": "No reports yet"}).json()
    response = client.get(f"/api/v1/sessions/{session['id']}/reports")
    assert response.status_code == 200
    assert response.json() == []


def test_get_report_not_found(client):
    response = client.get("/api/v1/reports/999")
    assert response.status_code == 404
