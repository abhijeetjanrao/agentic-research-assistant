"""
Tests for Module 1: config, logging, and app bootstrap.

These are intentionally simple -- they just prove the foundation loads
and the app is wired correctly. Later modules will test actual agent
logic and RAG behavior.
"""

import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def set_required_env(monkeypatch):
    """Provide fake required env vars so Settings() validates successfully
    without needing real API keys/credentials during tests."""
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.setenv("MYSQL_USER", "test-user")
    monkeypatch.setenv("MYSQL_PASSWORD", "test-pass")
    monkeypatch.setenv("MYSQL_DATABASE", "test-db")


def test_settings_load(set_required_env):
    from app.config import get_settings

    get_settings.cache_clear()  # ensure a fresh read picks up monkeypatched env
    settings = get_settings()

    assert settings.app_name == "Agentic Research Assistant"
    assert settings.mysql_url.startswith("mysql+pymysql://test-user:test-pass@")


def test_health_check(set_required_env):
    from app.config import get_settings

    get_settings.cache_clear()
    from app.main import app

    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
