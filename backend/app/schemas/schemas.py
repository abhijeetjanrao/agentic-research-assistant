"""
Pydantic request/response schemas for the API layer.

Why these are kept separate from the SQLAlchemy models (app/db/models.py):
    ORM models describe the database; these describe the wire format.
    Conflating them is a common shortcut that breaks the moment you need
    a field in the API response that isn't a DB column (e.g. agent_trace
    below), or need to hide a DB column from clients. Keeping them
    separate costs a little duplication now and avoids that class of bug
    later.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


# --- Sessions ---

class SessionCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)


class SessionResponse(BaseModel):
    id: int
    title: str
    status: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)  # allows Session.model_validate(orm_instance)


# --- Chat / research query ---

class ResearchQueryRequest(BaseModel):
    session_id: int
    query: str = Field(..., min_length=1)


class ResearchQueryResponse(BaseModel):
    session_id: int
    final_report: Optional[str]
    citations: Optional[List[Dict[str, Any]]]
    gaps: Optional[List[str]]
    agent_trace: List[Dict[str, Any]]


# --- Documents ---

class DocumentResponse(BaseModel):
    id: int
    session_id: int
    filename: str
    status: str
    num_chunks: Optional[int]
    error_message: Optional[str]

    model_config = ConfigDict(from_attributes=True)


# --- Reports ---

class ReportResponse(BaseModel):
    id: int
    session_id: int
    title: str
    content_markdown: str
    citations_json: Optional[str]
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
