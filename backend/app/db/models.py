"""
SQLAlchemy ORM models.

Design rationale:
    ResearchSession is the anchor entity. A "session" represents one
    research task from start to finish (e.g. "compare transformer vs
    mamba architectures") and everything else -- messages, uploaded
    documents, and the final report -- belongs to exactly one session.
    This mirrors how the LangGraph manager agent will scope its state:
    one graph run = one session_id.

    We use SQLAlchemy 2.0's typed declarative style (Mapped[...] +
    mapped_column) instead of the legacy Column() style -- it gives us
    IDE autocomplete and static type checking on model attributes, and
    is the current idiomatic approach as of SQLAlchemy 2.x.
"""

import enum
from datetime import datetime
from typing import List, Optional

from sqlalchemy import Enum as SAEnum
from sqlalchemy import ForeignKey, String, Text, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Shared declarative base for all models."""
    pass


class SessionStatus(str, enum.Enum):
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    AGENT = "agent"  # intermediate agent output, e.g. planner's plan, retriever's findings


class DocumentStatus(str, enum.Enum):
    UPLOADED = "uploaded"
    INGESTING = "ingesting"
    INGESTED = "ingested"
    FAILED = "failed"


class ResearchSession(Base):
    """One end-to-end research task. Anchor entity for messages, documents, reports."""

    __tablename__ = "research_sessions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    title: Mapped[str] = mapped_column(String(255))
    status: Mapped[SessionStatus] = mapped_column(
        SAEnum(SessionStatus, values_callable=lambda enum_cls: [e.value for e in enum_cls]),
        default=SessionStatus.ACTIVE,
    )
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )

    messages: Mapped[List["Message"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    documents: Mapped[List["Document"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    reports: Mapped[List["Report"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class Message(Base):
    """A single turn in the session -- user input, final assistant answer,
    or an intermediate agent step (kept for transparency/debugging and to
    feed the Memory Agent)."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("research_sessions.id"))
    role: Mapped[MessageRole] = mapped_column(
        SAEnum(MessageRole, values_callable=lambda enum_cls: [e.value for e in enum_cls])
    )
    agent_name: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    session: Mapped["ResearchSession"] = relationship(back_populates="messages")


class Document(Base):
    """Metadata for an uploaded PDF. The actual embeddings live in FAISS
    (Module 3) -- this row just tracks what was uploaded and its
    ingestion status, so the UI can show progress and so we know which
    FAISS index entries correspond to which source file."""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("research_sessions.id"))
    filename: Mapped[str] = mapped_column(String(255))
    file_path: Mapped[str] = mapped_column(String(512))
    status: Mapped[DocumentStatus] = mapped_column(
        SAEnum(DocumentStatus, values_callable=lambda enum_cls: [e.value for e in enum_cls]),
        default=DocumentStatus.UPLOADED,
    )
    num_chunks: Mapped[Optional[int]] = mapped_column(nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    session: Mapped["ResearchSession"] = relationship(back_populates="documents")


class Report(Base):
    """A final generated research report with citations, produced by the
    Report Generator agent at the end of a session's graph run."""

    __tablename__ = "reports"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("research_sessions.id"))
    title: Mapped[str] = mapped_column(String(255))
    content_markdown: Mapped[str] = mapped_column(Text)
    citations_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    session: Mapped["ResearchSession"] = relationship(back_populates="reports")
