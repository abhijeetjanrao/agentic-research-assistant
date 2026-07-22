"""
Chat / research routes.

Design decision: the graph is invoked synchronously within the request
for this project's scope. A production system with longer-running
research sessions (multiple gap-retries, large document sets) would want
this as a background task with polling or websockets -- noted here as a
known scaling limitation rather than silently pretending this is
production-final. For a portfolio-scale project (single user, bounded
gap retries, small corpora), synchronous is simpler and still responsive.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import crud
from app.db.models import MessageRole
from app.db.session import get_db
from app.graph.research_graph import get_research_graph
from app.graph.state import create_initial_state
from app.logging_config import logger
from app.schemas.schemas import (
    ResearchQueryRequest,
    ResearchQueryResponse,
    SessionCreateRequest,
    SessionResponse,
)

router = APIRouter(tags=["chat"])


@router.post("/sessions", response_model=SessionResponse)
def create_session(payload: SessionCreateRequest, db: Session = Depends(get_db)):
    session = crud.create_session(db, title=payload.title)
    return session


@router.get("/sessions/{session_id}", response_model=SessionResponse)
def get_session(session_id: int, db: Session = Depends(get_db)):
    session = crud.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.get("/sessions", response_model=list[SessionResponse])
def list_sessions(db: Session = Depends(get_db)):
    return crud.list_sessions(db)


@router.post("/research", response_model=ResearchQueryResponse)
def run_research_query(payload: ResearchQueryRequest, db: Session = Depends(get_db)):
    session = crud.get_session(db, payload.session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")

    crud.add_message(db, payload.session_id, MessageRole.USER, payload.query)

    logger.info(
        "Running research graph for session {sid}: {q}",
        sid=payload.session_id,
        q=payload.query,
    )

    graph = get_research_graph()
    initial_state = create_initial_state(session_id=payload.session_id, query=payload.query)

    try:
        final_state = graph.invoke(initial_state, config={"recursion_limit": 100})
    except Exception as e:
        logger.error("Research graph failed for session {sid}: {error}", sid=payload.session_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Research graph failed: {e}")

    if final_state.get("final_report"):
        crud.add_message(
            db,
            payload.session_id,
            MessageRole.ASSISTANT,
            final_state["final_report"],
            agent_name="report_generator_agent",
        )

    return ResearchQueryResponse(
        session_id=payload.session_id,
        final_report=final_state.get("final_report"),
        citations=final_state.get("citations"),
        gaps=final_state.get("gaps"),
        agent_trace=final_state.get("agent_trace", []),
    )
