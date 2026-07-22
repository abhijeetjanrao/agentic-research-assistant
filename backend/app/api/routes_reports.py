"""
Report retrieval routes.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import crud
from app.db.models import Report
from app.db.session import get_db
from app.schemas.schemas import ReportResponse

router = APIRouter(tags=["reports"])


@router.get("/sessions/{session_id}/reports", response_model=list[ReportResponse])
def list_reports(session_id: int, db: Session = Depends(get_db)):
    session = crud.get_session(db, session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return crud.get_session_reports(db, session_id)


@router.get("/reports/{report_id}", response_model=ReportResponse)
def get_report(report_id: int, db: Session = Depends(get_db)):
    # A by-id-only lookup (no session_id needed since report_id is
    # globally unique) -- inlined here rather than added to crud.py since
    # it's the only place that needs this exact query shape. Uses the
    # same Depends(get_db) pattern as every other route, rather than
    # opening its own SessionLocal, so it shares the request's connection
    # and is consistently testable via dependency override.
    report = db.get(Report, report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found")
    return report
