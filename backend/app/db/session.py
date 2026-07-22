"""
Database engine and session management.

Why a generator-based get_db() dependency:
    FastAPI's dependency injection system expects a generator that yields
    a session and cleans up afterward (closes it, even if the request
    raised an exception). This pattern guarantees every request gets its
    own session and no connection leaks across requests -- important
    once we have concurrent agent calls hitting the DB mid-graph-run.
"""

from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings

settings = get_settings()

# pool_pre_ping avoids "MySQL server has gone away" errors from stale
# connections that timed out while idle -- common with MySQL + long-lived
# pools in a dev environment where the app sits idle between requests.
engine = create_engine(settings.mysql_url, pool_pre_ping=True, pool_recycle=3600)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a DB session, closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
