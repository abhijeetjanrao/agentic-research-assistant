"""
Shared pytest fixtures.

`real_mysql_db_session` provides a session against an actual running
MySQL-compatible server (not SQLite) for tests that specifically need to
prove real persistence works -- e.g. Module 7's report-saving test. Most
DB tests (test_db.py) intentionally use SQLite for speed; this fixture
is reserved for the few tests where "does this actually work against
real MySQL" is the point.

Requires a MySQL/MariaDB server reachable with the connection details
below. Tests using this fixture will fail with a clear connection error
if no such server is running -- see the Module 7 README notes for how to
stand one up locally.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db.models import Base

TEST_MYSQL_URL = "mysql+pymysql://research_user:root@localhost:3306/research_assistant_test"


@pytest.fixture()
def real_mysql_db_session():
    engine = create_engine(TEST_MYSQL_URL, pool_pre_ping=True)
    Base.metadata.create_all(engine)
    TestingSessionLocal = sessionmaker(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()
