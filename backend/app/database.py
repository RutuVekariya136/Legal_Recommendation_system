"""Database engine + session management.

SQLite for local development (file-based, zero setup); the models are written so
moving to PostgreSQL is only a `DATABASE_URL` change (PROJECT_NOTES.md §4).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import get_settings
from app.models import Base  # registers all tables


def make_engine(url: str | None = None) -> Engine:
    settings = get_settings()
    url = url or settings.database_url_resolved
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, future=True, connect_args=connect_args)


engine: Engine = make_engine()
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False, future=True)


def init_db(target_engine: Engine | None = None) -> None:
    """Create all tables that don't exist yet."""
    Base.metadata.create_all(target_engine or engine)


def bind(target_engine: Engine) -> None:
    """Repoint the module-level session factory (used by tests)."""
    global engine
    engine = target_engine
    SessionLocal.configure(bind=target_engine)


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional scope for scripts / background work."""
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def get_db() -> Iterator[Session]:
    """FastAPI dependency: a request-scoped session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
