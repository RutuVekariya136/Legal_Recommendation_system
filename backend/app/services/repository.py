"""Data-access helpers between the routers (Step 18) and the ORM.

Thin, dependency-light functions - no FastAPI imports here.
"""

from __future__ import annotations

import uuid
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Case, RecommendationResult, RecommendationRun, UploadedDocument


# --------------------------------------------------------------------------- #
# Cases                                                                       #
# --------------------------------------------------------------------------- #

def upsert_cases_from_corpus(db: Session, corpus) -> dict:
    """Insert/update one Case row per corpus document. Idempotent."""
    existing = {c.doc_id: c for c in db.scalars(select(Case)).all()}
    created = updated = 0
    for doc_id in corpus.doc_ids:
        doc = corpus[doc_id]
        text = corpus.text(doc_id)
        fields = dict(
            title=doc.title, year=doc.year, splits=list(doc.splits),
            has_title=bool(doc.title), char_len=len(text),
            source_path=str(doc.path) if getattr(doc, "path", None) else None,
            sha1=doc.sha1() if hasattr(doc, "sha1") else None,
        )
        row = existing.get(doc_id)
        if row is None:
            db.add(Case(doc_id=doc_id, **fields))
            created += 1
        else:
            for k, v in fields.items():
                setattr(row, k, v)
            updated += 1
    db.commit()
    return {"created": created, "updated": updated, "total": db.scalar(select(func.count(Case.id)))}


def get_case(db: Session, doc_id: str) -> Optional[Case]:
    return db.scalar(select(Case).where(Case.doc_id == doc_id))


def list_cases(db: Session, *, q: Optional[str] = None, year_min: Optional[int] = None,
               year_max: Optional[int] = None, limit: int = 50, offset: int = 0
               ) -> tuple[list[Case], int]:
    stmt = select(Case)
    if q:
        stmt = stmt.where(Case.title.ilike(f"%{q}%"))
    if year_min is not None:
        stmt = stmt.where(Case.year >= year_min)
    if year_max is not None:
        stmt = stmt.where(Case.year <= year_max)
    total = db.scalar(select(func.count()).select_from(stmt.subquery()))
    rows = db.scalars(stmt.order_by(Case.year.desc(), Case.doc_id).limit(limit).offset(offset)).all()
    return list(rows), int(total or 0)


def case_stats(db: Session) -> dict:
    total = db.scalar(select(func.count(Case.id))) or 0
    with_title = db.scalar(select(func.count(Case.id)).where(Case.has_title.is_(True))) or 0
    yr_min = db.scalar(select(func.min(Case.year)))
    yr_max = db.scalar(select(func.max(Case.year)))
    return {"cases": total, "with_title": with_title, "year_min": yr_min, "year_max": yr_max}


# --------------------------------------------------------------------------- #
# Uploads                                                                     #
# --------------------------------------------------------------------------- #

def create_upload(db: Session, *, filename: str, source_type: str, raw_text: str,
                  page_count: Optional[int] = None, needs_ocr: bool = False,
                  warnings: Optional[list] = None, year_hint: Optional[int] = None
                  ) -> UploadedDocument:
    doc = UploadedDocument(
        id=uuid.uuid4().hex, filename=filename, source_type=source_type,
        raw_text=raw_text, char_count=len(raw_text), page_count=page_count,
        needs_ocr=needs_ocr, warnings=list(warnings or []), year_hint=year_hint,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def get_upload(db: Session, upload_id: str) -> Optional[UploadedDocument]:
    return db.get(UploadedDocument, upload_id)


# --------------------------------------------------------------------------- #
# Recommendation runs                                                         #
# --------------------------------------------------------------------------- #

def record_run(db: Session, *, query_type: str, method: str, top_k: int,
               params: dict, results: list, query_case_doc_id: Optional[str] = None,
               query_upload_id: Optional[str] = None) -> RecommendationRun:
    """`results` is a list of app.ml.recommendation.Recommendation."""
    run = RecommendationRun(
        id=uuid.uuid4().hex, query_type=query_type, method=method, top_k=top_k,
        params=params, query_case_doc_id=query_case_doc_id, query_upload_id=query_upload_id,
    )
    for r in results:
        run.results.append(RecommendationResult(
            rank=r.rank, target_doc_id=r.doc_id, hybrid_score=r.hybrid_score,
            text_score=r.text_score, network_score=r.network_score,
            rationale=r.rationale, explanation=r.to_dict(),
        ))
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def get_run(db: Session, run_id: str) -> Optional[RecommendationRun]:
    return db.get(RecommendationRun, run_id)


def list_recent_runs(db: Session, limit: int = 20) -> list[RecommendationRun]:
    return list(db.scalars(
        select(RecommendationRun).order_by(RecommendationRun.created_at.desc()).limit(limit)
    ).all())
