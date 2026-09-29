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

def upsert_cases_from_corpus(db: Session, corpus, *, corpus_name: str = "base",
                             store_text: bool = True) -> dict:
    """Insert/update one Case row per corpus document. Idempotent."""
    existing = {c.doc_id: c for c in db.scalars(select(Case)).all()}
    created = updated = 0
    for doc_id in corpus.doc_ids:
        doc = corpus[doc_id]
        text = corpus.text(doc_id)
        fields = dict(
            corpus=corpus_name, title=doc.title, year=doc.year, splits=list(doc.splits),
            has_title=bool(doc.title), char_len=len(text),
            source_path=str(doc.path) if getattr(doc, "path", None) else None,
            sha1=doc.sha1() if hasattr(doc, "sha1") else None,
            text=text if store_text else None,
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


def bulk_insert_documents(db: Session, rows: list[dict], *, corpus_name: str,
                          skip_existing: bool = True, batch: int = 1000) -> dict:
    """Insert many judgment rows. Each dict: doc_id, text, [title, year, source_path].
    Dedups on doc_id and on sha1."""
    import hashlib

    have_ids = {r[0] for r in db.execute(select(Case.doc_id)).all()}
    have_sha = {r[0] for r in db.execute(select(Case.sha1)).all() if r[0]}
    inserted = skipped = 0
    pending: list[Case] = []
    for r in rows:
        text = (r.get("text") or "").strip()
        doc_id = str(r["doc_id"]).strip()
        if not text or not doc_id:
            skipped += 1
            continue
        sha = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if skip_existing and (doc_id in have_ids or sha in have_sha):
            skipped += 1
            continue
        have_ids.add(doc_id)
        have_sha.add(sha)
        title = r.get("title")
        pending.append(Case(
            doc_id=doc_id, corpus=corpus_name, title=title,
            year=r.get("year"), splits=r.get("splits") or [],
            has_title=bool(title), char_len=len(text),
            source_path=r.get("source_path"), sha1=sha, text=text,
        ))
        if len(pending) >= batch:
            db.add_all(pending)
            db.commit()
            inserted += len(pending)
            pending = []
    if pending:
        db.add_all(pending)
        db.commit()
        inserted += len(pending)
    total = db.scalar(select(func.count(Case.id)))
    by_corpus = dict(db.execute(select(Case.corpus, func.count(Case.id)).group_by(Case.corpus)).all())
    return {"inserted": inserted, "skipped": skipped, "total": total, "by_corpus": by_corpus}


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


def facets(db: Session) -> dict:
    """Year-decade histogram + split counts over the whole corpus."""
    rows = db.execute(select(Case.year, Case.splits)).all()
    by_decade: dict[str, int] = {}
    splits: dict[str, int] = {}
    for year, sp in rows:
        if year:
            key = f"{(year // 10) * 10}s"
            by_decade[key] = by_decade.get(key, 0) + 1
        for s in (sp or []):
            splits[s] = splits.get(s, 0) + 1
    return {
        "total": len(rows),
        "years": dict(sorted(by_decade.items())),
        "splits": dict(sorted(splits.items())),
    }


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
                  warnings: Optional[list] = None, year_hint: Optional[int] = None,
                  category: Optional[str] = None, category_query: Optional[str] = None,
                  ) -> UploadedDocument:
    doc = UploadedDocument(
        id=uuid.uuid4().hex, filename=filename, source_type=source_type,
        raw_text=raw_text, char_count=len(raw_text), page_count=page_count,
        needs_ocr=needs_ocr, warnings=list(warnings or []), year_hint=year_hint,
        category=category, category_query=category_query,
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
