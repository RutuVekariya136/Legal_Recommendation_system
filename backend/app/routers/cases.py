"""Browse the indexed corpus."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.models import CaseDetail, CaseListResponse, CaseSummary, TreatmentSummaryResponse
from app.services import repository as repo
from app.services.recommender import RecommenderService, get_service

router = APIRouter(prefix="/cases", tags=["cases"])


@router.get("", response_model=CaseListResponse)
def list_cases(
    q: str | None = Query(None, description="substring match on the case title"),
    year_min: int | None = None,
    year_max: int | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: Session = Depends(get_db),
) -> CaseListResponse:
    rows, total = repo.list_cases(db, q=q, year_min=year_min, year_max=year_max,
                                  limit=limit, offset=offset)
    return CaseListResponse(
        total=total, limit=limit, offset=offset,
        items=[CaseSummary(doc_id=c.doc_id, title=c.title, year=c.year,
                           splits=c.splits or [], has_title=c.has_title,
                           char_len=c.char_len) for c in rows],
    )


@router.get("/{doc_id}", response_model=CaseDetail)
def get_case(doc_id: str, db: Session = Depends(get_db),
             svc: RecommenderService = Depends(get_service)) -> CaseDetail:
    c = repo.get_case(db, doc_id)
    if c is None:
        raise HTTPException(404, f"case {doc_id!r} not found")
    preview = svc.corpus.text(doc_id)[:1200] if doc_id in svc.corpus else ""
    return CaseDetail(doc_id=c.doc_id, title=c.title, year=c.year,
                      splits=c.splits or [], has_title=c.has_title,
                      char_len=c.char_len, preview=preview)


@router.get("/{doc_id}/text")
def get_case_text(doc_id: str, svc: RecommenderService = Depends(get_service)) -> dict:
    if doc_id not in svc.corpus:
        raise HTTPException(404, f"case {doc_id!r} not found")
    return {"doc_id": doc_id, "text": svc.corpus.text(doc_id)}


@router.get("/{doc_id}/treatment", response_model=TreatmentSummaryResponse)
def get_case_treatment(doc_id: str, svc: RecommenderService = Depends(get_service)) -> TreatmentSummaryResponse:
    if doc_id not in svc.corpus:
        raise HTTPException(404, f"case {doc_id!r} not found")
    summary = svc.treatment_for(doc_id)
    if summary is None:
        return TreatmentSummaryResponse(doc_id=doc_id, overall_status="no_data")
    return TreatmentSummaryResponse(
        doc_id=summary.doc_id,
        overall_status=summary.overall_status,
        signals=[s.__dict__ for s in summary.signals],
        n_resolved_citations_checked=summary.n_resolved_citations_checked,
    )
