"""Pairwise comparison / explanation for two indexed cases."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.models import ExplanationResponse
from app.services import repository as repo
from app.services.recommender import RecommenderService, get_service

router = APIRouter(prefix="/compare", tags=["compare"])


@router.get("", response_model=ExplanationResponse)
def compare(
    a: str = Query(..., description="query case doc_id"),
    b: str = Query(..., description="target case doc_id"),
    svc: RecommenderService = Depends(get_service),
) -> ExplanationResponse:
    for doc_id in (a, b):
        if doc_id not in svc.corpus:
            raise HTTPException(404, f"case {doc_id!r} not found")
    ex = svc.explain_pair(a, b)
    return ExplanationResponse(**ex.to_dict())


@router.get("/upload/{upload_id}", response_model=ExplanationResponse)
def compare_upload(
    upload_id: str,
    target: str = Query(..., description="corpus case doc_id to explain against"),
    db: Session = Depends(get_db),
    svc: RecommenderService = Depends(get_service),
) -> ExplanationResponse:
    doc = repo.get_upload(db, upload_id)
    if doc is None:
        raise HTTPException(404, f"upload {upload_id!r} not found")
    if target not in svc.corpus:
        raise HTTPException(404, f"case {target!r} not found")
    ex = svc.explain_text(doc.raw_text, target, year=doc.year_hint)
    return ExplanationResponse(**ex.to_dict())
