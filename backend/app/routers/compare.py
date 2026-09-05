"""Pairwise comparison / explanation for two indexed cases."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.schemas.models import ExplanationResponse
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
