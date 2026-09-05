"""Search the indexed corpus (title match + semantic content search)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.models import SearchResponse
from app.services import repository as repo
from app.services.recommender import RecommenderService, get_service

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchResponse)
def search(
    q: str = Query(..., min_length=1, description="search query"),
    mode: str = Query("semantic", pattern="^(semantic|title|hybrid)$"),
    year_min: int | None = None,
    year_max: int | None = None,
    split: str | None = Query(None, pattern="^(validation|test)$"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    with_facets: bool = Query(False, description="include corpus year/split facets"),
    db: Session = Depends(get_db),
    svc: RecommenderService = Depends(get_service),
) -> SearchResponse:
    result = svc.search(q, mode=mode, year_min=year_min, year_max=year_max,
                        split=split, limit=limit, offset=offset)
    if with_facets or offset == 0:
        result["facets"] = repo.facets(db)
    return SearchResponse(**result)


@router.get("/facets")
def facets(db: Session = Depends(get_db)) -> dict:
    return repo.facets(db)
