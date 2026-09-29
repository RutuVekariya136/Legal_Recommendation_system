"""Top-K hybrid recommendations for an indexed case."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.ml.recommendation import Filters
from app.schemas.models import RecommendationItem, RecommendResponse
from app.services import repository as repo
from app.services.recommender import RecommenderService, get_service

router = APIRouter(prefix="/recommend", tags=["recommend"])


def _to_items(result) -> list[RecommendationItem]:
    return [RecommendationItem(**r.to_dict()) for r in result.results]


@router.get("/case/{doc_id}", response_model=RecommendResponse)
def recommend_case(
    doc_id: str,
    k: int = Query(10, ge=1, le=50),
    year_min: int | None = None,
    year_max: int | None = None,
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    save: bool = Query(True, description="record this run in the history table"),
    db: Session = Depends(get_db),
    svc: RecommenderService = Depends(get_service),
) -> RecommendResponse:
    if doc_id not in svc.corpus:
        raise HTTPException(404, f"case {doc_id!r} not found")
    filters = Filters(year_min=year_min, year_max=year_max, min_hybrid_score=min_score)
    result = svc.recommend_case(doc_id, k=k, filters=filters)

    run_id = None
    if save:
        run = repo.record_run(
            db, query_type="case", method=result.method, top_k=k,
            params={"year_min": year_min, "year_max": year_max, "min_score": min_score},
            results=result.results, query_case_doc_id=doc_id,
        )
        run_id = run.id

    return RecommendResponse(
        query=result.query, query_in_corpus=result.query_in_corpus, method=result.method,
        k=result.k, n_candidates=result.n_candidates, n_filtered_out=result.n_filtered_out,
        run_id=run_id, results=_to_items(result),
    )


@router.post("/upload/{upload_id}", response_model=RecommendResponse)
def recommend_upload(
    upload_id: str,
    k: int = Query(10, ge=1, le=50),
    year_min: int | None = None,
    year_max: int | None = None,
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    category: str | None = Query(None, description="restrict to this predefined case category"),
    category_query: str | None = Query(None, description="restrict to cases whose text contains this phrase (used when category='other')"),
    save: bool = Query(True),
    db: Session = Depends(get_db),
    svc: RecommenderService = Depends(get_service),
) -> RecommendResponse:
    doc = repo.get_upload(db, upload_id)
    if doc is None:
        raise HTTPException(404, f"upload {upload_id!r} not found")
    # default to the category chosen at upload time; the caller can still
    # override (e.g. pass category="" ... ) or the frontend can pass explicit
    # values to change/clear it without re-uploading.
    if category is None and category_query is None:
        category, category_query = doc.category, doc.category_query
    filters = Filters(year_min=year_min, year_max=year_max, min_hybrid_score=min_score)
    result = svc.recommend_text(doc.raw_text, k=k, year=doc.year_hint, filters=filters,
                                category=category, category_query=category_query)

    run_id = None
    if save:
        run = repo.record_run(
            db, query_type="upload", method=result.method, top_k=k,
            params={"year_min": year_min, "year_max": year_max, "min_score": min_score,
                    "filename": doc.filename, "category": category, "category_query": category_query},
            results=result.results, query_upload_id=upload_id,
        )
        run_id = run.id

    return RecommendResponse(
        query=doc.filename, query_in_corpus=False, method=result.method, k=result.k,
        n_candidates=result.n_candidates, n_filtered_out=result.n_filtered_out,
        run_id=run_id, results=_to_items(result),
    )
