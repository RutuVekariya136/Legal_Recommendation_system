"""Top-K hybrid recommendations for an indexed case."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.database import get_db
from app.ml.case_category import classify
from app.ml.case_parameters import augment_text, match_details, validate_details
from app.ml.recommendation import Filters
from app.schemas.models import RecommendationItem, RecommendDetailsRequest, RecommendResponse
from app.services import repository as repo
from app.services.recommender import RecommenderService, get_service

router = APIRouter(prefix="/recommend", tags=["recommend"])


def _summarize(results) -> dict:
    """Top match and average calibrated accuracy ("reliability") of a result list."""
    if not results:
        return {"top_doc": None, "top_score": None, "top_accuracy": None, "mean_accuracy": None}
    rel = [r.reliability for r in results if r.reliability is not None]
    top = results[0]
    return {"top_doc": top.doc_id, "top_score": top.hybrid_score, "top_accuracy": top.reliability,
            "mean_accuracy": round(sum(rel) / len(rel), 4) if rel else None}


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
        run_id=run_id, results=_to_items(result), precedent_summary=result.precedent_summary,
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
    body: RecommendDetailsRequest | None = None,
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

    # manually entered case details (place, weapon, time ...): added to the case
    # text before matching, so scores and accuracy genuinely change; the plain
    # run is kept alongside so the effect can be shown
    param_cat = category or doc.category or classify(doc.raw_text)
    details, detail_issues = validate_details(body.details if body else None, param_cat)
    text = augment_text(doc.raw_text, details, param_cat)
    result = svc.recommend_text(text, k=k, year=doc.year_hint, filters=filters,
                                category=category, category_query=category_query)
    base = None
    if details:
        base = svc.recommend_text(doc.raw_text, k=k, year=doc.year_hint, filters=filters,
                                  category=category, category_query=category_query)
    fallback_note = None
    if details and not result.results and base is not None and base.results:
        # the added details pushed every candidate below the similarity threshold:
        # fall back to the document alone rather than returning nothing
        result = base
        details_applied = {}
        fallback_note = ("Your additional details did not match any case above the similarity threshold, "
                         "so these results are based on your document alone.")
    else:
        details_applied = details
    comparison = None
    if details and base is not None:
        comparison = {"n_details": len(details_applied),
                      "before": _summarize(base.results), "after": _summarize(result.results)}
    items = _to_items(result)
    if details_applied:
        for it in items:
            it.detail_matches = match_details(svc.corpus.text(it.doc_id), details_applied)

    run_id = None
    if save:
        run = repo.record_run(
            db, query_type="upload", method=result.method, top_k=k,
            params={"year_min": year_min, "year_max": year_max, "min_score": min_score,
                    "filename": doc.filename, "category": category, "category_query": category_query,
                    "details": details_applied or None},
            results=result.results, query_upload_id=upload_id,
        )
        run_id = run.id

    return RecommendResponse(
        query=doc.filename, query_in_corpus=False, method=result.method, k=result.k,
        n_candidates=result.n_candidates, n_filtered_out=result.n_filtered_out,
        run_id=run_id, results=items, precedent_summary=result.precedent_summary,
        details_used=details_applied, detail_issues=detail_issues, comparison=comparison,
        fallback_note=fallback_note,
    )
