"""Citation / statute-network neighbourhood of a case (for the graph view)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from app.services.recommender import RecommenderService, get_service

router = APIRouter(prefix="/graph", tags=["graph"])


@router.get("/case/{doc_id}")
def ego_graph(
    doc_id: str,
    similar: int = Query(6, ge=0, le=15),
    statutes: int = Query(8, ge=0, le=20),
    svc: RecommenderService = Depends(get_service),
) -> dict:
    try:
        return svc.ego_graph(doc_id, n_similar=similar, n_statutes=statutes)
    except KeyError:
        raise HTTPException(404, f"case {doc_id!r} not found")
