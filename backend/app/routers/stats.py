"""Corpus + model statistics."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.models import CorpusStats
from app.services import repository as repo
from app.services.recommender import RecommenderService, get_service

router = APIRouter(prefix="/stats", tags=["stats"])


@router.get("", response_model=CorpusStats)
def stats(db: Session = Depends(get_db),
          svc: RecommenderService = Depends(get_service)) -> CorpusStats:
    s = repo.case_stats(db)
    return CorpusStats(
        cases=s["cases"], with_title=s["with_title"],
        year_min=s["year_min"], year_max=s["year_max"],
        model={
            "text_method": svc.meta.text_method,
            "network_method": svc.meta.network_method,
            "hybrid_method": svc.meta.hybrid_method,
            "ann_backend": svc.meta.ann_backend,
            "rebuilt_on_start": svc.meta.rebuilt,
            "load_seconds": svc.meta.load_seconds,
        },
    )
