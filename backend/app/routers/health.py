"""Liveness / readiness."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.config import get_settings
from app.schemas.models import HealthResponse, ReadyResponse
from app.services.recommender import RecommenderService, get_service

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    s = get_settings()
    return HealthResponse(app=s.app_name, env=s.app_env)


@router.get("/health/ready", response_model=ReadyResponse)
def ready(svc: RecommenderService = Depends(get_service)) -> ReadyResponse:
    m = svc.meta
    return ReadyResponse(
        ready=True, text_method=m.text_method, network_method=m.network_method,
        hybrid_method=m.hybrid_method, n_cases=m.n_cases, rebuilt=m.rebuilt,
        load_seconds=m.load_seconds,
    )
