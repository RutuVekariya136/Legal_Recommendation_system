"""FastAPI application factory.

    uvicorn app.main:app --reload --app-dir backend

Routes are mounted under /api. The ML pipeline (RecommenderService) is loaded
lazily on the first request that needs it; set WARM_ON_STARTUP=1 to load it
during startup instead.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from app.config import get_settings
from app.database import init_db
from app.routers import all_routers


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if os.environ.get("WARM_ON_STARTUP") == "1":
        from app.services.recommender import get_service

        get_service()
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        description="Hybrid (text + citation/statute network) legal case similarity. "
                    "Implements Bhattacharya et al. (2022) from scratch; not legal advice.",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for router in all_routers:
        app.include_router(router, prefix="/api")

    @app.get("/", include_in_schema=False)
    def _root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    return app


app = create_app()
