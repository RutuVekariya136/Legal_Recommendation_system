"""FastAPI routers. `all_routers` is mounted by app.main."""

from app.routers import cases, compare, evaluate, graph, health, recommend, search, stats, upload

all_routers = [health.router, stats.router, cases.router, search.router,
               recommend.router, compare.router, upload.router, graph.router,
               evaluate.router]

__all__ = ["all_routers"]
