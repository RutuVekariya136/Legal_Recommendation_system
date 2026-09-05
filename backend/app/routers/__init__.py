"""FastAPI routers. `all_routers` is mounted by app.main."""

from app.routers import cases, compare, health, recommend, stats

all_routers = [health.router, stats.router, cases.router, recommend.router, compare.router]

__all__ = ["all_routers"]
