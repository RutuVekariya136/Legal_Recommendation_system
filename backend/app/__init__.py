"""Backend application package for the Legal Case Similarity system.

Subpackages:
    ml/        pure ML logic (text sim, citation extraction, network model,
               hybrid combination, recommendation) — no FastAPI dependency.
    routers/   FastAPI route handlers (added from Step 18).
    services/  orchestration between routers/ and ml/.
    models/    SQLAlchemy models.
    schemas/   Pydantic request/response schemas.
    utils/     shared helpers.
"""

__version__ = "0.1.0"
