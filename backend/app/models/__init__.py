"""SQLAlchemy models. Importing this package registers every table on `Base`."""

from app.models.base import Base, TimestampMixin
from app.models.case import Case
from app.models.recommendation import RecommendationResult, RecommendationRun
from app.models.upload import UploadedDocument

__all__ = [
    "Base",
    "TimestampMixin",
    "Case",
    "UploadedDocument",
    "RecommendationRun",
    "RecommendationResult",
]
