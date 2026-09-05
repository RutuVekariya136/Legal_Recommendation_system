"""Recommendation run history (useful for the demo UI and Step 22 evaluation)."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import JSON, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class RecommendationRun(Base, TimestampMixin):
    __tablename__ = "recommendation_runs"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # uuid4 hex
    query_type: Mapped[str] = mapped_column(String(8))             # "case" | "upload"
    query_case_doc_id: Mapped[Optional[str]] = mapped_column(String(64), index=True)
    query_upload_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("uploaded_documents.id", ondelete="SET NULL")
    )
    method: Mapped[str] = mapped_column(String(16))
    top_k: Mapped[int] = mapped_column(Integer, default=10)
    params: Mapped[dict] = mapped_column(JSON, default=dict)

    results: Mapped[list["RecommendationResult"]] = relationship(
        back_populates="run", cascade="all, delete-orphan",
        order_by="RecommendationResult.rank",
    )

    def __repr__(self) -> str:  # pragma: no cover
        return f"<RecommendationRun {self.id} {self.query_type}>"


class RecommendationResult(Base):
    __tablename__ = "recommendation_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("recommendation_runs.id", ondelete="CASCADE"), index=True
    )
    rank: Mapped[int] = mapped_column(Integer)
    target_doc_id: Mapped[str] = mapped_column(String(64), index=True)
    hybrid_score: Mapped[float] = mapped_column(Float)
    text_score: Mapped[float] = mapped_column(Float, default=0.0)
    network_score: Mapped[float] = mapped_column(Float, default=0.0)
    rationale: Mapped[Optional[str]] = mapped_column(String(1024))
    explanation: Mapped[Optional[dict]] = mapped_column(JSON)

    run: Mapped[RecommendationRun] = relationship(back_populates="results")
