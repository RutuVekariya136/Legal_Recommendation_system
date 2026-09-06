"""The indexed corpus: one row per unique judgment document."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import JSON, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class Case(Base):
    """One judgment. The 296 base-paper documents (`corpus="base"`) carry the
    expert-scored pairs; a bulk-ingested corpus (e.g. ILDC, `corpus="ildc"`) does
    not and is used only for search / recommendation."""

    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    doc_id: Mapped[str] = mapped_column(String(96), unique=True, index=True)
    corpus: Mapped[str] = mapped_column(String(32), default="base", index=True)
    title: Mapped[Optional[str]] = mapped_column(String(512))
    year: Mapped[Optional[int]] = mapped_column(Integer, index=True)
    splits: Mapped[list] = mapped_column(JSON, default=list)
    has_title: Mapped[bool] = mapped_column(default=False)
    char_len: Mapped[int] = mapped_column(Integer, default=0)
    source_path: Mapped[Optional[str]] = mapped_column(String(512))
    sha1: Mapped[Optional[str]] = mapped_column(String(40), index=True)
    text: Mapped[Optional[str]] = mapped_column(Text)  # full judgment text (scale mode)

    def __repr__(self) -> str:  # pragma: no cover
        return f"<Case {self.doc_id} ({self.year})>"
