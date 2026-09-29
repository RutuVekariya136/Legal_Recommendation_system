"""User-uploaded case documents (the upload workflow, Step 21)."""

from __future__ import annotations

from typing import Optional

from sqlalchemy import JSON, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base, TimestampMixin


class UploadedDocument(Base, TimestampMixin):
    __tablename__ = "uploaded_documents"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)  # uuid4 hex
    filename: Mapped[str] = mapped_column(String(255))
    source_type: Mapped[str] = mapped_column(String(8))  # "pdf" | "txt"
    raw_text: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    page_count: Mapped[Optional[int]] = mapped_column(Integer)
    needs_ocr: Mapped[bool] = mapped_column(default=False)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    year_hint: Mapped[Optional[int]] = mapped_column(Integer)
    category: Mapped[Optional[str]] = mapped_column(String(64))  # predefined id, or "other"
    category_query: Mapped[Optional[str]] = mapped_column(String(256))  # free text if category="other"

    def __repr__(self) -> str:  # pragma: no cover
        return f"<UploadedDocument {self.id} {self.filename!r}>"
