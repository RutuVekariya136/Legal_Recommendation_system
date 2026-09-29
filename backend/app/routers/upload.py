"""Upload a case document (PDF / .txt), extract its text, persist it."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.ml.text_extraction import ExtractionError, extract_text
from app.schemas.models import UploadResponse
from app.services import repository as repo

router = APIRouter(prefix="/upload", tags=["upload"])

_ALLOWED_SUFFIXES = {".pdf", ".txt"}


def _to_response(doc, *, preview_len: int = 1200) -> UploadResponse:
    return UploadResponse(
        id=doc.id, filename=doc.filename, source_type=doc.source_type,
        char_count=doc.char_count, page_count=doc.page_count, needs_ocr=doc.needs_ocr,
        warnings=doc.warnings or [], year_hint=doc.year_hint,
        category=doc.category, category_query=doc.category_query,
        created_at=doc.created_at.isoformat() if doc.created_at else None,
        preview=doc.raw_text[:preview_len],
    )


@router.post("", response_model=UploadResponse, status_code=201)
async def upload_case(
    file: UploadFile = File(...),
    year: int | None = Form(None),
    category: str | None = Form(None, description="predefined case-type id, or 'other'"),
    category_query: str | None = Form(None, description="free text if category='other'"),
    db: Session = Depends(get_db),
) -> UploadResponse:
    settings = get_settings()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(422, f"unsupported file type {suffix or '(none)'!r}; "
                                 f"expected one of {sorted(_ALLOWED_SUFFIXES)}")
    data = await file.read()
    if len(data) > settings.upload_max_bytes:
        raise HTTPException(413, f"file exceeds {settings.upload_max_bytes} bytes")
    if not data:
        raise HTTPException(422, "empty file")
    if suffix == ".pdf" and data[:5] != b"%PDF-":
        raise HTTPException(422, "file is named .pdf but is not a PDF")

    try:
        extracted = extract_text(data, source_name=file.filename or "upload")
    except ExtractionError as exc:
        raise HTTPException(422, str(exc)) from exc

    # "other" isn't a real corpus category label - store it as a free-text
    # query instead so the recommender's category_query path handles it.
    if category == "other":
        category = None
    else:
        category_query = None

    doc = repo.create_upload(
        db, filename=file.filename or "upload", source_type=extracted.source_type,
        raw_text=extracted.raw_text, page_count=extracted.page_count,
        needs_ocr=extracted.needs_ocr, warnings=extracted.warnings, year_hint=year,
        category=category, category_query=category_query,
    )
    return _to_response(doc)


@router.get("/{upload_id}", response_model=UploadResponse)
def get_upload(upload_id: str, db: Session = Depends(get_db)) -> UploadResponse:
    doc = repo.get_upload(db, upload_id)
    if doc is None:
        raise HTTPException(404, f"upload {upload_id!r} not found")
    return _to_response(doc)


@router.get("/{upload_id}/text")
def get_upload_text(upload_id: str, db: Session = Depends(get_db)) -> dict:
    doc = repo.get_upload(db, upload_id)
    if doc is None:
        raise HTTPException(404, f"upload {upload_id!r} not found")
    return {"id": doc.id, "filename": doc.filename, "text": doc.raw_text}
