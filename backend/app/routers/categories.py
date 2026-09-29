"""Case-type category taxonomy, for the upload form's category picker."""

from __future__ import annotations

from fastapi import APIRouter

from app.ml.case_category import category_choices

router = APIRouter(prefix="/categories", tags=["categories"])


@router.get("")
def list_categories() -> list[dict]:
    return category_choices()
