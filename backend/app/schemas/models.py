"""Response/request models. Deep explanation payloads are passed through as
`dict` (their shape is defined by the ML dataclasses' `to_dict()`)."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


# -- health / stats -------------------------------------------------

class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    env: str


class ReadyResponse(BaseModel):
    ready: bool
    text_method: Optional[str] = None
    network_method: Optional[str] = None
    hybrid_method: Optional[str] = None
    n_cases: Optional[int] = None
    rebuilt: list[str] = []
    load_seconds: Optional[float] = None


class CorpusStats(BaseModel):
    cases: int
    with_title: int
    year_min: Optional[int]
    year_max: Optional[int]
    model: dict[str, Any] = {}


# -- cases -------------------------------------------------------

class CaseSummary(BaseModel):
    doc_id: str
    title: Optional[str]
    year: Optional[int]
    splits: list[str] = []
    has_title: bool = False
    char_len: int = 0


class CaseDetail(CaseSummary):
    preview: str = ""


class CaseListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    items: list[CaseSummary]


# -- recommend -------------------------------------------------

class RecommendationItem(BaseModel):
    rank: int
    doc_id: str
    title: Optional[str]
    year: Optional[int]
    hybrid_score: float
    text_score: float
    network_score: float
    embedding_score: float
    statute_overlap: float
    citation_overlap: float
    shared_acts: list[dict] = []
    shared_sections: list[dict] = []
    shared_citations: list[str] = []
    cocitation_count: int = 0
    rationale: str = ""


class RecommendResponse(BaseModel):
    query: str
    query_in_corpus: bool
    method: str
    k: int
    n_candidates: int
    n_filtered_out: int
    run_id: Optional[str] = None
    results: list[RecommendationItem]


class RecommendTextRequest(BaseModel):
    text: str = Field(min_length=20)
    k: int = 10
    year: Optional[int] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    min_score: float = 0.0


# -- upload ----------------------------------------------------

class UploadResponse(BaseModel):
    id: str
    filename: str
    source_type: str
    char_count: int
    page_count: Optional[int] = None
    needs_ocr: bool = False
    warnings: list[str] = []
    year_hint: Optional[int] = None
    created_at: Optional[str] = None
    preview: str = ""


# -- search ----------------------------------------------------

class SearchHit(BaseModel):
    doc_id: str
    title: Optional[str]
    year: Optional[int]
    splits: list[str] = []
    score: float
    title_match: bool = False
    snippet: str = ""


class SearchResponse(BaseModel):
    query: str
    mode: str
    total: int
    limit: int
    offset: int
    items: list[SearchHit]
    facets: Optional[dict[str, Any]] = None


# -- compare / explain ------------------------------------

class ExplanationResponse(BaseModel):
    query_id: str
    target_id: str
    query_in_corpus: bool
    method: str
    hybrid_score: float
    text_score: float
    network_score: float
    embedding_score: float
    statute_overlap: float
    citation_overlap: float
    contributions: list[dict]
    shared_statutes: list[dict]
    shared_raw_citations: list[str]
    shared_cited_cases: list[dict]
    cocitation_count: int
    summary: str
