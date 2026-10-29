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
    searchable_cases: int
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


# -- citation treatment ("is this case still good law?") --------

TREATMENT_COVERAGE_NOTE = (
    "Based only on citations we could automatically resolve to a document in "
    "this corpus (about 1.2% of raw citation mentions). No warning shown does "
    "not confirm this case is still good law - it may simply mean no later "
    "citing case was resolved."
)


class TreatmentSignal(BaseModel):
    citing_doc_id: str
    citing_year: Optional[int] = None
    status: str
    context: str


class TreatmentSummaryResponse(BaseModel):
    doc_id: str
    overall_status: str
    signals: list[TreatmentSignal] = []
    n_resolved_citations_checked: int = 0
    coverage_note: str = TREATMENT_COVERAGE_NOTE


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
    reliability: Optional[float] = None
    reliability_n: Optional[int] = None
    detail_matches: dict[str, bool] = {}
    exact_match: bool = False
    treatment_status: Optional[str] = None


class RecommendResponse(BaseModel):
    query: str
    query_in_corpus: bool
    method: str
    k: int
    n_candidates: int
    n_filtered_out: int
    run_id: Optional[str] = None
    results: list[RecommendationItem]
    precedent_summary: list[dict] = []
    details_used: dict[str, str] = {}
    detail_issues: dict[str, str] = {}
    fallback_note: Optional[str] = None
    comparison: Optional[dict] = None


class RecommendDetailsRequest(BaseModel):
    details: dict[str, str] = {}


class ParameterListResponse(BaseModel):
    category: Optional[str] = None
    category_label: Optional[str] = None
    category_inferred: bool = False
    coverage_sample: int = 0
    parameters: list[dict] = []


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
    category: Optional[str] = None
    category_query: Optional[str] = None
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
    shared_terms: list[str] = []
    reliability: Optional[float] = None
    reliability_n: Optional[int] = None
    summary: str
