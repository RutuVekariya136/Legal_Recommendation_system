""""Is this case still good law?" - a heuristic citation-treatment signal.

A later case citing an earlier one sometimes says, in passing, whether that
precedent still stands ("no longer good law", "set aside") or was followed
("affirmed", "upheld"). This module classifies the text snippet already
captured around each resolved citation (``CitationEdge.context``, Step 7)
into a coarse status with simple keyword/regex rules - no ML model, same
convention as ``case_parameters.py``.

**Coverage caveat (do not drop this when using the module):** only a small
fraction of raw "A v B" mentions resolve to a document in this corpus (see
``citation_extraction.py`` module docstring). A case with no signals here
is *not* confirmed good law - it only means no later citing case was
resolved. Every consumer of ``TreatmentSummary`` must surface this caveat,
never silence-as-reassurance.

No FastAPI / DB dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from app.ml.citation_extraction import CitationEdge

NEGATIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\bno longer (?:good|valid) law\b",
        r"\boverrul(?:ed|ing)\b",
        r"\bset aside\b",
        r"\breversed\b",
        r"\boverturned\b",
        r"\bdoes not (?:lay down|reflect|state) (?:the )?correct law\b",
        r"\bincorrectly decided\b",
        r"\bnot good law\b",
    )
]
POSITIVE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\baffirm(?:ed|ing)\b",
        r"\bupheld\b",
        r"\bapprov(?:ed|ing)\b",
        r"\bfollow(?:ed|ing)\b",
        r"\breaffirm(?:ed|ing)\b",
    )
]
DISTINGUISHED_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\bdistinguish(?:ed|able)\b",
    )
]


def classify_signal(context: str) -> str:
    """Classify one citation's context snippet: "negative" | "distinguished"
    | "positive" | "neutral". Checked in that order so a mixed or ambiguous
    snippet never under-reports the most serious (negative) signal."""
    text = context or ""
    if any(p.search(text) for p in NEGATIVE_PATTERNS):
        return "negative"
    if any(p.search(text) for p in DISTINGUISHED_PATTERNS):
        return "distinguished"
    if any(p.search(text) for p in POSITIVE_PATTERNS):
        return "positive"
    return "neutral"


@dataclass(frozen=True)
class TreatmentSignal:
    citing_doc_id: str
    citing_year: Optional[int]
    status: str
    context: str


_STATUS_PRIORITY = ("negative", "distinguished", "positive", "neutral")


@dataclass
class TreatmentSummary:
    doc_id: str
    overall_status: str  # negative | distinguished | positive | neutral | no_data
    signals: list[TreatmentSignal] = field(default_factory=list)
    n_resolved_citations_checked: int = 0


def _overall_status(signals: list[TreatmentSignal]) -> str:
    statuses = {s.status for s in signals}
    for status in _STATUS_PRIORITY:
        if status in statuses:
            return status
    return "no_data"


def build_treatment_index(
    citation_edges: list[CitationEdge], corpus=None,
) -> dict[str, TreatmentSummary]:
    """Group resolved citation edges by the case they cite (``dst_doc_id``),
    classify each one's context, and aggregate into one TreatmentSummary per
    cited doc_id. A doc_id with no incoming resolved edges is simply absent;
    callers must treat absence as ``"no_data"``, not "confirmed clean"."""
    by_dst: dict[str, list[CitationEdge]] = {}
    for edge in citation_edges:
        by_dst.setdefault(edge.dst_doc_id, []).append(edge)

    index: dict[str, TreatmentSummary] = {}
    for doc_id, edges in by_dst.items():
        signals = [
            TreatmentSignal(
                citing_doc_id=e.src_doc_id,
                citing_year=corpus[e.src_doc_id].year if corpus is not None and e.src_doc_id in corpus else None,
                status=classify_signal(e.context),
                context=e.context,
            )
            for e in edges
        ]
        index[doc_id] = TreatmentSummary(
            doc_id=doc_id,
            overall_status=_overall_status(signals),
            signals=signals,
            n_resolved_citations_checked=len(signals),
        )
    return index
