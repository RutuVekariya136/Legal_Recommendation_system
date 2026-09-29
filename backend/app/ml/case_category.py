"""Case-type category classification (keyword/statute based).

A coarse case-type taxonomy on top of the existing statute-extraction
pipeline: courts don't file judgments by "category", so this is a simple
keyword/Act-name scorer, not a learned classifier. Each category has a list
of case-insensitive signal phrases; a document's category is whichever one
has the most matches, provided at least ``MIN_SCORE`` phrases hit. Below
that, or with no signal, the document is left uncategorised (``None``).

Used two ways:
  * offline, to backfill ``Case.category`` for the indexed corpus
    (``scripts/classify_categories.py``);
  * online, to let a judge pick a category on upload and have the
    recommender restrict candidates to it (Step 21 extension).
"""

from __future__ import annotations

import re

MIN_SCORE = 1

CATEGORIES: list[tuple[str, str, list[str]]] = [
    ("cheque_bounce", "Cheque bounce (NI Act)", [
        "negotiable instruments act", "section 138", "cheque was dishonoured",
        "cheque bounce", "dishonour of cheque", "cheque bearing no",
    ]),
    ("motor_accident", "Motor accident claims", [
        "motor vehicles act", "motor accident claims tribunal", "m.a.c.t",
        "third party insurance", "compensation for the death", "accident claim",
    ]),
    ("criminal_appeal", "Criminal appeal", [
        "indian penal code", "code of criminal procedure", "cr.p.c", "crpc",
        "conviction", "acquittal", "sentence of imprisonment", "bail application",
        "murder", "dying declaration",
    ]),
    ("constitutional", "Constitutional / writ", [
        "constitution of india", "writ petition", "article 32", "article 226",
        "article 14", "article 21", "fundamental right", "public interest litigation",
    ]),
    ("tax", "Tax / revenue", [
        "income tax act", "income-tax act", "sales tax", "customs act",
        "central excise", "assessing officer", "assessee", "assessment order",
        "value added tax",
    ]),
    ("property_land", "Property / land", [
        "transfer of property act", "land acquisition act", "easement",
        "eviction", "tenancy", "specific performance", "partition suit",
        "possession of the suit property",
    ]),
    ("contract_commercial", "Contract / commercial", [
        "indian contract act", "sale of goods act", "companies act",
        "arbitration and conciliation act", "breach of contract",
        "arbitral award", "winding up",
    ]),
    ("service_labour", "Service / labour", [
        "industrial disputes act", "service matter", "departmental inquiry",
        "compulsory retirement", "termination of service", "workman",
        "disciplinary proceedings",
    ]),
    ("family_matrimonial", "Family / matrimonial", [
        "hindu marriage act", "divorce", "restitution of conjugal rights",
        "maintenance under section 125", "custody of the child", "special marriage act",
    ]),
]

_PATTERNS = {
    cat_id: [re.compile(re.escape(p), re.IGNORECASE) for p in phrases]
    for cat_id, _, phrases in CATEGORIES
}
LABELS = {cat_id: label for cat_id, label, _ in CATEGORIES}


def classify(text: str) -> str | None:
    """Return the best-matching category id, or None if nothing clears MIN_SCORE."""
    if not text:
        return None
    best_id, best_score = None, 0
    for cat_id, _, _ in CATEGORIES:
        score = sum(1 for pat in _PATTERNS[cat_id] if pat.search(text))
        if score > best_score:
            best_id, best_score = cat_id, score
    return best_id if best_score >= MIN_SCORE else None


def category_choices() -> list[dict]:
    """[{id, label}, ...] for the frontend's category picker, plus "other"."""
    return [{"id": cat_id, "label": label} for cat_id, label, _ in CATEGORIES] + [
        {"id": "other", "label": "Other"}
    ]
